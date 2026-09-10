"""
batch_eval.py
--------------
Runs the VishingGuard-ZSL LLMEngine over the KorCCViD development split
(dev_m16.csv) and reports accuracy, per-tier breakdown, router distribution,
and latency stats. This is the script behind the "tested on 20 transcripts"
/ "prompt produces valid JSON for 90%+" success-checklist items -- point it
at the full dev split (or a --limit subset) once your API key is set.

CHANGE FROM ORIGINAL: now also writes the 7 raw per-attribute scores
(authority_score, urgency_score, etc.) alongside fused_risk/llm_final_risk.
This lets fusion weights be re-tuned OFFLINE against captured LLM output
(see offline_weight_search.py) instead of re-calling the (rate-limited,
non-deterministic) LLM every time you want to test a new weight set.

Expected CSV columns (from dev_m16.csv): id, transcript, label, source, category
  - label: '0' = benign, '1' = vishing (adjust LABEL_VISHING_VALUE below if
    your actual encoding differs -- the script prints a warning if it never
    sees more than one distinct label value, which usually means a mismatch).

Usage:
    python batch_eval.py dev_m16.csv
    python batch_eval.py dev_m16.csv --limit 20          # quick smoke test
    python batch_eval.py dev_m16.csv --limit 20 --seed 7 # different random subset
    python batch_eval.py dev_m16.csv --out results.csv

Respects Groq's free-tier rate limit (30 requests/minute) with a small
delay between calls, and writes results incrementally so a Ctrl+C or a
crash partway through doesn't lose progress already made.
"""

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

from llm_engine import LLMEngine
from prompt_framework import REQUIRED_ATTRIBUTES

LABEL_VISHING_VALUE = "1"  # adjust if your dataset encodes it differently

# Groq's free tier for the current GPT-OSS models is 30 RPM but only ~8,000
# tokens/minute combined input+output (see https://console.groq.com/docs/rate-limits).
# Long transcripts (some KorCCViD entries run to 1,000+ tokens) mean the token
# budget, not the request-count budget, is usually the real bottleneck. A few
# seconds between calls keeps you comfortably under 30 RPM; llm_engine's own
# backoff handles the rest if you still hit a 429.
SECONDS_BETWEEN_CALLS = 4.0


def load_rows(csv_path: str):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    missing = {"id", "transcript", "label"} - set(rows[0].keys())
    if missing:
        raise ValueError(
            f"Expected columns 'id', 'transcript', 'label' but this file's columns are "
            f"{list(rows[0].keys())}. Missing: {missing}"
        )
    return rows


def is_true_vishing(label_value: str) -> bool:
    return str(label_value).strip() == LABEL_VISHING_VALUE


def run(csv_path: str, limit: int, seed: int, out_path: str):
    rows = load_rows(csv_path)
    print(f"Loaded {len(rows)} rows from {csv_path}")

    distinct_labels = {r["label"] for r in rows}
    print(f"Distinct label values seen: {sorted(distinct_labels)}")
    if len(distinct_labels) == 1:
        print("WARNING: only one distinct label value found -- check the CSV / column mapping.")

    if limit and limit < len(rows):
        random.seed(seed)
        rows = random.sample(rows, limit)
        print(f"Sampling {limit} rows (seed={seed}) for this run.")

    engine = LLMEngine()

    attr_fieldnames = [f"{attr}_score" for attr in REQUIRED_ATTRIBUTES]

    fieldnames = [
        "id", "true_label", "true_is_vishing", "predicted_tier", "predicted_is_vishing",
        "fused_risk", "llm_final_risk", "model_used", "retries", "fallback_triggered",
        "latency_seconds", "error", "correct",
    ] + attr_fieldnames

    out_file = Path(out_path)
    write_header = not out_file.exists()
    csv_out = open(out_file, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(csv_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    n = len(rows)
    correct = 0
    scored = 0
    tier_counts = {"Low": 0, "Moderate": 0, "High": 0}
    model_counts = {}
    latencies = []
    errors = 0

    start_time = time.time()
    for i, row in enumerate(rows, start=1):
        transcript = row["transcript"]
        true_vishing = is_true_vishing(row["label"])

        result = engine.analyze_transcript(transcript)
        meta = result.get("_meta", {})

        tier = result.get("risk_tier", "Low")
        predicted_vishing = tier in ("Moderate", "High")
        is_correct = predicted_vishing == true_vishing

        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        model_counts[meta.get("model_used", "unknown")] = model_counts.get(meta.get("model_used", "unknown"), 0) + 1
        latencies.append(meta.get("latency_seconds", 0.0))
        if meta.get("error"):
            errors += 1
        else:
            scored += 1
            if is_correct:
                correct += 1

        row_out = {
            "id": row["id"],
            "true_label": row["label"],
            "true_is_vishing": true_vishing,
            "predicted_tier": tier,
            "predicted_is_vishing": predicted_vishing,
            "fused_risk": result.get("fused_risk"),
            "llm_final_risk": result.get("final_risk"),
            "model_used": meta.get("model_used"),
            "retries": meta.get("retries"),
            "fallback_triggered": meta.get("fallback_triggered"),
            "latency_seconds": meta.get("latency_seconds"),
            "error": meta.get("error"),
            "correct": is_correct,
        }
        for attr in REQUIRED_ATTRIBUTES:
            node = result.get(attr, {})
            row_out[f"{attr}_score"] = node.get("score") if isinstance(node, dict) else None

        writer.writerow(row_out)
        csv_out.flush()  # write progress immediately, don't lose it on interruption

        elapsed = time.time() - start_time
        eta = (elapsed / i) * (n - i)
        print(f"[{i}/{n}] id={row['id']} true={'VISHING' if true_vishing else 'benign':7s} "
              f"pred={tier:8s} model={meta.get('model_used')} "
              f"latency={meta.get('latency_seconds', 0):.2f}s "
              f"({'OK' if is_correct else 'WRONG' if not meta.get('error') else 'ERROR'}) "
              f"| ETA {eta/60:.1f} min", flush=True)

        if i < n:
            time.sleep(SECONDS_BETWEEN_CALLS)

    csv_out.close()

    print("\n" + "=" * 60)
    print(f"Done. Results appended to: {out_path}")
    print(f"Total transcripts: {n}  |  Scored successfully: {scored}  |  Hard errors: {errors}")
    if scored:
        print(f"Accuracy (risk_tier != Low  ==  vishing) on scored rows: {correct}/{scored} = {100*correct/scored:.1f}%")
    print(f"Risk tier distribution: {tier_counts}")
    print(f"Model routing distribution: {model_counts}")
    if model_counts:
        total_routed = sum(model_counts.values())
        for model, count in model_counts.items():
            print(f"  {model}: {100*count/total_routed:.1f}%  (target: primary ~80-90%, fallback ~10-20%)")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Average latency: {avg_latency:.2f}s  (target: under 2s)")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-evaluate VishingGuard-ZSL on a CSV of transcripts.")
    parser.add_argument("csv_path", help="Path to dev_m16.csv (or any CSV with id/transcript/label columns)")
    parser.add_argument("--limit", type=int, default=0, help="Randomly sample this many rows instead of running the whole file (0 = run all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --limit sampling")
    parser.add_argument("--out", default="batch_eval_results.csv", help="Output CSV path (appended to if it already exists)")
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    run(args.csv_path, args.limit, args.seed, args.out)
