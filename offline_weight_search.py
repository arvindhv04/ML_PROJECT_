"""
offline_weight_search.py
-------------------------
Tests candidate fusion_weights / risk_tier thresholds against ALREADY
CAPTURED LLM attribute scores (from batch_eval.py's per-attribute columns),
with zero additional API calls. This is the fast inner loop for tuning
fusion weights: run batch_eval.py once to build up a results CSV with
enough rows, then iterate on weights here in milliseconds instead of
re-hitting the (rate-limited, non-deterministic) LLM every time.

Requires batch_eval_results.csv rows written by the UPDATED batch_eval.py
that includes the 7 *_score columns (authority_score, urgency_score, etc).
Rows with a logged error, or missing attribute scores, are skipped -- they
reflect LLM/pipeline failures, not fusion-weight behavior, and mixing them
in would make weight comparisons noisy.

Usage:
    python offline_weight_search.py batch_eval_results.csv
    python offline_weight_search.py batch_eval_results.csv --grid-search
"""

import argparse
import csv
import sys
from pathlib import Path

ATTRIBUTES = [
    "information_sensitivity", "urgency", "authority",
    "threat", "reward", "scarcity", "social_proof",
]

# Current production weights (config.py), for baseline comparison.
CURRENT_WEIGHTS = {
    "information_sensitivity": 0.25,
    "urgency": 0.25,
    "authority": 0.15,
    "threat": 0.15,
    "reward": 0.07,
    "scarcity": 0.07,
    "social_proof": 0.06,
}

# A few hand-picked alternatives worth comparing directly. Add your own here.
CANDIDATE_WEIGHTS = {
    "current": CURRENT_WEIGHTS,
    "boosted_authority": {
        "information_sensitivity": 0.25,
        "urgency": 0.15,
        "authority": 0.25,
        "threat": 0.15,
        "reward": 0.06,
        "scarcity": 0.06,
        "social_proof": 0.08,
    },
    "authority_info_dominant": {
        "information_sensitivity": 0.30,
        "urgency": 0.15,
        "authority": 0.25,
        "threat": 0.15,
        "reward": 0.05,
        "scarcity": 0.05,
        "social_proof": 0.05,
    },
}

DEFAULT_MODERATE_THRESHOLD = 0.4
DEFAULT_HIGH_THRESHOLD = 0.7


def load_scored_rows(csv_path: str):
    """Load rows that have usable attribute scores (skip hard-failed calls)."""
    rows = []
    skipped = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("error"):
                skipped += 1
                continue
            scores = {}
            ok = True
            for attr in ATTRIBUTES:
                col = f"{attr}_score"
                val = r.get(col)
                if val is None or val == "":
                    ok = False
                    break
                try:
                    scores[attr] = float(val)
                except ValueError:
                    ok = False
                    break
            if not ok:
                skipped += 1
                continue
            rows.append({
                "id": r["id"],
                "true_is_vishing": str(r.get("true_is_vishing", "")).strip().lower() == "true",
                "scores": scores,
            })
    if skipped:
        print(f"Skipped {skipped} rows (errored calls or missing attribute scores).")
    return rows


def fuse(weights: dict, scores: dict) -> float:
    total = sum(scores.get(attr, 0.0) * w for attr, w in weights.items())
    return round(min(max(total, 0.0), 1.0), 4)


def tier(risk: float, moderate_thresh: float, high_thresh: float) -> str:
    if risk > high_thresh:
        return "High"
    if risk >= moderate_thresh:
        return "Moderate"
    return "Low"


def evaluate(weights, rows, moderate_thresh, high_thresh):
    tp = fp = tn = fn = 0
    fn_ids = []
    fp_ids = []
    for r in rows:
        risk = fuse(weights, r["scores"])
        predicted_vishing = tier(risk, moderate_thresh, high_thresh) != "Low"
        true_vishing = r["true_is_vishing"]
        if predicted_vishing and true_vishing:
            tp += 1
        elif predicted_vishing and not true_vishing:
            fp += 1
            fp_ids.append(r["id"])
        elif not predicted_vishing and not true_vishing:
            tn += 1
        else:
            fn += 1
            fn_ids.append(r["id"])

    total = tp + fp + tn + fn
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    accuracy = (tp + tn) / total if total else float("nan")
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")

    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "recall": recall, "precision": precision,
        "accuracy": accuracy, "fpr": fpr,
        "fn_ids": fn_ids, "fp_ids": fp_ids,
    }


def print_report(name, weights, metrics, moderate_thresh, high_thresh):
    print(f"\n--- {name} ---")
    print(f"weights: {weights}")
    print(f"thresholds: moderate>={moderate_thresh}, high>{high_thresh}")
    print(f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(f"Recall (catch rate on true vishing):  {metrics['recall']*100:.1f}%")
    print(f"Precision (of flagged, how many real): {metrics['precision']*100:.1f}%")
    print(f"False positive rate (benign flagged):  {metrics['fpr']*100:.1f}%")
    print(f"Overall accuracy:                      {metrics['accuracy']*100:.1f}%")
    if metrics["fn_ids"]:
        print(f"Missed vishing (false negatives): {metrics['fn_ids']}")
    if metrics["fp_ids"]:
        print(f"Benign flagged (false positives):  {metrics['fp_ids']}")


def grid_search(rows, moderate_thresh, high_thresh, max_fpr=0.15):
    """
    Sweep authority weight up, rebalancing information_sensitivity and
    urgency down proportionally to keep the total at 1.0, and report the
    setting with the best recall subject to a false-positive-rate cap.
    Simple 1D sweep -- meant as a starting point, not an exhaustive search.
    """
    print(f"\n=== Grid search: sweeping authority weight (max_fpr={max_fpr*100:.0f}%) ===")
    best = None
    for authority_w in [round(x * 0.01, 2) for x in range(10, 41, 2)]:  # 0.10 to 0.40
        remaining = 1.0 - authority_w - 0.15 - 0.07 - 0.07 - 0.06  # threat/reward/scarcity/social_proof fixed
        if remaining <= 0:
            continue
        # split remaining between information_sensitivity and urgency, weighted 55/45
        info_w = round(remaining * 0.55, 4)
        urgency_w = round(remaining - info_w, 4)
        weights = {
            "information_sensitivity": info_w,
            "urgency": urgency_w,
            "authority": authority_w,
            "threat": 0.15,
            "reward": 0.07,
            "scarcity": 0.07,
            "social_proof": 0.06,
        }
        metrics = evaluate(weights, rows, moderate_thresh, high_thresh)
        marker = ""
        if metrics["fpr"] <= max_fpr or metrics["fpr"] != metrics["fpr"]:  # nan-safe
            if best is None or metrics["recall"] > best[1]["recall"]:
                best = (weights, metrics)
                marker = "  <-- best so far"
        print(f"authority={authority_w:.2f} info={info_w:.2f} urgency={urgency_w:.2f} "
              f"=> recall={metrics['recall']*100:.1f}% fpr={metrics['fpr']*100:.1f}%{marker}")

    if best:
        print_report("BEST (within FPR cap)", best[0], best[1], moderate_thresh, high_thresh)
    else:
        print("No weight setting in the sweep satisfied the FPR cap. Try raising --max-fpr.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline fusion-weight testing against captured LLM scores.")
    parser.add_argument("csv_path", help="Path to batch_eval_results.csv (must include *_score columns)")
    parser.add_argument("--moderate-threshold", type=float, default=DEFAULT_MODERATE_THRESHOLD)
    parser.add_argument("--high-threshold", type=float, default=DEFAULT_HIGH_THRESHOLD)
    parser.add_argument("--grid-search", action="store_true", help="Sweep authority weight to find a better balance")
    parser.add_argument("--max-fpr", type=float, default=0.15, help="Max acceptable false-positive rate for grid search")
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    rows = load_scored_rows(args.csv_path)
    print(f"Loaded {len(rows)} usable rows (with valid attribute scores) for offline evaluation.")
    if not rows:
        print("No usable rows -- make sure you ran the UPDATED batch_eval.py that writes *_score columns.")
        sys.exit(1)

    for name, weights in CANDIDATE_WEIGHTS.items():
        metrics = evaluate(weights, rows, args.moderate_threshold, args.high_threshold)
        print_report(name, weights, metrics, args.moderate_threshold, args.high_threshold)

    if args.grid_search:
        grid_search(rows, args.moderate_threshold, args.high_threshold, args.max_fpr)
