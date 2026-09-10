"""
inspect_one.py
---------------
Pulls one specific transcript out of a CSV by its id, runs it through
LLMEngine, and prints the FULL result (every attribute's score+evidence,
final_risk, fused_risk, risk_tier, and -- most importantly -- the model's
own reasoning_trace) so you can see exactly why it scored a transcript the
way it did.

Usage:
    python inspect_one.py dev_m16.csv VISHING_2731
    python inspect_one.py dev_m16.csv VISHING_2731 --version v2   # try an older prompt version for comparison
"""

import argparse
import csv
import json

from llm_engine import LLMEngine
import prompt_framework as pf


def find_row(csv_path: str, target_id: str) -> dict:
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["id"] == target_id:
                return row
    raise SystemExit(f"No row with id={target_id!r} found in {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("transcript_id")
    parser.add_argument("--version", default="latest", help="Prompt version to use (v1/v2/v3/latest)")
    args = parser.parse_args()

    row = find_row(args.csv_path, args.transcript_id)
    print("=" * 70)
    print(f"id: {row['id']}   true_label: {row['label']}   category: {row.get('category')}")
    print("=" * 70)
    print("TRANSCRIPT (first 500 chars):")
    print(row["transcript"][:500] + ("..." if len(row["transcript"]) > 500 else ""))
    print(f"\n(full transcript length: {len(row['transcript'])} characters)")
    print("=" * 70)

    engine = LLMEngine(prompt_version=args.version)
    result = engine.analyze_transcript(row["transcript"])

    print(f"\nUsing prompt version: {pf.get_prompt(args.version).version}\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
