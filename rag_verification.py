"""
rag_verification.py
-------------------
Member 3 — RAG verification for VishingGuard-ZSL.

Runs AFTER Member 2's LLM engine. Retrieves the top-3 nearest KorCCViD
patterns and checks whether that evidence agrees with the LLM verdict.

Public API:
    verify_decision(llm_result, transcript, store=None) -> dict

Return dict always contains:
    verdict: CONFIRMED | REDUCE_CONFIDENCE | FLAG_FOR_REVIEW
    verification_score: float in [0, 1]
    adjusted_fused_risk, adjusted_risk_tier
    retrieved: top-3 neighbour records
    needs_manual_review: bool
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

from vector_store import VectorStore, get_default_store

logger = logging.getLogger("vishingguard.rag")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

TOP_K = int(os.getenv("RAG_TOP_K", "3"))
# Neighbours farther than this cosine-distance are treated as unrelated.
MAX_NEIGHBOR_DISTANCE = float(os.getenv("RAG_MAX_NEIGHBOR_DISTANCE", "0.65"))
MIN_SUPPORTING_NEIGHBORS = int(os.getenv("RAG_MIN_SUPPORTING_NEIGHBORS", "1"))
MODERATE_RISK_THRESHOLD = float(os.getenv("MODERATE_RISK_THRESHOLD", "0.4"))
HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.7"))
REDUCE_FACTOR = float(os.getenv("RAG_REDUCE_FACTOR", "0.65"))

CONFIRMED = "CONFIRMED"
REDUCE_CONFIDENCE = "REDUCE_CONFIDENCE"
FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"

_STORE: Optional[VectorStore] = None


def _shared_store() -> VectorStore:
    global _STORE
    if _STORE is None:
        _STORE = get_default_store(in_memory=False)
    return _STORE


def llm_says_vishing(llm_result: Dict[str, Any]) -> bool:
    """
    Mirror Member 2 / batch_eval.py: Moderate or High == predicted vishing.
    Falls back to fused_risk, then final_risk, if risk_tier is missing.
    """
    tier = str(llm_result.get("risk_tier") or "").strip()
    if tier in ("Moderate", "High"):
        return True
    if tier == "Low":
        return False
    risk = llm_result.get("fused_risk", llm_result.get("final_risk", 0.0))
    try:
        return float(risk) >= MODERATE_RISK_THRESHOLD
    except (TypeError, ValueError):
        return False


def _risk_tier(score: float) -> str:
    if score > HIGH_RISK_THRESHOLD:
        return "High"
    if score >= MODERATE_RISK_THRESHOLD:
        return "Moderate"
    return "Low"


def _usable_neighbors(retrieved: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [h for h in retrieved if h.get("distance", 1.0) <= MAX_NEIGHBOR_DISTANCE]


def _evidence_vote(neighbors: List[Dict[str, Any]]) -> Optional[bool]:
    """True = vishing majority, False = benign majority, None = tie / empty."""
    if not neighbors:
        return None
    vishing = sum(1 for h in neighbors if h.get("is_vishing"))
    benign = len(neighbors) - vishing
    if vishing > benign:
        return True
    if benign > vishing:
        return False
    return None


def ensure_store_loaded(store: Optional[VectorStore] = None, csv_path: str = "final_dataset_m16.csv") -> VectorStore:
    """Ensure a usable vector store is available; populate it if it is empty."""
    store = store or _shared_store()
    if store.count() == 0:
        logger.info("Vector store is empty; populating from %s", csv_path)
        store.populate_from_csv(csv_path=csv_path, reset=False)
    return store


def verify_decision(
    llm_result: Dict[str, Any],
    transcript: str,
    store: Optional[VectorStore] = None,
    n_results: int = TOP_K,
    csv_path: str = "final_dataset_m16.csv",
) -> Dict[str, Any]:
    """
    Compare Member 2's JSON result against nearest stored patterns.

    Logic:
        LLM vishing + retrieved vishing -> CONFIRMED
        LLM benign  + retrieved benign  -> CONFIRMED
        LLM vishing + retrieved benign  -> REDUCE_CONFIDENCE
        LLM benign  + retrieved vishing -> FLAG_FOR_REVIEW
        not enough similar neighbours   -> FLAG_FOR_REVIEW
    """
    store = ensure_store_loaded(store, csv_path=csv_path)
    retrieved = store.query_similar(transcript or "", n_results=n_results)
    neighbors = _usable_neighbors(retrieved)
    llm_vishing = llm_says_vishing(llm_result or {})

    try:
        fused = float((llm_result or {}).get("fused_risk", (llm_result or {}).get("final_risk", 0.0)) or 0.0)
    except (TypeError, ValueError):
        fused = 0.0
    fused = max(0.0, min(1.0, fused))

    mean_sim = 0.0
    if neighbors:
        mean_sim = sum(h.get("similarity", 0.0) for h in neighbors) / len(neighbors)

    evidence_vishing = _evidence_vote(neighbors)
    needs_review = False
    reason = ""

    if len(neighbors) < MIN_SUPPORTING_NEIGHBORS or evidence_vishing is None:
        verdict = FLAG_FOR_REVIEW
        verification_score = round(0.25 * mean_sim, 4)
        adjusted = fused
        needs_review = True
        reason = (
            f"Insufficient retrieval support ({len(neighbors)} usable of {len(retrieved)} retrieved; "
            f"need >= {MIN_SUPPORTING_NEIGHBORS} neighbours within distance {MAX_NEIGHBOR_DISTANCE})."
        )
    elif llm_vishing and evidence_vishing:
        verdict = CONFIRMED
        verification_score = round(min(1.0, 0.75 + 0.25 * mean_sim), 4)
        adjusted = fused
        reason = "LLM vishing verdict agrees with retrieved vishing patterns."
    elif (not llm_vishing) and (not evidence_vishing):
        verdict = CONFIRMED
        verification_score = round(min(1.0, 0.75 + 0.25 * mean_sim), 4)
        adjusted = fused
        reason = "LLM benign verdict agrees with retrieved benign patterns."
    elif llm_vishing and not evidence_vishing:
        verdict = REDUCE_CONFIDENCE
        verification_score = round(max(0.15, 0.45 * mean_sim), 4)
        adjusted = round(fused * REDUCE_FACTOR, 4)
        reason = "LLM says vishing but nearest patterns are benign — lowering fused_risk."
    else:
        # LLM benign + retrieved vishing: do not silently accept a Low tier.
        verdict = FLAG_FOR_REVIEW
        verification_score = round(max(0.2, 0.55 * mean_sim), 4)
        adjusted = fused
        needs_review = True
        reason = "LLM says benign but nearest patterns are vishing — flag for human review (possible evasion)."

    adjusted = max(0.0, min(1.0, adjusted))
    result = {
        "verdict": verdict,
        "verification_score": verification_score,
        "reason": reason,
        "llm_is_vishing": llm_vishing,
        "evidence_is_vishing": evidence_vishing,
        "retrieved": retrieved,
        "usable_neighbor_count": len(neighbors),
        "mean_similarity": round(mean_sim, 4),
        "original_fused_risk": fused,
        "adjusted_fused_risk": adjusted,
        "original_risk_tier": (llm_result or {}).get("risk_tier"),
        "adjusted_risk_tier": _risk_tier(adjusted),
        "needs_manual_review": needs_review or verdict == FLAG_FOR_REVIEW,
        "llm_result": llm_result,
    }
    logger.info(
        "RAG %s llm_vishing=%s evidence_vishing=%s score=%.3f adjusted_risk=%.3f",
        verdict,
        llm_vishing,
        evidence_vishing,
        verification_score,
        adjusted,
    )
    return result


def attach_verification(llm_result: Dict[str, Any], verification: Dict[str, Any]) -> Dict[str, Any]:
    """Merge RAG fields onto the LLM's dict so the final result is a single JSON object."""
    merged = dict(llm_result or {})
    merged["fused_risk"] = verification["adjusted_fused_risk"]
    merged["risk_tier"] = verification["adjusted_risk_tier"]
    merged["rag"] = {
        "verdict": verification["verdict"],
        "verification_score": verification["verification_score"],
        "reason": verification["reason"],
        "needs_manual_review": verification["needs_manual_review"],
        "retrieved_ids": [h.get("id") for h in verification.get("retrieved", [])],
        "original_fused_risk": verification["original_fused_risk"],
        "original_risk_tier": verification["original_risk_tier"],
    }
    return merged


def run_rag_verification(
    llm_result: Dict[str, Any],
    transcript: str,
    store: Optional[VectorStore] = None,
    n_results: int = TOP_K,
    csv_path: str = "final_dataset_m16.csv",
) -> Dict[str, Any]:
    """Standalone verification entrypoint for this project without Member 4."""
    verification = verify_decision(llm_result, transcript, store=store, n_results=n_results, csv_path=csv_path)
    return attach_verification(llm_result, verification)


def _load_json_arg(raw: Optional[str]) -> Dict[str, Any]:
    if raw is None or raw == "":
        return {}
    if os.path.exists(raw):
        with open(raw, "r", encoding="utf-8") as fh:
            return json.load(fh)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON input: {raw}") from exc


def _load_transcript_from_csv(csv_path: str, row_index: int = 0) -> str:
    """Read the transcript from final_dataset_m16.csv (or another CSV file) by row index."""
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        import csv
        rows = list(csv.DictReader(fh))
    if not rows:
        raise ValueError(f"CSV file {csv_path} is empty")
    if row_index < 0:
        row_index = 0
    if row_index >= len(rows):
        row_index = len(rows) - 1
    transcript = (rows[row_index].get("transcript") or "").strip()
    if not transcript:
        raise ValueError(f"Row {row_index} in {csv_path} does not include a transcript")
    return transcript


def _read_interactive_transcript() -> str:
    """Read a pasted transcript until the user enters an empty line."""
    print("Paste the call transcript below.")
    print("Press Enter on an empty line when you are finished.")
    lines: List[str] = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    transcript = "\n".join(lines).strip()
    if not transcript:
        raise ValueError("No transcript was entered.")
    return transcript


def _analyze_transcript(transcript: str) -> Dict[str, Any]:
    """Run the project's LLM scorer when no precomputed result is supplied."""
    from llm_engine import LLMEngine

    return LLMEngine().analyze_transcript(transcript)


def _build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paste a call transcript to analyze it with LLM scoring and RAG verification."
    )
    parser.add_argument("csv_path", nargs="?", default=None, help="Optional path to the CSV dataset. If omitted, final_dataset_m16.csv is used.")
    parser.add_argument("--transcript", help="Transcript text. If omitted, interactive paste mode is used.")
    parser.add_argument("--llm-result", help="Optional legacy JSON dict or path to JSON file for a precomputed LLM output.")
    parser.add_argument("--llm-result-file", help="Alias for --llm-result.")
    parser.add_argument("--csv", dest="csv_option", help="Alias for the CSV path.")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of retrieved neighbours to inspect.")
    parser.add_argument("--row-index", type=int, default=0, help="Row index to read from the CSV when --transcript is omitted.")
    return parser


def _normalize_cli_args(argv: List[str]) -> List[str]:
    """Accept a pasted one-line shell continuation before an option."""
    return ["--csv" if arg == " --csv" else arg for arg in argv]


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args(_normalize_cli_args(sys.argv[1:]))

    # Some users paste a line-continuation slash into a one-line command,
    # where the shell passes it as a literal positional argument.
    positional_csv = None if args.csv_path in ("\\", "/") else args.csv_path
    csv_path = positional_csv or args.csv_option or "final_dataset_m16.csv"
    llm_payload = args.llm_result or args.llm_result_file
    if args.transcript:
        transcript = args.transcript
    elif llm_payload or positional_csv or args.csv_option:
        transcript = _load_transcript_from_csv(csv_path, row_index=args.row_index)
    else:
        transcript = _read_interactive_transcript()

    llm_result = _load_json_arg(llm_payload) if llm_payload else _analyze_transcript(transcript)
    final = run_rag_verification(llm_result, transcript, n_results=args.top_k, csv_path=csv_path)
    print(json.dumps(final, indent=2, ensure_ascii=False))
