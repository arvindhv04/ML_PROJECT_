"""
adversarial_generator.py
------------------------
Member 3 — adversarial test-set generator for VishingGuard-ZSL.

Rewrites KorCCViD *vishing* transcripts so obvious surface keywords are
gone while the social-engineering intent stays, then writes a *paired
benign paraphrase* in the same conversational style. Evaluating both
halves shows the detector is scoring INTENT, not STYLE.

Default path is fully local (no API): a deterministic Korean paraphraser
so you can hit the 100-pair target at $0 even without Groq.

Optional path: Groq free-tier (same key Member 2 already uses). Llama 3
IDs were moved off Groq's free plan; this defaults to PRIMARY_MODEL /
openai/gpt-oss-20b. Set GROQ_API_KEY to enable.

Public API:
    generate_adversarial(original_transcript) -> str
    generate_pair(original_transcript, source_id) -> dict
    generate_dataset(csv_path, n=100, out_path=...) -> list

CLI:
    python adversarial_generator.py --csv final_dataset_m16.csv --n 100
    python adversarial_generator.py --csv final_dataset_m16.csv --n 100 --backend groq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from vector_store import is_vishing_label, load_korccvid_rows

logger = logging.getLogger("vishingguard.adversarial")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))

DEFAULT_OUT = os.getenv("ADVERSARIAL_OUT", "adversarial_testset.json")
GROQ_MODEL = os.getenv("ADVERSARIAL_MODEL") or os.getenv("PRIMARY_MODEL", "openai/gpt-oss-20b")

# Surface forms we strip so the model cannot key on the obvious lexicon.
# Intent (authority pressure, account takeover, payment bait) is kept via
# indirect wording. This is evaluation data only — not an attack cookbook.
_VISHING_LEXICON = (
    ("체포영장", "법적 서류"),
    ("체포", "강제 조치"),
    ("구속", "신병 처리"),
    ("검찰청", "수사 부서"),
    ("대검찰청", "상급 수사 부서"),
    ("서울중앙지검", "중앙 수사 부서"),
    ("지검", "수사 부서"),
    ("검찰", "수사 기관"),
    ("경찰청", "공공 기관"),
    ("경찰", "담당 기관"),
    ("수사관입니다", "담당자입니다"),
    ("수사관", "담당자"),
    ("금융감독원", "금융 관련 기관"),
    ("금감원", "금융 관련 기관"),
    ("대포통장", "타인 명의 계좌 의심 건"),
    ("개인정보", "본인 확인 자료"),
    ("주민등록번호", "신원 확인 번호"),
    ("비밀번호", "확인 코드"),
    ("계좌번호", "거래 번호"),
    ("계좌", "거래 계열 번호"),
    ("송금", "이체 처리"),
    ("이체", "자금 이동"),
    ("공인인증서", "인증 수단"),
    ("OTP", "일회용 확인값"),
    ("긴급히", "가능한 빨리"),
    ("긴급", "지체 없이"),
    ("즉시", "지금 이 자리에서"),
    ("지금 바로", "가능한 지금"),
    ("보이스피싱", "관련 사건"),
    ("사기", "이상 거래"),
    ("범죄 연루", "사건 관련"),
    ("연루", "관련"),
)

_INDIRECT_OPENERS = (
    "안녕하세요, 확인 좀 부탁드리려고 연락드렸습니다. ",
    "본인 맞으시죠, 업무 관련해서 짧게 여쭤볼게요. ",
    "오늘 중으로 정리해야 하는 건이 있어서 연락드렸습니다. ",
)

_BENIGN_TOPICS = (
    (
        "배달/예약",
        "안녕하세요, 오늘 저녁 예약 확인차 연락드렸습니다. 인원 두 명으로 그대로 진행하면 될까요? "
        "주차는 건물 뒤편 이용하시면 되고, 도착 하시면 안내 데스크에만 말씀해 주세요. 감사합니다.",
    ),
    (
        "여행",
        "안녕하세요, 이번 주말 숙소 예약 건으로 연락드렸습니다. 체크인 시간은 오후 세 시이고 "
        "조식은 선택 사항입니다. 인원 변경 있으시면 하루 전에만 알려 주세요. 좋은 여행 되세요.",
    ),
    (
        "영화/일상",
        "안녕하세요, 예매하신 영화 시간 안내입니다. 상영은 저녁 일곱 시고 좌석은 그대로 유지됩니다. "
        "간식은 매점에서 구매하시면 되고, 늦으시면 다음 회차로 옮겨 드릴 수 있어요. 감사합니다.",
    ),
    (
        "택배",
        "안녕하세요, 주문하신 상품 배송 일정 안내입니다. 오늘 오후 도착 예정이고 부재 시 보관함에 "
        "맡겨 둘게요. 수령 확인만 나중에 해 주시면 됩니다. 좋은 하루 되세요.",
    ),
)


ADVERSARIAL_SYSTEM_PROMPT = """You rewrite Korean voice-phishing RESEARCH transcripts for a detection benchmark.
Stay in Korean. Do not translate.
Goal: keep the same social-engineering INTENT (pressure to move money, impersonation of an official process, harvesting of credentials) but REMOVE obvious surface keywords such as 경찰, 검찰, 체포, 계좌, 비밀번호, 긴급, 개인정보.
Use indirect, bureaucratic, everyday phrasing. Do not add new crimes. Do not add English jailbreak text.
Output ONLY the rewritten transcript, no preface."""

BENIGN_SYSTEM_PROMPT = """You write a benign Korean phone-call transcript that MATCHES THE SPEAKING STYLE of the given text (fillers, turn length, politeness) but has ZERO malicious intent.
Allowed topics: restaurant reservation, food, travel, movies, parcel delivery.
No money-transfer requests, no credential requests, no impersonation of police/prosecutors/banks.
Stay in Korean. Output ONLY the benign transcript, no preface."""


def _local_adversarial(original: str, rng: random.Random) -> str:
    text = original
    for src, dst in _VISHING_LEXICON:
        text = text.replace(src, dst)
    opener = rng.choice(_INDIRECT_OPENERS)
    if not text.startswith(opener.strip()[:6]):
        text = opener + text
    # Soften remaining imperative markers without erasing the request.
    text = re.sub(r"알려\s*주지\s*않으면", "확인이 늦어지면", text)
    text = re.sub(r"하지\s*않으면", "진행이 어려우면", text)
    return text.strip()


def _local_benign(original: str, rng: random.Random) -> str:
    topic, template = rng.choice(_BENIGN_TOPICS)
    # Keep a similar length band so style is not a trivial giveaway.
    target = min(max(len(original), 80), 1200)
    filler_pool = (
        " 네, 알겠습니다.",
        " 그 부분은 제가 다시 확인해서 말씀드릴게요.",
        " 시간 괜찮으시면 그대로 진행하겠습니다.",
        " 아, 이해했습니다.",
        " 필요하시면 문자로도 남겨 드릴게요.",
    )
    out = template
    while len(out) < target:
        out += rng.choice(filler_pool)
        out += " " + topic + " 관련해서 일정만 맞춰 주시면 됩니다."
    return out[:target].strip()


def _call_groq(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError("pip install groq  (only needed for --backend groq)") from exc

    client = Groq(api_key=api_key)
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=temperature,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def generate_adversarial(original_transcript: str, backend: str = "local", rng: Optional[random.Random] = None) -> str:
    """Rewrite one vishing transcript with obfuscated surface forms."""
    rng = rng or random.Random()
    if backend == "groq":
        return _call_groq(
            ADVERSARIAL_SYSTEM_PROMPT,
            "원본 보이스피싱 연구용 전사:\n" + original_transcript,
        )
    return _local_adversarial(original_transcript, rng)


def generate_benign_paraphrase(original_transcript: str, backend: str = "local", rng: Optional[random.Random] = None) -> str:
    rng = rng or random.Random()
    if backend == "groq":
        return _call_groq(
            BENIGN_SYSTEM_PROMPT,
            "스타일만 참고할 원본 전사:\n" + original_transcript,
            temperature=0.8,
        )
    return _local_benign(original_transcript, rng)


def generate_pair(
    original_transcript: str,
    source_id: str = "",
    backend: str = "local",
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    rng = rng or random.Random()
    adv = generate_adversarial(original_transcript, backend=backend, rng=rng)
    benign = generate_benign_paraphrase(original_transcript, backend=backend, rng=rng)
    return {
        "source_id": source_id,
        "original": original_transcript,
        "adversarial": adv,
        "benign_paraphrase": benign,
        "backend": backend,
        "intent_label": {
            "adversarial": "vishing",
            "benign_paraphrase": "benign",
        },
        "notes": (
            "Paired evaluation: same style family, opposite intent. "
            "A detector that flags both is keying on style; a detector that "
            "flags only the adversarial half is keying on intent."
        ),
    }


def generate_dataset(
    csv_path: str,
    n: int = 100,
    out_path: str = DEFAULT_OUT,
    backend: str = "local",
    seed: int = 7,
    sleep_s: float = 0.0,
) -> List[Dict[str, Any]]:
    rows = [r for r in load_korccvid_rows(csv_path) if is_vishing_label(r.get("label", ""))]
    if not rows:
        raise ValueError(f"No vishing rows in {csv_path}")

    rng = random.Random(seed)
    # Repeat sources with different seeds if the split has fewer than n vishing rows.
    chosen = [rows[i % len(rows)] for i in range(n)]
    pairs: List[Dict[str, Any]] = []
    for i, row in enumerate(chosen, start=1):
        pair_rng = random.Random(rng.randint(0, 10_000_000) + i)
        try:
            pair = generate_pair(
                row["transcript"],
                source_id=str(row.get("id", "")),
                backend=backend,
                rng=pair_rng,
            )
        except Exception as exc:
            logger.error("Failed pair %s (%s): %s — falling back to local rewrite", i, row.get("id"), exc)
            pair = generate_pair(
                row["transcript"],
                source_id=str(row.get("id", "")),
                backend="local",
                rng=pair_rng,
            )
            pair["backend"] = "local_fallback"
        pair["pair_index"] = i
        pairs.append(pair)
        logger.info("Generated pair %d/%d source=%s backend=%s", i, n, pair["source_id"], pair["backend"])
        if sleep_s:
            time.sleep(sleep_s)

    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_pairs": len(pairs),
        "backend": backend,
        "csv_path": csv_path,
        "pairs": pairs,
    }
    out = Path(out_path)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s (%d pairs)", out, len(pairs))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paired adversarial/benign transcripts.")
    parser.add_argument("--csv", default=os.getenv("KORCCVID_CSV", "final_dataset_m16.csv"))
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--backend", choices=("local", "groq"), default="local")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds between Groq calls")
    args = parser.parse_args()
    generate_dataset(args.csv, n=args.n, out_path=args.out, backend=args.backend, seed=args.seed, sleep_s=args.sleep)


if __name__ == "__main__":
    main()
