"""
prompt_framework.py
--------------------
Chain-of-Thought prompt template(s) for VishingGuard-ZSL, JSON schema
validation, and a small version registry so prompt iterations are tracked
and swappable without touching llm_engine.py.

Public API:
    get_prompt(version="latest") -> PromptVersion
    build_messages(transcript, version="latest") -> list[dict]  (system+user messages)
    validate_output(payload: dict) -> (bool, list[str] errors)
    REQUIRED_ATTRIBUTES: the 7 canonical attribute names
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

REQUIRED_ATTRIBUTES = (
    "authority",
    "urgency",
    "information_sensitivity",
    "threat",
    "reward",
    "scarcity",
    "social_proof",
)

TOP_LEVEL_KEYS = REQUIRED_ATTRIBUTES + ("final_risk", "overall_explanation", "reasoning_trace")


@dataclass(frozen=True)
class PromptVersion:
    version: str
    system_prompt: str
    changelog: str  # what changed vs. previous version, and why


# ---------------------------------------------------------------------------
# VERSION HISTORY
# ---------------------------------------------------------------------------
# Each entry is a full, standalone system prompt (never a diff) so any
# version can be loaded and run in isolation for A/B testing.

_V1 = PromptVersion(
    version="v1",
    changelog=(
        "Initial draft, directly from the spec. Free-form 'reason step-by-step, "
        "then output JSON' instruction. Baseline for measuring improvements."
    ),
    system_prompt="""You are a cybersecurity expert analyzing Korean phone call transcripts for vishing (voice phishing) attacks.

Analyze this transcript across 7 social engineering attributes. For each: score 0.0 (absent) to 1.0 (very strong), and quote EXACT evidence phrases from the transcript.

Attributes:
- authority: Does the caller impersonate authority (police, bank, government)?
- urgency: Is false time pressure created?
- information_sensitivity: Is personal/financial data requested?
- threat: Are threats or intimidation used?
- reward: Are fake rewards promised?
- scarcity: Is artificial scarcity created?
- social_proof: Is fake social proof used?

First, reason step-by-step about the transcript. Then output ONLY valid JSON with the schema provided. No markdown, no extra text.""",
)

_V2 = PromptVersion(
    version="v2",
    changelog=(
        "v1 failure analysis (dev split) showed: (a) models sometimes wrapped JSON in "
        "```json fences despite instructions, (b) 'evidence' was occasionally a paraphrase "
        "instead of an exact quote, (c) benign transcripts about legitimate bank/delivery "
        "calls were over-scored on 'authority' just for mentioning a bank name. "
        "Fixes: explicit schema in-prompt, explicit 'no markdown fences' instruction, "
        "explicit rule that evidence must be a verbatim substring, and a calibration note "
        "distinguishing legitimate authority mentions from impersonation attempts."
    ),
    system_prompt="""You are a cybersecurity expert analyzing Korean phone call transcripts for vishing (voice phishing) attacks. You must be precise and avoid false positives on ordinary, benign calls (e.g. real bank customer service, delivery notifications, restaurant reservations).

Score the transcript across exactly 7 social engineering attributes, each from 0.0 (completely absent) to 1.0 (very strong / textbook example):

- authority: Caller IMPERSONATES police, bank fraud department, prosecutor's office, or government agency to gain trust or compliance. A real bank agent verifying an existing transaction is NOT high authority; only impersonation or unverifiable claims of authority count.
- urgency: Caller manufactures false time pressure ("act within 10 minutes", "your account will be frozen today").
- information_sensitivity: Caller requests OTP, PIN, full card number, resident registration number, or asks the victim to move money / install an app / share screen.
- threat: Caller threatens arrest, lawsuit, account freeze, or other legal/financial consequence to coerce compliance.
- reward: Caller promises a prize, refund, unrealistic investment return, or says "you've won/qualified for X".
- scarcity: Caller claims limited slots, limited time offers, or "last chance" framing unrelated to legitimate urgency.
- social_proof: Caller cites fake testimonials, "many other customers have already done this", or fabricated popularity/consensus.

Rules:
1. Reason step-by-step first, in a short internal analysis, BEFORE assigning any scores.
2. Every "evidence" value must be an EXACT, VERBATIM substring copied from the transcript (or "" if score is 0.0). Do not paraphrase or translate it.
3. final_risk is your overall calibrated probability (0.0-1.0) that this is a vishing attack, considering all attributes together, not just their average.
4. Output ONLY raw JSON matching the schema below. No markdown code fences, no commentary before or after, no trailing text.

Schema:
{
  "authority": {"score": 0.0, "evidence": ""},
  "urgency": {"score": 0.0, "evidence": ""},
  "information_sensitivity": {"score": 0.0, "evidence": ""},
  "threat": {"score": 0.0, "evidence": ""},
  "reward": {"score": 0.0, "evidence": ""},
  "scarcity": {"score": 0.0, "evidence": ""},
  "social_proof": {"score": 0.0, "evidence": ""},
  "final_risk": 0.0,
  "overall_explanation": "1-2 sentence human-readable verdict",
  "reasoning_trace": "step-by-step reasoning that led to the scores above"
}""",
)

_V3 = PromptVersion(
    version="v3",
    changelog=(
        "v2 dev-split analysis showed two remaining issues: (a) occasional score inflation "
        "on 'threat' when a transcript merely mentioned legal terms in a non-threatening, "
        "informational context (e.g. explaining a legitimate loan's terms), and (b) a small "
        "number of responses still produced malformed JSON when the transcript contained "
        "quotation marks, breaking the JSON string. Fixes: added a worked few-shot example "
        "(one vishing, one benign) to anchor calibration, and an explicit instruction on "
        "escaping quotes inside evidence strings."
    ),
    system_prompt="""You are a cybersecurity expert analyzing Korean phone call transcripts for vishing (voice phishing) attacks. You must be precise and avoid false positives on ordinary, benign calls (e.g. real bank customer service, delivery notifications, restaurant reservations, informational calls that merely mention legal or financial terms).

Score the transcript across exactly 7 social engineering attributes, each from 0.0 (completely absent) to 1.0 (very strong / textbook example):

- authority: Caller IMPERSONATES police, bank fraud department, prosecutor's office, or government agency to gain trust or compliance. A real bank agent verifying an existing transaction is NOT high authority; only impersonation or unverifiable claims of authority count.
- urgency: Caller manufactures false time pressure ("act within 10 minutes", "your account will be frozen today").
- information_sensitivity: Caller requests OTP, PIN, full card number, resident registration number, or asks the victim to move money / install an app / share screen.
- threat: Caller threatens arrest, lawsuit, account freeze, or other legal/financial consequence AS COERCION. Merely mentioning a legal/financial term in a neutral, informational way (e.g. explaining standard loan terms) is NOT a threat and should score near 0.0.
- reward: Caller promises a prize, refund, unrealistic investment return, or says "you've won/qualified for X".
- scarcity: Caller claims limited slots, limited time offers, or "last chance" framing unrelated to legitimate urgency.
- social_proof: Caller cites fake testimonials, "many other customers have already done this", or fabricated popularity/consensus.

Calibration examples (for reasoning guidance only, do not copy their content):
- VISHING pattern: caller claims to be from the prosecutor's office, says the listener's bank account is linked to a crime, demands they move all funds to a "safe account" within the hour and never hang up the phone -> high authority, high urgency, high threat, high information_sensitivity, high final_risk.
- BENIGN pattern: caller confirms a restaurant reservation time and party size, asks if there are any allergies, thanks the customer -> all attributes near 0.0, low final_risk.

Rules:
1. Reason step-by-step first, in a short internal analysis, BEFORE assigning any scores.
2. Every "evidence" value must be an EXACT, VERBATIM substring copied from the transcript (or "" if score is 0.0). If the evidence text itself contains a double-quote character, escape it as \\" so the JSON stays valid.
3. final_risk is your overall calibrated probability (0.0-1.0) that this is a vishing attack, considering all attributes together, not just their average.
4. Output ONLY raw JSON matching the schema below. No markdown code fences (no ```), no commentary before or after, no trailing text -- the response must start with `{` and end with `}`.

Schema:
{
  "authority": {"score": 0.0, "evidence": ""},
  "urgency": {"score": 0.0, "evidence": ""},
  "information_sensitivity": {"score": 0.0, "evidence": ""},
  "threat": {"score": 0.0, "evidence": ""},
  "reward": {"score": 0.0, "evidence": ""},
  "scarcity": {"score": 0.0, "evidence": ""},
  "social_proof": {"score": 0.0, "evidence": ""},
  "final_risk": 0.0,
  "overall_explanation": "1-2 sentence human-readable verdict",
  "reasoning_trace": "step-by-step reasoning that led to the scores above"
}""",
)

_V4 = PromptVersion(
    version="v4",
    changelog=(
        "v3 dev-split analysis surfaced a whole missed SUBTYPE: illegal loan-brokering / "
        "credit-repair-fee vishing (e.g. 'guaranteed loan approval regardless of your credit "
        "history', 'we can raise your credit grade for a fee', suspiciously low rates with no "
        "underwriting). These calls scored ~0.0 on EVERY attribute because (a) 'reward' was "
        "defined narrowly around prizes/lottery/investment-return framing and didn't cover "
        "guaranteed-approval or paid credit-score manipulation bait, and (b) there was no "
        "calibration example for this pattern -- only the prosecutor-impersonation example, "
        "which shares none of this pattern's surface features (no authority claim, no threat, "
        "no urgency). Fix: broadened the 'reward' definition to explicitly include loan/credit "
        "bait, and added a second VISHING calibration example anchored to that pattern so the "
        "model has something to generalize from beyond prosecutor-impersonation scams."
    ),
    system_prompt="""You are a cybersecurity expert analyzing Korean phone call transcripts for vishing (voice phishing) attacks. You must be precise and avoid false positives on ordinary, benign calls (e.g. real bank customer service, delivery notifications, restaurant reservations, informational calls that merely mention legal or financial terms). Vishing comes in more than one style -- do not assume every scam call impersonates police or a prosecutor; illegal loan-brokering and credit-repair-fee scams are just as common and often use no authority claim, no threat, and no urgency at all.

Score the transcript across exactly 7 social engineering attributes, each from 0.0 (completely absent) to 1.0 (very strong / textbook example):

- authority: Caller IMPERSONATES police, bank fraud department, prosecutor's office, or government agency to gain trust or compliance. A real bank agent verifying an existing transaction is NOT high authority; only impersonation or unverifiable claims of authority count.
- urgency: Caller manufactures false time pressure ("act within 10 minutes", "your account will be frozen today").
- information_sensitivity: Caller requests OTP, PIN, full card number, resident registration number, or asks the victim to move money / install an app / share screen.
- threat: Caller threatens arrest, lawsuit, account freeze, or other legal/financial consequence AS COERCION. Merely mentioning a legal/financial term in a neutral, informational way (e.g. explaining standard loan terms) is NOT a threat and should score near 0.0.
- reward: Caller promises a prize, refund, unrealistic investment return, says "you've won/qualified for X", OR offers a guaranteed loan/credit approval "regardless of your credit history", a suspiciously low interest rate with no real underwriting, or offers to artificially raise the listener's credit score/grade for a fee. All of these are reward-type bait even without explicit prize or lottery framing -- the common thread is an unrealistically good financial outcome offered with no legitimate basis.
- scarcity: Caller claims limited slots, limited time offers, or "last chance" framing unrelated to legitimate urgency.
- social_proof: Caller cites fake testimonials, "many other customers have already done this", or fabricated popularity/consensus.

Calibration examples (for reasoning guidance only, do not copy their content):
- VISHING pattern (authority impersonation): caller claims to be from the prosecutor's office, says the listener's bank account is linked to a crime, demands they move all funds to a "safe account" within the hour and never hang up the phone -> high authority, high urgency, high threat, high information_sensitivity, high final_risk.
- VISHING pattern (illegal loan brokering): caller offers a large loan "regardless of your credit rating," quotes a below-market interest rate, and offers to raise the listener's credit grade for a fee before disbursing the loan -> low/zero authority, low/zero urgency, low/zero threat, but high reward (guaranteed-approval and pay-to-improve-credit-score bait) and high final_risk. Do not let the absence of authority/urgency/threat pull final_risk down when reward alone is a strong, textbook signal.
- BENIGN pattern: caller confirms a restaurant reservation time and party size, asks if there are any allergies, thanks the customer -> all attributes near 0.0, low final_risk.

Rules:
1. Reason step-by-step first, in a short internal analysis, BEFORE assigning any scores.
2. Every "evidence" value must be an EXACT, VERBATIM substring copied from the transcript (or "" if score is 0.0). If the evidence text itself contains a double-quote character, escape it as \\" so the JSON stays valid.
3. final_risk is your overall calibrated probability (0.0-1.0) that this is a vishing attack, considering all attributes together, not just their average. A single very strong attribute (e.g. reward = 0.9) can justify a high final_risk on its own, even if every other attribute is 0.0 -- vishing subtypes do not all rely on the same combination of tactics.
4. Output ONLY raw JSON matching the schema below. No markdown code fences (no ```), no commentary before or after, no trailing text -- the response must start with `{` and end with `}`.

Schema:
{
  "authority": {"score": 0.0, "evidence": ""},
  "urgency": {"score": 0.0, "evidence": ""},
  "information_sensitivity": {"score": 0.0, "evidence": ""},
  "threat": {"score": 0.0, "evidence": ""},
  "reward": {"score": 0.0, "evidence": ""},
  "scarcity": {"score": 0.0, "evidence": ""},
  "social_proof": {"score": 0.0, "evidence": ""},
  "final_risk": 0.0,
  "overall_explanation": "1-2 sentence human-readable verdict",
  "reasoning_trace": "step-by-step reasoning that led to the scores above"
}""",
)

_VERSIONS: Dict[str, PromptVersion] = {"v1": _V1, "v2": _V2, "v3": _V3, "v4": _V4}
_LATEST = "v4"


def get_prompt(version: str = "latest") -> PromptVersion:
    """Look up a prompt version by name. 'latest' resolves to the current recommended version."""
    key = _LATEST if version == "latest" else version
    if key not in _VERSIONS:
        raise KeyError(f"Unknown prompt version '{version}'. Available: {list(_VERSIONS) + ['latest']}")
    return _VERSIONS[key]


def list_versions() -> List[str]:
    return list(_VERSIONS.keys())


def build_messages(transcript: str, version: str = "latest") -> List[Dict[str, str]]:
    """Build the [system, user] message list ready to hand to an LLM chat completion call."""
    prompt = get_prompt(version)
    return [
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": transcript},
    ]


def validate_output(payload: Any) -> Tuple[bool, List[str]]:
    """
    Validate a parsed JSON payload against the required VishingGuard schema.
    Returns (is_valid, list_of_error_messages). Does not raise.
    """
    errors: List[str] = []

    if not isinstance(payload, dict):
        return False, [f"Top-level payload must be a JSON object, got {type(payload).__name__}"]

    for key in TOP_LEVEL_KEYS:
        if key not in payload:
            errors.append(f"Missing required key: '{key}'")

    for attr in REQUIRED_ATTRIBUTES:
        if attr not in payload:
            continue  # already reported above
        node = payload[attr]
        if not isinstance(node, dict):
            errors.append(f"'{attr}' must be an object with 'score' and 'evidence', got {type(node).__name__}")
            continue
        if "score" not in node:
            errors.append(f"'{attr}.score' is missing")
        else:
            score = node["score"]
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                errors.append(f"'{attr}.score' must be numeric, got {type(score).__name__}")
            elif not (0.0 <= float(score) <= 1.0):
                errors.append(f"'{attr}.score' must be in [0.0, 1.0], got {score}")
        if "evidence" not in node:
            errors.append(f"'{attr}.evidence' is missing")
        elif not isinstance(node["evidence"], str):
            errors.append(f"'{attr}.evidence' must be a string, got {type(node['evidence']).__name__}")

    if "final_risk" in payload:
        fr = payload["final_risk"]
        if not isinstance(fr, (int, float)) or isinstance(fr, bool):
            errors.append(f"'final_risk' must be numeric, got {type(fr).__name__}")
        elif not (0.0 <= float(fr) <= 1.0):
            errors.append(f"'final_risk' must be in [0.0, 1.0], got {fr}")

    if "overall_explanation" in payload and not isinstance(payload["overall_explanation"], str):
        errors.append("'overall_explanation' must be a string")

    if "reasoning_trace" in payload and not isinstance(payload["reasoning_trace"], str):
        errors.append("'reasoning_trace' must be a string")

    return (len(errors) == 0), errors


if __name__ == "__main__":
    # Smoke test with a deliberately broken payload
    bad = {"authority": {"score": 1.5, "evidence": 123}, "final_risk": "high"}
    ok, errs = validate_output(bad)
    print("valid:", ok)
    for e in errs:
        print(" -", e)
    print("\nAvailable versions:", list_versions())