"""
config.py
---------
Central configuration for VishingGuard-ZSL LLM & Prompt Engineering module.

ALL secrets (API keys) are loaded from environment variables. Nothing is
hardcoded. Copy `.env.example` to `.env` and fill in your keys, or export
the variables in your shell before running.

Required environment variables:
    GROQ_API_KEY    - API key for Groq (free tier; powers both primary and
                      fallback models by default -- see "Model names" below
                      for why these are no longer literally Llama 3 models)
    OPENAI_API_KEY  - API key for OpenAI (only needed if FALLBACK_PROVIDER=openai,
                      e.g. to use the paid GPT-4o-mini fallback)

Optional environment variables (all have sane defaults below):
    PRIMARY_MODEL, FALLBACK_MODEL, LLM_TEMPERATURE, MAX_TOKENS,
    ROUTER_COMPLEXITY_THRESHOLD, ROUTER_LEN_THRESHOLD_1, ROUTER_LEN_THRESHOLD_2,
    REQUEST_TIMEOUT_SECONDS, MAX_RETRIES, LOG_LEVEL
"""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


load_dotenv()


def _env_float(name: str, default: float) -> float:
    val = os.getenv(name)
    return float(val) if val is not None else default


def _env_int(name: str, default: int) -> int:
    val = os.getenv(name)
    return int(val) if val is not None else default


def _env_str(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val is not None else default


@dataclass(frozen=True)
class Settings:
    # ---- API keys (never hardcode; loaded from env at import time) ----
    groq_api_key: str = field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    # Optional: only required if PRIMARY_MODEL/FALLBACK_MODEL is set to a
    # gemini-* model name (see llm_engine._call_gemini). Free key, no credit
    # card needed: https://aistudio.google.com/apikey
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # ---- Model names ----
    # Both primary and fallback run on Groq's free tier (no credit card required).
    # As of the current Groq model catalog, Llama 3.1 8B / Llama 3.3 70B have
    # moved to Enterprise-only ("Contact Sales") and are NOT available on the
    # free plan. The free-tier text models are OpenAI's open-weight GPT-OSS
    # models (and a couple of Qwen models). gpt-oss-20b is the fast/cheap
    # primary; gpt-oss-120b is the stronger fallback for ambiguous transcripts.
    # If a paid budget becomes available later, set FALLBACK_PROVIDER=openai
    # and FALLBACK_MODEL=gpt-4o-mini via env vars to switch to a paid fallback.
    # Verify current model IDs/limits anytime at https://console.groq.com/docs/models
    primary_model: str = field(default_factory=lambda: _env_str("PRIMARY_MODEL", "openai/gpt-oss-20b"))
    fallback_model: str = field(default_factory=lambda: _env_str("FALLBACK_MODEL", "openai/gpt-oss-120b"))
    fallback_provider: str = field(default_factory=lambda: _env_str("FALLBACK_PROVIDER", "groq"))  # "groq" or "openai"

    # ---- LLM generation settings ----
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    # Bumped 1536 -> 2048 (2026-09-09): dev-split debugging showed several
    # openai/gpt-oss-20b calls returning `json_validate_failed` with a
    # `failed_generation` payload cut off mid-string inside `reasoning_trace`
    # (e.g. VISHING_2795 -- the JSON was well-formed up to the point where it
    # simply ran out of tokens). Reasoning models spend part of the budget on
    # hidden "thinking" even at reasoning_effort=low, so 1536 wasn't always
    # enough headroom to finish writing the visible JSON afterward. This also
    # showed up as run-to-run SCORE INSTABILITY on the same transcript at
    # temperature=0 (e.g. VISHING_2731 scoring final_risk anywhere from 0.0 to
    # 0.85 across identical calls) -- plausibly because a tight token budget
    # makes the model's hidden reasoning phase get cut off at different points
    # depending on minor timing/sampling variance, even though the *visible*
    # temperature is 0. More headroom should reduce both failure modes; if
    # instability persists after this change, the next thing to try is
    # REASONING_EFFORT=medium (slower but more consistent).
    max_tokens: int = field(default_factory=lambda: _env_int("MAX_TOKENS", 2048))
    # openai/gpt-oss-20b and openai/gpt-oss-120b are REASONING models: they spend
    # tokens on hidden "thinking" before writing the JSON answer. At the default
    # effort ("medium") this consistently burns most of max_tokens on reasoning
    # alone, causing high latency (20-30s+) and "max completion tokens reached
    # before generating a valid document" errors on long transcripts. "low"
    # effort keeps reasoning brief while still following the CoT instructions
    # in the prompt itself. Only applied when the model name contains "gpt-oss"
    # (see llm_engine._call_groq) since other Groq models don't support it.
    reasoning_effort: str = field(default_factory=lambda: _env_str("REASONING_EFFORT", "low"))
    # JSON mode (response_format=json_object) requires reasoning to be routed
    # out of the main content field -- "hidden" drops it, "parsed" returns it
    # separately. We don't need the reasoning trace itself (the prompt already
    # asks for a "reasoning_trace" field inside the JSON), so hidden is simplest.
    reasoning_format: str = field(default_factory=lambda: _env_str("REASONING_FORMAT", "hidden"))

    # ---- Complexity router thresholds ----
    # Transcript length breakpoints (characters) that add to complexity_score
    router_len_threshold_1: int = field(default_factory=lambda: _env_int("ROUTER_LEN_THRESHOLD_1", 300))
    router_len_threshold_2: int = field(default_factory=lambda: _env_int("ROUTER_LEN_THRESHOLD_2", 500))
    # complexity_score >= this value routes to the fallback (GPT-4o-mini)
    router_complexity_threshold: int = field(
        default_factory=lambda: _env_int("ROUTER_COMPLEXITY_THRESHOLD", 3)
    )

    # ---- Reliability settings ----
    request_timeout_seconds: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT_SECONDS", 15.0)
    )
    max_retries: int = field(default_factory=lambda: _env_int("MAX_RETRIES", 1))  # retries on malformed JSON

    # ---- Fusion weights (must sum to 1.0) ----
    # UPDATED 2026-09-09: dev-split false-negative analysis (offline grid search
    # over captured attribute scores, offline_weight_search.py) showed the
    # original weights (info_sensitivity/urgency=0.25, authority/threat=0.15,
    # reward/scarcity=0.07, social_proof=0.06) under-weighted 'authority' badly
    # enough that even a NEAR-PERFECT LLM read of a textbook prosecutor-
    # impersonation transcript (authority=0.9, info_sensitivity=0.8) fused to
    # only 0.385 -- under the 0.4 Moderate threshold. The grid search found
    # recall kept improving all the way to authority=0.40 on our small sample,
    # but that sample was only 6 vishing rows total, so treating 0.40 as final
    # would be overfitting to 6 examples. Landing on authority=0.22 as a more
    # conservative middle ground: meaningfully higher than the original 0.15
    # (fixes the near-miss case above -- 0.9*0.22 + 0.8*0.25 = 0.398, right at
    # the threshold; combined with the v4 prompt's calibration fix this should
    # clear it) without swinging as far as the small-sample-optimal 0.40.
    # RE-VALIDATE these weights once a larger batch (20-30+ vishing rows) is
    # available -- see offline_weight_search.py --grid-search.
    fusion_weights: dict = field(default_factory=lambda: {
        "information_sensitivity": 0.25,
        "urgency": 0.18,
        "authority": 0.22,
        "threat": 0.15,
        "reward": 0.09,
        "scarcity": 0.06,
        "social_proof": 0.05,
    })

    # ---- Single-signal floor (see llm_engine.compute_fusion_risk) ----
    # Guarantees fused_risk >= (highest single attribute score * this factor),
    # so a vishing subtype that legitimately only lights up ONE attribute
    # (e.g. loan-scam bait scoring high on 'reward' alone) isn't diluted to
    # near-zero by a flat weighted sum across all 7 attributes. Calibrated
    # against dev-split data where reward=0.8-0.9 loan-scam cases needed to
    # clear the 0.4 Moderate threshold; re-validate once more benign edge
    # cases (formal/urgent legitimate calls) are available -- see the
    # docstring on compute_fusion_risk for the full reasoning and caveats.
    single_signal_floor_factor: float = field(
        default_factory=lambda: _env_float("SINGLE_SIGNAL_FLOOR_FACTOR", 0.5)
    )

    # ---- Decision thresholds ----
    high_risk_threshold: float = field(default_factory=lambda: _env_float("HIGH_RISK_THRESHOLD", 0.7))
    moderate_risk_threshold: float = field(default_factory=lambda: _env_float("MODERATE_RISK_THRESHOLD", 0.4))

    # ---- Logging ----
    log_level: str = field(default_factory=lambda: _env_str("LOG_LEVEL", "INFO"))

    # ---- Router keyword lists (Korean) ----
    ambiguity_patterns: tuple = ("불명확", "확인 필요", "여러 가능성")
    vishing_keywords: tuple = ("긴급", "경찰", "계좌", "정지", "개인정보")
    benign_keywords: tuple = ("예약", "문의", "배송", "감사합니다")

    def validate(self) -> None:
        """Raise a clear error early if required secrets/config are missing or malformed."""
        problems = []
        primary_is_gemini = "gemini" in self.primary_model.lower()
        fallback_is_gemini = "gemini" in self.fallback_model.lower()

        if primary_is_gemini or fallback_is_gemini:
            if not self.gemini_api_key:
                problems.append("GEMINI_API_KEY is not set but a Gemini model is configured.")

        if not primary_is_gemini and not self.groq_api_key:
            problems.append("GROQ_API_KEY is not set but the primary model is configured for Groq.")

        if not fallback_is_gemini:
            if self.fallback_provider == "openai" and not self.openai_api_key:
                problems.append("FALLBACK_PROVIDER is 'openai' but OPENAI_API_KEY is not set.")
            elif self.fallback_provider == "groq" and not self.groq_api_key:
                problems.append("GROQ_API_KEY is not set but the fallback model is configured for Groq.")
        if abs(sum(self.fusion_weights.values()) - 1.0) > 1e-6:
            problems.append(f"fusion_weights must sum to 1.0, got {sum(self.fusion_weights.values())}")
        if not (0.0 <= self.moderate_risk_threshold < self.high_risk_threshold <= 1.0):
            problems.append("Risk thresholds must satisfy 0 <= moderate < high <= 1.")
        if problems:
            raise ValueError("Config validation failed:\n- " + "\n- ".join(problems))


settings = Settings()


if __name__ == "__main__":
    # Quick manual check: `python config.py`
    try:
        settings.validate()
        print("Config OK.")
    except ValueError as e:
        print(e)
