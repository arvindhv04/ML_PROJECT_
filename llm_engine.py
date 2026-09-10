"""
llm_engine.py
-------------
Core LLM integration for VishingGuard-ZSL: complexity routing, Groq's free
gpt-oss-20b as the primary scorer, and Groq's free gpt-oss-120b as the
fallback for ambiguous transcripts (both models run on Groq's no-credit-card
free tier, so the whole pipeline costs $0 by default). An OpenAI/GPT-4o-mini
path and a Gemini path are also wired in and can be enabled via config/env
vars if you want to switch providers -- neither is required to run this
module.

NOTE ON MODEL NAMES: Groq's free-tier model catalog changes over time (e.g.
llama3-8b-8192 and llama-3.3-70b-versatile were moved to Enterprise-only
partway through this project). If you get a "model_decommissioned" or
"model_not_found" error, check https://console.groq.com/docs/models for the
current free-tier lineup and update PRIMARY_MODEL/FALLBACK_MODEL in your
environment (or the defaults in config.py) accordingly.

SWITCHING TO GEMINI: Groq's free tier is capped at ~8,000 tokens/minute,
which is easy to blow through with reasoning models and 2000+ max_tokens
budgets, causing long rate-limit stalls. Gemini's free tier (as of Sep 2026)
gives ~250,000 tokens/minute on Flash models -- a much bigger buffer -- at
the cost of a lower per-minute REQUEST count (~10-15 RPM vs Groq's higher
ceiling). Since this pipeline's calls are 2-30s each anyway, RPM is rarely
the binding constraint, so Gemini's much larger TPM budget is usually a net
win for this workload. To switch:
    setx GEMINI_API_KEY "your-key-here"      (from https://aistudio.google.com/apikey)
    setx PRIMARY_MODEL "gemini-2.5-flash"
    setx FALLBACK_MODEL "gemini-2.5-flash"   (or a different Gemini model)
The model name is auto-detected: any model name containing "gemini" routes
through the Gemini REST API instead of Groq. No other code changes needed.
Gemini's free-tier rate limits and model lineup change over time -- verify
current numbers/model IDs at https://ai.google.dev/gemini-api/docs/rate-limits
and https://ai.google.dev/gemini-api/docs/models before relying on this in
a time-sensitive run.

Also includes: JSON parsing with one retry on malformed output, automatic
backoff-and-retry on 429 rate-limit responses, per-request latency
tracking, error logging, and weighted fusion/decision logic.

Usage:
    from llm_engine import LLMEngine

    engine = LLMEngine()
    result = engine.analyze_transcript(korean_transcript_text)
    # result is a dict: 7 attributes + final_risk + fused_risk + risk_tier +
    # overall_explanation + reasoning_trace + _meta (model used, latency, retries)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from config import settings
from prompt_framework import build_messages, validate_output, REQUIRED_ATTRIBUTES

logger = logging.getLogger("vishingguard.llm_engine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_handler)
logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# ---------------------------------------------------------------------------
# Complexity Router
# ---------------------------------------------------------------------------

def compute_fusion_risk(payload: Dict[str, Any]) -> float:
    """
    Recompute final_risk as a weighted fusion of the 7 attribute scores,
    per config.settings.fusion_weights. This is deliberately independent of
    whatever `final_risk` the LLM itself produced, so a miscalibrated
    top-level number from the model can't silently drive the risk tier --
    the LLM's own final_risk is kept in the payload for comparison/logging.

    SINGLE-SIGNAL FLOOR (added 2026-09-09): dev-split analysis found whole
    vishing SUBTYPES that legitimately light up only ONE attribute strongly
    with everything else at a genuine 0.0 -- e.g. illegal loan-brokering
    scams score high on 'reward' alone (guaranteed-approval / pay-to-raise-
    your-credit-score bait) with zero authority, urgency, or threat, because
    that tactic doesn't use those levers at all. A flat weighted sum caps
    such cases at (top_score * that_attribute's_weight), which for a weight
    like reward=0.09 means even a perfect reward=1.0 read can only ever
    contribute 0.09 -- nowhere near the Moderate threshold, REGARDLESS of
    how obviously risky a human would find it. The floor below guarantees a
    minimum fused_risk of (highest single attribute * FLOOR_FACTOR), so one
    very strong, unambiguous signal can't be diluted to near-zero just
    because the OTHER six attributes are correctly, legitimately at 0.0.
    FLOOR_FACTOR=0.5 was chosen because it clears the Moderate threshold
    (0.4) for the loan-scam cases we found (reward=0.8-0.9) while a lower
    single-attribute score (e.g. 0.3-0.4, more ambiguous) stays under it.
    Validated against 15 benign dev-split rows, where every attribute
    scored 0.0 across the board -- so the floor never fired on benign data
    in that sample. CAVEAT: that benign sample didn't include "hard" edge
    cases (e.g. a legitimate bank call using formal/urgent language for a
    real reason), so this factor should be re-validated once such cases are
    available, not treated as permanently tuned.
    """
    total = 0.0
    top_score = 0.0
    for attr, weight in settings.fusion_weights.items():
        node = payload.get(attr, {})
        score = node.get("score", 0.0) if isinstance(node, dict) else 0.0
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        total += score * weight
        if score > top_score:
            top_score = score

    floor = top_score * settings.single_signal_floor_factor
    fused = max(total, floor)
    return round(min(max(fused, 0.0), 1.0), 4)


def risk_tier(risk_score: float) -> str:
    """Map a 0-1 risk score to Low / Moderate / High per config thresholds."""
    if risk_score > settings.high_risk_threshold:
        return "High"
    if risk_score >= settings.moderate_risk_threshold:
        return "Moderate"
    return "Low"


def route_transcript(transcript: str) -> str:
    """
    Decide which model should handle a transcript.

    Returns the model name string (settings.primary_model or
    settings.fallback_model). Target distribution: 80-90% primary (free
    Llama 3), 10-20% fallback (GPT-4o-mini), reserved for ambiguous /
    mixed-signal transcripts.
    """
    complexity_score = 0

    if len(transcript) > settings.router_len_threshold_1:
        complexity_score += 1
    if len(transcript) > settings.router_len_threshold_2:
        complexity_score += 1

    if any(p in transcript for p in settings.ambiguity_patterns):
        complexity_score += 1

    has_vishing_kw = any(k in transcript for k in settings.vishing_keywords)
    has_benign_kw = any(k in transcript for k in settings.benign_keywords)
    if has_vishing_kw and has_benign_kw:
        complexity_score += 2  # mixed signals = ambiguous, worth the stronger model

    if complexity_score >= settings.router_complexity_threshold:
        return settings.fallback_model
    return settings.primary_model


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    payload: Dict[str, Any]
    model_used: str
    latency_seconds: float
    retries: int
    fallback_triggered: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = dict(self.payload)
        out["_meta"] = {
            "model_used": self.model_used,
            "latency_seconds": round(self.latency_seconds, 3),
            "retries": self.retries,
            "fallback_triggered": self.fallback_triggered,
            "error": self.error,
        }
        return out


def _empty_error_payload(message: str) -> Dict[str, Any]:
    """A schema-shaped payload used when every attempt fails, so downstream
    code (fusion, RAG verification) never has to special-case a missing dict."""
    payload = {attr: {"score": 0.0, "evidence": ""} for attr in REQUIRED_ATTRIBUTES}
    payload.update({
        "final_risk": 0.0,
        "fused_risk": 0.0,
        "risk_tier": "Low",
        "overall_explanation": f"Analysis failed: {message}",
        "reasoning_trace": "",
    })
    return payload


def _extract_json(raw_text: str) -> Any:
    """Parse a model response into JSON, tolerating stray markdown fences."""
    text = raw_text.strip()
    if text.startswith("```"):
        # strip ```json ... ``` or ``` ... ``` wrappers some models add anyway
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)


def _is_gemini_model(model_name: str) -> bool:
    return "gemini" in model_name.lower()


# ---------------------------------------------------------------------------
# LLMEngine
# ---------------------------------------------------------------------------

class LLMEngine:
    """
    Wraps Groq (primary), OpenAI (optional fallback), and Gemini (optional
    provider, auto-detected by model name) chat completion calls behind a
    single `analyze_transcript` method, with routing, retry-on-malformed-
    JSON, latency tracking, and error logging.

    Groq/OpenAI clients are constructed lazily so importing this module
    (e.g. for unit tests) never requires API keys or network access. The
    Gemini path uses stdlib urllib directly (no extra pip dependency).
    """

    def __init__(self, prompt_version: str = "latest"):
        self.prompt_version = prompt_version
        self._groq_client = None
        self._openai_client = None

    # -- lazy client construction -----------------------------------------

    def _get_groq_client(self):
        if self._groq_client is None:
            from groq import Groq  # imported here so the module loads without the package installed
            self._groq_client = Groq(api_key=settings.groq_api_key)
        return self._groq_client

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI(api_key=settings.openai_api_key)
        return self._openai_client

    # -- raw model calls -----------------------------------------------

    def _call_groq(self, model_name: str, messages, max_rate_limit_retries: int = 4) -> str:
        """
        Call Groq with automatic backoff on 429 rate-limit responses. The
        free tier has fairly tight per-model RPM/TPM budgets (see
        https://console.groq.com/docs/rate-limits), so hitting a 429 under
        normal use is expected, not exceptional -- this waits (honoring the
        server's retry-after header when present) and retries in place
        rather than treating it as a hard failure.
        """
        client = self._get_groq_client()
        last_exc = None
        for attempt in range(max_rate_limit_retries + 1):
            try:
                kwargs = dict(
                    model=model_name,
                    messages=messages,
                    temperature=settings.temperature,
                    max_tokens=settings.max_tokens,
                    response_format={"type": "json_object"},
                )
                if "gpt-oss" in model_name:
                    # Reasoning models: keep the hidden "thinking" phase short so
                    # it doesn't eat the whole max_tokens budget before the model
                    # ever writes the JSON answer.
                    kwargs["reasoning_effort"] = settings.reasoning_effort
                    kwargs["reasoning_format"] = settings.reasoning_format
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as exc:
                status_code = getattr(exc, "status_code", None)
                if status_code == 429 and attempt < max_rate_limit_retries:
                    retry_after = None
                    response_obj = getattr(exc, "response", None)
                    if response_obj is not None:
                        retry_after = response_obj.headers.get("retry-after")
                    wait_seconds = float(retry_after) if retry_after else (2 ** attempt) * 2.0
                    logger.warning(
                        "Rate limited on %s (attempt %d/%d), waiting %.1fs before retry.",
                        model_name, attempt + 1, max_rate_limit_retries, wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    last_exc = exc
                    continue
                raise
        raise last_exc

    def _call_openai(self, messages) -> str:
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=settings.fallback_model,
            messages=messages,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _call_gemini(self, model_name: str, messages: List[Dict[str, str]], max_rate_limit_retries: int = 4) -> str:
        """
        Call the Gemini API (generateContent) via plain HTTP (stdlib urllib,
        no google-generativeai dependency required). Converts the
        [system, user] message list built by prompt_framework.build_messages
        into Gemini's system_instruction + contents shape, and asks for JSON
        output directly via responseMimeType.

        Retries on 429 with backoff, honoring a retry-after-style delay from
        the error body when Gemini provides one (quota exceeded errors often
        include a RetryInfo with a suggested delay in the JSON error body,
        unlike Groq which uses a plain retry-after header).
        """
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and set it as an "
                "environment variable before using a gemini-* model."
            )

        system_prompt = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Concatenate any non-system turns (user + prior assistant
                # retries) into the single user turn Gemini expects here --
                # this mirrors how llm_engine's JSON/schema retry logic
                # appends corrective follow-up messages for Groq/OpenAI.
                user_content += ("\n\n" if user_content else "") + msg["content"]

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model_name}:generateContent?key={settings.gemini_api_key}"
        )
        body = {
            "contents": [{"role": "user", "parts": [{"text": user_content}]}],
            "generationConfig": {
                "temperature": settings.temperature,
                "maxOutputTokens": settings.max_tokens,
                "responseMimeType": "application/json",
            },
        }
        if system_prompt:
            body["system_instruction"] = {"parts": [{"text": system_prompt}]}

        payload = json.dumps(body).encode("utf-8")
        last_exc = None

        for attempt in range(max_rate_limit_retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=settings.request_timeout_seconds) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates: {data}")
                parts = candidates[0].get("content", {}).get("parts", [])
                if not parts or "text" not in parts[0]:
                    raise RuntimeError(f"Gemini candidate had no text part: {candidates[0]}")
                return parts[0]["text"]
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                last_exc = RuntimeError(f"Gemini HTTPError {exc.code}: {error_body}")
                if exc.code == 429 and attempt < max_rate_limit_retries:
                    wait_seconds = (2 ** attempt) * 3.0  # simple exponential backoff
                    try:
                        # Gemini often nests a suggested retry delay inside the
                        # error body's RetryInfo detail, e.g. "retryDelay": "23s"
                        err_json = json.loads(error_body)
                        for detail in err_json.get("error", {}).get("details", []):
                            if "retryDelay" in detail:
                                wait_seconds = float(str(detail["retryDelay"]).rstrip("s"))
                                break
                    except Exception:
                        pass
                    logger.warning(
                        "Rate limited on %s (attempt %d/%d), waiting %.1fs before retry.",
                        model_name, attempt + 1, max_rate_limit_retries, wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue
                raise last_exc
            except Exception as exc:
                last_exc = exc
                raise
        raise last_exc

    def _call_model(self, model_name: str, messages) -> str:
        if _is_gemini_model(model_name):
            return self._call_gemini(model_name, messages)
        if model_name == settings.fallback_model and settings.fallback_provider == "openai":
            return self._call_openai(messages)
        # Default path: both primary and fallback are Groq-hosted, free-tier
        # models -- same client, just a different model name (see config.py
        # for current defaults, since Groq's free-tier lineup can change).
        return self._call_groq(model_name, messages)

    # -- public API ----------------------------------------------------

    def analyze_transcript(self, transcript: str) -> Dict[str, Any]:
        """
        Score a single Korean transcript across the 7 vishing attributes.

        Flow: route -> call model -> parse+validate JSON -> on malformed
        JSON, retry once against the SAME model with a corrective note ->
        on any hard failure of the primary model, escalate once to the
        fallback model -> if everything fails, return a schema-shaped
        error payload so downstream code never crashes.
        """
        if not transcript or not transcript.strip():
            logger.warning("Received empty transcript.")
            return AnalysisResult(
                payload=_empty_error_payload("empty transcript"),
                model_used="none",
                latency_seconds=0.0,
                retries=0,
                error="empty_transcript",
            ).to_dict()

        model_name = route_transcript(transcript)
        messages = build_messages(transcript, version=self.prompt_version)

        start = time.perf_counter()
        retries = 0
        fallback_triggered = False
        last_error: Optional[str] = None

        # Attempt sequence: (model, is_retry_of_malformed_json)
        for attempt_model, is_json_retry in self._attempt_plan(model_name):
            try:
                raw = self._call_model(attempt_model, messages)
            except Exception as exc:  # network/auth/rate-limit errors etc.
                last_error = f"{type(exc).__name__}: {exc}"
                logger.error("Model call failed (model=%s): %s", attempt_model, last_error)
                if attempt_model != settings.fallback_model:
                    fallback_triggered = True
                continue  # try next entry in the attempt plan

            if is_json_retry:
                retries += 1

            try:
                parsed = _extract_json(raw)
            except json.JSONDecodeError as exc:
                last_error = f"JSONDecodeError: {exc}"
                logger.warning("Malformed JSON from %s (attempt will retry once): %s", attempt_model, last_error)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Your previous response was not valid JSON. Reply again with ONLY the "
                        "raw JSON object matching the required schema, no other text."
                    )},
                ]
                continue

            is_valid, errors = validate_output(parsed)
            if not is_valid:
                last_error = "SchemaValidationError: " + "; ".join(errors)
                logger.warning("Schema validation failed from %s: %s", attempt_model, last_error)
                messages = messages + [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": (
                        "Your JSON did not match the required schema (" + "; ".join(errors) + "). "
                        "Reply again with ONLY corrected raw JSON."
                    )},
                ]
                continue

            latency = time.perf_counter() - start
            logger.info("analyze_transcript succeeded model=%s latency=%.3fs retries=%d",
                        attempt_model, latency, retries)
            # Fusion & Decision: recompute a weighted risk score independent of
            # whatever final_risk the LLM produced, and attach the tier.
            fused = compute_fusion_risk(parsed)
            parsed["fused_risk"] = fused
            parsed["risk_tier"] = risk_tier(fused)
            return AnalysisResult(
                payload=parsed,
                model_used=attempt_model,
                latency_seconds=latency,
                retries=retries,
                fallback_triggered=fallback_triggered or (attempt_model != model_name),
                error=None,
            ).to_dict()

        # Every attempt failed
        latency = time.perf_counter() - start
        logger.error("analyze_transcript FAILED after all attempts. last_error=%s", last_error)
        return AnalysisResult(
            payload=_empty_error_payload(last_error or "unknown error"),
            model_used=model_name,
            latency_seconds=latency,
            retries=retries,
            fallback_triggered=fallback_triggered,
            error=last_error,
        ).to_dict()

    def _attempt_plan(self, primary_choice: str):
        """
        Yields (model_name, is_json_retry) pairs in the order they should be
        attempted: primary model, one retry on the primary model (for
        malformed JSON / schema errors), then a single escalation to the
        other model as a last resort.
        """
        other = settings.fallback_model if primary_choice == settings.primary_model else settings.primary_model
        yield (primary_choice, False)
        for _ in range(settings.max_retries):
            yield (primary_choice, True)
        yield (other, False)


if __name__ == "__main__":
    # Manual smoke test (requires GROQ_API_KEY, or GEMINI_API_KEY if
    # PRIMARY_MODEL is set to a gemini-* model, in env; will raise otherwise).
    sample = "안녕하세요, 서울중앙지검 검사입니다. 귀하의 계좌가 범죄에 연루되어 즉시 확인이 필요합니다."
    engine = LLMEngine()
    print(json.dumps(engine.analyze_transcript(sample), ensure_ascii=False, indent=2))
