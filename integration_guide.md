# Integration Guide — LLM & Prompt Engineering Module (Member 2)

**For:** Member 4 (API / FastAPI integration)
**Pipeline position:** `[Member 3: Input Sanitization] → [Complexity Router] → [CoT Attribute Scoring] → [Member 3: RAG Verification] → [Fusion & Decision] → Output`
This module owns the middle three boxes: Complexity Router, LLM Scoring, and Fusion.

---

## 1. Install dependencies

```bash
pip install groq
```

By default, **both** the primary and fallback models run on Groq's free tier (no credit card ever required) — `openai` is only needed if you later set `FALLBACK_PROVIDER=openai` to use a paid GPT-4o-mini fallback. Until then, skip installing it.

## 2. Environment variables to set

| Variable | Required | Default | Notes |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Free, no credit card. Powers both primary and fallback models by default |
| `OPENAI_API_KEY` | No | — | Only needed if `FALLBACK_PROVIDER=openai` |
| `PRIMARY_MODEL` | No | `llama3-8b-8192` | Free Groq model |
| `FALLBACK_MODEL` | No | `llama-3.3-70b-versatile` | Free Groq model, used for ambiguous transcripts |
| `FALLBACK_PROVIDER` | No | `groq` | Set to `openai` later if a paid budget becomes available |
| `LLM_TEMPERATURE` | No | `0.0` | |
| `MAX_TOKENS` | No | `1024` | |
| `ROUTER_COMPLEXITY_THRESHOLD` | No | `3` | Raise to send less traffic to the fallback model |
| `HIGH_RISK_THRESHOLD` | No | `0.7` | |
| `MODERATE_RISK_THRESHOLD` | No | `0.4` | |
| `LOG_LEVEL` | No | `INFO` | |

Set these in your shell, a `.env` file loaded by your process manager, or your deployment platform's secrets manager. **Never commit keys to the repo.**

## 3. Input format expected

A single **Korean-language transcript string** (the full call transcript, or the sanitized output of Member 3's input-sanitization step). No translation needed — pass Korean text directly.

```python
transcript = "여보세요, 서울중앙지검 수사관입니다. 계좌 확인이 긴급합니다..."
```

## 4. How to call it

```python
from llm_engine import LLMEngine

engine = LLMEngine()  # instantiate once, reuse across requests
result = engine.analyze_transcript(transcript)
```

For a FastAPI endpoint:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from llm_engine import LLMEngine

app = FastAPI()
engine = LLMEngine()  # module-level singleton; safe to share across requests

class TranscriptRequest(BaseModel):
    transcript: str

@app.post("/analyze")
def analyze(req: TranscriptRequest):
    return engine.analyze_transcript(req.transcript)
```

## 5. Output format guaranteed

`analyze_transcript()` **always** returns a `dict` — it never raises, even on total API failure (network errors, malformed JSON after retries, etc.). Downstream code can rely on every key below being present:

```jsonc
{
  "authority":                {"score": 0.0, "evidence": ""},
  "urgency":                  {"score": 0.0, "evidence": ""},
  "information_sensitivity":  {"score": 0.0, "evidence": ""},
  "threat":                   {"score": 0.0, "evidence": ""},
  "reward":                   {"score": 0.0, "evidence": ""},
  "scarcity":                 {"score": 0.0, "evidence": ""},
  "social_proof":             {"score": 0.0, "evidence": ""},
  "final_risk": 0.0,            // the LLM's own holistic estimate (0-1)
  "fused_risk": 0.0,             // weighted recomputation from the 7 scores — USE THIS for the risk decision
  "risk_tier": "Low",            // "Low" | "Moderate" | "High", derived from fused_risk
  "overall_explanation": "...",
  "reasoning_trace": "...",
  "_meta": {
    "model_used": "llama3-8b-8192",   // or "llama-3.3-70b-versatile" for escalated/ambiguous cases (both free Groq models)
    "latency_seconds": 0.842,
    "retries": 0,
    "fallback_triggered": false,
    "error": null                      // non-null string if every attempt ultimately failed
  }
}
```

**Recommendation for Member 4:** use `fused_risk` / `risk_tier` for the actual product decision (these are computed deterministically from the 7 attribute scores via `config.fusion_weights`, independent of the LLM's own possibly-uncalibrated `final_risk`). Surface `final_risk` and `reasoning_trace` in the UI as supporting explanation. If `_meta.error` is non-null, treat the result as low-confidence (all scores will be `0.0` / tier `"Low"`) and consider surfacing a "could not analyze" state rather than silently trusting a Low verdict — check `_meta.error` before treating a Low tier as a real Low.

## 6. Where this module fits vs. Member 3's RAG verification

This module's output feeds into Member 3's RAG verification step next. Pass the full dict (including `_meta`) through — Member 3's step may want `reasoning_trace` and per-attribute `evidence` for cross-checking against retrieved reference cases.

## 7. Performance expectations

- Target average latency: **under 2 seconds** per transcript. Both models run on Groq's LPU hardware (very fast); the ~10-20% of transcripts routed to the larger `llama-3.3-70b-versatile` fallback will be somewhat slower than the 8B primary model, but still free and still fast.
- Expect occasional retries (one automatic retry on malformed JSON/schema errors) and rare fallback escalations on primary-model outages — both are handled transparently inside `analyze_transcript()`; no special handling needed by Member 4 beyond checking `_meta`.
- No rate-limit handling beyond the built-in retry/fallback is implemented; if Groq free-tier limits become an issue at demo time, that's a natural extension point in `LLMEngine._call_groq`.
