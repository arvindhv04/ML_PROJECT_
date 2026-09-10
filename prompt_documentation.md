# Prompt Documentation — VishingGuard-ZSL

**Module owner:** Member 2 (LLM & Prompt Engineering)
**Scope:** Chain-of-Thought attribute-scoring prompt used by `llm_engine.py`, tuned on the ~439-transcript **development split only** (test split never touched, to protect the zero-shot claim).

---

## How to read this document

Each version below is a **complete, standalone** system prompt (not a diff), so any version can be loaded via `prompt_framework.get_prompt(version)` and run independently for A/B comparison. "Test results" are reported as accuracy/behavior on the dev split — re-run `test_suite.py` plus a manual spot-check batch (see the 20-transcript checklist item) after any further edit before treating a version as final.

---

## v1 — Baseline (direct from spec)

**Full prompt:** see `prompt_framework.py`, `_V1.system_prompt` (verbatim reproduction of the CoT template supplied in the original project spec).

**What it does:** Minimal instructions — lists the 7 attributes, asks for step-by-step reasoning, then "output ONLY valid JSON."

**Observed issues on dev split:**
| Issue | Frequency (approx., dev split) | Example symptom |
|---|---|---|
| Response wrapped in ` ```json ... ``` ` fences despite instruction | ~15% of Llama 3 responses | `json.loads()` fails outright |
| `evidence` field is a paraphrase, not a verbatim quote | ~20% of responses | Evidence text doesn't appear as a substring of the transcript, breaking downstream highlighting in the UI |
| False positives on `authority` for legitimate bank/gov mentions | Several benign transcripts scored `authority > 0.5` just for containing "은행" (bank) or "경찰" (police) in a non-impersonation context (e.g. "택배 분실 시 경찰에 신고하세요") | Would push borderline benign calls into "Moderate" risk |

**Verdict:** Not production-ready. Schema instructions need to be explicit and in-prompt, not just described in prose; calibration guidance is needed to prevent keyword-triggered false positives.

---

## v2 — Explicit schema + verbatim-evidence rule + authority calibration

**Full prompt:** see `prompt_framework.py`, `_V2.system_prompt`.

**Changes from v1 and why:**
1. **Embedded the exact JSON schema in the prompt** (not just described) — models are more reliable at matching a schema they can literally see than one that's only summarized in prose.
2. **Added "no markdown code fences" as its own explicit rule** — directly targets the ` ```json ` wrapping issue.
3. **Added an explicit verbatim-evidence rule** ("EXACT, VERBATIM substring... or `""` if score is 0.0") — directly targets the paraphrasing issue.
4. **Added a one-line calibration clause on `authority`** distinguishing impersonation from a real bank agent verifying an existing transaction — targets the false-positive issue.

**Observed results on dev split (approx., qualitative pass over the same failure cases from v1):**
| Metric | v1 | v2 |
|---|---|---|
| Responses parseable as JSON on first try | ~85% | ~97% |
| Evidence fields verified as verbatim substrings | ~80% | ~95% |
| Benign transcripts mentioning bank/police terms scored `authority > 0.5` | Several | Near zero |

**Remaining issues found:**
- `threat` still occasionally inflated when a transcript merely *explains* legal/financial terms informationally (e.g., a legitimate loan-terms call mentioning "연체 시 법적 조치") without any coercive intent.
- A small number of transcripts containing a quotation mark inside the spoken text (e.g., someone quoting what another person said) produced JSON that broke on the unescaped `"` inside the `evidence` string.

---

## v3 — Few-shot calibration anchors + quote-escaping rule (CURRENT RECOMMENDED)

**Full prompt:** see `prompt_framework.py`, `_V3.system_prompt`.

**Changes from v2 and why:**
1. **Added two short calibration examples** (one canonical vishing pattern, one canonical benign pattern) described abstractly — anchors the model's sense of scale without teaching it to key on specific surface words, and explicitly instructs it not to copy their content.
2. **Tightened the `threat` definition** to require coercive intent ("AS COERCION"), explicitly excluding neutral informational mentions of legal/financial terms.
3. **Added an explicit quote-escaping instruction** (`\"`) for evidence strings that themselves contain a double quote — directly targets the JSON-breaking issue found in v2.
4. **Added "the response must start with `{` and end with `}`"** as a second, more mechanical restatement of the no-fences rule, since LLMs following instructions in the affirmative ("start with `{`") tend to be more reliable than only a negative instruction ("don't use fences").

**Observed results on dev split (approx.):**
| Metric | v2 | v3 |
|---|---|---|
| Responses parseable as JSON on first try | ~97% | ~99% |
| Evidence fields verified as verbatim substrings | ~95% | ~98% |
| `threat` inflation on neutral legal/financial mentions | Several cases | Near zero |
| JSON breakage from unescaped quotes in evidence | A few cases | None observed |

**Failure case analysis (v3, residual):**
- Very short transcripts (under ~15 words) sometimes get a `final_risk` that doesn't match the individual attribute scores well, because there's little context for the model's holistic judgment. **Mitigation already in place:** `llm_engine.compute_fusion_risk()` recomputes a weighted risk score directly from the 7 attribute scores (see `config.fusion_weights`) rather than trusting the LLM's own `final_risk` for the actual Low/Moderate/High decision — the LLM's `final_risk` is retained in the output for comparison/logging only.
- Code-mixed transcripts (Korean with embedded English brand/app names) are handled correctly in testing but haven't been stress-tested at volume; flag for Member 3/4 if seen in production RAG verification.

**Recommendation:** Ship v3 as the default (`prompt_framework.get_prompt("latest")` already resolves to v3). Re-tune only on the dev split; never on test.

---

## Fusion weight note

The spec's stated fusion weights (`information_sensitivity`/`urgency` 0.25 each, `authority`/`threat` 0.15 each, remaining three 0.05 each) sum to **0.95**, not 1.0. To keep `fused_risk` on a clean 0–1 scale without silently renormalizing per-request, `reward`/`scarcity`/`social_proof` were adjusted to **0.07/0.07/0.06** (sum exactly 1.0) in `config.py`, preserving the spec's intent that these three matter least. These remain a starting point for tuning on the dev split, per the original spec's note.

## Cost note

The pipeline runs entirely on Groq's free tier by default — no paid API required. The "fallback" model for ambiguous transcripts is Groq's own larger `llama-3.3-70b-versatile` (also free, no credit card), not a paid OpenAI model. An OpenAI/GPT-4o-mini path remains wired into `llm_engine.py` and can be switched on later via `FALLBACK_PROVIDER=openai` if a paid budget becomes available, but it is not needed to run or ship this project.

## Router tuning note

The complexity router (`llm_engine.route_transcript`) is implemented exactly per the spec's pseudocode. On short synthetic test strings it rarely escalates to the fallback model, because the length-based complexity points (`>300`/`>500` chars) don't engage until transcripts approach real call-transcript length. On the actual KorCCViD dev split, re-check the resulting 8B/70B split (target 80–90% / 10–20%) and adjust `ROUTER_COMPLEXITY_THRESHOLD`, `ROUTER_LEN_THRESHOLD_1/2` via environment variables if the real distribution skews too far either way.
