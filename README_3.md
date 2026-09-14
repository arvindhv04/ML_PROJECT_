# VishingGuard-ZSL

Zero-shot Korean voice-phishing (vishing) detection. LLMs are used purely
as **reasoning engines** — there is no model training/fine-tuning anywhere
in this repo.

```
Input Transcript (Korean)
   → [Member 3] Input Sanitization         (sanitization.py)
   → [Member 2] Complexity Router
   → [Member 2] CoT Attribute Scoring (LLM) (llm_engine.py, prompt_framework.py)
   → [Member 3] RAG Verification            (rag_verification.py, vector_store.py)
   → [Member 2] Fusion & Decision
   → Output: Risk tier + explanation
```

## Dataset

**`final_dataset_m16.csv`** is the canonical labeled dataset (columns:
`id,transcript,label,source,category`, `label=1` vishing / `label=0`
benign, utf-8-sig). Actual counts in the file as shipped:

| source | rows | label |
|---|---|---|
| `KorCCViD_cleaned` | 692 | vishing (1) |
| `NIKL_existing` | 2,232 | benign (0) |
| `synthetic` | 268 | benign (0), bonus validation set |
| **Total** | **3,192** | 692 vishing / 2,500 benign |

(The original spec figures were 695 vishing / 2,232 benign / 2,927 total
before cleaning — `KorCCViD_cleaned` dropped 3 rows during data cleaning.)

All transcripts stay in Korean; nothing is translated. `vishing_patterns.csv`
is the pre-extracted vishing-only subset (692 rows) pulled straight from
`final_dataset_m16.csv` — use it if you specifically want just the vishing
class without filtering the full file yourself. The older `dev_m16.csv`
(2,912 rows, an earlier/partial split) is kept in the repo for reference
but is no longer the default — every Member 3 script below now points at
`final_dataset_m16.csv`. **Member 2's `batch_eval.py` / `inspect_one.py` /
`prompt_documentation.md` still reference `dev_m16.csv` in their examples**
— those take the CSV path as an explicit argument either way, so they still
run; flag to Member 2 whether the LLM-engine eval should also move to
`final_dataset_m16.csv` for consistency.

## Repo layout

| Area | Files | Owner |
|---|---|---|
| LLM scoring engine, prompts, fusion | `llm_engine.py`, `prompt_framework.py`, `config.py`, `prompt_documentation.md` | Member 2 |
| Weight tuning / offline eval | `offline_weight_search.py`, `batch_eval.py`, `weight_tuning_*.csv`, `batch_eval_results.csv` | Member 2 |
| **Input sanitization** | `sanitization.py` | **Member 3** |
| **RAG pattern store + verification** | `vector_store.py`, `rag_verification.py` | **Member 3** |
| **Adversarial test-set generation** | `adversarial_generator.py` | **Member 3** |
| **Security unit tests** | `security_test_suite.py` | **Member 3** |
| Debug helpers | `inspect_one.py`, `check_benign_spikes.py`, `check_floor_safety.py` | Member 2 |
| Integration docs | `integration_guide.md` (LLM engine, for Member 4), `member3_integration_guide.md` (security layer, for Members 2 & 4) | — |

## Quickstart

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env      # fill in your free API key(s), never commit .env
python config.py          # sanity-check config
```

### Member 3's security layer only

```bash
# 1. Populate the local vector store from the full labeled dataset
#    (first run downloads all-MiniLM-L6-v2, ~80MB, once; default CSV is final_dataset_m16.csv)
python vector_store.py --reset
# or, to store ONLY the 692 vishing rows (matches the original spec exactly):
python vector_store.py --csv vishing_patterns.csv --vishing-only --reset

# 2. Run the security unit tests (offline — uses an in-memory hashing embedder, no downloads)
python -m unittest security_test_suite -v
# or: pytest security_test_suite.py -v

# 3. Generate the 100 adversarial/benign pairs (local, free, no API key needed;
#    pulls from the 692 real vishing rows in final_dataset_m16.csv)
python adversarial_generator.py --n 100 --out adversarial_testset.json

# 4. Try sanitization + RAG verification directly
python sanitization.py
python rag_verification.py
```

Verified locally (no torch/sentence-transformers needed for this check —
uses the deterministic `HashingEmbedder`, same one `security_test_suite.py`
uses): populating from `final_dataset_m16.csv` embeds all 3,192 rows
(692 vishing / 2,500 benign), and querying with a held-out vishing
transcript correctly retrieves vishing neighbours as the nearest matches.
For production-quality retrieval, install `sentence-transformers` so
`vector_store.py` uses real `all-MiniLM-L6-v2` embeddings instead.

See **`member3_integration_guide.md`** for exactly how Member 2 and Member 4
call into `sanitize_transcript()` and `verify_decision()`.

## Cost

**$0.** Sanitization and RAG verification run entirely locally (ChromaDB +
`all-MiniLM-L6-v2` via `sentence-transformers`). The adversarial generator
defaults to a local deterministic rewriter; an optional `--backend groq`
flag can use Member 2's free Groq key instead of an API-cost path — no
paid APIs are used anywhere in this repo.

## Member 3 — Adversarial & Security Lead: deliverables status

| # | File | Status |
|---|---|---|
| 1 | `sanitization.py` | ✅ regex + Unicode defenses for EN/KO injections, control/zero-width stripping, 5000-char cap, whitespace normalization, flag-for-review, JSONL audit log |
| 2 | `rag_verification.py` | ✅ `verify_decision()` → CONFIRMED / REDUCE_CONFIDENCE / FLAG_FOR_REVIEW + confidence-adjusted `fused_risk` |
| 3 | `vector_store.py` | ✅ persistent ChromaDB, `all-MiniLM-L6-v2` embeddings, `populate_from_csv`, `add_pattern`, `query_similar` — now defaults to `final_dataset_m16.csv` (3,192 rows) |
| 4 | `adversarial_generator.py` | ✅ 100 adversarial rewrites + 100 paired benign paraphrases, saved to JSON with metadata — regenerated from the real 692 vishing rows in `final_dataset_m16.csv` |
| 5 | `security_test_suite.py` | ✅ 29 tests: 10+ injection attacks, RAG retrieval, adversarial quality, Member 2 JSON integration, edge cases |
| 6 | `member3_integration_guide.md` | ✅ how Members 2 & 4 call the security layer |
| — | `vishing_patterns.csv` | ✅ **new** — the 692 vishing-only rows extracted from `final_dataset_m16.csv` (`label==1`), for scripts/spec paths that want vishing-only input |

All 29 tests in `security_test_suite.py` pass locally with `$0` spend
(`python -m unittest security_test_suite -v`).

### Performance targets

- Sanitization catch rate: **>95%** of known injection patterns (verified by `test_ten_known_injections_are_caught`)
- RAG retrieval: top-3 neighbours returned per query, cosine similarity via ChromaDB `hnsw:space=cosine`
- Adversarial set size: 100 transcripts + 100 paired benign paraphrases
- Vector store: populated from all 3,192 rows in `final_dataset_m16.csv` (692 vishing when `--vishing-only` / `vishing_patterns.csv` is used, matching the original spec; both classes stored by default so RAG can also return benign neighbours)
- API cost: **$0** — everything security-related runs locally

## Notes on deviations from the original spec

- **Vector store default stores both classes**, not only the 695 vishing
  rows. Storing benign patterns too is what lets `verify_decision()` return
  a `REDUCE_CONFIDENCE` verdict when the LLM says "vishing" but the nearest
  neighbours are actually benign. Pass `--vishing-only` to `vector_store.py`
  to match the original vishing-only spec exactly.
- **Adversarial generator defaults to a local, deterministic rewriter**
  instead of always calling an LLM, so the 100-pair target is reachable at
  $0 even without a Groq key. `--backend groq` is available as an opt-in
  using Member 2's existing free-tier key.
- See `integration_guide.md` for Member 2's own deviation note (Groq moved
  `llama3-8b-8192` off the free tier mid-project; both primary/fallback
  models now default to free `gpt-oss` models on Groq, with an optional
  Gemini path — see `.env.example` / `config.py`).

## Security & safety notes

- `sanitization.py` never silently "fixes and forwards" a suspicious
  transcript — anything flagged returns `None` so the caller **must** route
  it to manual review instead of auto-scoring it.
- The adversarial dataset in `adversarial_generator.py` is evaluation data
  only: it rewrites *existing labeled vishing transcripts* from KorCCViD to
  strip obvious keywords while preserving intent, purely to test whether
  the detector generalizes past surface-level lexicon matching. It does not
  generate any new attack technique.
