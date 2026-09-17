# RAG Verification Guide (Standalone)

This project can run the RAG verification step without Member 4.

The verifier:
- loads the vector store
- populates it from `final_dataset_m16.csv` if empty
- retrieves the nearest matching transcripts
- compares the retrieved evidence against the LLM verdict
- adjusts the risk score and returns a verdict

## Files involved
- `rag_verification.py` — standalone verification logic
- `vector_store.py` — ChromaDB vector store and CSV loader
- `final_dataset_m16.csv` — canonical labeled transcript dataset

## 1) Make sure the dataset exists
The default CSV is:

```bash
final_dataset_m16.csv
```

This file contains the transcript rows used for retrieval.

## 2) Populate the vector store
If the store is empty, the script will populate it automatically. But you can also do it explicitly:

```bash
python3 vector_store.py --csv final_dataset_m16.csv --reset
```

This creates and loads the ChromaDB collection from the labeled transcript rows.

## 3) Run analysis with a transcript
### Recommended: paste the transcript

You do not need to create JSON or calculate a risk score yourself. Run:

```bash
python3 rag_verification.py
```

Then paste the transcript exactly as it was transcribed. Press **Enter on an
empty line** when you are finished. For example:

```text
Caller: This is the bank security team. Your account is at risk.
Caller: Tell me the one-time password we just sent to your phone.
```

The script sends the transcript through the configured LLM scorer and then
checks its result against the local RAG evidence store.

### Option A: transcript on the command line
```bash
python3 rag_verification.py --transcript "Caller asks for bank details and OTP."
```

### Recommended for a dataset row

To analyze the first transcript in the dataset, run:

```bash
python3 rag_verification.py \ --csv final_dataset_m16.csv
```

The script reads row `0`, sends its transcript through the configured LLM
scorer, and then runs RAG verification. Row `0` is a benign travel
conversation, so zero scores for all seven attack attributes are correct.

To test a vishing transcript with nonzero attribute scores, use the first
vishing-labeled row:

```bash
python3 rag_verification.py \ --csv final_dataset_m16.csv --row-index 2500
```

To use any other row, replace `2500` with its zero-based row index.

Do not add `--llm-result`; the script calculates the LLM result automatically.

### Option C: replaying an existing LLM result

This is kept for integrations and testing when the LLM result already exists:

```bash
python3 rag_verification.py \
  --transcript "caller asks for bank details and OTP" \
  --llm-result-file llm_result.json
```

## 4) What the output means
The script returns a JSON object with fields like:

```json
{
  "fused_risk": 0.533,
  "risk_tier": "Moderate",
  "rag": {
    "verdict": "REDUCE_CONFIDENCE",
    "verification_score": 0.4398,
    "reason": "LLM says vishing but nearest patterns are benign — lowering fused_risk.",
    "needs_manual_review": false,
    "retrieved_ids": ["BENIGN_1", "BENIGN_1697", "BENIGN_1622"]
  }
}
```

Possible verdicts:
- `CONFIRMED` — LLM verdict agrees with retrieved evidence
- `REDUCE_CONFIDENCE` — LLM says vishing but nearby patterns are benign
- `FLAG_FOR_REVIEW` — retrieval is weak or contradictory

## 5) Python API
You can also call it directly in Python:

```python
from rag_verification import run_rag_verification

llm_result = {
    "fused_risk": 0.82,
    "risk_tier": "High"
}

transcript = "caller asks for bank details and OTP"

result = run_rag_verification(llm_result, transcript)
print(result)
```

## Notes
- This does not require Member 4.
- The project uses the vector store as the evidence layer.
- The default retrieval threshold is similarity-based and the verifier automatically ignores distant neighbors.
