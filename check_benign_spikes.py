import csv

rows = list(csv.DictReader(open('weight_tuning_gemini.csv', encoding='utf-8-sig')))
attrs = ['authority_score', 'urgency_score', 'information_sensitivity_score',
         'threat_score', 'reward_score', 'scarcity_score', 'social_proof_score']

print("Checking benign rows for any single attribute >= 0.5...")
found_any = False
for r in rows:
    if r['true_is_vishing'] == 'False':
        scores = {a: float(r[a]) for a in attrs}
        top_attr = max(scores, key=scores.get)
        top_val = scores[top_attr]
        if top_val >= 0.5:
            found_any = True
            print(f"  {r['id']}: {top_attr}={top_val}  (all: {scores})")

if not found_any:
    print("  None found -- no benign row has any attribute >= 0.5.")
