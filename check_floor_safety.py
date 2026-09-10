import csv

rows = list(csv.DictReader(open('weight_tuning_gemini.csv', encoding='utf-8-sig')))
attrs = ['authority_score', 'urgency_score', 'information_sensitivity_score',
         'threat_score', 'reward_score', 'scarcity_score', 'social_proof_score']

benign_max = 0.0
benign_max_row = None
for r in rows:
    if r['true_is_vishing'] == 'False':
        scores = {a: float(r[a]) for a in attrs}
        top_attr = max(scores, key=scores.get)
        top_val = scores[top_attr]
        if top_val > benign_max:
            benign_max = top_val
            benign_max_row = (r['id'], top_attr, top_val)

print(f"Highest single attribute score across ALL benign rows: {benign_max}")
if benign_max_row:
    print(f"  -> {benign_max_row[0]}: {benign_max_row[1]} = {benign_max_row[2]}")

vishing_scores = []
for r in rows:
    if r['true_is_vishing'] == 'True':
        scores = {a: float(r[a]) for a in attrs}
        top_attr = max(scores, key=scores.get)
        top_val = scores[top_attr]
        vishing_scores.append((r['id'], r['predicted_tier'], top_attr, top_val))

print("\nVishing rows -- top single attribute each:")
for id_, tier, attr, val in vishing_scores:
    print(f"  {id_}: pred={tier:8s} top_attr={attr}={val}")

print(f"\nSafety margin: for a floor rule (max_attr * FACTOR) to never false-positive on benign,")
print(f"FACTOR must be < {round(0.4/benign_max, 3) if benign_max > 0 else 'inf'} "
      f"(so that {benign_max} * FACTOR stays under the 0.4 Moderate threshold).")
