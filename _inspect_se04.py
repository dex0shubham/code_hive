import json
from collections import defaultdict

by_model = defaultdict(list)
with open('results/raw_responses/SE-04.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        by_model[rec['model_display']].append(rec)

for m, rows in by_model.items():
    temps = sorted({r['temperature'] for r in rows})
    print(f"{m}: {len(rows)} samples, temps={temps}, family={rows[0]['model_family']}")
