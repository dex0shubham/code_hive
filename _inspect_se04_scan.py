import json
from collections import defaultdict, Counter

scan_path = 'results/proof_security/sec_20260505T211023Z/scan_results.jsonl'

by_model = defaultdict(list)
with open(scan_path, encoding='utf-8') as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec['prompt_id'] != 'SE-04':
            continue
        by_model[rec['model_display']].append(rec)

print("SE-04 vulnerability per model:\n")
for m, rows in by_model.items():
    n = len(rows)
    nv = sum(1 for r in rows if r['is_vulnerable'])
    cwes = Counter(c for r in rows for c in r['cwe_set'])
    sigs = Counter(tuple(tuple(p) for p in r['pattern_signature']) for r in rows)
    print(f"  {m}: {nv}/{n} vulnerable")
    print(f"    CWEs: {dict(cwes)}")
    for sig, cnt in sigs.most_common(3):
        print(f"    sig({cnt}): {sig}")
    print()
