import json
target_models = {"Gemini 2.5 Pro", "Claude 3.5 Haiku", "Gemini 2.5 Flash"}
counts = {m: 0 for m in target_models}
with open('results/raw_responses/SE-04.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        if rec['model_display'] not in target_models: continue
        counts[rec['model_display']] += 1
        if rec['sample_index'] in (0, 1, 5):
            print("-" * 80)
            print(f"{rec['model_display']} idx={rec['sample_index']}  "
                  f"finish={rec.get('finish_reason')}  "
                  f"len(text)={len(rec.get('response_text') or '')}")
            print(repr(rec.get('response_text', ''))[:400])
print()
print("counts:", counts)
