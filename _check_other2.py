import json
target_models = {"Gemini 2.5 Pro", "Claude 3.5 Haiku", "Gemini 2.5 Flash"}
with open('results/raw_responses/SE-04.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        if rec['model_display'] not in target_models: continue
        if rec['sample_index'] != 0: continue
        print("=" * 80)
        print(f"{rec['model_display']} sample 0 (raw response_text):")
        print("=" * 80)
        print(rec['response_text'][:1500])
        print()
