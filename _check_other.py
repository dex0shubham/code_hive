import json, re
from collections import defaultdict

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.MULTILINE)
def extract_code(text):
    m = CODE_FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()

target_models = {"Gemini 2.5 Pro", "Claude 3.5 Haiku", "Gemini 2.5 Flash"}

with open('results/raw_responses/SE-04.jsonl', encoding='utf-8') as f:
    for line in f:
        if not line.strip(): continue
        rec = json.loads(line)
        if rec['model_display'] not in target_models: continue
        if rec['sample_index'] != 0: continue
        code = extract_code(rec['response_text'])
        print("=" * 80)
        print(f"{rec['model_display']} sample 0:")
        print("=" * 80)
        print(code[:2000])
        print()
