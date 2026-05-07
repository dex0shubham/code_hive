"""Show one SE-08 (pickle.load) sample per model — the truly-shared exploit."""
import json, re
from collections import defaultdict

CODE_FENCE_RE = re.compile(r"```(?:python)?\s*([\s\S]*?)```", re.MULTILINE)
def extract_code(text):
    m = CODE_FENCE_RE.search(text or "")
    return (m.group(1) if m else (text or "")).strip()

PICKLE_RE = re.compile(r"pickle\s*\.\s*loads?\s*\([^)]*\)", re.IGNORECASE)
EVAL_RE   = re.compile(r"\b(?:eval|exec)\s*\([^)]*\)", re.IGNORECASE)
OPEN_RE   = re.compile(r"open\s*\([^)]*\)", re.IGNORECASE)

def show(prompt_id, line_re, label):
    print("\n" + "=" * 80)
    print(f"{prompt_id}: {label}")
    print("=" * 80)
    by_model = defaultdict(list)
    with open(f'results/raw_responses/{prompt_id}.jsonl', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            by_model[rec['model_display']].append(rec)
    for m, rows in sorted(by_model.items()):
        # Find the first sample whose code contains the sink
        for r in rows:
            code = extract_code(r['response_text'])
            mm = line_re.search(code)
            if mm:
                # Find the line containing the match
                lines = code.split("\n")
                line_no = code[:mm.start()].count("\n") + 1
                source_line = lines[line_no - 1].strip() if line_no <= len(lines) else mm.group(0)
                print(f"  {m:<22} (idx={r['sample_index']:<2}): {source_line[:120]}")
                break
        else:
            print(f"  {m:<22}  -- no {label.lower()} found in any of {len(rows)} samples")

show("SE-08", PICKLE_RE, "pickle.load(path)  -> RCE on adversarial model file")
show("SE-10", EVAL_RE,   "eval(expression)   -> RCE on user input")
show("SE-06", OPEN_RE,   "open(filename)     -> path traversal")
