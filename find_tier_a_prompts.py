#!/usr/bin/env python3
"""
Scan the Pool-B Python prompts and surface candidates that are likely to
elicit "Tier A" (high-severity / exploit-shaped) vulnerabilities from LLMs.

We score each prompt across several Tier-A CWE categories using keyword
heuristics, then pick the highest-scoring distinct prompts per category so
you get a diverse, security-relevant subset for code generation.

Outputs a JSON manifest listing the chosen prompts.

Usage:
  python find_tier_a_prompts.py
  python find_tier_a_prompts.py --top-n 5 --out local_datasets/sample_tier_a_prompts.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "local_datasets" / "pool_b_open_ended_candidates" / "python_tasks" / "data.jsonl"


# Each category lists keyword regexes that suggest the prompt invites that risk class.
# Patterns are matched case-insensitively against the prompt text.
TIER_A_CATEGORIES: dict[str, list[str]] = {
    "RCE_command_injection": [
        r"\bexecute\s+(?:a|the)?\s*(?:shell\s+)?command",
        r"\brun\s+(?:a|the)?\s*(?:shell|system)?\s*command",
        r"\bsubprocess\b",
        r"\bsystem\(",
        r"\bping\b.*\b(host|ip|url)\b",
        r"\bspawn\s+a\s+process",
        r"\bgit\s+commands?\b",
        r"\bbash\b|\bshell\b",
    ],
    "deserialization": [
        r"\bpickle\b|\bunpickle\b",
        r"\byaml\b.*\b(load|parse)\b",
        r"\bmarshal\b",
        r"\bdeserial(ise|ize)",
        r"\bload(?:s)?\s+(?:from\s+)?(?:a\s+)?(?:pickle|yaml|file)\b",
        r"\bsave\s+(?:and\s+)?(?:load|restore)\s+(?:object|state|model)\b",
    ],
    "sql_injection": [
        r"\bsql\b",
        r"\bsqlite\b|\bsqlite3\b",
        r"\bmysql\b|\bpostgres\b|\bpostgresql\b",
        r"\b(?:execute|run)\s+(?:a\s+)?(?:sql\s+)?query\b",
        r"\bwhere\s+\w+\s*=\s*['\"]?\s*\{",
        r"\bsearch\s+(?:by|for)\s+\w+\s+in\s+(?:the\s+)?database",
    ],
    "eval_exec": [
        r"\beval\b",
        r"\bexec\b",
        r"\bcalculator\b",
        r"\b(?:evaluate|interpret)\s+(?:user|the)\s+(?:input|expression)\b",
        r"\bmath\s+expression\b|\barithmetic\s+expression\b",
        r"\bdsl\b|\bdomain[- ]specific language\b",
    ],
    "path_traversal_ssrf": [
        r"\bdownload\s+(?:a\s+)?file\b",
        r"\bupload\s+(?:a\s+)?file\b",
        r"\bopen\s+(?:a\s+)?file\s+from\s+(?:user|the\s+user|input)",
        r"\bproxy\s+(?:server|request)",
        r"\bfetch\s+(?:the\s+)?url\b",
        r"\bredirect\s+to\s+(?:url|user)",
        r"\bserve\s+(?:a\s+)?file\b|\bstatic\s+file\s+server\b",
    ],
    "weak_crypto_auth": [
        r"\bhash\s+(?:the\s+)?password\b",
        r"\bencrypt\b.*\bpassword\b",
        r"\bjwt\b|\btoken\b.*\bauth(?:entication|orize)?\b",
        r"\bsession\s+(?:id|token)\b",
        r"\bgenerate\s+(?:a\s+)?(?:secure\s+)?(?:random\s+)?(?:token|key|password)\b",
        r"\botp\b|\bone[- ]time\s+password\b",
    ],
    "xxe_xpath": [
        r"\bxml\b.*\b(parse|load|process)\b",
        r"\bxpath\b",
        r"\bsoap\b",
    ],
    "ssti_template": [
        r"\b(jinja|template)\b.*\brender\b",
        r"\brender\s+(?:a\s+)?template\s+with\s+user",
    ],
    "ldap_no_sql": [
        r"\bldap\b",
        r"\bmongo(?:db)?\b",
        r"\bnosql\b",
    ],
}


def score_prompt(prompt: str) -> dict[str, int]:
    text = prompt.lower()
    scores: dict[str, int] = {}
    for cat, patterns in TIER_A_CATEGORIES.items():
        scores[cat] = sum(1 for p in patterns if re.search(p, text))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=5,
                    help="How many prompts to keep (default 5).")
    ap.add_argument("--out", type=str,
                    default="local_datasets/sample_tier_a_prompts.json")
    ap.add_argument("--min-prompt-length", type=int, default=40)
    ap.add_argument("--max-prompt-length", type=int, default=600)
    args = ap.parse_args()

    candidates: list[dict] = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("task_language") != "PYTHON":
                continue
            prompt = (rec.get("prompt") or "").strip()
            if not prompt:
                continue
            if len(prompt) < args.min_prompt_length or len(prompt) > args.max_prompt_length:
                continue
            scores = score_prompt(prompt)
            total = sum(scores.values())
            if total == 0:
                continue
            top_cat = max(scores, key=lambda k: scores[k])
            candidates.append({
                "row_index":       rec.get("row_index"),
                "dataset":         rec.get("dataset"),
                "label":           rec.get("label"),
                "openness_score":  rec.get("openness_score"),
                "prompt":          prompt,
                "tier_a_scores":   scores,
                "tier_a_total":    total,
                "primary_category": top_cat,
            })

    # Pick top-1 per primary_category, then fill with highest-total remaining
    candidates.sort(key=lambda c: (-c["tier_a_total"], -len(c["prompt"])))
    picked: list[dict] = []
    used_categories: set[str] = set()
    for c in candidates:
        if c["primary_category"] not in used_categories:
            picked.append(c)
            used_categories.add(c["primary_category"])
        if len(picked) >= args.top_n:
            break
    if len(picked) < args.top_n:
        for c in candidates:
            if c not in picked:
                picked.append(c)
            if len(picked) >= args.top_n:
                break

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "n_candidates_scored":  len(candidates),
        "n_picked":             len(picked),
        "categories_covered":   sorted(used_categories),
        "prompts":              picked,
    }, indent=2), encoding="utf-8")

    print(f"Scored {len(candidates)} candidate prompts; picked {len(picked)}.")
    print(f"Categories covered: {sorted(used_categories)}")
    for i, p in enumerate(picked, 1):
        print(f"\n[{i}] row_index={p['row_index']}  category={p['primary_category']}  "
              f"score={p['tier_a_total']}")
        print(f"    {p['prompt'][:160]}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
