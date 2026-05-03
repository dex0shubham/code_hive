#!/usr/bin/env python3
"""
fetch_human_baseline.py
=======================
Pull human-written reference solutions from the APPS dataset
(Hendrycks et al. 2021, https://huggingface.co/datasets/codeparrot/apps)
and emit them as `<human-dir>/<PROMPT_ID>.jsonl` files compatible with
`proof_homogeneity.py`.

Why APPS: ~10k programming problems with up to ~50 human Python solutions
each, so we get *intra-task* human diversity for the LLM-vs-human comparison
in Pillar 2. We match by question-text keywords (no LLM-based matching).

This script is INDEPENDENT of the rest of the pipeline. It only needs:
  pip install datasets

Usage
-----
  # Default: AL-01, AL-03, AL-05 -> results/human_baseline/
  python fetch_human_baseline.py

  # Custom prompts and output dir, with a per-prompt cap
  python fetch_human_baseline.py \
      --prompts AL-01 AL-03 AL-05 \
      --out-dir results/human_baseline \
      --max-solutions-per-prompt 30 \
      --max-problems-per-prompt 5

How matching works
------------------
For each prompt id we have a small dictionary of must-match keywords
(`MUST_HAVE`) and at-least-one keywords (`ANY_OF`). A problem is included
only if its `question` text contains all MUST_HAVE words and at least one
ANY_OF word. From matching problems we pull up to N solutions, lightly
deduplicate, and lint-pass them (must `ast.parse`).

Output schema (per JSONL line)
------------------------------
  {
    "prompt_id":   "AL-01",
    "source":      "APPS",
    "problem_id":  <APPS index>,
    "split":       "train" | "test",
    "code":        "<python source>",
    "n_chars":     <int>,
    "match_keywords": [...]
  }
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# Each prompt id: keywords that must ALL be present, plus at least one of ANY_OF.
@dataclass
class MatchSpec:
    must_have: list[str]
    any_of: list[str]


PROMPT_MATCH: dict[str, MatchSpec] = {
    # AL-01: "find all anagram groups in a list of words"
    "AL-01": MatchSpec(
        must_have=["anagram"],
        any_of=["group", "list", "words", "strings"],
    ),
    # AL-03: "detect cycles in a directed graph"
    "AL-03": MatchSpec(
        must_have=["cycle"],
        any_of=["graph", "directed", "adjacency", "node"],
    ),
    # AL-05: "generate all valid combinations of n pairs of balanced parentheses"
    "AL-05": MatchSpec(
        must_have=["parenthes"],
        any_of=["balanced", "valid", "generate", "combinations"],
    ),
    # Common alternative phrasing (kept here as easy add-ons for later prompts)
    "OD-01": MatchSpec(
        must_have=["cache"],
        any_of=["expir", "ttl", "lru", "memor"],
    ),
}


def _normalise_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).lower()


def _matches(question: str, spec: MatchSpec) -> list[str]:
    """Return the matched keywords if the question matches, else []."""
    qn = _normalise_text(question)
    matched: list[str] = []
    for kw in spec.must_have:
        if kw not in qn:
            return []
        matched.append(kw)
    if spec.any_of:
        any_hits = [kw for kw in spec.any_of if kw in qn]
        if not any_hits:
            return []
        matched.extend(any_hits)
    return matched


def _is_python(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _hash(code: str) -> str:
    return hashlib.sha1(code.strip().encode("utf-8")).hexdigest()


def _parse_str_or_list(v):
    """Some HF dumps store list-typed columns as JSON strings."""
    if v is None:
        return []
    if isinstance(v, str):
        try:
            x = json.loads(v)
        except json.JSONDecodeError:
            return []
        return list(x) if isinstance(x, (list, tuple)) else []
    if isinstance(v, (list, tuple)):
        return list(v)
    return []


def _map_apps(row, idx):
    return {
        "problem_id": row.get("problem_id", idx),
        "split": row.get("split", "?"),
        "question": row.get("question") or "",
        "solutions": _parse_str_or_list(row.get("solutions")),
    }


def _map_taco(row, idx):
    return {
        "problem_id": row.get("source") or row.get("id") or idx,
        "split": row.get("split", "?"),
        "question": row.get("question") or "",
        "solutions": _parse_str_or_list(row.get("solutions")),
    }


def _map_code_contests(row, idx):
    """deepmind/code_contests stores solutions as a dict
    {"language": [int...], "solution": [str...]}. Filter to Python (3=PY3, 1=PY2)."""
    sols = row.get("solutions") or {}
    if isinstance(sols, dict):
        langs = sols.get("language") or []
        codes = sols.get("solution") or []
        py = [c for lg, c in zip(langs, codes) if lg in (1, 3)]
    else:
        py = []
    return {
        "problem_id": row.get("name") or idx,
        "split": "train",
        "question": row.get("description") or "",
        "solutions": py,
    }


# (loader_callable, label, mapper)  — tried in order until one yields rows.
def _candidate_loaders():
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)

    def _try(repo: str, mapper, **kwargs):
        # Try common split names; first one that loads wins.
        for split in ("train", "test", "all", "validation"):
            try:
                ds = load_dataset(repo, split=split, **kwargs)
                return ds, split
            except Exception:
                continue
        return None, None

    yield ("codeparrot/apps (parquet revision)", _map_apps,
           lambda: _try("codeparrot/apps", _map_apps,
                        revision="refs/convert/parquet"))
    yield ("BAAI/TACO (parquet revision)", _map_taco,
           lambda: _try("BAAI/TACO", _map_taco,
                        revision="refs/convert/parquet"))
    yield ("deepmind/code_contests", _map_code_contests,
           lambda: _try("deepmind/code_contests", _map_code_contests))


def _iter_apps_problems():
    """Yield (problem_id, split, question, solutions: list[str]) tuples
    from the first dataset source that loads successfully."""
    last_errs: list[str] = []
    for label, mapper, loader in _candidate_loaders():
        print(f"Trying {label}...")
        try:
            ds, split = loader()
        except Exception as e:
            last_errs.append(f"  {label}: {type(e).__name__}: {str(e)[:200]}")
            print(last_errs[-1])
            continue
        if ds is None:
            last_errs.append(f"  {label}: no usable split")
            print(last_errs[-1])
            continue
        print(f"  OK: {label}, split={split}")
        for idx, row in enumerate(ds):
            rec = mapper(row, idx)
            yield (rec["problem_id"], rec["split"], rec["question"],
                   rec["solutions"])
        return
    print("\nERROR: every dataset source failed:", file=sys.stderr)
    for e in last_errs:
        print(e, file=sys.stderr)
    print("\nIf this is a `datasets>=4.0` script-removal issue, you can also pin\n"
          "  pip install 'datasets<4.0'\n"
          "and re-run.", file=sys.stderr)
    sys.exit(1)


def fetch_for_prompts(prompt_ids: list[str], out_dir: Path,
                      max_solutions_per_prompt: int,
                      max_problems_per_prompt: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = {pid: PROMPT_MATCH[pid] for pid in prompt_ids
             if pid in PROMPT_MATCH}
    missing = [pid for pid in prompt_ids if pid not in PROMPT_MATCH]
    if missing:
        print(f"  [warn] no APPS match-spec for prompts: {missing}; skipping")

    if not specs:
        print("Nothing to fetch.")
        return

    # Per-prompt collectors
    rows_by_prompt: dict[str, list[dict]] = {pid: [] for pid in specs}
    seen_hashes: dict[str, set[str]] = {pid: set() for pid in specs}
    problems_seen: dict[str, int] = {pid: 0 for pid in specs}

    print(f"Streaming APPS dataset (this can take a few minutes the first time)...")
    for problem_id, split, question, solutions in _iter_apps_problems():
        if not question or not solutions:
            continue
        for pid, spec in specs.items():
            if problems_seen[pid] >= max_problems_per_prompt:
                continue
            if len(rows_by_prompt[pid]) >= max_solutions_per_prompt:
                continue
            kws = _matches(question, spec)
            if not kws:
                continue
            problems_seen[pid] += 1
            print(f"  [{pid}] match: problem_id={problem_id} ({split})  "
                  f"keywords={kws}  ({len(solutions)} candidate solutions)")
            for code in solutions:
                if not isinstance(code, str) or not code.strip():
                    continue
                if not _is_python(code):
                    continue
                h = _hash(code)
                if h in seen_hashes[pid]:
                    continue
                seen_hashes[pid].add(h)
                rows_by_prompt[pid].append({
                    "prompt_id": pid,
                    "source": "APPS",
                    "problem_id": problem_id,
                    "split": split,
                    "code": code,
                    "n_chars": len(code),
                    "match_keywords": kws,
                })
                if len(rows_by_prompt[pid]) >= max_solutions_per_prompt:
                    break

        # Early exit: every prompt full?
        if all(len(rows_by_prompt[pid]) >= max_solutions_per_prompt
               for pid in specs):
            print("  All prompts have reached the solution cap; stopping.")
            break

    # Write per-prompt JSONLs
    for pid, rows in rows_by_prompt.items():
        out_path = out_dir / f"{pid}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {out_path}  ({len(rows)} solutions)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--prompts", nargs="+", default=["AL-01", "AL-03", "AL-05"])
    p.add_argument("--out-dir", type=Path,
                   default=Path("results") / "human_baseline")
    p.add_argument("--max-solutions-per-prompt", type=int, default=30)
    p.add_argument("--max-problems-per-prompt", type=int, default=5,
                   help="Stop scanning after K matching problems per prompt.")
    return p.parse_args()


def main():
    args = parse_args()
    fetch_for_prompts(args.prompts, args.out_dir,
                      args.max_solutions_per_prompt,
                      args.max_problems_per_prompt)


if __name__ == "__main__":
    main()
