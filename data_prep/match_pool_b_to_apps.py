#!/usr/bin/env python3
"""match_pool_b_to_apps.py
=========================
For every open-ended Python prompt in
``local_datasets/pool_b_open_ended_candidates/python_tasks/data.jsonl`` find the
closest APPS (Hendrycks et al. 2021) problem by TF-IDF cosine similarity over
the question text. Optionally, replace ``PB-01..N`` in
``local_datasets/homogeneity_pool_b_curated.json`` with the top matches and
emit per-prompt human JSONL files (compatible with
``proof_homogeneity_v2.py --human-dir ...``).

Why
---
Pool B prompts come from open-instruction LLM datasets; APPS ships with up to
~50 *human* Python solutions per problem. Aligning the two lets us bring real
human reference code into Pillar 1/2 even for Pool B-style task descriptions.

Outputs (always)
----------------
- ``local_datasets/pool_b_apps_mapping.json``
    Full ranking of pool_b_prompt -> best APPS problem (similarity, n_solutions, ...).
- ``local_datasets/_apps_python_cache.jsonl``
    Per-problem cache of (problem_id, split, question, [py_solutions]) so we
    do not stream APPS twice. Delete to force a re-fetch.

Outputs (when ``--apply``)
--------------------------
- Rewrites ``local_datasets/homogeneity_pool_b_curated.json`` so PB-01..PB-N
  point at the top-N matched Pool B prompts. Each entry records the APPS
  problem id, split, similarity, and human-solution count so reruns are
  reproducible.
- Writes ``<human-dir>/PB-0X.jsonl`` with the matched APPS solutions.

Usage
-----
::

  # 1) Build the mapping JSON only (cheap, idempotent after first APPS load)
  python data_prep/match_pool_b_to_apps.py

  # 2) Pick top 3 mapped prompts, rewrite curated JSON + write human JSONLs
  python data_prep/match_pool_b_to_apps.py --apply --top-n 3 \
      --human-dir results/human_baseline \
      --min-similarity 0.30 --min-solutions 15
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

# ── Project paths ──────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
POOL_B_PATH = ROOT / "local_datasets" / "pool_b_open_ended_candidates" / \
    "python_tasks" / "data.jsonl"
APPS_CACHE_PATH = ROOT / "local_datasets" / "_apps_python_cache.jsonl"
MAPPING_OUT_PATH = ROOT / "local_datasets" / "pool_b_apps_mapping.json"
CURATED_PATH = ROOT / "local_datasets" / "homogeneity_pool_b_curated.json"

# Reuse the APPS streaming + Python-validity helpers from fetch_human_baseline
sys.path.insert(0, str(ROOT))
from fetch_human_baseline import _iter_apps_problems, _is_python  # noqa: E402


# ── 1. APPS cache ──────────────────────────────────────────────────────────

def build_or_load_apps_cache(force_refresh: bool = False) -> list[dict]:
    """Load APPS Python solutions, caching to a local JSONL on first run.

    Each cached row::

        {"problem_id": int|str,
         "split": "train"|"test"|...,
         "question": str,
         "solutions": [py_str, ...]}    # only solutions that ast.parse cleanly
    """
    if APPS_CACHE_PATH.exists() and not force_refresh:
        rows = []
        with APPS_CACHE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        print(f"[apps cache] loaded {len(rows)} problems from {APPS_CACHE_PATH}")
        return rows

    print(f"[apps cache] streaming APPS via fetch_human_baseline._iter_apps_problems ...")
    APPS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with APPS_CACHE_PATH.open("w", encoding="utf-8") as f:
        for problem_id, split, question, solutions in _iter_apps_problems():
            if not question or not solutions:
                continue
            py = []
            seen = set()
            for s in solutions:
                if not isinstance(s, str) or not s.strip():
                    continue
                if not _is_python(s):
                    continue
                key = s.strip()
                if key in seen:
                    continue
                seen.add(key)
                py.append(s)
            if not py:
                continue  # skip problems without any usable Python solutions
            rec = {
                "problem_id": problem_id,
                "split": split,
                "question": question,
                "solutions": py,
            }
            f.write(json.dumps(rec) + "\n")
            rows.append(rec)
    print(f"[apps cache] wrote {len(rows)} problems with Python solutions "
          f"-> {APPS_CACHE_PATH}")
    return rows


# ── 2. Pool B candidates ───────────────────────────────────────────────────

def iter_pool_b_open_ended_python(min_chars: int = 200,
                                   max_chars: int = 3000) -> Iterable[dict]:
    """Yield Pool B records with label=OPEN_ENDED and language=PYTHON in a
    sensible prompt-length band (avoids 1-line trivial prompts and giant
    instructions that no model could complete in 2k tokens)."""
    if not POOL_B_PATH.exists():
        raise FileNotFoundError(f"Pool B JSONL not found at {POOL_B_PATH}")
    with POOL_B_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("label") != "OPEN_ENDED":
                continue
            if rec.get("task_language") != "PYTHON":
                continue
            prompt = (rec.get("prompt") or "").strip()
            if not (min_chars <= len(prompt) <= max_chars):
                continue
            yield rec


# ── 3. Matching ────────────────────────────────────────────────────────────

def match_pool_b_to_apps(pool_b_rows: list[dict], apps_rows: list[dict],
                          *, batch: int = 256) -> list[dict]:
    """For each Pool B prompt, find the top-1 APPS problem by TF-IDF cosine.

    Uses word n-grams (1-2) on the union vocabulary; English stop-words
    removed; sub-linear TF; L2-normalized for cosine via simple matmul.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    apps_q = [r["question"] for r in apps_rows]
    pool_q = [r["prompt"] for r in pool_b_rows]
    print(f"[match] vectorising {len(pool_q)} pool prompts and "
          f"{len(apps_q)} APPS questions ...")

    vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, lowercase=True, stop_words="english",
        max_features=200_000,
    )
    vec.fit(apps_q + pool_q)
    A = normalize(vec.transform(apps_q), copy=False)        # (Na, V) sparse
    P = normalize(vec.transform(pool_q), copy=False)        # (Np, V) sparse

    matches: list[dict] = []
    for start in range(0, P.shape[0], batch):
        sub = P[start:start + batch]
        # cosine sim because both halves are L2 normalised
        sims = (sub @ A.T).toarray()
        top_idx = sims.argmax(axis=1)
        top_sim = sims.max(axis=1)
        for i, (j, s) in enumerate(zip(top_idx, top_sim)):
            pb = pool_b_rows[start + i]
            ap = apps_rows[int(j)]
            matches.append({
                "pool_b_key":      f"{pb.get('dataset')}:{pb.get('split')}:{pb.get('row_index')}",
                "pool_b_dataset":  pb.get("dataset"),
                "pool_b_split":    pb.get("split"),
                "pool_b_row":      pb.get("row_index"),
                "pool_b_prompt":   pb.get("prompt"),
                "openness_score":  pb.get("openness_score"),
                "open_ended_dimensions": pb.get("open_ended_dimensions"),
                "apps_problem_id": ap.get("problem_id"),
                "apps_split":      ap.get("split"),
                "apps_question":   ap.get("question"),
                "n_human_solutions": len(ap.get("solutions") or []),
                "similarity":      float(s),
            })
        if (start // batch) % 10 == 0:
            print(f"  matched {start + sub.shape[0]}/{P.shape[0]}  "
                  f"latest top-sim={top_sim.max():.3f}")
    matches.sort(key=lambda m: (-m["similarity"], -m["n_human_solutions"]))
    return matches


# ── 4. Selection of "best" matches for replacement ─────────────────────────

def pick_top_matches(matches: list[dict], *, top_n: int,
                     min_similarity: float, min_solutions: int) -> list[dict]:
    """Greedy top-N pick that:

    - drops near-duplicates (same APPS problem already used, or near-identical
      Pool B prompt text already chosen);
    - rejects matches below the similarity / solutions thresholds;
    - prefers (similarity * log(1 + n_solutions)) so a strong match with many
      solutions beats a slightly stronger match that has too few.
    """
    used_apps_ids: set = set()
    used_prompt_starts: set[str] = set()
    picks: list[dict] = []
    ranked = sorted(matches,
                    key=lambda m: -(m["similarity"]
                                    * math.log1p(m["n_human_solutions"])))
    for m in ranked:
        if len(picks) >= top_n:
            break
        if m["similarity"] < min_similarity:
            continue
        if m["n_human_solutions"] < min_solutions:
            continue
        if m["apps_problem_id"] in used_apps_ids:
            continue
        head = (m["pool_b_prompt"] or "")[:120].lower()
        if head in used_prompt_starts:
            continue
        used_apps_ids.add(m["apps_problem_id"])
        used_prompt_starts.add(head)
        picks.append(m)
    return picks


# ── 5. Apply: rewrite curated JSON + write human JSONLs ────────────────────

def write_curated_json(picks: list[dict]) -> None:
    """Replace PB-01..PB-N entries in homogeneity_pool_b_curated.json with the
    top picks, preserving the schema understood by ``prompt_suite.py``.

    The original file is overwritten; back it up yourself if needed.
    """
    out = {
        "name": "homogeneity_pool_b_curated",
        "description": (
            "Pool B Python prompts that have a paired APPS problem with "
            "Python human solutions (mapped via TF-IDF cosine similarity). "
            "Use --human-dir results/human_baseline when running "
            "proof_homogeneity_v2.py to enable Pillar 1 human pool / Pillar 2 "
            "ordering test."
        ),
        "source_file": "local_datasets/pool_b_open_ended_candidates/python_tasks/data.jsonl",
        "mapping_file": "local_datasets/pool_b_apps_mapping.json",
        "prompts": [],
    }
    for i, pk in enumerate(picks, start=1):
        pid = f"PB-{i:02d}"
        out["prompts"].append({
            "id": pid,
            "category": "POOL_B_CURATED",
            "language": "python",
            "expected_diversity": "high",
            "source_dataset":   pk.get("pool_b_dataset"),
            "source_split":     pk.get("pool_b_split"),
            "source_row_index": pk.get("pool_b_row"),
            "openness_score":   pk.get("openness_score"),
            "open_ended_dimensions": pk.get("open_ended_dimensions"),
            "prompt": pk.get("pool_b_prompt"),
            "human_baseline": {
                "source": "APPS",
                "apps_problem_id":   pk.get("apps_problem_id"),
                "apps_split":        pk.get("apps_split"),
                "apps_question":     pk.get("apps_question"),
                "match_similarity":  pk.get("similarity"),
                "n_human_solutions": pk.get("n_human_solutions"),
            },
        })
    CURATED_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[apply] rewrote {CURATED_PATH} with {len(picks)} prompts "
          f"({', '.join(p['id'] for p in out['prompts'])})")


def write_human_jsonls(picks: list[dict], apps_rows: list[dict],
                        human_dir: Path,
                        max_solutions_per_prompt: int) -> None:
    """For each picked Pool B prompt, write a JSONL with the matched APPS
    problem's Python human solutions, in the schema understood by
    ``proof_homogeneity.load_human_responses``."""
    human_dir.mkdir(parents=True, exist_ok=True)
    by_id: dict = {r["problem_id"]: r for r in apps_rows}
    for i, pk in enumerate(picks, start=1):
        pid = f"PB-{i:02d}"
        ap = by_id.get(pk["apps_problem_id"])
        if ap is None:
            print(f"  [warn] {pid}: APPS problem {pk['apps_problem_id']} missing "
                  f"in cache; skipping")
            continue
        sols = ap["solutions"][:max_solutions_per_prompt]
        out_path = human_dir / f"{pid}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for code in sols:
                rec = {
                    "prompt_id": pid,
                    "source": "APPS",
                    "problem_id": ap["problem_id"],
                    "split": ap["split"],
                    "code": code,
                    "n_chars": len(code),
                    "match_similarity": pk["similarity"],
                }
                f.write(json.dumps(rec) + "\n")
        print(f"[apply] {pid}: wrote {len(sols)} human solutions -> {out_path}")


# ── 6. CLI ─────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--force-refresh-apps", action="store_true",
                   help="Ignore the local APPS cache and re-stream from HF.")
    p.add_argument("--apply", action="store_true",
                   help="Replace PB-01..N in the curated JSON and emit human "
                        "JSONLs to --human-dir.")
    p.add_argument("--top-n", type=int, default=3,
                   help="Number of replacement PB-* slots when --apply is set.")
    p.add_argument("--min-similarity", type=float, default=0.30,
                   help="Reject matches below this cosine similarity.")
    p.add_argument("--min-solutions", type=int, default=15,
                   help="Reject matches whose APPS problem has fewer than "
                        "this many Python solutions.")
    p.add_argument("--max-solutions-per-prompt", type=int, default=50,
                   help="Cap human solutions per prompt when writing JSONLs.")
    p.add_argument("--human-dir", type=Path,
                   default=ROOT / "results" / "human_baseline",
                   help="Directory for per-prompt human JSONLs (created on apply).")
    p.add_argument("--max-pool-prompts", type=int, default=8000,
                   help="Stop after this many Pool B candidates (memory cap).")
    return p.parse_args()


def main():
    args = parse_args()

    apps_rows = build_or_load_apps_cache(force_refresh=args.force_refresh_apps)
    if not apps_rows:
        print("ERROR: no APPS rows; aborting.", file=sys.stderr)
        sys.exit(2)

    pool_b_rows = []
    for rec in iter_pool_b_open_ended_python():
        pool_b_rows.append(rec)
        if len(pool_b_rows) >= args.max_pool_prompts:
            break
    print(f"[pool_b] kept {len(pool_b_rows)} OPEN_ENDED Python candidates "
          f"(prompt length 200..3000 chars)")

    matches = match_pool_b_to_apps(pool_b_rows, apps_rows)
    MAPPING_OUT_PATH.write_text(json.dumps(matches[:5000], indent=2),
                                 encoding="utf-8")
    print(f"[mapping] wrote top 5000 matches -> {MAPPING_OUT_PATH}")

    # Quick at-a-glance head
    print("\n[mapping] top 10 by similarity:")
    for m in matches[:10]:
        print(f"  sim={m['similarity']:.3f}  n_sols={m['n_human_solutions']:>3}  "
              f"apps={m['apps_problem_id']} ({m['apps_split']})  "
              f"pool=[{m['pool_b_key']}]\n     pb : {(m['pool_b_prompt'] or '')[:120]!r}\n     app: {(m['apps_question'] or '')[:120]!r}")

    if not args.apply:
        print("\n(no --apply; not modifying curated JSON or writing human files)")
        return

    picks = pick_top_matches(matches, top_n=args.top_n,
                              min_similarity=args.min_similarity,
                              min_solutions=args.min_solutions)
    if not picks:
        print(f"\nERROR: no matches passed filters "
              f"(min_similarity={args.min_similarity}, "
              f"min_solutions={args.min_solutions}). Try lowering the "
              f"thresholds.", file=sys.stderr)
        sys.exit(3)

    print(f"\n[apply] picked {len(picks)} replacement prompts:")
    for i, pk in enumerate(picks, start=1):
        print(f"  PB-{i:02d}  sim={pk['similarity']:.3f}  "
              f"n_sols={pk['n_human_solutions']}  "
              f"apps={pk['apps_problem_id']} ({pk['apps_split']})")
        print(f"     pb : {(pk['pool_b_prompt'] or '')[:120]!r}")
        print(f"     app: {(pk['apps_question'] or '')[:120]!r}")

    write_curated_json(picks)
    write_human_jsonls(picks, apps_rows, args.human_dir,
                        args.max_solutions_per_prompt)
    print("\n[done] Replacement complete. Next:")
    print("  1) Delete results/raw_responses/PB-0*.jsonl  (stale LLM outputs)")
    print("  2) python run.py --collect --samples 20 --temps 0.0 0.7 --prompts PB-01 PB-02 PB-03")
    print("  3) python proof_homogeneity_v2.py --prompts PB-01 PB-02 PB-03 \\")
    print("       --temperatures 0.0 0.7 --human-dir results/human_baseline \\")
    print("       --embedder codet5p --center-embeddings center")


if __name__ == "__main__":
    main()
