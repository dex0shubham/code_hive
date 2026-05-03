#!/usr/bin/env python3
"""
Logic-level homogeneity analysis.

For a chosen list of prompt IDs (and an optional temperature filter), this
script measures three complementary "same idea?" signals:

  1. Logic / data-flow features (deterministic, AST-based).
     - emits a numeric feature vector and a coarse strategy label
     - reports dominance, entropy, mean pairwise cosine similarity

  2. Code-aware embeddings (UniXcoder / CodeBERT).
     - mean pairwise cosine over snippets, captures semantic similarity
       beyond surface text and AST shape

  3. Trace-based behavioral equivalence (optional).
     - executes runnable snippets in a sandboxed subprocess with a timeout
     - compares stdout pairwise (exact match / set match / Jaccard)

Outputs:
  <out-dir>/per_sample.csv        per (prompt, model, sample) features + label
  <out-dir>/per_prompt.json       per-prompt metrics (label dist, similarities)
  <out-dir>/summary.csv           publication-friendly table

Usage:
  python logic_analysis.py --prompts DSB-01 DSB-02 --temperature 1.0 \
      --out-dir results/sample_analysis/dataset_pool_b/v2/logic_t1 \
      --embeddings --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from logic_features import (  # noqa: E402
    LogicFeatures,
    cosine_matrix,
    derive_logic_label,
    extract_logic_features,
)


RAW_DIR = ROOT / "results" / "raw_responses"
_FENCE = re.compile(r"```(?:python|py|javascript|js)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    matches = _FENCE.findall(text or "")
    return "\n".join(matches) if matches else (text or "").strip()


def load_responses(prompt_ids: list[str], temperature: float | None) -> list[dict]:
    rows: list[dict] = []
    for pid in prompt_ids:
        fp = RAW_DIR / f"{pid}.jsonl"
        if not fp.exists():
            print(f"  WARN: {fp} not found")
            continue
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                rec = json.loads(ln)
                if temperature is not None and abs(rec.get("temperature", 0.0) - temperature) > 1e-6:
                    continue
                rows.append(rec)
    return rows


def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    p = np.array([c / total for c in counts if c > 0], dtype=float)
    return float(-(p * np.log2(p)).sum())


def normalised_entropy(counts: list[int]) -> float:
    n = len([c for c in counts if c > 0])
    if n <= 1:
        return 0.0
    return shannon_entropy(counts) / np.log2(n)


def dominant_ratio(counts: list[int]) -> float:
    total = sum(counts)
    return max(counts) / total if total else 0.0


def _serialize(o):
    if isinstance(o, dict):
        return {str(k): _serialize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_serialize(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return float(o) if isinstance(o, np.floating) else int(o)
    if isinstance(o, LogicFeatures):
        return asdict(o)
    return o


def main():
    ap = argparse.ArgumentParser(description="Logic-level homogeneity analysis")
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Filter to a single temperature, e.g. 1.0. Default: all.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embeddings", action="store_true",
                    help="Compute code-aware embedding similarity (UniXcoder).")
    ap.add_argument("--embedding-model", type=str, default="microsoft/unixcoder-base")
    ap.add_argument("--execute", action="store_true",
                    help="Execute runnable snippets in subprocess and compare stdout. "
                         "WARNING: runs LLM-generated code locally without network sandbox.")
    ap.add_argument("--exec-timeout", type=float, default=8.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_responses(args.prompts, args.temperature)
    if not rows:
        print("No responses loaded; aborting.")
        sys.exit(1)
    print(f"Loaded {len(rows)} responses across {len(args.prompts)} prompts")

    # ─── 1. logic features ────────────────────────────────────────────
    samples = []
    for r in rows:
        code = extract_code(r.get("response_text", ""))
        feats = extract_logic_features(code)
        samples.append({
            "prompt_id":    r["prompt_id"],
            "model":        r["model_display"],
            "family":       r.get("model_family"),
            "temperature":  r.get("temperature"),
            "sample_index": r.get("sample_index"),
            "code":         code,
            "feats":        feats,
            "label":        derive_logic_label(feats),
        })

    # ─── 2. optional code embeddings ──────────────────────────────────
    embedder = None
    if args.embeddings:
        from code_embeddings import CodeEmbeddingComputer
        print(f"Loading code embedding model: {args.embedding_model}")
        embedder = CodeEmbeddingComputer(model_name=args.embedding_model)

    # ─── 3. optional execution ────────────────────────────────────────
    if args.execute:
        print(f"  WARN: executing LLM-generated code locally (timeout={args.exec_timeout}s)",
              file=sys.stderr)
        from trace_equivalence import (  # noqa: E402
            compare_outputs,
            is_runnable_as_script,
            run_code,
        )
        for s in samples:
            s["runnable"] = is_runnable_as_script(s["code"])
            if s["runnable"]:
                rr = run_code(s["code"], timeout=args.exec_timeout)
                s["run"] = {
                    "success":    rr.success,
                    "stdout":     rr.stdout,
                    "stderr":     rr.stderr[:200],
                    "returncode": rr.returncode,
                    "elapsed_ms": rr.elapsed_ms,
                    "timed_out":  rr.timed_out,
                }
            else:
                s["run"] = None

    # ─── 4. per-prompt analyses ───────────────────────────────────────
    by_prompt: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_prompt[s["prompt_id"]].append(s)

    per_prompt: dict[str, dict] = {}
    for pid, plist in by_prompt.items():
        n = len(plist)
        labels = [s["label"] for s in plist]
        lc = Counter(labels)
        counts = list(lc.values())

        # logic-feature cosine similarity
        M = np.vstack([s["feats"].vector() for s in plist])
        cos = cosine_matrix(M)
        iu = np.triu_indices_from(cos, k=1)
        feat_metrics = {
            "mean_pairwise_cosine":   float(cos[iu].mean()) if iu[0].size else 0.0,
            "median_pairwise_cosine": float(np.median(cos[iu])) if iu[0].size else 0.0,
        }

        # cross-model agreement on logic label
        by_model: dict[str, list[int]] = defaultdict(list)
        for i, s in enumerate(plist):
            by_model[s["model"]].append(i)
        models = sorted(by_model.keys())
        cross = {}
        for a in models:
            cross[a] = {}
            for b in models:
                if a == b:
                    cross[a][b] = 1.0
                    continue
                la = [plist[i]["label"] for i in by_model[a]]
                lb = [plist[i]["label"] for i in by_model[b]]
                if not la or not lb:
                    cross[a][b] = 0.0
                    continue
                la_arr, lb_arr = np.array(la), np.array(lb)
                cross[a][b] = float((la_arr[:, None] == lb_arr[None, :]).mean())

        # code embedding similarity
        emb_metrics = None
        if embedder is not None:
            sim = embedder.cosine_sim_matrix([s["code"] for s in plist])
            if sim.size:
                ut = np.triu_indices_from(sim, k=1)
                emb_metrics = {
                    "mean_pairwise_cosine":   float(sim[ut].mean()) if ut[0].size else 0.0,
                    "median_pairwise_cosine": float(np.median(sim[ut])) if ut[0].size else 0.0,
                    "model": embedder.model_name,
                }

        # behavioral equivalence
        beh_metrics = None
        if args.execute:
            ok_idx = [i for i, s in enumerate(plist) if s.get("run") and s["run"]["success"]]
            n_run = sum(1 for s in plist if s.get("runnable"))
            pairs = exact = set_m = 0
            jaccs: list[float] = []
            for i in range(len(ok_idx)):
                for j in range(i + 1, len(ok_idx)):
                    a = plist[ok_idx[i]]["run"]["stdout"]
                    b = plist[ok_idx[j]]["run"]["stdout"]
                    cmp = compare_outputs(a, b)  # noqa: F821
                    pairs += 1
                    exact += int(cmp["exact_match"])
                    set_m += int(cmp["set_match"])
                    jaccs.append(cmp["jaccard"])
            beh_metrics = {
                "n_runnable":        n_run,
                "n_ran_successfully": len(ok_idx),
                "n_pairs":           pairs,
                "exact_match_rate":  (exact / pairs) if pairs else 0.0,
                "set_match_rate":    (set_m / pairs) if pairs else 0.0,
                "mean_jaccard":      float(np.mean(jaccs)) if jaccs else 0.0,
            }

        per_prompt[pid] = {
            "prompt_id": pid,
            "n":         n,
            "label_metrics": {
                "n_unique":           len(lc),
                "dominant_ratio":     dominant_ratio(counts),
                "shannon_entropy":    shannon_entropy(counts),
                "normalised_entropy": normalised_entropy(counts),
                "top5":               lc.most_common(5),
            },
            "feature_metrics":              feat_metrics,
            "embedding_metrics":            emb_metrics,
            "behavioral_metrics":           beh_metrics,
            "cross_model_label_agreement":  cross,
        }

    # ─── 5. write outputs ─────────────────────────────────────────────
    with open(out_dir / "per_prompt.json", "w", encoding="utf-8") as f:
        json.dump(_serialize(per_prompt), f, indent=2)
    print(f"Wrote {out_dir / 'per_prompt.json'}")

    sample_cols = [
        "prompt_id", "model", "family", "temperature", "sample_index",
        "label", "parse_ok", "num_for", "num_while", "num_if",
        "num_print_calls", "num_function_calls", "max_nesting_depth",
        "has_ascending_loop", "has_descending_loop", "has_recursion",
        "has_bitwise", "uses_comprehension", "has_unrolled_outputs",
        "runnable", "ran_ok", "stdout_lines",
    ]
    with open(out_dir / "per_sample.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(sample_cols)
        for s in samples:
            run = s.get("run")
            stdout_lines = len((run or {}).get("stdout", "").splitlines()) if run else 0
            feats: LogicFeatures = s["feats"]
            w.writerow([
                s["prompt_id"], s["model"], s["family"], s["temperature"], s["sample_index"],
                s["label"], int(feats.parse_ok),
                feats.num_for, feats.num_while, feats.num_if,
                feats.num_print_calls, feats.num_function_calls, feats.max_nesting_depth,
                int(feats.has_ascending_loop), int(feats.has_descending_loop),
                int(feats.has_recursion), int(feats.has_bitwise),
                int(feats.uses_comprehension), int(feats.has_unrolled_outputs),
                int(s.get("runnable", False)),
                int(bool(run and run["success"])),
                stdout_lines,
            ])
    print(f"Wrote {out_dir / 'per_sample.csv'}")

    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "prompt_id", "n_samples", "n_unique_labels", "dominant_ratio",
            "normalised_entropy", "feature_cosine_mean",
            "embedding_cosine_mean", "exec_exact_match_rate",
            "exec_set_match_rate", "n_runnable", "n_ran_ok",
        ])
        for pid in sorted(per_prompt.keys()):
            d = per_prompt[pid]
            lm = d["label_metrics"]; fm = d["feature_metrics"]
            em = d["embedding_metrics"] or {}
            bm = d["behavioral_metrics"] or {}
            w.writerow([
                pid, d["n"], lm["n_unique"],
                f"{lm['dominant_ratio']:.3f}",
                f"{lm['normalised_entropy']:.3f}",
                f"{fm['mean_pairwise_cosine']:.3f}",
                f"{em.get('mean_pairwise_cosine', 0):.3f}" if em else "",
                f"{bm.get('exact_match_rate', 0):.3f}" if bm else "",
                f"{bm.get('set_match_rate', 0):.3f}" if bm else "",
                bm.get("n_runnable", "") if bm else "",
                bm.get("n_ran_successfully", "") if bm else "",
            ])
    print(f"Wrote {out_dir / 'summary.csv'}")

    # Console summary
    print("\n" + "=" * 110)
    print("LOGIC HOMOGENEITY  -  per-prompt summary")
    print("=" * 110)
    print(f"{'Prompt':<10} {'N':>4} {'unique':>7} {'dom%':>7} {'normH':>7} "
          f"{'feat_cos':>9} {'emb_cos':>9} {'exec_exact%':>12} {'exec_set%':>10}")
    print("-" * 110)
    for pid in sorted(per_prompt.keys()):
        d = per_prompt[pid]; lm = d["label_metrics"]; fm = d["feature_metrics"]
        em = d["embedding_metrics"] or {}; bm = d["behavioral_metrics"] or {}
        emb_str = f"{em.get('mean_pairwise_cosine', 0):>9.3f}" if em else f"{'-':>9}"
        ex_exact = f"{bm.get('exact_match_rate', 0)*100:>11.1f}%" if bm else f"{'-':>12}"
        ex_set = f"{bm.get('set_match_rate', 0)*100:>9.1f}%" if bm else f"{'-':>10}"
        print(f"{pid:<10} {d['n']:>4d} {lm['n_unique']:>7d} "
              f"{lm['dominant_ratio']*100:>6.1f}% {lm['normalised_entropy']:>7.3f} "
              f"{fm['mean_pairwise_cosine']:>9.3f} {emb_str} {ex_exact} {ex_set}")


if __name__ == "__main__":
    main()
