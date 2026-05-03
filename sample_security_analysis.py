#!/usr/bin/env python3
"""
Hardened security analysis for a small set of prompts.

Compared to security_analysis.py (which scans the entire results corpus),
this script:
  - operates only on a chosen list of prompt IDs
  - filters to a chosen temperature
  - runs regex-based detectors (CWE patterns) AND optional Bandit
  - computes bootstrap 95% confidence intervals for the headline rates
    (vuln_rate, exact_cwe_match_rate, any_cwe_overlap_rate)

Output:
  <out-dir>/security_analysis.json
  <out-dir>/cwe_pair_matches.csv

Usage:
  python sample_security_analysis.py \
      --prompts DSB-01 DSB-02 \
      --temperature 1.0 \
      --bandit \
      --out-dir results/sample_analysis/dataset_pool_b/security_dsb_v2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from security_analysis import (  # noqa: E402  (after sys.path bump)
    extract_code,
    scan_code,
    check_risky_imports,
    run_bandit,
    VULN_PATTERNS,
)


RAW_DIR = ROOT / "results" / "raw_responses"


def load_responses(prompt_ids: list[str], temperature: float | None) -> list[dict]:
    rows: list[dict] = []
    for pid in prompt_ids:
        fp = RAW_DIR / f"{pid}.jsonl"
        if not fp.exists():
            print(f"  WARN: {fp} not found; skipping")
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


def bootstrap_ci(
    indicator: list[int],
    n_resamples: int = 2000,
    alpha: float = 0.05,
    rng: random.Random | None = None,
) -> tuple[float, float, float]:
    """Return (mean, lower, upper) for the mean of a 0/1 indicator vector."""
    if not indicator:
        return 0.0, 0.0, 0.0
    rng = rng or random.Random(42)
    n = len(indicator)
    means = []
    for _ in range(n_resamples):
        sample = [indicator[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return float(sum(indicator) / n), float(lo), float(hi)


def main():
    parser = argparse.ArgumentParser(description="Sample-scoped security analysis with Bandit + bootstrap CIs")
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--temperature", type=float, default=None,
                        help="Filter to a single temperature, e.g. 1.0. Default: include all.")
    parser.add_argument("--bandit", action="store_true",
                        help="Also run Bandit on each generated code snippet.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_responses(args.prompts, args.temperature)
    if not rows:
        print("No responses found for the given filters.")
        sys.exit(1)

    models = sorted({r["model_display"] for r in rows})
    prompts = sorted({r["prompt_id"] for r in rows})
    print(f"Loaded {len(rows)} responses across {len(prompts)} prompts and {len(models)} models")
    if args.bandit:
        print("  Bandit enabled (this can be slow on large samples)")

    findings: list[dict] = []
    for r in rows:
        code = extract_code(r.get("response_text", ""))
        regex_hits = scan_code(code)
        bandit_hits: list[dict] = []
        if args.bandit:
            try:
                bandit_results = run_bandit(code)
                for b in bandit_results:
                    bandit_hits.append({
                        "cwe": b.get("test_id", "B???"),
                        "name": f"Bandit: {b.get('test_name', '?')}",
                        "line": b.get("line_number", 0),
                        "severity": b.get("issue_severity", "?"),
                        "matched": (b.get("issue_text", "") or "")[:160],
                    })
            except Exception as e:
                print(f"  Bandit failed on one sample: {type(e).__name__}: {e}")
        all_hits = regex_hits + bandit_hits
        findings.append({
            "prompt_id": r["prompt_id"],
            "model": r["model_display"],
            "temperature": r.get("temperature"),
            "sample_index": r.get("sample_index"),
            "regex_hits": regex_hits,
            "bandit_hits": bandit_hits,
            "n_vulns": len(all_hits),
            "cwes": sorted({f["cwe"] for f in all_hits}),
            "cwe_set": frozenset(f["cwe"] for f in all_hits),
            "risky_imports": check_risky_imports(code),
        })

    total = len(findings)
    vuln_indicator = [1 if f["n_vulns"] > 0 else 0 for f in findings]
    vuln_rate, vuln_lo, vuln_hi = bootstrap_ci(vuln_indicator, args.bootstrap_resamples)

    cwe_counter = Counter(c for f in findings for c in f["cwes"])
    by_prompt: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for f in findings:
        by_prompt[f["prompt_id"]][f["model"]].append(f)

    pair_rows: list[dict] = []
    exact_indicator: list[int] = []
    overlap_indicator: list[int] = []
    prompt_homogeneity: list[dict] = []

    for pid in prompts:
        per_model = by_prompt[pid]
        mlist = sorted(per_model.keys())
        prompt_exact: list[int] = []
        prompt_overlap: list[int] = []
        for i in range(len(mlist)):
            for j in range(i + 1, len(mlist)):
                ma, mb = mlist[i], mlist[j]
                # one outcome per (model_a sample, model_b sample) pair
                for fa in per_model[ma]:
                    for fb in per_model[mb]:
                        a, b = fa["cwe_set"], fb["cwe_set"]
                        em = int(a == b and len(a) > 0)
                        ov = int(bool(a & b))
                        exact_indicator.append(em)
                        overlap_indicator.append(ov)
                        prompt_exact.append(em)
                        prompt_overlap.append(ov)
                        pair_rows.append({
                            "prompt_id": pid,
                            "model_a": ma,
                            "model_b": mb,
                            "sample_a": fa["sample_index"],
                            "sample_b": fb["sample_index"],
                            "exact_match": em,
                            "any_overlap": ov,
                            "overlap_cwes": ";".join(sorted(a & b)),
                        })

        if prompt_exact:
            er, er_lo, er_hi = bootstrap_ci(prompt_exact, args.bootstrap_resamples)
            ov_r, ov_lo, ov_hi = bootstrap_ci(prompt_overlap, args.bootstrap_resamples)
            prompt_homogeneity.append({
                "prompt_id": pid,
                "n_pairs": len(prompt_exact),
                "exact_rate": er,
                "exact_rate_ci95": [er_lo, er_hi],
                "overlap_rate": ov_r,
                "overlap_rate_ci95": [ov_lo, ov_hi],
            })

    exact_rate, exact_lo, exact_hi = bootstrap_ci(exact_indicator, args.bootstrap_resamples)
    overlap_rate, overlap_lo, overlap_hi = bootstrap_ci(overlap_indicator, args.bootstrap_resamples)

    # Per-model rates (point estimate; CI not always meaningful with very small n per model)
    per_model = {}
    for m in models:
        mf = [f for f in findings if f["model"] == m]
        if not mf:
            continue
        ind = [1 if f["n_vulns"] > 0 else 0 for f in mf]
        rate = sum(ind) / len(ind)
        per_model[m] = {
            "n": len(mf),
            "vuln_rate": rate,
            "mean_cwes": float(np.mean([f["n_vulns"] for f in mf])),
            "cwe_breakdown": dict(Counter(c for f in mf for c in f["cwes"])),
        }

    # Risky-import convergence (point estimate, names of shared risky packages)
    risky_by_model: dict[str, Counter] = defaultdict(Counter)
    for f in findings:
        for ri in f["risky_imports"]:
            risky_by_model[f["model"]][ri["import"]] += 1
    all_risky = {imp for c in risky_by_model.values() for imp in c}
    shared_risky = []
    for imp in sorted(all_risky):
        users = sorted([m for m in models if imp in risky_by_model[m]])
        if len(users) > 1:
            shared_risky.append({"import": imp, "users": users})

    out = {
        "scope": {
            "prompt_ids": prompts,
            "temperature_filter": args.temperature,
            "n_responses": total,
            "n_models": len(models),
            "models": models,
            "bandit_enabled": bool(args.bandit),
        },
        "overall": {
            "vuln_rate": vuln_rate,
            "vuln_rate_ci95": [vuln_lo, vuln_hi],
            "exact_cwe_match_rate": exact_rate,
            "exact_cwe_match_rate_ci95": [exact_lo, exact_hi],
            "any_cwe_overlap_rate": overlap_rate,
            "any_cwe_overlap_rate_ci95": [overlap_lo, overlap_hi],
            "n_pairs": len(exact_indicator),
        },
        "per_model": per_model,
        "cwe_distribution": dict(cwe_counter.most_common()),
        "prompt_homogeneity": prompt_homogeneity,
        "shared_risky_imports": shared_risky,
    }

    with open(out_dir / "security_analysis.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    with open(out_dir / "cwe_pair_matches.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "prompt_id", "model_a", "model_b", "sample_a", "sample_b",
            "exact_match", "any_overlap", "overlap_cwes",
        ])
        w.writeheader()
        w.writerows(pair_rows)

    print()
    print(f"Vuln rate:             {vuln_rate:.1%}  CI95=[{vuln_lo:.1%}, {vuln_hi:.1%}]")
    print(f"Exact CWE-set match:   {exact_rate:.1%}  CI95=[{exact_lo:.1%}, {exact_hi:.1%}]")
    print(f"Any CWE overlap:       {overlap_rate:.1%}  CI95=[{overlap_lo:.1%}, {overlap_hi:.1%}]")
    print(f"Top CWEs: {cwe_counter.most_common(8)}")
    print(f"Wrote {out_dir / 'security_analysis.json'}")
    print(f"Wrote {out_dir / 'cwe_pair_matches.csv'}")


if __name__ == "__main__":
    main()
