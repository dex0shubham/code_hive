#!/usr/bin/env python3
"""
sensitivity_codebert.py
=======================
Sensitivity analysis comparing CodeT5+ and CodeBERT embeddings on the
diversity suite.  For a small subset of prompts, computes Vendi/N under
three kernel families (token, CodeT5+, CodeBERT) for the inter-LLM and
human pools, and verifies that the ranking  inter-LLM < human  is
preserved under both neural embedders.

The token kernel is shared across both embedders (it does not depend on
the neural model) and serves as a consistency / sanity check.

Outputs
-------
  - Console: side-by-side comparison table
  - JSON:    results/sensitivity_codebert_results.json

Usage
-----
  python sensitivity_codebert.py                        # defaults
  python sensitivity_codebert.py --device cuda           # GPU
  python sensitivity_codebert.py --prompts AL-01 AL-05   # subset

Requires: torch, transformers, numpy  (same deps as proof_homogeneity.py)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Import primitives from the existing homogeneity proof scripts
# ---------------------------------------------------------------------------
from proof_homogeneity import (
    extract_code,
    load_llm_responses,
    load_human_responses,
    build_pools,
    PromptPools,
    kernel_token,
    kernel_codebert,
    kernel_codet5p_embed,
    vendi_score,
)

ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _vendi_norm(codes: list[str],
                kernel_fn,
                n_boot: int,
                rng: np.random.Generator) -> dict:
    """Compute Vendi / N with bootstrap CI for a single pool + kernel.

    Returns {"vendi": float, "vendi_norm": float, "ci_norm": [lo, hi], "n": int}.
    """
    n = len(codes)
    if n < 2:
        v = float(n)
        vn = 1.0 if n >= 1 else 0.0
        return {"vendi": v, "vendi_norm": vn, "ci_norm": [vn, vn], "n": n}

    K = kernel_fn(codes)
    point = vendi_score(K)
    boots: list[float] = []
    N = K.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        Kb = K[np.ix_(idx, idx)]
        boots.append(vendi_score(Kb) / N)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return {
        "vendi": float(point),
        "vendi_norm": float(point) / N,
        "ci_norm": [float(lo), float(hi)],
        "n": N,
    }


def build_pools_for_prompts(
    prompts: list[str],
    raw_dir: Path,
    human_dir: Path | None,
    temperature: float,
    rng: np.random.Generator,
    samples_per_model_for_inter: int = 4,
    cap_per_pool: int = 60,
) -> dict[str, PromptPools]:
    """Load data and build per-prompt pools (mirrors v2 helper)."""
    llm_all = load_llm_responses(raw_dir, prompts, temperature=None)
    human = load_human_responses(human_dir, prompts)

    # Filter to requested temperature
    llm_at_temp: dict[str, list[dict]] = {
        pid: [r for r in rs if abs(r.get("temperature", 0.0) - temperature) < 1e-6]
        for pid, rs in llm_all.items()
    }

    pools: dict[str, PromptPools] = {}
    for pid in prompts:
        other_codes: list[str] = []
        for opid, rs in llm_at_temp.items():
            if opid == pid:
                continue
            for r in rs:
                c = extract_code(r.get("response_text", ""))
                if c:
                    other_codes.append(c)
        pools[pid] = build_pools(
            pid,
            llm_at_temp.get(pid, []),
            human.get(pid, []),
            other_codes,
            rng,
            samples_per_model_for_inter=samples_per_model_for_inter,
            cap_per_pool=cap_per_pool,
        )
    return pools


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_sensitivity(args) -> dict:
    """Run the full sensitivity comparison and return structured results."""
    rng = np.random.default_rng(args.seed)

    print(f"[sensitivity] prompts:     {args.prompts}")
    print(f"[sensitivity] temperature: {args.temperature}")
    print(f"[sensitivity] device:      {args.device}")
    print(f"[sensitivity] n_boot:      {args.n_boot}")
    print()

    # Build pools once (shared across all kernels)
    pools = build_pools_for_prompts(
        args.prompts,
        args.raw_dir,
        args.human_dir,
        args.temperature,
        rng,
        samples_per_model_for_inter=args.samples_per_model_for_inter,
        cap_per_pool=args.cap_per_pool,
    )

    for pid, p in pools.items():
        print(f"  pools[{pid}]  inter_llm={len(p.inter_llm_codes)}  "
              f"human={len(p.human_codes)}  null={len(p.null_codes)}")
    print()

    # Define the three kernel functions
    device = args.device
    center = args.center_embeddings

    kernel_fns = {
        "token":   lambda codes: kernel_token(codes, n=3),
        "codet5p": lambda codes: kernel_codet5p_embed(
            codes, device=device, center=center),
        "codebert": lambda codes: kernel_codebert(
            codes, device=device, center=center),
    }

    # Compute Vendi/N for each (prompt, kernel, pool) combination
    results: dict = {
        "config": {
            "prompts": args.prompts,
            "temperature": args.temperature,
            "device": device,
            "center_embeddings": center,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "per_prompt": {},
        "ranking_preserved": {},
    }

    for pid, pool in pools.items():
        per_kernel: dict = {}
        for kname, kfn in kernel_fns.items():
            print(f"  [{pid}] kernel={kname} ...", end=" ", flush=True)

            inter = _vendi_norm(pool.inter_llm_codes, kfn, args.n_boot, rng)
            human = (_vendi_norm(pool.human_codes, kfn, args.n_boot, rng)
                     if pool.human_codes else None)

            per_kernel[kname] = {
                "inter_llm": inter,
                "human": human,
            }

            # Print inline summary
            h_str = f"{human['vendi_norm']:.3f}" if human else "N/A"
            print(f"inter={inter['vendi_norm']:.3f}  human={h_str}")

        results["per_prompt"][pid] = per_kernel

    # Check ranking preservation: inter-LLM Vendi/N < human Vendi/N
    print()
    for kname in kernel_fns:
        all_preserved = True
        for pid in args.prompts:
            pk = results["per_prompt"].get(pid, {}).get(kname, {})
            inter_vn = pk.get("inter_llm", {}).get("vendi_norm")
            human_vn = (pk.get("human", {}) or {}).get("vendi_norm")
            if inter_vn is None or human_vn is None:
                all_preserved = False
                continue
            if inter_vn >= human_vn:
                all_preserved = False
        results["ranking_preserved"][kname] = all_preserved

    return results


def print_table(results: dict):
    """Print a formatted comparison table to stdout."""
    prompts = results["config"]["prompts"]
    kernels = ["token", "codet5p", "codebert"]

    # Header
    print()
    print("=" * 90)
    print("SENSITIVITY ANALYSIS: CodeT5+ vs CodeBERT embeddings")
    print("=" * 90)
    print()

    # Table header
    header = f"{'Prompt':<8} {'Kernel':<10} {'Pool':<12} {'N':>4} {'Vendi/N':>9} {'CI_lo':>8} {'CI_hi':>8}"
    print(header)
    print("-" * len(header))

    for pid in prompts:
        pk = results["per_prompt"].get(pid, {})
        for kname in kernels:
            kdata = pk.get(kname, {})
            for pool_name in ("inter_llm", "human"):
                pdata = kdata.get(pool_name)
                if pdata is None:
                    print(f"{pid:<8} {kname:<10} {pool_name:<12} {'--':>4} {'--':>9} {'--':>8} {'--':>8}")
                    continue
                ci = pdata.get("ci_norm", [0, 0])
                print(f"{pid:<8} {kname:<10} {pool_name:<12} "
                      f"{pdata['n']:>4} "
                      f"{pdata['vendi_norm']:>9.4f} "
                      f"{ci[0]:>8.4f} {ci[1]:>8.4f}")
        print()

    # Ranking summary
    print("-" * 60)
    print("RANKING PRESERVATION: inter-LLM Vendi/N < human Vendi/N ?")
    print("-" * 60)
    for kname in kernels:
        preserved = results["ranking_preserved"].get(kname)
        tag = "YES" if preserved else "NO / INCOMPLETE"
        print(f"  {kname:<12}  {tag}")

    # Cross-embedder consistency
    print()
    print("-" * 60)
    print("CROSS-EMBEDDER CONSISTENCY (mean Vendi/N across prompts)")
    print("-" * 60)
    for pool_name in ("inter_llm", "human"):
        vals_by_kernel: dict[str, list[float]] = {k: [] for k in kernels}
        for pid in prompts:
            pk = results["per_prompt"].get(pid, {})
            for kname in kernels:
                pdata = (pk.get(kname, {}) or {}).get(pool_name)
                if pdata and pdata.get("vendi_norm") is not None:
                    vals_by_kernel[kname].append(pdata["vendi_norm"])
        print(f"  {pool_name}:")
        for kname in kernels:
            vals = vals_by_kernel[kname]
            if vals:
                print(f"    {kname:<12}  mean={np.mean(vals):.4f}  "
                      f"std={np.std(vals):.4f}  n_prompts={len(vals)}")
            else:
                print(f"    {kname:<12}  (no data)")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensitivity analysis: CodeT5+ vs CodeBERT on diversity suite",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--prompts", nargs="+", default=["AL-01", "AL-03", "AL-05"],
                   help="Prompt IDs to evaluate (default: AL-01, AL-03, AL-05)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature to filter LLM responses (default: 1.0)")
    p.add_argument("--raw-dir", type=Path,
                   default=ROOT / "results" / "raw_responses",
                   help="Directory containing per-prompt JSONL files")
    p.add_argument("--human-dir", type=Path,
                   default=ROOT / "results" / "human_baseline",
                   help="Directory containing human baseline JSONL files")
    p.add_argument("--out-json", type=Path,
                   default=ROOT / "results" / "sensitivity_codebert_results.json",
                   help="Path for output JSON")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Torch device for neural embedders (default: auto)")
    p.add_argument("--n-boot", type=int, default=1000,
                   help="Number of bootstrap resamples (default: 1000)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-model-for-inter", type=int, default=4)
    p.add_argument("--cap-per-pool", type=int, default=60)
    p.add_argument("--center-embeddings", default="none",
                   choices=["none", "center", "abtt"],
                   help="Anisotropy correction for neural embedders (default: none)")
    return p.parse_args()


def resolve_device(device: str) -> str:
    """Resolve 'auto' to cuda/cpu based on torch availability."""
    if device != "auto":
        if device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    print("  [warn] --device cuda but no CUDA available; using cpu")
                    return "cpu"
            except ImportError:
                print("  [warn] torch not installed; using cpu")
                return "cpu"
        return device
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"  [device] auto -> cuda ({name})")
            return "cuda"
        print("  [device] auto -> cpu")
        return "cpu"
    except ImportError:
        print("  [device] auto -> cpu (torch not available)")
        return "cpu"


def main():
    args = parse_args()
    args.device = resolve_device(args.device)

    results = run_sensitivity(args)
    print_table(results)

    # Save JSON
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
