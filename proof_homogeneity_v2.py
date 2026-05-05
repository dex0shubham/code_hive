#!/usr/bin/env python3
"""
proof_homogeneity_v2.py
=======================
Bulletproof v2 of the homogeneity proof. Imports core primitives from
proof_homogeneity.py (v1) and adds three reviewer-hardening features:

  1. Temperature sweep   — Pillar 1+2(+3) run at each temperature in a list,
                           plus a `temp_sweep_summary` showing how Vendi /
                           ordering changes with temperature.
  2. Family ablation     — leave-one-family-out: for each model family
                           (OpenAI, Anthropic, Google, ...), recompute the
                           inter-LLM Vendi with that family removed. If the
                           gap to human survives every leave-out, the result
                           is not driven by a single family.
  3. Mixed model — statsmodels MixedLM (fallback: OLS + prompt FE) for
                              vendi_norm ~ pool + prompt heterogeneity
                           per kernel, `human` as reference pool. JSON records
                           `model` and `fallback_reason` when MixedLM Hessian/CIs fail.

The v1 file (proof_homogeneity.py) is the load-bearing module — kernels,
Vendi, pool builders, harnesses, Pillar 1/2/3 all live there. v2 only
*orchestrates* across temperatures and adds two new analyses on top.

Usage
-----
  # Full v2 run (3 temps × 3 prompts) — defaults to CodeBERT on CUDA if a GPU
  # is available, otherwise falls back to CPU automatically.
  python proof_homogeneity_v2.py \
      --prompts PB-01 PB-02 PB-03 \
      --temperatures 0.0 0.7 1.0 \
      --human-dir results/human_baseline \
      --embedder codebert --device auto

  # Skip the slow pillars to iterate on stats only
  python proof_homogeneity_v2.py --skip-pillar3 --temperatures 1.0

Outputs (under --out-dir)
-------------------------
  summary.json                   master numeric results
  summary.md                     human-readable v2 summary
  fig_temperature_sweep.png      Vendi (LLM/human) vs temperature
  fig_family_ablation.png        inter-LLM Vendi with each family removed
  fig_mixed_effects.png          fixed-effect contrasts vs human
  t<TEMP>/                       per-temperature outputs (v1 pillars + figures)

  Default prompts PB-01..PB-03 are curated OPEN_ENDED rows from
  ``local_datasets/pool_b_open_ended_candidates/python_tasks/data.jsonl``
  (manifest: ``local_datasets/homogeneity_pool_b_curated.json``). Pillar 3
  has functional harnesses only for AL-*; with PB-* prompts Pillar 3 is
  skipped per prompt (``no_harness``). Use ``--prompts AL-01 AL-03 AL-05``
  for the full three-pillar pipeline on algorithmic tasks.
    summary.json
    summary.md
    fig_pillar1_vendi.png
    fig_pillar2_ordering.png
    fig_pillar3_traces.png

Dependencies (in addition to v1's deps)
---------------------------------------
  pip install statsmodels pandas         # for mixed-effects model
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

# ── v1 primitives ──────────────────────────────────────────────────
from proof_homogeneity import (  # type: ignore  # noqa: E402
    extract_code,
    load_llm_responses, load_human_responses,
    build_pools, PromptPools,
    KERNEL_FNS, _register_kernels,
    _pool_vendi_with_ci,
    run_pillar1, run_pillar2, run_pillar3,
    make_figures, write_summary,
    ROOT,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────
# 1. Pool builder helpers — pulled out of v1's main() to be reusable
# ─────────────────────────────────────────────────────────────────────

def _build_pools_for_temp(args, llm_at_temp: dict[str, list[dict]],
                          human: dict[str, list[dict]],
                          rng: np.random.Generator
                          ) -> dict[str, PromptPools]:
    """For one temperature's filtered LLM data, build per-prompt pools.

    Mirrors the inline logic in v1's main() but exposed as a function so the
    temperature sweep can call it in a loop.
    """
    pools: dict[str, PromptPools] = {}
    for pid in args.prompts:
        # null draws from OTHER prompts' codes only (within this temp)
        other_codes: list[str] = []
        for opid, rs in llm_at_temp.items():
            if opid == pid:
                continue
            for r in rs:
                c = extract_code(r.get("response_text", ""))
                if c:
                    other_codes.append(c)
        pools[pid] = build_pools(
            pid, llm_at_temp.get(pid, []), human.get(pid, []),
            other_codes, rng,
            samples_per_model_for_inter=args.samples_per_model_for_inter,
            cap_per_pool=args.cap_per_pool,
        )
    return pools


# ─────────────────────────────────────────────────────────────────────
# 2. Family ablation
# ─────────────────────────────────────────────────────────────────────

def run_family_ablation(args, llm_at_temp: dict[str, list[dict]],
                        human: dict[str, list[dict]],
                        kernels: list[str], seed: int) -> dict:
    """For each model family observed in the data, recompute the inter-LLM
    Vendi after removing that family. Returns:

        {
          "<temp>": {
             "families": [...],
             "per_family_excluded": {
                "OpenAI": {prompt: {kernel: {vendi, ci, vendi_norm, ci_norm, n}}},
                ...
             },
             "baseline_all": {prompt: {kernel: {...}}},
          }
        }
    """
    families: set[str] = set()
    for rs in llm_at_temp.values():
        for r in rs:
            families.add(r.get("model_family", "?"))
    families.discard("?")

    rng_seed_base = seed + 100

    # Baseline (no family removed) for reference
    rng = np.random.default_rng(rng_seed_base)
    base_pools = _build_pools_for_temp(args, llm_at_temp, human, rng)
    baseline: dict[str, dict] = {}
    for pid, pool in base_pools.items():
        per_kernel = {}
        for kname in kernels:
            per_kernel[kname] = _pool_vendi_with_ci(
                pool.inter_llm_codes, kname, args.n_boot, rng)
        baseline[pid] = per_kernel

    per_family_excluded: dict[str, dict] = {}
    for fam in sorted(families):
        # Filter out this family from all prompts
        llm_minus = {
            pid: [r for r in rs if r.get("model_family") != fam]
            for pid, rs in llm_at_temp.items()
        }
        rng = np.random.default_rng(rng_seed_base + hash(fam) % 1000)
        pools = _build_pools_for_temp(args, llm_minus, human, rng)
        per_prompt: dict[str, dict] = {}
        for pid, pool in pools.items():
            n_inter = len(pool.inter_llm_codes)
            per_kernel = {}
            for kname in kernels:
                per_kernel[kname] = _pool_vendi_with_ci(
                    pool.inter_llm_codes, kname, args.n_boot, rng)
            per_kernel["_n_models_remaining"] = len({
                r.get("model_display") for r in llm_minus.get(pid, [])
            })
            per_kernel["_n_inter_samples"] = n_inter
            per_prompt[pid] = per_kernel
        per_family_excluded[fam] = per_prompt
        print(f"  [ablation] excluded {fam}: "
              f"avg inter-Vendi (token) = "
              f"{np.mean([per_prompt[p]['token']['vendi'] for p in per_prompt]):.2f}")

    return {
        "families": sorted(families),
        "baseline_all": baseline,
        "per_family_excluded": per_family_excluded,
    }


# ─────────────────────────────────────────────────────────────────────
# 3. Mixed-effects model
# ─────────────────────────────────────────────────────────────────────

def _summarize_fit(res) -> dict:
    """Pull comparable inference tables from a fitted statsmodels result."""
    params = {k: float(v) for k, v in res.params.items()}
    pvals = {k: float(v) for k, v in res.pvalues.items()}
    ses = {k: float(v) for k, v in res.bse.items()}
    ci_df = res.conf_int()
    ci_clean = {k: [float(row[0]), float(row[1])] for k, row in ci_df.iterrows()}
    return {
        "params": params,
        "pvalues": pvals,
        "stderr": ses,
        "ci95": ci_clean,
        "loglik": float(res.llf),
    }


def run_mixed_effects(pillar1_by_temp: dict[float, dict]) -> dict:
    """Fit vendi_norm ~ C(pool, ref='human') + (1 | prompt) per kernel.

    Path A: statsmodels MixedLM (random intercept per prompt). With very
    few prompts, the Hessian for standard errors can be singular.

    Path B: OLS with prompt dummies — same pool contrasts, prompt absorbed
    as fixed effects. See summary fields ``model`` and ``fallback_reason``.
    """
    try:
        import statsmodels.formula.api as smf  # type: ignore
        import pandas as pd  # type: ignore
    except ImportError as e:
        return {"status": "skipped",
                "reason": f"statsmodels/pandas not installed: {e}"}

    rows: list[dict] = []
    for temp, p1 in pillar1_by_temp.items():
        for pid, kr in p1.get("per_prompt", {}).items():
            for kname, pp in kr.items():
                for pool in ("null", "human", "intra_llm_mean", "inter_llm"):
                    v = pp.get(pool)
                    if v is None:
                        continue
                    vn = v.get("vendi_norm")
                    if vn is None:
                        continue
                    rows.append({
                        "kernel": kname,
                        "prompt": pid,
                        "temperature": float(temp),
                        "pool": pool,
                        "vendi_norm": float(vn),
                        "n": int(v.get("n", v.get("n_models", 0))),
                    })
    if not rows:
        return {"status": "skipped", "reason": "no data"}

    df = pd.DataFrame(rows)

    out: dict = {"status": "ok", "per_kernel": {}, "n_obs_total": len(df)}
    for kname in sorted(df["kernel"].unique()):
        sub = df[df["kernel"] == kname].copy()
        sub["pool"] = pd.Categorical(
            sub["pool"],
            categories=["human", "null", "intra_llm_mean", "inter_llm"],
            ordered=False,
        )
        sub["prompt"] = pd.Categorical(sub["prompt"])

        ml_err: str | None = None
        try:
            m_mixed = smf.mixedlm(
                "vendi_norm ~ C(pool, Treatment(reference='human'))",
                sub,
                groups=sub["prompt"],
            )
            res = m_mixed.fit(reml=True, method="lbfgs", maxiter=200)
            # SEs / CIs touch the Hessian — often where singularity surfaces.
            _ = res.bse
            _ = res.conf_int()
            summ = _summarize_fit(res)
            out["per_kernel"][kname] = {
                **summ,
                "model": "MixedLM",
                "n_obs": int(len(sub)),
                "n_groups": int(sub["prompt"].nunique()),
            }
        except Exception as e:
            ml_err = f"{type(e).__name__}: {str(e)[:180]}"
            try:
                m_ols = smf.ols(
                    "vendi_norm ~ C(pool, Treatment(reference='human')) "
                    "+ C(prompt)",
                    sub,
                )
                res_o = m_ols.fit()
                summ = _summarize_fit(res_o)
                out["per_kernel"][kname] = {
                    **summ,
                    "model": "OLS_with_prompt_FE",
                    "fallback_reason": f"MixedLM failed: {ml_err}",
                    "n_obs": int(len(sub)),
                    "n_groups": int(sub["prompt"].nunique()),
                }
            except Exception as e2:
                fb = ml_err if ml_err else "unknown MixedLM failure"
                out["per_kernel"][kname] = {
                    "error": (
                        f"MixedLM failed ({fb}); OLS fallback failed: "
                        f"{type(e2).__name__}: {str(e2)[:180]}"
                    ),
                }
    return out


# ─────────────────────────────────────────────────────────────────────
# 4. Temperature sweep orchestrator
# ─────────────────────────────────────────────────────────────────────

def run_temperature_sweep(args, llm_all: dict[str, list[dict]],
                          human: dict[str, list[dict]],
                          kernels: list[str]) -> dict:
    """Run pillars at each temperature in args.temperatures."""
    by_temp: dict = {}
    for temp in args.temperatures:
        # Filter LLM data to this temperature
        llm_at_temp: dict[str, list[dict]] = {
            pid: [r for r in rs if abs(r.get("temperature", 0.0) - temp) < 1e-6]
            for pid, rs in llm_all.items()
        }
        n_total = sum(len(v) for v in llm_at_temp.values())
        if n_total == 0:
            print(f"  [warn] no LLM data at temp={temp}; skipping.")
            continue

        sub_dir = args.out_dir / f"t{temp:g}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*64}")
        print(f"=== TEMPERATURE = {temp}  ({n_total} samples)")
        print(f"{'='*64}")
        rng = np.random.default_rng(args.seed)
        pools = _build_pools_for_temp(args, llm_at_temp, human, rng)
        for pid, p in pools.items():
            print(f"  pools[{pid}]  inter={len(p.inter_llm_codes)}  "
                  f"intra_models={len(p.intra_llm_codes)}  "
                  f"human={len(p.human_codes)}  null={len(p.null_codes)}")

        print("\n[Pillar 1] Vendi over kernels x pools ...")
        pillar1 = run_pillar1(pools, kernels, args.n_boot, args.seed)

        print("\n[Pillar 2] LLM vs human ordering test ...")
        pillar2 = run_pillar2(pillar1, args.n_boot, args.seed + 1)

        if args.skip_pillar3:
            pillar3 = {"status": "skipped (--skip-pillar3)"}
        else:
            print("\n[Pillar 3] Functional convergence ...")
            pillar3 = run_pillar3(pools, sub_dir,
                                  timeout=args.harness_timeout)

        # Per-temp pillars: write to subfolder using v1's writer + figures
        try:
            shim_args = _pillar_writer_shim(args, sub_dir)
            write_summary(sub_dir, shim_args, pillar1, pillar2, pillar3)
            make_figures(pillar1, pillar2 if isinstance(pillar2, dict) else {},
                         pillar3 if isinstance(pillar3, dict) else {},
                         sub_dir)
        except Exception as e:
            print(f"  [warn] per-temp writer/figures failed: "
                  f"{type(e).__name__}: {e}")

        by_temp[temp] = {
            "pillar1": pillar1,
            "pillar2": pillar2,
            "pillar3": pillar3,
            "llm_at_temp": llm_at_temp,
        }
    return by_temp


def _pillar_writer_shim(parent_args, sub_dir: Path):
    """Build a tiny argparse.Namespace that v1's write_summary expects."""
    return argparse.Namespace(
        prompts=parent_args.prompts,
        temperature=None,  # written as multi-temp at the master level
        raw_dir=parent_args.raw_dir,
        human_dir=parent_args.human_dir,
        device=parent_args.device,
        n_boot=parent_args.n_boot,
        seed=parent_args.seed,
        samples_per_model_for_inter=parent_args.samples_per_model_for_inter,
        cap_per_pool=parent_args.cap_per_pool,
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Figures (v2-specific)
# ─────────────────────────────────────────────────────────────────────

def make_v2_figures(out_dir: Path, by_temp: dict, family_ablation: dict,
                    mixed_fx: dict, kernels: list[str]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping v2 figures")
        return
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200,
                         "savefig.bbox": "tight", "font.size": 9})

    # ── Fig: Vendi (inter-LLM, human, intra-LLM) vs temperature ──
    try:
        _fig_temp_sweep(plt, by_temp, kernels, out_dir)
    except Exception as e:
        print(f"  [warn] fig_temperature_sweep failed: {type(e).__name__}: {e}")

    # ── Fig: family ablation ──
    try:
        _fig_family_ablation(plt, family_ablation, kernels, out_dir)
    except Exception as e:
        print(f"  [warn] fig_family_ablation failed: {type(e).__name__}: {e}")

    # ── Fig: mixed-effects coefficients ──
    try:
        _fig_mixed_effects(plt, mixed_fx, out_dir)
    except Exception as e:
        print(f"  [warn] fig_mixed_effects failed: {type(e).__name__}: {e}")


def _fig_temp_sweep(plt, by_temp, kernels, out_dir):
    if not by_temp:
        return
    temps = sorted(by_temp.keys())
    fig, axes = plt.subplots(1, len(kernels),
                             figsize=(5 * len(kernels), 4),
                             squeeze=False)
    pool_styles = {
        "human":          {"color": "#1f77b4", "marker": "o", "label": "human"},
        "inter_llm":      {"color": "#d62728", "marker": "s", "label": "inter-LLM"},
        "intra_llm_mean": {"color": "#2ca02c", "marker": "^", "label": "intra-LLM"},
        "null":           {"color": "#888",    "marker": "x", "label": "null"},
    }
    for ax, kname in zip(axes[0], kernels):
        for pool, style in pool_styles.items():
            xs, ys, lo, hi = [], [], [], []
            for t in temps:
                p1 = by_temp[t]["pillar1"]
                # Average across prompts for a cleaner curve
                vals = []
                lows = []
                his  = []
                for pid, kr in p1.get("per_prompt", {}).items():
                    pp = kr.get(kname, {})
                    v = pp.get(pool)
                    if v is None or v.get("vendi_norm") is None:
                        continue
                    vals.append(v["vendi_norm"])
                    cn = v.get("ci_norm", [v["vendi_norm"], v["vendi_norm"]])
                    lows.append(cn[0]); his.append(cn[1])
                if not vals:
                    continue
                xs.append(t)
                ys.append(np.mean(vals))
                lo.append(np.mean(vals) - np.mean(lows))
                hi.append(np.mean(his) - np.mean(vals))
            if not xs:
                continue
            ax.errorbar(xs, ys, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)],
                        capsize=3, **style)
        ax.set_xlabel("Temperature")
        ax.set_ylabel("Vendi / N  (per-sample uniqueness)")
        ax.set_title(f"kernel = {kname}")
        ax.set_ylim(0, 1)
    axes[0][-1].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("Temperature sweep — does monoculture survive higher T?")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_temperature_sweep.png")
    plt.close(fig)


def _fig_family_ablation(plt, family_ablation, kernels, out_dir):
    """Bar chart: per (kernel, family-excluded), mean inter-LLM Vendi/N
    averaged over prompts. Includes baseline (no family removed) as a
    horizontal reference line."""
    if not family_ablation:
        return
    # Use the first available temperature's ablation data for the headline plot
    temps_with = [t for t, d in family_ablation.items() if d.get("families")]
    if not temps_with:
        return
    temp = sorted(temps_with)[-1]  # highest temp (most diversity expected)
    abl = family_ablation[temp]

    families = abl["families"]
    baseline = abl["baseline_all"]
    per_fam  = abl["per_family_excluded"]

    fig, axes = plt.subplots(1, len(kernels),
                             figsize=(5 * len(kernels), 4),
                             squeeze=False)
    for ax, kname in zip(axes[0], kernels):
        # Baseline mean across prompts
        base_vals = [baseline[p][kname]["vendi_norm"]
                     for p in baseline if baseline[p].get(kname)]
        base_mean = float(np.mean(base_vals)) if base_vals else 0.0

        bars = []
        for fam in families:
            vals = [per_fam[fam][p][kname]["vendi_norm"]
                    for p in per_fam[fam]
                    if per_fam[fam][p].get(kname)]
            bars.append(float(np.mean(vals)) if vals else 0.0)

        x = np.arange(len(families))
        ax.bar(x, bars, color="#d62728", alpha=0.8)
        ax.axhline(base_mean, color="black", linestyle="--", linewidth=0.8,
                   label=f"baseline (all): {base_mean:.3f}")
        ax.set_xticks(x)
        ax.set_xticklabels(families, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Inter-LLM Vendi / N  (per-sample uniqueness)")
        ax.set_title(f"kernel = {kname}  (T={temp})")
        ax.legend(frameon=False, fontsize=7)
    fig.suptitle(
        "Leave-one-family-out — bars near baseline = result not driven by a "
        "single family.")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_family_ablation.png")
    plt.close(fig)


def _fig_mixed_effects(plt, mixed_fx, out_dir):
    """Coefficient plot per kernel: contrasts vs human (the reference)."""
    if mixed_fx.get("status") != "ok":
        return
    per_kernel = mixed_fx.get("per_kernel", {})
    kernels = list(per_kernel.keys())
    if not kernels:
        return
    fig, axes = plt.subplots(1, len(kernels),
                             figsize=(5 * len(kernels), 4),
                             squeeze=False)
    for ax, kname in zip(axes[0], kernels):
        kdata = per_kernel[kname]
        if "error" in kdata:
            ax.text(0.5, 0.5, f"fit failed:\n{kdata['error']}",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=8)
            ax.set_title(f"kernel = {kname}")
            continue
        params = kdata["params"]
        cis   = kdata["ci95"]
        # Pull just the pool contrasts (skip Intercept and Group Var)
        contrast_keys = [k for k in params
                         if "pool" in k and "Intercept" not in k]
        labels = [k.split(".")[-1].rstrip("]") for k in contrast_keys]
        labels = [l.replace("T(", "").replace(")", "")
                  .replace("'", "").replace('"', "")
                  for l in labels]
        means = [params[k] for k in contrast_keys]
        lo    = [params[k] - cis[k][0] for k in contrast_keys]
        hi    = [cis[k][1] - params[k] for k in contrast_keys]
        y = np.arange(len(labels))
        ax.errorbar(means, y, xerr=[np.maximum(lo, 0), np.maximum(hi, 0)],
                    fmt="o", capsize=3, color="#d62728")
        ax.axvline(0, color="black", linewidth=0.5, linestyle=":")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Coefficient (Δ vendi_norm vs human)")
        mkind = kdata.get("model", "")
        suf = f"  [{mkind}]" if mkind else ""
        ax.set_title(f"kernel = {kname}  (n_obs={kdata['n_obs']}){suf}")
    fig.suptitle(
        "Pool contrasts vs human (MixedLM or OLS+prompt FE) — "
        "values < 0 mean less diverse than humans."
    )
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mixed_effects.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 6. Summary writer
# ─────────────────────────────────────────────────────────────────────

def write_v2_summary(out_dir: Path, args, by_temp: dict,
                     family_ablation: dict, mixed_fx: dict,
                     kernels: list[str]):
    summary = {
        "config": {
            "prompts": args.prompts,
            "temperatures": args.temperatures,
            "raw_dir": str(args.raw_dir),
            "human_dir": str(args.human_dir) if args.human_dir else None,
            "kernels": kernels,
            "device": args.device,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "samples_per_model_for_inter": args.samples_per_model_for_inter,
            "cap_per_pool": args.cap_per_pool,
            "embedder": args.embedder,
            "center_embeddings": getattr(args, "center_embeddings", "none"),
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "by_temperature": {str(t): {
            "pillar1": d["pillar1"],
            "pillar2": d["pillar2"],
            "pillar3": d["pillar3"],
        } for t, d in by_temp.items()},
        "family_ablation_by_temp": {str(t): d for t, d in family_ablation.items()},
        "mixed_effects": mixed_fx,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str), encoding="utf-8")

    # Markdown
    lines: list[str] = []
    lines.append("# Homogeneity Proof v2 — bulletproof summary\n")
    lines.append(f"- Prompts: {', '.join(args.prompts)}")
    lines.append(f"- Temperatures: {args.temperatures}")
    lines.append(f"- Kernels: {', '.join(kernels)}")
    lines.append(f"- Embedder: {args.embedder}")
    lines.append("")

    lines.append("## Headline — normalized Δ(LLM−human) by temperature × kernel\n")
    lines.append("| Temp | Kernel | Δ(LLM−human) | CI95 | p | Δ(intra-LLM−human) |")
    lines.append("|---|---|---|---|---|---|")
    for t in sorted(by_temp.keys()):
        p2 = by_temp[t]["pillar2"]
        if p2.get("status") != "ok":
            continue
        for kname, res in p2["per_kernel"].items():
            d  = res.get("llm_vs_human_norm") or {}
            di = res.get("intra_vs_human_norm") or {}
            ci = d.get("ci", [None, None])
            lines.append(
                f"| {t} | {kname} | "
                f"{d.get('delta_mean', 0):+.3f} | "
                f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | "
                f"{d.get('p', 0):.4f} | "
                f"{di.get('delta_mean', 0):+.3f} |"
            )
    lines.append("")

    lines.append("## Family ablation — inter-LLM Vendi / N with each family removed\n")
    for t, abl in family_ablation.items():
        if not abl.get("families"):
            continue
        lines.append(f"### Temperature = {t}")
        baseline = abl["baseline_all"]
        per_fam  = abl["per_family_excluded"]
        for kname in kernels:
            base_vals = [baseline[p][kname]["vendi_norm"]
                         for p in baseline if baseline[p].get(kname)]
            base_mean = float(np.mean(base_vals)) if base_vals else 0.0
            lines.append(f"- **kernel={kname}**  (baseline = {base_mean:.3f})")
            for fam in abl["families"]:
                vals = [per_fam[fam][p][kname]["vendi_norm"]
                        for p in per_fam[fam] if per_fam[fam][p].get(kname)]
                fam_mean = float(np.mean(vals)) if vals else 0.0
                delta = fam_mean - base_mean
                lines.append(f"    - excl {fam:<14}  vendi_norm={fam_mean:.3f}  "
                             f"(Δ = {delta:+.3f})")
        lines.append("")

    lines.append(
        "## Mixed model / OLS fallback — vendi_norm ~ pool + prompt heterogeneity\n"
    )
    if mixed_fx.get("status") != "ok":
        lines.append(f"- skipped: {mixed_fx.get('reason', 'unknown')}")
    else:
        lines.append(
            "Primary fit: MixedLM random intercept `(1 | prompt)`. If Hessian/CIs "
            "are singular (common with fewer than ~5 prompts), we use equivalent "
            "OLS with prompt fixed effects (`summary.json`: `model`, "
            "`fallback_reason`).\n\n"
            "Fixed-effect contrasts vs the reference pool (`human`). Negative "
            "coefficient = pool less diverse than humans.\n")
        for kname, kdata in mixed_fx["per_kernel"].items():
            lines.append(f"### kernel = {kname}")
            if "error" in kdata:
                lines.append(f"- fit error: {kdata['error']}")
                continue
            mod = kdata.get("model", "unknown")
            lines.append(f"- **model**: `{mod}`")
            if kdata.get("fallback_reason"):
                lines.append(f"- **fallback_reason**: {kdata['fallback_reason']}")
            params = kdata["params"]
            pvals  = kdata["pvalues"]
            cis    = kdata["ci95"]
            for k in params:
                if "Intercept" in k or "Var" in k or "pool" not in k:
                    continue
                lab = (k.split("[T.")[-1].rstrip("]") if "[T." in k
                       else k.split("(")[-1].rstrip(")"))
                lines.append(
                    f"- {lab:<20}  β = {params[k]:+.3f}   "
                    f"CI95 = [{cis[k][0]:+.3f}, {cis[k][1]:+.3f}]   "
                    f"p = {pvals[k]:.4f}")
            lines.append(f"  (n_obs = {kdata['n_obs']}, "
                         f"n_groups = {kdata['n_groups']}, "
                         f"loglik = {kdata['loglik']:.2f})\n")
    lines.append("")

    lines.append("## Per-temperature pillars\n")
    for t in sorted(by_temp.keys()):
        lines.append(f"- `t{t:g}/summary.md`")
    lines.append("")
    lines.append("## v2 figures\n")
    lines.append("- `fig_temperature_sweep.png`")
    lines.append("- `fig_family_ablation.png`")
    lines.append("- `fig_mixed_effects.png`")

    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 7. CLI / main
# ─────────────────────────────────────────────────────────────────────

def _resolve_device(device: str, embedder: str) -> str:
    """Resolve --device 'auto' to 'cuda' if torch.cuda.is_available(), else 'cpu'.

    For non-neural embedders (tfidf/none) the device is irrelevant; we still
    return a concrete value for logging/JSON. We only import torch when needed
    so tfidf-only runs don't pay the import cost or fail on torch-less envs.
    """
    if device != "auto":
        if device == "cuda" and embedder in ("codebert", "unixcoder", "codet5p"):
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    print("  [warn] --device cuda requested but torch.cuda."
                          "is_available() is False; falling back to CPU.")
                    return "cpu"
            except Exception as e:
                print(f"  [warn] could not import torch ({e}); falling back to CPU.")
                return "cpu"
        return device

    # device == "auto"
    if embedder not in ("codebert", "unixcoder", "codet5p"):
        return "cpu"  # neural device doesn't matter for tfidf/none
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
                print(f"  [device] auto-selected cuda  ({name})")
            except Exception:
                print(f"  [device] auto-selected cuda")
            return "cuda"
        print("  [device] auto-selected cpu (torch.cuda.is_available()=False)")
        return "cpu"
    except Exception as e:
        print(f"  [device] auto -> cpu (torch import failed: {e})")
        return "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--prompts", nargs="+", default=["PB-01", "PB-02", "PB-03"],
                   help="Prompt IDs from prompt_suite (defaults: Pool-B-curated "
                        "PB-01..03). Use AL-01 AL-03 AL-05 for Pillar-3 harnesses.")
    p.add_argument("--temperatures", nargs="+", type=float, default=[1.0],
                   help="One or more temperatures to sweep over.")
    p.add_argument("--raw-dir", type=Path,
                   default=ROOT / "results" / "raw_responses")
    p.add_argument("--human-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: results/proof_v2/v2_<UTC>_<embedder>/ "
                        "(embedder suffix avoids overwriting codebert vs tfidf runs).")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="Compute device for neural embedders (codebert/unixcoder). "
                        "'auto' (default) uses CUDA if torch.cuda.is_available(), "
                        "else CPU.")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-model-for-inter", type=int, default=4)
    p.add_argument("--cap-per-pool", type=int, default=60)
    p.add_argument("--embedder", default="codebert",
                   choices=["unixcoder", "codebert", "codet5p", "tfidf", "none"],
                   help="Embedding kernel. Default is 'codebert' (microsoft/"
                        "codebert-base) on CUDA when available. 'codet5p' "
                        "uses Salesforce/codet5p-110m-embedding (a "
                        "contrastively trained code retriever, recommended "
                        "for diversity studies — its cosine has wider "
                        "dynamic range than vanilla MLM encoders). 'tfidf' "
                        "is a no-torch sklearn-only fallback; 'none' skips "
                        "neural embeddings entirely (token + AST only).")
    p.add_argument("--center-embeddings", dest="center_embeddings",
                   default="none", choices=["none", "center", "abtt"],
                   help="Anisotropy correction for neural embedders before "
                        "cosine similarity. 'center' subtracts the column "
                        "mean (BERT-Whitening step 1). 'abtt' additionally "
                        "projects out the top principal direction "
                        "(All-But-The-Top, Mu & Viswanath 2018). Ignored "
                        "for tfidf/none.")
    p.add_argument("--skip-pillar3", action="store_true")
    p.add_argument("--skip-family-ablation", action="store_true")
    p.add_argument("--skip-mixed-effects", action="store_true")
    p.add_argument("--harness-timeout", type=float, default=6.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate setup and exit before computing kernels.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.out_dir = (
            ROOT / "results" / "proof_v2" / f"v2_{ts}_{args.embedder}"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    args.device = _resolve_device(args.device, args.embedder)

    print(f"== proof_homogeneity_v2 ==")
    print(f"  prompts:       {args.prompts}")
    print(f"  temperatures:  {args.temperatures}")
    print(f"  raw_dir:       {args.raw_dir}")
    print(f"  human_dir:     {args.human_dir}")
    print(f"  out_dir:       {args.out_dir}")
    print(f"  embedder:      {args.embedder}")
    print(f"  device:        {args.device}")
    print()

    # Load all temperatures' LLM data; we filter per-temp inside the sweep
    llm_all = load_llm_responses(args.raw_dir, args.prompts, temperature=None)
    n_total = sum(len(v) for v in llm_all.values())
    print(f"Loaded {n_total} LLM responses across {len(llm_all)} prompts "
          f"(all temperatures)")
    if n_total == 0:
        print(f"\nERROR: no LLM responses in {args.raw_dir} for "
              f"prompts={args.prompts}. Run multi_model_sampler first.")
        sys.exit(2)

    human = load_human_responses(args.human_dir, args.prompts)
    n_human = sum(len(v) for v in human.values())
    print(f"Loaded {n_human} human responses")

    if args.dry_run:
        print("\n--dry-run: setup OK, exiting before kernels.")
        return

    # Register kernels
    _register_kernels(embedder=args.embedder, device=args.device,
                      center=getattr(args, "center_embeddings", "none"))
    kernels = list(KERNEL_FNS.keys())
    print(f"Active kernels: {kernels}")

    # 1. Temperature sweep (also runs Pillar 1+2+3 at each T)
    by_temp = run_temperature_sweep(args, llm_all, human, kernels)

    # 2. Family ablation per-temperature
    family_ablation: dict = {}
    if not args.skip_family_ablation:
        print(f"\n[Family ablation]")
        for temp in args.temperatures:
            llm_at_temp = {
                pid: [r for r in rs if abs(r.get("temperature", 0.0) - temp) < 1e-6]
                for pid, rs in llm_all.items()
            }
            if not any(llm_at_temp.values()):
                continue
            print(f"  temp={temp}:")
            family_ablation[temp] = run_family_ablation(
                args, llm_at_temp, human, kernels, args.seed)

    # 3. Mixed-effects across all temperatures and prompts
    mixed_fx: dict = {"status": "skipped (--skip-mixed-effects)"}
    if not args.skip_mixed_effects:
        print(f"\n[Mixed-effects model]")
        pillar1_by_temp = {t: d["pillar1"] for t, d in by_temp.items()}
        mixed_fx = run_mixed_effects(pillar1_by_temp)
        if mixed_fx.get("status") == "ok":
            print(f"  fit OK across {mixed_fx['n_obs_total']} observations")
        else:
            print(f"  skipped: {mixed_fx.get('reason')}")

    # Write summary first (so plot bugs never destroy numbers)
    print("\n[Summary] writing ...")
    write_v2_summary(args.out_dir, args, by_temp, family_ablation, mixed_fx, kernels)

    # Then v2 figures
    print("[Figures] writing ...")
    make_v2_figures(args.out_dir, by_temp, family_ablation, mixed_fx, kernels)

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()
