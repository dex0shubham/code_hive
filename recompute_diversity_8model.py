#!/usr/bin/env python3
"""
recompute_diversity_8model.py
=============================
Recompute diversity-pillar Vendi/N (Table 2, Figure 1, Table 3, Table 4)
on the 8 producing models only, by filtering the raw PB-01..20 response
data to exclude Claude 3.5 Haiku, Gemini 2.5 Flash, Gemini 2.5 Pro,
then running proof_homogeneity_v2.py on the filtered data.

Usage
-----
  python recompute_diversity_8model.py \\
      --raw-dir "C:/Users/RushiHiray/Downloads/latest_final_result-20260506T195609Z-3-001/latest_final_result" \\
      --human-dir "results/human_baseline" \\
      --out-dir "results/proof_v2_8model" \\
      --device cuda

To also recompute the SE-* bridge values:
  python recompute_diversity_8model.py \\
      --raw-dir "C:/Users/RushiHiray/Downloads/latest_final_result-20260506T195609Z-3-001/latest_final_result/_x/SE_responses" \\
      --suite SE \\
      --out-dir "results/se_bridge_8model" \\
      --device cuda
(adjust SE raw-dir path to match where the SE-*.jsonl files live)
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXCLUDED_MODELS = {"Claude 3.5 Haiku", "Gemini 2.5 Flash", "Gemini 2.5 Pro"}


def get_prompts(suite: str) -> list[str]:
    if suite == "PB":
        return [f"PB-{i:02d}" for i in range(1, 21)]
    if suite == "SE":
        return [f"SE-{i:02d}" for i in range(1, 31)]
    raise ValueError(f"unknown suite {suite!r}")


def filter_raw_dir(src: Path, dst: Path, prompts: list[str]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    total_in = 0
    total_out = 0
    missing = []
    for pid in prompts:
        in_fp = src / f"{pid}.jsonl"
        if not in_fp.exists():
            missing.append(pid)
            continue
        out_fp = dst / f"{pid}.jsonl"
        with open(in_fp, encoding="utf-8") as fin, \
             open(out_fp, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip():
                    continue
                rec = json.loads(line)
                total_in += 1
                if rec.get("model_display") in EXCLUDED_MODELS:
                    continue
                fout.write(line)
                total_out += 1
    print(f"  filtered {total_in} -> {total_out} records "
          f"(dropped {total_in - total_out} from {sorted(EXCLUDED_MODELS)})")
    if missing:
        print(f"  [warn] missing prompt files: {missing}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="Directory with {prompt_id}.jsonl files for the full pilot pool")
    ap.add_argument("--suite", default="PB", choices=["PB", "SE"],
                    help="Prompt suite to process")
    ap.add_argument("--human-dir", default="results/human_baseline",
                    help="Human baseline directory (only used for PB suite)")
    ap.add_argument("--out-dir", default="results/proof_v2_8model",
                    help="Output directory")
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.0, 0.7])
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--embedder", default="codet5p",
                    choices=["codet5p", "codebert", "tfidf"])
    ap.add_argument("--center-embeddings", default="kernel",
                    choices=["kernel", "embedding", "none"],
                    help="Match the main paper's pipeline (default: kernel)")
    ap.add_argument("--keep-filtered", action="store_true")
    args = ap.parse_args()

    src = Path(args.raw_dir)
    if not src.exists():
        sys.exit(f"raw-dir not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = get_prompts(args.suite)

    filtered_dir = out_dir / "_filtered_raw"
    print(f"[1/2] Filtering raw responses ({args.suite} suite) -> {filtered_dir}")
    filter_raw_dir(src, filtered_dir, prompts)

    print(f"[2/2] Running proof_homogeneity_v2 on filtered data")
    cmd = [
        sys.executable, "proof_homogeneity_v2.py",
        "--prompts", *prompts,
        "--temperatures", *[str(t) for t in args.temperatures],
        "--raw-dir", str(filtered_dir),
        "--out-dir", str(out_dir / "v2_run"),
        "--embedder", args.embedder,
        "--device", args.device,
        "--center-embeddings", args.center_embeddings,
    ]
    if args.suite == "PB":
        cmd += ["--human-dir", args.human_dir]
    # SE suite: omit --human-dir; v2 defaults to None (skipped) for SE-* prompts
    print("  $", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"proof_homogeneity_v2 exited with code {rc}")

    if not args.keep_filtered:
        shutil.rmtree(filtered_dir, ignore_errors=True)

    print(f"\nDone. Results in: {out_dir / 'v2_run'}")
    print("  - summary.md     -> headline + family ablation (Tables 3, 4)")
    print("  - t0.7/summary.md -> per-prompt Vendi/N at T=0.7 (Table 2)")
    print("  - t0/summary.md   -> per-prompt Vendi/N at T=0   (Table 2)")


if __name__ == "__main__":
    main()
