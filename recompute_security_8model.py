#!/usr/bin/env python3
"""
recompute_security_8model.py
============================
Recompute the security pillar (raw + calibrated) on the 8 producing models.

Pipeline:
  1. Filter SE-01..30 raw response data to drop the 3 excluded models.
  2. Run security_pillar.py on the filtered data        -> raw scan_results
  3. Run security_pillar_calibrated.py on those results -> calibrated outputs

Usage
-----
  python recompute_security_8model.py \\
      --raw-dir "C:/Users/RushiHiray/Downloads/latest_final_result-20260506T195609Z-3-001/latest_final_result/_x/SE_responses" \\
      --out-dir "results/proof_security_8model"

Outputs (under --out-dir)
-------------------------
  raw/scan_results.jsonl, raw/summary.json, raw/summary.md, raw/figures
      -> uncalibrated static analysis
  calibrated/scan_results.jsonl, calibrated/summary.json, calibrated/summary.md,
  calibrated/calibration_audit.json
      -> calibrated (5-rule overlay) analysis - this is what feeds Section 6.1
         (vuln rate, CWE counts, calibration fires) and Pillar S2 tables.
"""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

EXCLUDED_MODELS = {"Claude 3.5 Haiku", "Gemini 2.5 Flash", "Gemini 2.5 Pro"}
SE_PROMPTS = [f"SE-{i:02d}" for i in range(1, 31)]


def filter_raw_dir(src: Path, dst: Path, prompts: list[str]) -> tuple[int, int]:
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
    if missing:
        print(f"  [warn] missing prompt files: {missing}")
    return total_in, total_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="Directory with SE-01.jsonl..SE-30.jsonl (full 11-model pilot pool)")
    ap.add_argument("--out-dir", default="results/proof_security_8model",
                    help="Output directory")
    ap.add_argument("--temperatures", nargs="*", type=float, default=None,
                    help="Optional temperature filter (default: all)")
    ap.add_argument("--bandit", action="store_true",
                    help="Also run Bandit scanner (matches main paper run)")
    ap.add_argument("--pypi-live-check", action="store_true",
                    help="Live PyPI dependency check (matches main paper run)")
    ap.add_argument("--keep-filtered", action="store_true",
                    help="Keep the temporary filtered raw dir for inspection")
    args = ap.parse_args()

    src = Path(args.raw_dir)
    if not src.exists():
        sys.exit(f"raw-dir not found: {src}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_out = out_dir / "raw"
    calib_out = out_dir / "calibrated"
    filtered_dir = out_dir / "_filtered_raw"

    # ---------------------------------------------------------------
    # Step 1: filter raw SE responses to drop excluded models
    # ---------------------------------------------------------------
    print(f"[1/3] Filtering SE responses -> {filtered_dir}")
    total_in, total_out = filter_raw_dir(src, filtered_dir, SE_PROMPTS)
    print(f"  filtered {total_in} -> {total_out} records "
          f"(dropped {total_in - total_out} from {sorted(EXCLUDED_MODELS)})")

    # ---------------------------------------------------------------
    # Step 2: run security_pillar.py (raw static analysis)
    # ---------------------------------------------------------------
    print(f"\n[2/3] Running security_pillar.py (raw)")
    cmd = [
        sys.executable, "security_pillar.py",
        "--prompts", *SE_PROMPTS,
        "--raw-dir", str(filtered_dir),
        "--out-dir", str(raw_out),
    ]
    if args.temperatures is not None:
        cmd += ["--temperatures", *[str(t) for t in args.temperatures]]
    if args.bandit:
        cmd += ["--bandit"]
    if args.pypi_live_check:
        cmd += ["--pypi-live-check"]
    print("  $", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"security_pillar.py exited with code {rc}")

    # ---------------------------------------------------------------
    # Step 3: run security_pillar_calibrated.py (5-rule overlay)
    # ---------------------------------------------------------------
    raw_scan = raw_out / "scan_results.jsonl"
    if not raw_scan.exists():
        sys.exit(f"raw scan_results not found at {raw_scan}")

    print(f"\n[3/3] Running security_pillar_calibrated.py (5-rule overlay)")
    cmd = [
        sys.executable, "security_pillar_calibrated.py",
        "--scan-results", str(raw_scan),
        "--out-dir", str(calib_out),
    ]
    print("  $", " ".join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"security_pillar_calibrated.py exited with code {rc}")

    if not args.keep_filtered:
        shutil.rmtree(filtered_dir, ignore_errors=True)

    print(f"\nDone. Outputs under: {out_dir}")
    print(f"  raw/        -> uncalibrated static analysis")
    print(f"  calibrated/ -> calibrated (5-rule) analysis  <-- feeds the paper")
    print(f"")
    print(f"Key files to read for paper updates:")
    print(f"  {calib_out / 'summary.md'}            (vuln rate, CWE counts, S2 homogeneity)")
    print(f"  {calib_out / 'summary.json'}          (full numeric results)")
    print(f"  {calib_out / 'calibration_audit.json'} (rule-fire counts -> calibration sentence)")


if __name__ == "__main__":
    main()
