#!/usr/bin/env python3
"""Dedupe results/raw_responses/PB-*.jsonl in place.

Two parallel collectors (one zombie + one explicit resume) both wrote to
PB-08..PB-20.jsonl, producing duplicate rows keyed by
``(model_id, prompt_id, temperature, sample_idx)``.

This script keeps the *first* occurrence of each key per file, rewrites the
per-prompt JSONL, and regenerates ``all_responses.jsonl``.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results" / "raw_responses"
KEY = ("model_id", "prompt_id", "temperature", "sample_index")


def dedupe_file(path: Path) -> tuple[int, int]:
    seen: set = set()
    rows_in: list[dict] = []
    rows_out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_in.append(row)
            k = tuple(row.get(f) for f in KEY)
            if k in seen:
                continue
            seen.add(k)
            rows_out.append(row)
    if len(rows_out) != len(rows_in):
        with path.open("w", encoding="utf-8") as fh:
            for r in rows_out:
                fh.write(json.dumps(r) + "\n")
    return len(rows_in), len(rows_out)


def main() -> None:
    pb_files = sorted(RAW.glob("PB-*.jsonl"))
    if not pb_files:
        print(f"[dedup] no PB-*.jsonl found under {RAW}")
        return
    grand_in, grand_out = 0, 0
    summary: list[tuple[str, int, int]] = []
    for f in pb_files:
        n_in, n_out = dedupe_file(f)
        summary.append((f.name, n_in, n_out))
        grand_in += n_in
        grand_out += n_out

    width = max(len(name) for name, _, _ in summary)
    print(f"{'file'.ljust(width)}  {'in':>5}  {'out':>5}  {'dropped':>7}")
    for name, n_in, n_out in summary:
        print(f"{name.ljust(width)}  {n_in:>5}  {n_out:>5}  {(n_in - n_out):>7}")
    print("-" * (width + 28))
    print(f"{'TOTAL'.ljust(width)}  {grand_in:>5}  {grand_out:>5}  "
          f"{(grand_in - grand_out):>7}")

    all_path = RAW / "all_responses.jsonl"
    with all_path.open("w", encoding="utf-8") as fh:
        for f in pb_files:
            for line in f.open("r", encoding="utf-8"):
                if line.strip():
                    fh.write(line if line.endswith("\n") else line + "\n")
    print(f"[dedup] rewrote {all_path}")


if __name__ == "__main__":
    main()
