#!/usr/bin/env python3
"""
Collect a tiny multi-model sample from local dataset prompts.

Usage:
  python dataset_sample_collect.py --num-prompts 2 --samples 1 --temp 1.0
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from config import MODELS, SamplingConfig
from multi_model_sampler import Response, sample_one
from prompt_suite import CodePrompt


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "local_datasets" / "pool_b_open_ended_candidates" / "python_tasks" / "data.jsonl"
RAW_DIR = ROOT / "results" / "raw_responses"
DEFAULT_OUT_DIR = ROOT / "results" / "sample_analysis" / "dataset_pool_b"


def parse_row_specs(specs: list[str]) -> list[tuple[str, str, int]]:
    """Parse tokens like DATASET:SPLIT:ROW_INDEX (slash / in dataset name is allowed)."""
    parsed: list[tuple[str, str, int]] = []
    for s in specs:
        parts = s.split(":")
        if len(parts) != 3:
            raise ValueError(f"Bad row spec {s!r} — expected dataset:split:row_index")
        ds, spl, idx_s = parts[0].strip(), parts[1].strip(), parts[2].strip()
        try:
            idx = int(idx_s)
        except ValueError as e:
            raise ValueError(f"Bad row_index in spec {s!r}") from e
        parsed.append((ds, spl, idx))
    return parsed


def load_prompts_by_specs(row_specs: list[tuple[str, str, int]], id_prefix: str = "DS") -> list[CodePrompt]:
    wanted = set(row_specs)
    found: dict[tuple[str, str, int], CodePrompt] = {}
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec.get("dataset", ""), rec.get("split", ""), rec.get("row_index"))
            if key not in wanted:
                continue
            prompt_text = (rec.get("prompt") or "").strip()
            if not prompt_text:
                continue
            ds, spl, idx = key
            found[key] = CodePrompt(
                id="",
                category="DATASET_SAMPLE",
                prompt=prompt_text,
                language="python",
                expected_diversity="high",
                notes=f"source_dataset={ds} split={spl} row_index={idx}",
            )

    prompts: list[CodePrompt] = []
    for i, key in enumerate(row_specs, start=1):
        if key not in found:
            ds, spl, idx = key
            raise ValueError(f"could not locate {ds}:{spl}:{idx} in {DATASET_PATH}")
        p = found[key]
        prompts.append(
            CodePrompt(
                id=f"{id_prefix}-{i:02d}",
                category=p.category,
                prompt=p.prompt,
                language=p.language,
                expected_diversity=p.expected_diversity,
                notes=p.notes,
            )
        )
    return prompts


def load_open_ended_prompts(limit: int, id_prefix: str = "DS") -> list[CodePrompt]:
    prompts = []
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("task_language") != "PYTHON":
                continue
            if rec.get("label") != "OPEN_ENDED":
                continue
            prompt_text = (rec.get("prompt") or "").strip()
            if not prompt_text:
                continue
            pid = f"{id_prefix}-{len(prompts) + 1:02d}"
            prompts.append(
                CodePrompt(
                    id=pid,
                    category="DATASET_SAMPLE",
                    prompt=prompt_text,
                    language="python",
                    expected_diversity="high",
                    notes=f"source_row_index={rec.get('row_index')}",
                )
            )
            if len(prompts) >= limit:
                break
    return prompts


def load_prompts_by_row_indices(row_indices: list[int], id_prefix: str = "DS") -> list[CodePrompt]:
    """Legacy mode: select by row_index only — DANGEROUS when row_index collides across sources.

    If multiple records share the same row_index, the *last* one encountered wins.
    Prefer --rows dataset:split:row_index instead.
    """
    wanted = set(row_indices)
    found: dict[int, CodePrompt] = {}
    with open(DATASET_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            row_index = rec.get("row_index")
            if row_index not in wanted:
                continue
            prompt_text = (rec.get("prompt") or "").strip()
            if not prompt_text:
                continue
            found[row_index] = CodePrompt(
                id="",
                category="DATASET_SAMPLE",
                prompt=prompt_text,
                language="python",
                expected_diversity="high",
                notes=(
                    f"source_row_index={row_index} dataset={rec.get('dataset')} "
                    f"split={rec.get('split')}"
                ),
            )

    prompts: list[CodePrompt] = []
    for i, row_index in enumerate(row_indices, start=1):
        if row_index not in found:
            raise ValueError(f"row_index={row_index} not found in dataset")
        p = found[row_index]
        prompts.append(
            CodePrompt(
                id=f"{id_prefix}-{i:02d}",
                category=p.category,
                prompt=p.prompt,
                language=p.language,
                expected_diversity=p.expected_diversity,
                notes=p.notes,
            )
        )
    return prompts


async def collect(
    prompts: list[CodePrompt],
    cfg: SamplingConfig,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    providers: set[str] | None = None,
):
    import httpx

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    all_responses: list[Response] = []
    sem = asyncio.Semaphore(10)

    active_models = [
        (p, mid, dn, fm) for p, mid, dn, fm in MODELS
        if not providers or p in providers
    ]
    total_calls = (
        len(prompts) * len(active_models)
        * len(cfg.temperatures) * cfg.samples_per_model_per_temp
    )

    print(f"Collecting {len(prompts)} dataset prompts across {len(active_models)} models")
    async with httpx.AsyncClient() as client:
        for prompt in prompts:
            print(f"[{prompt.id}] {prompt.prompt[:70]}...")
            prompt_responses: list[Response] = []

            for provider, model_id, display, family in active_models:
                for temp in cfg.temperatures:
                    tasks = []
                    for s in range(cfg.samples_per_model_per_temp):
                        async def _task(p=provider, m=model_id, d=display, f=family, t=temp, si=s):
                            async with sem:
                                return await sample_one(client, p, m, d, f, prompt, cfg, t, si)
                        tasks.append(_task())
                    results = await asyncio.gather(*tasks)
                    valid = [r for r in results if r is not None]
                    prompt_responses.extend(valid)
                    print(f"  {display} t={temp}: {len(valid)}/{len(tasks)} OK")

            outpath = RAW_DIR / f"{prompt.id}.jsonl"
            with open(outpath, "w", encoding="utf-8") as f:
                for r in prompt_responses:
                    f.write(json.dumps(asdict(r)) + "\n")
            print(f"  Saved {len(prompt_responses)} -> {outpath}")
            all_responses.extend(prompt_responses)

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_path": str(DATASET_PATH),
        "num_prompts": len(prompts),
        "prompt_ids": [p.id for p in prompts],
        "prompts": [{"id": p.id, "prompt": p.prompt, "notes": p.notes} for p in prompts],
        "samples_per_model": cfg.samples_per_model_per_temp,
        "temperatures": cfg.temperatures,
        "providers_filter": sorted(providers) if providers else None,
        "responses_collected": len(all_responses),
    }
    with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Manifest -> {out_dir / 'run_manifest.json'}")


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument(
        "--rows",
        nargs="*",
        help="Fully-qualified row keys: dataset:split:row_index "
             "(recommended — row_index alone is ambiguous in this merged JSONL)",
    )
    parser.add_argument("--row-indices", nargs="*", type=int, default=None)
    parser.add_argument("--id-prefix", type=str, default="DS")
    parser.add_argument("--samples", type=int, default=2,
                        help="Samples per model per temperature.")
    parser.add_argument("--temps", nargs="*", type=float, default=[0.0, 1.0],
                        help="Temperatures to sweep (default: 0.0 and 1.0).")
    parser.add_argument("--providers", nargs="*",
                        help="Subset of providers to call: openai anthropic google together")
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
                        help="Where to write the run manifest (raw responses always go to results/raw_responses).")
    args = parser.parse_args()

    if args.rows and args.row_indices:
        print("ERROR: pass either --rows or --row-indices, not both", file=sys.stderr)
        sys.exit(1)
    if args.rows:
        specs = parse_row_specs(args.rows)
        prompts = load_prompts_by_specs(specs, args.id_prefix)
    elif args.row_indices:
        print("WARNING: --row-indices is ambiguous in merged pool-b JSONL; prefer --rows", file=sys.stderr)
        prompts = load_prompts_by_row_indices(args.row_indices, args.id_prefix)
    else:
        prompts = load_open_ended_prompts(args.num_prompts, args.id_prefix)
    if not prompts:
        print("No prompts loaded.")
        sys.exit(1)

    providers = set(args.providers) if args.providers else None
    if providers:
        unk = providers - {"openai", "anthropic", "google", "together"}
        if unk:
            print(f"ERROR: Unknown provider(s): {sorted(unk)}", file=sys.stderr)
            sys.exit(1)

    cfg = SamplingConfig(temperatures=args.temps, samples_per_model_per_temp=args.samples)
    asyncio.run(collect(prompts, cfg, Path(args.out_dir), providers=providers))


if __name__ == "__main__":
    main()
