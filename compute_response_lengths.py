"""
Compute response (code) lengths per model from scan_results.jsonl.
Groups models into zero-output vs producing and prints a summary table.
"""

import json
import statistics
import os

INPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "results", "proof_security", "sec_calibrated_v2", "scan_results.jsonl",
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(__file__),
    "results", "response_length_analysis.json",
)

ZERO_OUTPUT_MODELS = {
    "Claude 3.5 Haiku",
    "Gemini 2.5 Flash",
    "Gemini 2.5 Pro",
}


def main():
    # ---------- load data ----------
    records = []
    with open(INPUT_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    print(f"Loaded {len(records)} records total.\n")

    # ---------- group by model ----------
    from collections import defaultdict
    model_codes = defaultdict(list)
    for rec in records:
        code = rec.get("_code", "")
        model_codes[rec["model_display"]].append(code)

    # ---------- compute stats per model ----------
    model_stats = {}
    for model, codes in sorted(model_codes.items()):
        char_lens = [len(c) for c in codes]
        token_lens = [len(c.split()) for c in codes]
        n = len(codes)
        stats = {
            "n_samples": n,
            "mean_char_len": round(statistics.mean(char_lens), 1),
            "median_char_len": round(statistics.median(char_lens), 1),
            "mean_token_len": round(statistics.mean(token_lens), 1),
            "median_token_len": round(statistics.median(token_lens), 1),
            "min_char_len": min(char_lens),
            "max_char_len": max(char_lens),
        }
        group = "zero-output" if model in ZERO_OUTPUT_MODELS else "producing"
        stats["group"] = group
        model_stats[model] = stats

    # ---------- zero-output refusal analysis ----------
    zero_total = 0
    zero_short = 0  # code < 50 chars
    for model in ZERO_OUTPUT_MODELS:
        for code in model_codes.get(model, []):
            zero_total += 1
            if len(code) < 50:
                zero_short += 1

    zero_short_pct = (zero_short / zero_total * 100) if zero_total else 0.0

    # ---------- print summary table ----------
    hdr = (
        f"{'Model':<28} {'Group':<13} {'N':>5} "
        f"{'Mean(ch)':>9} {'Med(ch)':>9} {'Mean(tok)':>10} {'Med(tok)':>9} "
        f"{'Min(ch)':>8} {'Max(ch)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for model, s in sorted(model_stats.items()):
        print(
            f"{model:<28} {s['group']:<13} {s['n_samples']:>5} "
            f"{s['mean_char_len']:>9.1f} {s['median_char_len']:>9.1f} "
            f"{s['mean_token_len']:>10.1f} {s['median_token_len']:>9.1f} "
            f"{s['min_char_len']:>8} {s['max_char_len']:>8}"
        )

    print()
    print("=== Zero-output model refusal / stub analysis ===")
    print(f"  Total samples from zero-output models : {zero_total}")
    print(f"  Samples with code < 50 chars          : {zero_short}")
    print(f"  Percentage                             : {zero_short_pct:.1f}%")

    # ---------- write JSON ----------
    output = {
        "per_model": model_stats,
        "zero_output_refusal_analysis": {
            "zero_output_models": sorted(ZERO_OUTPUT_MODELS),
            "total_samples": zero_total,
            "samples_under_50_chars": zero_short,
            "pct_under_50_chars": round(zero_short_pct, 2),
        },
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(f"\nJSON written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
