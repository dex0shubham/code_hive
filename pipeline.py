"""
Code Hivemind — Analysis Pipeline
===================================
Computes intra-model and inter-model diversity metrics.
"""

# -- Path bootstrap (so cross-folder imports work on Windows) --
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------

import json
import os
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from config import MODELS, OUTPUT_DIR, RAW_RESPONSES_DIR, METRICS_DIR, FIGURES_DIR
from diversity_metrics import (
    extract_code_block, compute_diversity, EmbeddingComputer, DiversityReport,
)
from prompt_suite import PROMPTS


def load_responses(prompt_id):
    path = Path(RAW_RESPONSES_DIR) / f"{prompt_id}.jsonl"
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def group_responses(responses, key="model_display"):
    groups = defaultdict(list)
    for r in responses:
        groups[r[key]].append(r)
    return dict(groups)


# -- ANALYSIS 1: Intra-Model Diversity --

def analyze_intra_model(prompt_id, temperature=1.0, embedder=None):
    responses = load_responses(prompt_id)
    responses = [r for r in responses if abs(r["temperature"] - temperature) < 0.01]
    by_model = group_responses(responses, "model_display")
    reports = []
    for model_name, model_responses in by_model.items():
        code_samples = [extract_code_block(r["response_text"]) for r in model_responses]
        report = compute_diversity(code_samples, prompt_id, f"intra:{model_name}", embedder)
        reports.append(report)
    return reports


# -- ANALYSIS 2: Inter-Model Homogeneity --

def analyze_inter_model(prompt_id, temperature=1.0, embedder=None, samples_per_model=1):
    responses = load_responses(prompt_id)
    responses = [r for r in responses if abs(r["temperature"] - temperature) < 0.01]
    by_model = group_responses(responses, "model_display")

    rng = np.random.RandomState(42)
    selected = []
    for model_name, model_responses in by_model.items():
        if len(model_responses) >= samples_per_model:
            chosen = rng.choice(len(model_responses), samples_per_model, replace=False)
            for idx in chosen:
                selected.append(model_responses[idx])

    code_samples = [extract_code_block(r["response_text"]) for r in selected]
    return compute_diversity(code_samples, prompt_id, "inter:all_models", embedder)


# -- ANALYSIS 3: Family Clustering --

def analyze_family_clustering(prompt_id, temperature=1.0, embedder=None):
    responses = load_responses(prompt_id)
    responses = [r for r in responses if abs(r["temperature"] - temperature) < 0.01]
    by_family = group_responses(responses, "model_family")

    within_sims = {}
    for family, family_responses in by_family.items():
        if len(family_responses) < 2:
            continue
        codes = [extract_code_block(r["response_text"]) for r in family_responses]
        report = compute_diversity(codes, prompt_id, f"family:{family}", embedder)
        within_sims[family] = report.mean_embedding_similarity

    rng = np.random.RandomState(42)
    by_model = group_responses(responses, "model_display")
    canonical = {}
    for model_name, model_responses in by_model.items():
        idx = rng.randint(0, len(model_responses))
        canonical[model_name] = extract_code_block(model_responses[idx]["response_text"])

    model_families = {m[2]: m[3] for m in MODELS}
    across_sims = []
    model_names = list(canonical.keys())
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            m_i, m_j = model_names[i], model_names[j]
            if model_families.get(m_i) != model_families.get(m_j):
                if embedder:
                    sim = embedder.mean_pairwise_similarity([canonical[m_i], canonical[m_j]])
                    across_sims.append(sim)

    return {
        "within_family": within_sims,
        "across_family_mean": float(np.mean(across_sims)) if across_sims else 0,
        "across_family_std": float(np.std(across_sims)) if across_sims else 0,
    }


# -- ANALYSIS 4: Temperature Effect --

def analyze_temperature_effect(prompt_id, embedder=None):
    results = {}
    for temp in [0.0, 1.0]:
        intra_reports = analyze_intra_model(prompt_id, temp, embedder)
        if not intra_reports:
            continue
        mean_hivemind = np.mean([r.hivemind_score for r in intra_reports])
        mean_embed_sim = np.mean([r.mean_embedding_similarity for r in intra_reports])
        results[f"t={temp}"] = {
            "mean_hivemind_score": float(mean_hivemind),
            "mean_embedding_sim": float(mean_embed_sim),
            "n_models": len(intra_reports),
        }
    return results


# -- FULL RUNNER --

def run_full_analysis():
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    embedder = EmbeddingComputer()
    all_results = {}

    for prompt in PROMPTS:
        pid = prompt.id
        print(f"\n{'='*60}")
        print(f"Analyzing {pid}: {prompt.category}")
        print(f"{'='*60}")

        responses = load_responses(pid)
        if not responses:
            print(f"  No data for {pid}, skipping")
            continue

        print(f"  Loaded {len(responses)} responses")

        print("  -> Intra-model diversity...")
        intra = analyze_intra_model(pid, 1.0, embedder)
        print(f"    {len(intra)} models analyzed")

        print("  -> Inter-model homogeneity...")
        inter = analyze_inter_model(pid, 1.0, embedder)
        print(f"    Hivemind score: {inter.hivemind_score:.3f}")

        print("  -> Family clustering...")
        family = analyze_family_clustering(pid, 1.0, embedder)
        print(f"    Within-family sims: {family['within_family']}")

        print("  -> Temperature effects...")
        temp_fx = analyze_temperature_effect(pid, embedder)
        for t, vals in temp_fx.items():
            print(f"    {t}: hivemind={vals['mean_hivemind_score']:.3f}")

        all_results[pid] = {
            "prompt_id": pid,
            "category": prompt.category,
            "intra_model": [asdict(r) for r in intra],
            "inter_model": asdict(inter),
            "family_clustering": family,
            "temperature_effect": temp_fx,
        }

    outpath = Path(METRICS_DIR) / "full_analysis.json"
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved -> {outpath}")
    print_summary_table(all_results)
    return all_results


def print_summary_table(results):
    print("\n" + "=" * 90)
    print("SUMMARY: Code Hivemind Effect by Category")
    print("=" * 90)
    header = f"{'Category':<16} {'Prompt':<8} {'Inter-Hivemind':>14} {'Intra-Mean':>12} {'Embed-Sim':>12} {'Naming-Sim':>12}"
    print(header)
    print("-" * 90)

    by_category = defaultdict(list)
    for pid, data in results.items():
        inter = data["inter_model"]
        intra_scores = [r["hivemind_score"] for r in data["intra_model"]]
        by_category[data["category"]].append({
            "pid": pid,
            "inter_hivemind": inter["hivemind_score"],
            "intra_mean": np.mean(intra_scores) if intra_scores else 0,
            "embed_sim": inter["mean_embedding_similarity"],
            "naming_sim": inter["mean_naming_similarity"],
        })

    for category in sorted(by_category.keys()):
        for item in by_category[category]:
            print(
                f"{category:<16} {item['pid']:<8} "
                f"{item['inter_hivemind']:>14.3f} "
                f"{item['intra_mean']:>12.3f} "
                f"{item['embed_sim']:>12.3f} "
                f"{item['naming_sim']:>12.3f}"
            )
        avg_inter = np.mean([x["inter_hivemind"] for x in by_category[category]])
        avg_intra = np.mean([x["intra_mean"] for x in by_category[category]])
        print(f"  {'AVG':<14} {'':>8} {avg_inter:>14.3f} {avg_intra:>12.3f}")
        print()

    all_inter = [data["inter_model"]["hivemind_score"] for data in results.values()]
    print(f"{'OVERALL':>24} mean inter-model hivemind: {np.mean(all_inter):.3f} +/- {np.std(all_inter):.3f}")


if __name__ == "__main__":
    run_full_analysis()