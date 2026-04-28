#!/usr/bin/env python3
"""
Generate analysis tables and figures for Code Hivemind.

Typical usage:
  python make_report.py              # run/load metrics, then write all outputs
  python make_report.py --raw-only   # only write response coverage outputs
  python make_report.py --force      # recompute full_analysis.json first
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import FIGURES_DIR as CONFIG_FIGURES_DIR
from config import METRICS_DIR, RAW_RESPONSES_DIR
from diversity_metrics import compute_diversity, extract_code_block
from pipeline import run_full_analysis
from prompt_suite import PROMPTS


RESULTS_DIR = Path("results")
RAW_DIR = Path(RAW_RESPONSES_DIR)
METRICS_PATH = Path(METRICS_DIR) / "full_analysis.json"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = Path(CONFIG_FIGURES_DIR)

METRIC_COLUMNS = [
    "inter_hivemind",
    "inter_embedding_similarity",
    "inter_ast_similarity",
    "inter_naming_similarity",
    "inter_ngram_similarity",
    "intra_hivemind_mean",
]


def ensure_output_dirs() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_raw_responses() -> pd.DataFrame:
    all_path = RAW_DIR / "all_responses.jsonl"
    if all_path.exists():
        rows = read_jsonl(all_path)
    else:
        rows = []
        for path in sorted(RAW_DIR.glob("*.jsonl")):
            if path.name != "all_responses.jsonl":
                rows.extend(read_jsonl(path))

    if not rows:
        raise FileNotFoundError(f"No response JSONL files found in {RAW_DIR}")

    return pd.DataFrame(rows)


class NoEmbeddingComputer:
    """Fallback embedder for offline report generation."""

    def mean_pairwise_similarity(self, texts: list[str]) -> float:
        return 0.0


def group_records(records: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups = defaultdict(list)
    for record in records:
        groups[record[key]].append(record)
    return dict(groups)


def compute_reports_for_temperature(
    prompt_id: str,
    records: list[dict[str, Any]],
    temperature: float,
    embedder: Any,
) -> list[Any]:
    temp_records = [r for r in records if abs(r["temperature"] - temperature) < 0.01]
    reports = []
    for model_name, model_records in group_records(temp_records, "model_display").items():
        code_samples = [extract_code_block(r["response_text"]) for r in model_records]
        reports.append(compute_diversity(code_samples, prompt_id, f"intra:{model_name}", embedder))
    return reports


def run_local_analysis(raw_df: pd.DataFrame, use_embeddings: bool) -> dict[str, Any]:
    if use_embeddings:
        from diversity_metrics import EmbeddingComputer

        embedder = EmbeddingComputer()
    else:
        embedder = NoEmbeddingComputer()

    rng = np.random.RandomState(42)
    results = {}
    prompt_meta = {p.id: p for p in PROMPTS}

    for prompt_id, prompt in prompt_meta.items():
        records = raw_df[raw_df["prompt_id"] == prompt_id].to_dict("records")
        if not records:
            continue

        print(f"Analyzing {prompt_id}: {prompt.category} ({len(records)} responses)")

        intra = compute_reports_for_temperature(prompt_id, records, 1.0, embedder)

        temp_one_records = [r for r in records if abs(r["temperature"] - 1.0) < 0.01]
        selected = []
        for _model_name, model_records in group_records(temp_one_records, "model_display").items():
            if model_records:
                selected.append(model_records[rng.randint(0, len(model_records))])
        inter_codes = [extract_code_block(r["response_text"]) for r in selected]
        inter = compute_diversity(inter_codes, prompt_id, "inter:all_models", embedder)

        family = {}
        for family_name, family_records in group_records(temp_one_records, "model_family").items():
            if len(family_records) >= 2:
                family_codes = [extract_code_block(r["response_text"]) for r in family_records]
                family_report = compute_diversity(family_codes, prompt_id, f"family:{family_name}", embedder)
                family[family_name] = family_report.mean_embedding_similarity

        temp_effect = {}
        for temp in [0.0, 0.6, 1.0]:
            reports = compute_reports_for_temperature(prompt_id, records, temp, embedder)
            if not reports:
                continue
            temp_effect[f"t={temp}"] = {
                "mean_hivemind_score": float(np.mean([r.hivemind_score for r in reports])),
                "mean_embedding_sim": float(np.mean([r.mean_embedding_similarity for r in reports])),
                "n_models": len(reports),
            }

        results[prompt_id] = {
            "prompt_id": prompt_id,
            "category": prompt.category,
            "intra_model": [asdict(r) for r in intra],
            "inter_model": asdict(inter),
            "family_clustering": {
                "within_family": family,
                "across_family_mean": 0.0,
                "across_family_std": 0.0,
            },
            "temperature_effect": temp_effect,
        }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    suffix = "" if use_embeddings else "_no_embeddings"
    outpath = METRICS_PATH if use_embeddings else METRICS_PATH.with_name(f"full_analysis{suffix}.json")
    with outpath.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Analysis saved -> {outpath}")

    return results


def load_or_run_analysis(force: bool, raw_df: pd.DataFrame, no_embeddings: bool) -> dict[str, Any]:
    if no_embeddings:
        return run_local_analysis(raw_df, use_embeddings=False)

    if force or not METRICS_PATH.exists():
        try:
            return run_full_analysis()
        except Exception as exc:
            print(f"Embedding analysis failed: {type(exc).__name__}: {exc}")
            print("Falling back to non-embedding metrics. Re-run without network issues for semantic scores.")
            return run_local_analysis(raw_df, use_embeddings=False)

    with METRICS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def flatten_prompt_summary(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for prompt_id, data in results.items():
        inter = data["inter_model"]
        intra_scores = [r["hivemind_score"] for r in data["intra_model"]]

        rows.append(
            {
                "prompt_id": prompt_id,
                "category": data["category"],
                "inter_hivemind": inter["hivemind_score"],
                "inter_embedding_similarity": inter["mean_embedding_similarity"],
                "inter_ast_similarity": inter["mean_ast_similarity"],
                "inter_naming_similarity": inter["mean_naming_similarity"],
                "inter_ngram_similarity": inter["mean_ngram_jaccard_3"],
                "intra_hivemind_mean": float(np.mean(intra_scores)) if intra_scores else np.nan,
                "intra_hivemind_std": float(np.std(intra_scores)) if intra_scores else np.nan,
                "n_inter_samples": inter["n_samples"],
            }
        )

    return pd.DataFrame(rows).sort_values(["category", "prompt_id"])


def flatten_intra_model(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for prompt_id, data in results.items():
        for report in data["intra_model"]:
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "category": data["category"],
                    "model": report["group_label"].replace("intra:", ""),
                    "n_samples": report["n_samples"],
                    "hivemind_score": report["hivemind_score"],
                    "embedding_similarity": report["mean_embedding_similarity"],
                    "ast_similarity": report["mean_ast_similarity"],
                    "naming_similarity": report["mean_naming_similarity"],
                    "ngram_similarity": report["mean_ngram_jaccard_3"],
                    "pattern_entropy": report["pattern_diversity"].get("entropy"),
                    "unique_pattern_sets": report["pattern_diversity"].get("unique_pattern_sets"),
                    "library_entropy": report["library_diversity"].get("entropy"),
                    "unique_import_sets": report["library_diversity"].get("unique_import_sets"),
                }
            )

    return pd.DataFrame(rows).sort_values(["category", "prompt_id", "model"])


def flatten_temperature(results: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for prompt_id, data in results.items():
        for temp_label, values in data["temperature_effect"].items():
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "category": data["category"],
                    "temperature": float(temp_label.replace("t=", "")),
                    "mean_hivemind_score": values["mean_hivemind_score"],
                    "mean_embedding_similarity": values["mean_embedding_sim"],
                    "n_models": values["n_models"],
                }
            )

    return pd.DataFrame(rows).sort_values(["category", "prompt_id", "temperature"])


def write_raw_tables(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage = (
        raw_df.groupby(["prompt_id", "model_display", "temperature"])
        .size()
        .reset_index(name="responses")
        .sort_values(["prompt_id", "model_display", "temperature"])
    )

    by_model = (
        raw_df.groupby(["model_provider", "model_family", "model_display"])
        .agg(
            responses=("response_text", "size"),
            prompts=("prompt_id", "nunique"),
            mean_latency_ms=("latency_ms", "mean"),
            mean_input_tokens=("input_tokens", "mean"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
        .sort_values("responses", ascending=False)
    )

    by_prompt = (
        raw_df.groupby("prompt_id")
        .agg(
            responses=("response_text", "size"),
            models=("model_display", "nunique"),
            temperatures=("temperature", "nunique"),
            mean_output_tokens=("output_tokens", "mean"),
        )
        .reset_index()
        .sort_values("prompt_id")
    )

    coverage.to_csv(TABLES_DIR / "response_coverage_by_prompt_model_temp.csv", index=False)
    by_model.to_csv(TABLES_DIR / "response_summary_by_model.csv", index=False)
    by_prompt.to_csv(TABLES_DIR / "response_summary_by_prompt.csv", index=False)

    return coverage, by_model, by_prompt


def write_metric_tables(
    prompt_df: pd.DataFrame, intra_df: pd.DataFrame, temp_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prompt_df.to_csv(TABLES_DIR / "prompt_summary.csv", index=False)
    intra_df.to_csv(TABLES_DIR / "intra_model_summary.csv", index=False)
    temp_df.to_csv(TABLES_DIR / "temperature_effects.csv", index=False)

    category_summary = (
        prompt_df.groupby("category", as_index=False)
        .agg(
            prompts=("prompt_id", "count"),
            inter_hivemind_mean=("inter_hivemind", "mean"),
            inter_hivemind_std=("inter_hivemind", "std"),
            intra_hivemind_mean=("intra_hivemind_mean", "mean"),
            embedding_similarity_mean=("inter_embedding_similarity", "mean"),
            ast_similarity_mean=("inter_ast_similarity", "mean"),
            naming_similarity_mean=("inter_naming_similarity", "mean"),
        )
        .sort_values("inter_hivemind_mean", ascending=False)
    )

    model_summary = (
        intra_df.groupby("model", as_index=False)
        .agg(
            prompts=("prompt_id", "nunique"),
            hivemind_mean=("hivemind_score", "mean"),
            hivemind_std=("hivemind_score", "std"),
            embedding_similarity_mean=("embedding_similarity", "mean"),
            ast_similarity_mean=("ast_similarity", "mean"),
            naming_similarity_mean=("naming_similarity", "mean"),
            pattern_entropy_mean=("pattern_entropy", "mean"),
        )
        .sort_values("hivemind_mean", ascending=False)
    )

    category_summary.to_csv(TABLES_DIR / "category_summary.csv", index=False)
    model_summary.to_csv(TABLES_DIR / "model_summary.csv", index=False)

    return category_summary, model_summary


def plot_response_coverage(coverage: pd.DataFrame, by_model: pd.DataFrame) -> None:
    pivot = coverage.pivot_table(
        index="prompt_id", columns="model_display", values="responses", aggfunc="sum", fill_value=0
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="Blues", cbar_kws={"label": "Responses"})
    plt.title("Response Coverage by Prompt and Model")
    plt.xlabel("Model")
    plt.ylabel("Prompt")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "response_coverage_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.barplot(data=by_model, x="model_display", y="responses", hue="model_provider", dodge=False)
    plt.title("Responses Collected by Model")
    plt.xlabel("Model")
    plt.ylabel("Responses")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "responses_by_model.png", dpi=200)
    plt.close()


def plot_category_hivemind(category_summary: pd.DataFrame) -> None:
    long = category_summary.melt(
        id_vars="category",
        value_vars=["inter_hivemind_mean", "intra_hivemind_mean"],
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(10, 5))
    sns.barplot(data=long, x="category", y="score", hue="metric")
    plt.title("Hivemind Score by Prompt Category")
    plt.xlabel("Category")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "category_hivemind_scores.png", dpi=200)
    plt.close()


def plot_prompt_hivemind(prompt_df: pd.DataFrame) -> None:
    matrix = prompt_df.pivot_table(
        index="category", columns="prompt_id", values="inter_hivemind", aggfunc="mean"
    )

    plt.figure(figsize=(16, 5))
    sns.heatmap(matrix, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
    plt.title("Inter-Model Hivemind by Prompt")
    plt.xlabel("Prompt")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "prompt_hivemind_heatmap.png", dpi=200)
    plt.close()


def plot_model_intra(model_summary: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 5))
    sns.barplot(data=model_summary, x="model", y="hivemind_mean")
    plt.title("Mean Intra-Model Hivemind Score")
    plt.xlabel("Model")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_intra_hivemind.png", dpi=200)
    plt.close()


def plot_temperature_effect(temp_df: pd.DataFrame) -> None:
    plt.figure(figsize=(9, 5))
    sns.lineplot(
        data=temp_df,
        x="temperature",
        y="mean_hivemind_score",
        hue="category",
        marker="o",
        errorbar="sd",
    )
    plt.title("Temperature Effect on Intra-Model Hivemind")
    plt.xlabel("Temperature")
    plt.ylabel("Mean score")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "temperature_effect.png", dpi=200)
    plt.close()


def plot_metric_correlation(prompt_df: pd.DataFrame) -> None:
    corr = prompt_df[METRIC_COLUMNS].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Metric Correlation Across Prompts")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metric_correlation.png", dpi=200)
    plt.close()


def plot_metric_distributions(prompt_df: pd.DataFrame) -> None:
    long = prompt_df.melt(
        id_vars=["prompt_id", "category"],
        value_vars=METRIC_COLUMNS,
        var_name="metric",
        value_name="score",
    )

    plt.figure(figsize=(12, 5))
    sns.boxplot(data=long, x="metric", y="score")
    plt.title("Distribution of Similarity and Hivemind Metrics")
    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "metric_distributions.png", dpi=200)
    plt.close()


def write_readme(raw_df: pd.DataFrame, prompt_df: pd.DataFrame | None) -> None:
    lines = [
        "# Code Hivemind Report Outputs",
        "",
        f"- Raw responses: {len(raw_df):,}",
        f"- Models with responses: {raw_df['model_display'].nunique()}",
        f"- Prompts with responses: {raw_df['prompt_id'].nunique()}",
        "",
        "## Tables",
        "",
        "- `response_coverage_by_prompt_model_temp.csv`",
        "- `response_summary_by_model.csv`",
        "- `response_summary_by_prompt.csv`",
    ]

    if prompt_df is not None:
        lines.extend(
            [
                "- `prompt_summary.csv`",
                "- `category_summary.csv`",
                "- `model_summary.csv`",
                "- `intra_model_summary.csv`",
                "- `temperature_effects.csv`",
                "",
                "## Headline",
                "",
                f"- Mean inter-model hivemind: {prompt_df['inter_hivemind'].mean():.3f}",
                f"- Mean intra-model hivemind: {prompt_df['intra_hivemind_mean'].mean():.3f}",
            ]
        )

    (RESULTS_DIR / "REPORT_README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Code Hivemind report outputs")
    parser.add_argument("--force", action="store_true", help="Recompute full analysis before reporting")
    parser.add_argument("--raw-only", action="store_true", help="Only generate raw response coverage outputs")
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip sentence-transformers and generate all non-semantic metrics",
    )
    args = parser.parse_args()

    ensure_output_dirs()
    sns.set_theme(style="whitegrid", context="paper")

    raw_df = load_raw_responses()
    coverage, by_model, _by_prompt = write_raw_tables(raw_df)
    plot_response_coverage(coverage, by_model)

    prompt_df = None
    if not args.raw_only:
        results = load_or_run_analysis(args.force, raw_df, args.no_embeddings)
        prompt_df = flatten_prompt_summary(results)
        intra_df = flatten_intra_model(results)
        temp_df = flatten_temperature(results)

        category_summary, model_summary = write_metric_tables(prompt_df, intra_df, temp_df)
        plot_category_hivemind(category_summary)
        plot_prompt_hivemind(prompt_df)
        plot_model_intra(model_summary)
        plot_temperature_effect(temp_df)
        plot_metric_correlation(prompt_df)
        plot_metric_distributions(prompt_df)

    write_readme(raw_df, prompt_df)

    print(f"Wrote tables to {TABLES_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")
    print(f"Wrote summary to {RESULTS_DIR / 'REPORT_README.md'}")


if __name__ == "__main__":
    main()
