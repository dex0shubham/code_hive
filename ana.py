#!/usr/bin/env python3
"""
Code Hivemind — Analysis & Figure Generation
==============================================
Run after collecting data with `python run.py --collect`.

Usage:
  python analyze.py                      # full analysis
  python analyze.py --skip-embeddings    # skip if no sentence-transformers

Outputs (in results/figures/):
  - fig1_smoking_gun_pca.png        The headline figure (PCA clusters)
  - fig2_inter_model_heatmap.png    Pairwise similarity between models
  - fig3_category_breakdown.png     Hivemind score by prompt category
  - fig4_temperature_effect.png     Does temperature help diversity?
  - fig5_library_choices.png        Import convergence across models
  - fig6_naming_vs_structure.png    Naming sim vs structural sim scatter
  - table1_summary.csv              Main results table
  - table2_per_prompt.csv           Per-prompt breakdown
"""

import ast
import argparse
import hashlib
import itertools
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── Optional imports (graceful degradation) ──
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not found. Install with: pip install matplotlib")
    print("         Skipping figure generation, will still output CSV tables.\n")

try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False


# ── Config ──
RAW_DIR = os.path.join("results", "raw_responses")
FIG_DIR = os.path.join("results", "figures")
METRICS_DIR = os.path.join("results", "metrics")

# NeurIPS-quality figure settings
if HAS_MPL:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })


###############################################################################
# METRICS (copied from run.py to keep this self-contained)
###############################################################################

def extract_code_block(text):
    matches = re.findall(r"```(?:python|javascript|js|py)?\s*\n(.*?)```", text, re.DOTALL)
    return "\n".join(matches) if matches else text.strip()

def token_ngrams(text, n):
    tokens = re.findall(r"[a-zA-Z_]\w*|[^\s\w]", text)
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

def jaccard(a, b):
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0

def ngram_jaccard(a, b, n=3):
    return jaccard(set(token_ngrams(a, n).keys()), set(token_ngrams(b, n).keys()))

def ast_fingerprint(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return hashlib.md5(code.encode()).hexdigest()
    def ser(node, d=0):
        if d > 30: return "..."
        return f"({' '.join([type(node).__name__] + [ser(c, d+1) for c in ast.iter_child_nodes(node)])})"
    return ser(tree)

def ast_similarity(a, b):
    return jaccard(set(re.findall(r"\w+", ast_fingerprint(a))), set(re.findall(r"\w+", ast_fingerprint(b))))

def naming_similarity(a, b):
    def names(code):
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"[a-zA-Z_]\w{2,}", code))
        ns = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)): ns.add(node.name)
            elif isinstance(node, ast.ClassDef): ns.add(node.name)
            elif isinstance(node, ast.arg) and node.arg != "self": ns.add(node.arg)
            elif isinstance(node, ast.Name) and node.id not in dir(__builtins__): ns.add(node.id)
        return ns
    return jaccard(names(a), names(b))

def extract_imports(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names: imports.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    return imports

def detect_patterns(code):
    pats = set()
    if re.search(r"\bclass\s+\w+", code): pats.add("class-based")
    elif re.search(r"\bdef\s+\w+", code): pats.add("functional")
    for pat, name in [(r"@dataclass","dataclass"),(r"@\w+","decorators"),(r"\basync\s","async"),
                       (r"typing\.|Optional|Union","type-hints"),(r"with\s+\w+","context-mgr"),
                       (r"\[.*for\s+\w+\s+in","list-comp"),(r"try:.*except","error-handling"),
                       (r"logging\.|logger\.","logging"),(r"\bEnum\b","enum")]:
        if re.search(pat, code, re.DOTALL): pats.add(name)
    return pats


###############################################################################
# DATA LOADING
###############################################################################

def load_all():
    """Load all responses from JSONL files."""
    path = Path(RAW_DIR) / "all_responses.jsonl"
    if not path.exists():
        # Try loading from individual files
        responses = []
        for f in Path(RAW_DIR).glob("*.jsonl"):
            if f.name == "all_responses.jsonl": continue
            with open(f) as fh:
                for line in fh:
                    if line.strip():
                        responses.append(json.loads(line))
        return responses
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def describe_data(responses):
    """Print data shape summary."""
    prompts = set(r["prompt_id"] for r in responses)
    models = set(r["model_display"] for r in responses)
    families = set(r["model_family"] for r in responses)
    temps = set(r["temperature"] for r in responses)

    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print(f"  Total responses:  {len(responses)}")
    print(f"  Prompts:          {len(prompts)} ({sorted(prompts)[:5]}{'...' if len(prompts)>5 else ''})")
    print(f"  Models:           {len(models)} ({', '.join(sorted(models)[:5])}{'...' if len(models)>5 else ''})")
    print(f"  Families:         {len(families)} ({', '.join(sorted(families))})")
    print(f"  Temperatures:     {sorted(temps)}")

    # Samples per model per prompt
    counts = Counter((r["prompt_id"], r["model_display"], r["temperature"]) for r in responses)
    samples = [v for v in counts.values()]
    print(f"  Samples/cell:     min={min(samples)}, max={max(samples)}, median={sorted(samples)[len(samples)//2]}")
    print()
    return {"prompts": sorted(prompts), "models": sorted(models), "families": sorted(families), "temps": sorted(temps)}


###############################################################################
# CORE ANALYSES
###############################################################################

def compute_pairwise_metrics(responses, temp=1.0, max_pairs=200):
    """Compute all pairwise metrics for inter-model comparison."""
    filtered = [r for r in responses if abs(r["temperature"] - temp) < 0.01]
    by_model = defaultdict(list)
    for r in filtered:
        by_model[r["model_display"]].append(extract_code_block(r["response_text"]))

    # Pick one representative sample per model per prompt
    by_prompt_model = defaultdict(dict)
    for r in filtered:
        key = r["prompt_id"]
        model = r["model_display"]
        if model not in by_prompt_model[key]:
            by_prompt_model[key][model] = extract_code_block(r["response_text"])

    results = []
    for pid, model_codes in by_prompt_model.items():
        models = sorted(model_codes.keys())
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ca, cb = model_codes[models[i]], model_codes[models[j]]
                results.append({
                    "prompt_id": pid,
                    "model_a": models[i],
                    "model_b": models[j],
                    "ngram_sim": ngram_jaccard(ca, cb),
                    "ast_sim": ast_similarity(ca, cb),
                    "naming_sim": naming_similarity(ca, cb),
                    "same_imports": 1.0 if extract_imports(ca) == extract_imports(cb) else 0.0,
                    "same_patterns": jaccard(detect_patterns(ca), detect_patterns(cb)),
                })
    return results


def compute_intra_model_diversity(responses, temp=1.0):
    """For each model x prompt, measure diversity of that model's OWN outputs."""
    filtered = [r for r in responses if abs(r["temperature"] - temp) < 0.01]
    by_mp = defaultdict(list)
    for r in filtered:
        by_mp[(r["model_display"], r["prompt_id"])].append(extract_code_block(r["response_text"]))

    results = []
    for (model, pid), codes in by_mp.items():
        if len(codes) < 2: continue
        pairs = list(itertools.combinations(range(len(codes)), 2))
        if len(pairs) > 100:
            rng = np.random.RandomState(42)
            pairs = [pairs[i] for i in rng.choice(len(pairs), 100, replace=False)]

        ng = np.mean([ngram_jaccard(codes[i], codes[j]) for i, j in pairs])
        ast_s = np.mean([ast_similarity(codes[i], codes[j]) for i, j in pairs])
        nm = np.mean([naming_similarity(codes[i], codes[j]) for i, j in pairs])
        results.append({
            "model": model, "prompt_id": pid, "n_samples": len(codes),
            "ngram_sim": float(ng), "ast_sim": float(ast_s), "naming_sim": float(nm),
        })
    return results


def compute_embeddings(responses, temp=1.0):
    """Compute embeddings for all code samples (for PCA/heatmap)."""
    if not HAS_SBERT:
        return None, None, None
    print("  Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    filtered = [r for r in responses if abs(r["temperature"] - temp) < 0.01]
    # One sample per model per prompt
    by_pm = defaultdict(dict)
    for r in filtered:
        if r["model_display"] not in by_pm[r["prompt_id"]]:
            by_pm[r["prompt_id"]][r["model_display"]] = r

    samples = []
    for pid in sorted(by_pm.keys()):
        for model in sorted(by_pm[pid].keys()):
            r = by_pm[pid][model]
            samples.append({
                "prompt_id": pid,
                "model": model,
                "family": r["model_family"],
                "code": extract_code_block(r["response_text"]),
            })

    texts = [s["code"] for s in samples]
    print(f"  Embedding {len(texts)} samples...")
    embeddings = model.encode(texts, show_progress_bar=True)
    return samples, embeddings, model


###############################################################################
# FIGURE GENERATION
###############################################################################

def fig1_smoking_gun(samples, embeddings, save_path):
    """THE headline figure: PCA of code embeddings colored by model."""
    if not HAS_MPL or not HAS_SKLEARN or samples is None:
        print("  Skipping fig1 (needs matplotlib + sklearn + sentence-transformers)")
        return

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)

    models = sorted(set(s["model"] for s in samples))
    cmap = plt.cm.get_cmap("tab10", len(models))
    color_map = {m: cmap(i) for i, m in enumerate(models)}

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in models:
        idx = [i for i, s in enumerate(samples) if s["model"] == m]
        ax.scatter(coords[idx, 0], coords[idx, 1], label=m, alpha=0.6, s=20,
                   color=color_map[m], edgecolors="white", linewidths=0.3)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title("Code Embeddings Across Models\n(Each point = one model's response to one prompt)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False, markerscale=1.5)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig2_heatmap(pairwise, models, save_path):
    """Pairwise inter-model similarity heatmap."""
    if not HAS_MPL: return

    n = len(models)
    mat = np.zeros((n, n))
    counts = np.zeros((n, n))
    m2i = {m: i for i, m in enumerate(models)}

    for p in pairwise:
        i, j = m2i.get(p["model_a"]), m2i.get(p["model_b"])
        if i is None or j is None: continue
        val = (p["ngram_sim"] + p["ast_sim"] + p["naming_sim"]) / 3
        mat[i, j] += val; mat[j, i] += val
        counts[i, j] += 1; counts[j, i] += 1

    counts = np.clip(counts, 1, None)
    mat = mat / counts
    np.fill_diagonal(mat, 1.0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0.2, vmax=0.9)
    ax.set_xticks(range(n)); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(models)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if mat[i,j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, label="Mean Similarity (ngram + AST + naming)")
    ax.set_title("Inter-Model Code Similarity")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig3_category_breakdown(pairwise, save_path):
    """Bar chart: similarity by prompt category."""
    if not HAS_MPL: return

    # Infer category from prompt ID prefix
    cat_map = {"OD": "Open Design", "AL": "Algorithm", "RF": "Refactor",
               "NM": "Naming", "CT": "Creative Tool", "SD": "System Design"}

    by_cat = defaultdict(lambda: {"ngram": [], "ast": [], "naming": []})
    for p in pairwise:
        prefix = p["prompt_id"].split("-")[0]
        cat = cat_map.get(prefix, prefix)
        by_cat[cat]["ngram"].append(p["ngram_sim"])
        by_cat[cat]["ast"].append(p["ast_sim"])
        by_cat[cat]["naming"].append(p["naming_sim"])

    cats = sorted(by_cat.keys())
    x = np.arange(len(cats))
    w = 0.25

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - w, [np.mean(by_cat[c]["ngram"]) for c in cats], w, label="N-gram", color="#4361ee")
    ax.bar(x,     [np.mean(by_cat[c]["ast"]) for c in cats], w, label="AST Structure", color="#f72585")
    ax.bar(x + w, [np.mean(by_cat[c]["naming"]) for c in cats], w, label="Naming", color="#4cc9f0")

    ax.set_xticks(x); ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("Mean Pairwise Similarity"); ax.set_ylim(0, 1)
    ax.set_title("Code Hivemind by Task Category")
    ax.legend(frameon=False)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig4_temperature(responses, save_path):
    """Temperature effect on intra-model diversity."""
    if not HAS_MPL: return

    temps = sorted(set(r["temperature"] for r in responses))
    if len(temps) < 2:
        print("  Skipping fig4 (need 2+ temperatures)")
        return

    temp_data = {}
    for t in temps:
        intra = compute_intra_model_diversity(responses, t)
        if intra:
            temp_data[t] = {
                "ast": np.mean([r["ast_sim"] for r in intra]),
                "naming": np.mean([r["naming_sim"] for r in intra]),
                "ngram": np.mean([r["ngram_sim"] for r in intra]),
            }

    if len(temp_data) < 2: return

    fig, ax = plt.subplots(figsize=(5, 4))
    ts = sorted(temp_data.keys())
    ax.plot(ts, [temp_data[t]["ast"] for t in ts], "o-", label="AST Structure", color="#f72585")
    ax.plot(ts, [temp_data[t]["naming"] for t in ts], "s-", label="Naming", color="#4cc9f0")
    ax.plot(ts, [temp_data[t]["ngram"] for t in ts], "^-", label="N-gram", color="#4361ee")

    ax.set_xlabel("Temperature")
    ax.set_ylabel("Mean Intra-Model Similarity\n(lower = more diverse)")
    ax.set_title("Does Temperature Increase Code Diversity?")
    ax.legend(frameon=False)
    ax.set_ylim(0, 1)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig5_library_choices(responses, save_path, temp=1.0):
    """Stacked bar: which libraries each model picks."""
    if not HAS_MPL: return

    filtered = [r for r in responses if abs(r["temperature"] - temp) < 0.01]
    by_model = defaultdict(list)
    for r in filtered:
        by_model[r["model_display"]].append(extract_imports(extract_code_block(r["response_text"])))

    # Count top imports per model
    all_imports = Counter()
    for model, import_lists in by_model.items():
        for imp_set in import_lists:
            all_imports.update(imp_set)

    top_imports = [imp for imp, _ in all_imports.most_common(10)]
    models = sorted(by_model.keys())

    data = np.zeros((len(models), len(top_imports)))
    for i, model in enumerate(models):
        total = len(by_model[model])
        for j, imp in enumerate(top_imports):
            count = sum(1 for imp_set in by_model[model] if imp in imp_set)
            data[i, j] = count / total if total else 0

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data.T, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_yticks(range(len(top_imports))); ax.set_yticklabels(top_imports)
    for i in range(len(models)):
        for j in range(len(top_imports)):
            ax.text(i, j, f"{data[i,j]:.0%}", ha="center", va="center", fontsize=7,
                    color="white" if data[i,j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, label="Fraction of responses using library")
    ax.set_title("Library Usage Across Models")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def fig6_naming_vs_structure(pairwise, save_path):
    """Scatter: naming similarity vs structural similarity."""
    if not HAS_MPL: return

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter([p["naming_sim"] for p in pairwise],
               [p["ast_sim"] for p in pairwise],
               alpha=0.15, s=8, color="#7209b7")
    ax.set_xlabel("Naming Similarity")
    ax.set_ylabel("AST Structural Similarity")
    ax.set_title("Do Models Converge on Names, Structure, or Both?")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.2)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(save_path)
    plt.close(fig)
    print(f"  Saved: {save_path}")


###############################################################################
# TABLES
###############################################################################

def save_summary_table(pairwise, intra, models, save_path):
    """Table 1: Per-model summary stats."""
    rows = ["model,inter_ngram,inter_ast,inter_naming,inter_import_match,intra_ast,intra_naming"]

    for m in models:
        inter = [p for p in pairwise if p["model_a"] == m or p["model_b"] == m]
        intra_m = [r for r in intra if r["model"] == m]

        if inter:
            ing = np.mean([p["ngram_sim"] for p in inter])
            iast = np.mean([p["ast_sim"] for p in inter])
            inm = np.mean([p["naming_sim"] for p in inter])
            iimp = np.mean([p["same_imports"] for p in inter])
        else:
            ing = iast = inm = iimp = 0

        if intra_m:
            ia_ast = np.mean([r["ast_sim"] for r in intra_m])
            ia_nm = np.mean([r["naming_sim"] for r in intra_m])
        else:
            ia_ast = ia_nm = 0

        rows.append(f"{m},{ing:.3f},{iast:.3f},{inm:.3f},{iimp:.3f},{ia_ast:.3f},{ia_nm:.3f}")

    with open(save_path, "w") as f:
        f.write("\n".join(rows))
    print(f"  Saved: {save_path}")


def save_per_prompt_table(pairwise, save_path):
    """Table 2: Per-prompt breakdown."""
    by_prompt = defaultdict(list)
    for p in pairwise:
        by_prompt[p["prompt_id"]].append(p)

    rows = ["prompt_id,category,n_pairs,mean_ngram,mean_ast,mean_naming,import_match_rate"]
    cat_map = {"OD": "Open Design", "AL": "Algorithm", "RF": "Refactor",
               "NM": "Naming", "CT": "Creative Tool", "SD": "System Design"}

    for pid in sorted(by_prompt.keys()):
        pairs = by_prompt[pid]
        prefix = pid.split("-")[0]
        cat = cat_map.get(prefix, prefix)
        rows.append(f"{pid},{cat},{len(pairs)},"
                     f"{np.mean([p['ngram_sim'] for p in pairs]):.3f},"
                     f"{np.mean([p['ast_sim'] for p in pairs]):.3f},"
                     f"{np.mean([p['naming_sim'] for p in pairs]):.3f},"
                     f"{np.mean([p['same_imports'] for p in pairs]):.3f}")

    with open(save_path, "w") as f:
        f.write("\n".join(rows))
    print(f"  Saved: {save_path}")


###############################################################################
# MAIN
###############################################################################

def main():
    parser = argparse.ArgumentParser(description="Code Hivemind Analysis")
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Skip embedding-based analysis (no sentence-transformers needed)")
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)

    # 1. Load data
    print("Loading data...")
    responses = load_all()
    if not responses:
        print(f"ERROR: No data found in {RAW_DIR}/")
        print("Run `python run.py --collect` or `python run.py --demo` first.")
        sys.exit(1)

    meta = describe_data(responses)

    # 2. Compute pairwise inter-model metrics
    print("Computing inter-model pairwise metrics...")
    pairwise = compute_pairwise_metrics(responses, temp=max(meta["temps"]))
    print(f"  {len(pairwise)} pairs computed")

    # Headline number
    mean_ast = np.mean([p["ast_sim"] for p in pairwise])
    mean_naming = np.mean([p["naming_sim"] for p in pairwise])
    mean_ngram = np.mean([p["ngram_sim"] for p in pairwise])
    import_match = np.mean([p["same_imports"] for p in pairwise])

    print(f"\n  *** HEADLINE RESULTS ***")
    print(f"  Mean inter-model AST similarity:    {mean_ast:.3f}")
    print(f"  Mean inter-model naming similarity: {mean_naming:.3f}")
    print(f"  Mean inter-model n-gram similarity: {mean_ngram:.3f}")
    print(f"  Import set match rate:              {import_match:.1%}")
    print()

    # 3. Compute intra-model diversity
    print("Computing intra-model diversity...")
    intra = compute_intra_model_diversity(responses, temp=max(meta["temps"]))
    print(f"  {len(intra)} model-prompt cells computed")

    intra_ast = np.mean([r["ast_sim"] for r in intra])
    intra_naming = np.mean([r["naming_sim"] for r in intra])
    print(f"  Mean intra-model AST similarity:    {intra_ast:.3f}")
    print(f"  Mean intra-model naming similarity: {intra_naming:.3f}")
    print()

    # 4. Embeddings (optional)
    samples, embeddings, emb_model = None, None, None
    if not args.skip_embeddings and HAS_SBERT:
        print("Computing embeddings...")
        try:
            samples, embeddings, emb_model = compute_embeddings(responses, temp=max(meta["temps"]))
            if embeddings is not None:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                normed = embeddings / np.clip(norms, 1e-8, None)
                sim_mat = normed @ normed.T
                mask = np.triu(np.ones_like(sim_mat, dtype=bool), k=1)
                mean_cos = float(sim_mat[mask].mean())
                print(f"  Mean embedding cosine similarity:   {mean_cos:.3f}")
                print(f"  (Hivemind paper concern threshold:  > 0.80)")
        except Exception as e:
            print(f"  Embedding failed ({e}), continuing without.")
    else:
        print("Skipping embeddings (use --skip-embeddings to suppress this message)")
    print()

    # 5. Generate figures
    print("Generating figures...")
    fig1_smoking_gun(samples, embeddings, os.path.join(FIG_DIR, "fig1_smoking_gun_pca.png"))
    fig2_heatmap(pairwise, meta["models"], os.path.join(FIG_DIR, "fig2_inter_model_heatmap.png"))
    fig3_category_breakdown(pairwise, os.path.join(FIG_DIR, "fig3_category_breakdown.png"))
    fig4_temperature(responses, os.path.join(FIG_DIR, "fig4_temperature_effect.png"))
    fig5_library_choices(responses, os.path.join(FIG_DIR, "fig5_library_choices.png"))
    fig6_naming_vs_structure(pairwise, os.path.join(FIG_DIR, "fig6_naming_vs_structure.png"))
    print()

    # 6. Generate tables
    print("Generating tables...")
    save_summary_table(pairwise, intra, meta["models"], os.path.join(METRICS_DIR, "table1_summary.csv"))
    save_per_prompt_table(pairwise, os.path.join(METRICS_DIR, "table2_per_prompt.csv"))

    # 7. Print final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures:  {FIG_DIR}/")
    print(f"  Tables:   {METRICS_DIR}/")
    print()
    print("  Key findings for your paper:")
    print(f"  - {len(meta['models'])} models from {len(meta['families'])} families tested on {len(meta['prompts'])} tasks")
    print(f"  - Inter-model AST structural similarity: {mean_ast:.1%}")
    print(f"  - Inter-model naming convergence:        {mean_naming:.1%}")
    print(f"  - Library import match rate:              {import_match:.1%}")
    print(f"  - Intra-model self-similarity:            {intra_ast:.1%} (AST)")
    if embeddings is not None:
        print(f"  - Embedding cosine similarity:            {mean_cos:.1%}")
    print()

    # Suggest paper framing
    if mean_ast > 0.6:
        print("  STRONG HIVEMIND DETECTED: Models produce structurally near-identical code.")
    elif mean_ast > 0.4:
        print("  MODERATE HIVEMIND: Significant structural overlap but some differentiation.")
    else:
        print("  WEAK HIVEMIND: Models show meaningful structural diversity.")

    if import_match > 0.5:
        print("  LIBRARY MONOCULTURE: >50% of model pairs choose identical imports.")

    if intra_ast > mean_ast:
        print("  NOTE: Models are MORE similar to themselves than to each other")
        print("        (expected — but the inter-model similarity is the news)")


if __name__ == "__main__":
    main()