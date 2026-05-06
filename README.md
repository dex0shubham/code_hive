# Code Hivemind

**Frontier LLMs Generate Code With Half the Effective Diversity of Independent Human Authors**

A three-kernel Vendi-Score evaluation framework that disentangles lexical, structural, and semantic diversity in LLM-generated code, paired with a three-pillar security analysis (vulnerability rate, cross-model CWE-pattern homogeneity, slopsquatting).

Extends [Artificial Hivemind](https://arxiv.org/abs/2510.22954) (Jiang et al., NeurIPS 2025 Best Paper) to code generation.

## Research Question

> Do frontier LLMs converge on the same code? Across 8,799 samples from 11 LLMs spanning 6 families on 20 open-ended Python tasks, the cross-model ensemble shows 2.3--3.5x less lexical/semantic diversity than 50 independent human authors per task -- while being structurally indistinguishable at the AST level.

## Prerequisites

- Python 3.10+
- (Optional) CUDA-capable GPU for neural embedding kernels

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd code_hive

# Install core dependencies
pip install -r requirements.txt

# Optional: neural embedding kernels (CodeT5+, UniXcoder, CodeBERT)
pip install torch transformers

# Optional: APTED tree-edit distance (AST kernel, falls back to node-bag Jaccard without it)
pip install apted

# Optional: Meta CodeShield / Insecure Code Detector (security pillar primary backend)
pip install codeshield

# Optional: mixed-effects statistical model (proof_homogeneity_v2.py)
pip install statsmodels
```

## Environment Setup

API keys are required for data collection. Create a `.env` file in the project root or export them directly:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
```

The `.env` file is loaded automatically via `python-dotenv` and is excluded from version control by `.gitignore`.

## Quick Start (Demo)

No API keys needed — generates synthetic responses from 5 models and runs the legacy diversity metrics on them:

```bash
python run.py --demo
```

This writes mock JSONL to `results/raw_responses/OD-01.jsonl` and prints pairwise naming similarity, library-choice entropy, design-pattern detection, and embedding cosine similarity.

---

## Running the Full Pipeline

The pipeline has **6 scripts** that run in sequence. Each step produces outputs consumed by later steps. Below is the exact execution order.

### Step 1 — Collect LLM Responses (`run.py --collect`)

Calls 11 models across 4 API providers (OpenAI, Anthropic, Google, Together) and writes per-prompt JSONL files to `results/raw_responses/`.

```bash
python run.py --collect --samples 20 --temps 0.0 0.7
```

| Flag | Default | Description |
|------|---------|-------------|
| `--samples N` | 50 | Samples per model per temperature |
| `--temps T [T ...]` | 0.0 1.0 | Temperature values to sweep |
| `--prompts ID [ID ...]` | all 60 | Subset of prompt IDs (e.g. `AL-01 OD-03 SE-12`) |
| `--models NAME [NAME ...]` | all 11 | Subset of model display names |

**Requires**: API keys in `.env` (see [Environment Setup](#environment-setup)).
**Produces**: `results/raw_responses/<PROMPT_ID>.jsonl` — one line per (model, temperature, sample).

### Step 2 — Fetch Human Baseline (`fetch_human_baseline.py`)

Downloads human-written Python solutions from the APPS dataset (Hendrycks et al., 2021) and writes them in the same JSONL schema so the diversity proof can compare LLM vs. human pools.

```bash
python fetch_human_baseline.py \
    --prompts AL-01 AL-03 AL-05 \
    --out-dir results/human_baseline \
    --max-solutions-per-prompt 30
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts ID [ID ...]` | AL-01 AL-03 AL-05 | Prompt IDs to fetch baselines for |
| `--out-dir PATH` | `results/human_baseline` | Output directory |
| `--max-solutions-per-prompt N` | 30 | Cap human solutions per prompt |
| `--max-problems-per-prompt N` | 5 | Max APPS problems matched per prompt |

**Requires**: `pip install datasets` (already in `requirements.txt`).
**Produces**: `results/human_baseline/<PROMPT_ID>.jsonl`.

### Step 3 — Diversity Proof v1 (`proof_homogeneity.py`)

The paper's core three-pillar analysis on a single temperature. Computes Vendi Scores under three kernels across four pools (shuffled-null, human, inter-LLM, intra-LLM), runs a paired ordering test (LLM vs. human), and evaluates functional convergence on algorithmic prompts.

```bash
python proof_homogeneity.py \
    --prompts AL-01 AL-03 AL-05 \
    --temperature 0.7 \
    --human-dir results/human_baseline \
    --embedder codet5p \
    --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts ID [ID ...]` | AL-01 AL-03 AL-05 | Prompt IDs to analyze |
| `--temperature T` | 1.0 | Single temperature to analyze |
| `--raw-dir PATH` | `results/raw_responses` | Directory with LLM JSONL files |
| `--human-dir PATH` | *(none)* | Directory with human JSONL files |
| `--out-dir PATH` | auto-generated | Output directory |
| `--embedder NAME` | unixcoder | Embedding model: `codet5p`, `unixcoder`, `codebert`, or `none` |
| `--device` | cpu | `cpu` or `cuda` |
| `--n-boot N` | 1000 | Bootstrap resamples for CIs |
| `--skip-pillar3` | false | Skip functional-convergence tests |
| `--dry-run` | false | Print configuration and exit |

**Requires**: `pip install torch transformers` (for neural kernels), `pip install apted` (for AST kernel; falls back to node-bag Jaccard without it).
**Produces** (in `--out-dir`):

| File | Contents |
|------|----------|
| `summary.json` | Full numeric results |
| `summary.md` | Human-readable report |
| `fig_pillar1_vendi.png` | Grouped bars: pools x kernels x prompts |
| `fig_pillar2_ordering.png` | Paired scatter: LLM vs. human Vendi |
| `fig_pillar3_traces.png` | Stacked trace-cluster bars per prompt |

### Step 4 — Diversity Proof v2 (`proof_homogeneity_v2.py`)

Wraps v1 and adds temperature sweep, leave-one-family-out ablation, and a mixed-effects regression. This is the script that produces the paper's main diversity figures.

```bash
python proof_homogeneity_v2.py \
    --prompts PB-01 PB-02 PB-03 \
    --temperatures 0.0 0.7 \
    --human-dir results/human_baseline \
    --embedder codebert \
    --device auto
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts ID [ID ...]` | PB-01 PB-02 PB-03 | Prompt IDs |
| `--temperatures T [T ...]` | 1.0 | Temperatures to sweep |
| `--embedder NAME` | codebert | `codet5p`, `unixcoder`, `codebert`, or `none` |
| `--device` | auto | `auto`, `cpu`, or `cuda` |
| `--skip-pillar3` | false | Skip functional convergence |
| `--skip-family-ablation` | false | Skip leave-one-family-out |
| `--skip-mixed-effects` | false | Skip regression model |
| `--dry-run` | false | Print config and exit |

**Requires**: all v1 deps + `pip install statsmodels pandas` (for mixed-effects model).
**Produces** (in `results/proof_v2/`):

| File | Contents |
|------|----------|
| `summary.json` | Master results across all temperatures |
| `summary.md` | Human-readable v2 report |
| `fig_temperature_sweep.png` | Vendi (LLM/human) vs. temperature |
| `fig_family_ablation.png` | Inter-LLM Vendi with each family removed |
| `fig_mixed_effects.png` | Fixed-effect contrasts vs. human pool |
| `t<TEMP>/` | Per-temperature v1 outputs and figures |

### Step 5 — Security Analysis (`security_pillar.py`)

Three security pillars: vulnerability rate (S1), cross-model CWE-pattern homogeneity (S2), and slopsquatting (S3).

```bash
python security_pillar.py \
    --prompts SE-01 SE-02 SE-03 SE-04 SE-05 \
    --temperatures 0.0 0.7 \
    --bandit \
    --pypi-live-check \
    --human-dir results/human_baseline \
    --out-dir results/proof_security
```

| Flag | Default | Description |
|------|---------|-------------|
| `--prompts ID [ID ...]` | all available | Prompt IDs to scan |
| `--temperatures T [T ...]` | auto-detect | Temperatures to include |
| `--bandit` | false | Also run Bandit SAST and report consensus |
| `--pypi-live-check` | false | Live PyPI/npm existence check for S3 |
| `--human-dir PATH` | *(none)* | Human baseline for comparison |
| `--out-dir PATH` | auto-generated | Output directory |
| `--from-cache PATH` | *(none)* | Reuse cached scan results |
| `--dry-run` | false | Print config and exit |

**Requires**: `pip install codeshield` (primary backend; falls back to regex scanner), `pip install bandit` (if `--bandit` flag used).
**Produces** (in `--out-dir`):

| File | Contents |
|------|----------|
| `summary.json` | Full security results |
| `summary.md` | Human-readable report |
| `fig_pillar_s1_vuln_rates.png` | Per-model + per-CWE + severity-weighted |
| `fig_pillar_s2_homogeneity.png` | Vendi/N over CWE-pattern signatures |
| `fig_pillar_s3_slopsquatting.png` | Deterministic-vs-stochastic rates |
| `cwe_signatures.csv` | Per-(prompt, model, sample) CWE signatures |
| `slop_findings.jsonl` | Per-sample hallucinated package names |

### Step 6 — Legacy Analysis (`run.py --analyze`)

Runs the older `pipeline.py` analysis (intra/inter-model diversity, family clustering, temperature effects) using the 4-layer `diversity_metrics.py` framework. Not used for paper figures but retained for exploratory analysis.

```bash
python run.py --analyze
```

**Produces**: outputs in `results/metrics/` and `results/figures/`.

---

## Execution Order at a Glance

```
┌──────────────────────────────────────────────────────────────┐
│  1. run.py --collect          → results/raw_responses/       │
│  2. fetch_human_baseline.py   → results/human_baseline/      │
│  3. proof_homogeneity.py      → results/proof/v1/            │
│  4. proof_homogeneity_v2.py   → results/proof_v2/            │
│  5. security_pillar.py        → results/proof_security/      │
│  6. run.py --analyze          → results/metrics/ + figures/  │
└──────────────────────────────────────────────────────────────┘
Steps 1-2 produce the data. Steps 3-5 produce the paper's analysis.
Step 6 is optional (legacy exploratory pipeline).
```

## Viewing Results

All outputs are written to subdirectories under `results/`:

| Directory | What's There | How to View |
|-----------|-------------|-------------|
| `results/raw_responses/` | One JSONL per prompt, one line per (model, temp, sample) | Any text editor or `python -m json.tool` |
| `results/human_baseline/` | Human JSONL files from APPS | Same as above |
| `results/proof/v1/` | v1 diversity proof: `summary.json`, `summary.md`, PNG figures | Open `summary.md` for a narrative; open PNGs directly |
| `results/proof_v2/` | v2 diversity proof: temperature sweep, ablation, regression | Same — `summary.md` is the entry point |
| `results/proof_security/` | Security pillars: `summary.md`, CSVs, PNGs | `summary.md` + `cwe_signatures.csv` for details |
| `results/metrics/` | Legacy pipeline outputs | JSON files |
| `results/figures/` | Legacy pipeline plots | PNG files |

The `summary.md` files are self-contained human-readable reports with all key numbers, statistical tests, and references to the figures.

## Project Structure

```
code_hive/
├── config.py                  # Model registry (11 models, 6 families), sampling params
├── run.py                     # CLI entry point (collect / analyze / demo)
├── prompt_suite.py            # 60 prompts: 30 diversity + 30 security-eliciting (SE-01..30)
├── multi_model_sampler.py     # Async multi-provider API collector (OpenAI, Anthropic, Google, Together)
├── requirements.txt
│
├── proof_homogeneity.py       # Three-pillar diversity proof (Vendi Score, 3 kernels, 4 pools)
├── proof_homogeneity_v2.py    # V2: temperature sweep, family ablation, mixed-effects model
├── security_pillar.py         # Security pillars S1 (vuln rate), S2 (CWE homogeneity), S3 (slopsquatting)
├── security_analysis.py       # Regex-based CWE scanner (fallback when CodeShield unavailable)
│
├── fetch_human_baseline.py    # APPS dataset keyword-matcher for human reference solutions
├── pypi_check.py              # Live PyPI/npm existence checker (Pillar S3)
├── cwe_to_cvss.json           # CWE -> CVSS base score severity table
│
├── diversity_metrics.py       # Legacy 4-layer metrics (surface, structural, semantic, behavioral)
├── pipeline.py                # Legacy intra/inter-model analysis pipeline
│
├── data_prep/                 # Dataset curation (Pool A/B from HuggingFace)
├── local_datasets/            # Prompt manifests (JSON)
├── docs/                      # Extended documentation
└── results/                   # Generated outputs (gitignored)
    ├── raw_responses/         #   Per-prompt JSONL sample files
    ├── human_baseline/        #   Human reference solutions from APPS
    ├── proof_v2/              #   Diversity analysis outputs + figures
    └── proof_security/        #   Security analysis outputs + figures
```

## Diversity Kernels (Paper Methodology)

The paper's three-kernel Vendi-Score framework measures diversity along independent axes:

| Kernel | Similarity Function | Axis Measured |
|--------|-------------------|---------------|
| Token n-gram Jaccard | Jaccard over 3-gram token sets | Lexical: identifiers, imports, idioms |
| AST tree-edit (APTED) | 1 - normalized tree-edit distance | Structural: control flow, nesting, decomposition |
| CodeT5+ embedding cosine | Cosine of mean-pooled CodeT5+ vectors | Semantic: algorithmic intent and strategy |

Each kernel produces a Vendi Score (effective number of distinct items) normalized by pool size to give **per-sample effective uniqueness** in [0, 1].

## Security Pillars

| Pillar | Measurement | Key Finding |
|--------|------------|-------------|
| S1 | Per-model vulnerability rate, CVSS-B weighted | 42.9% overall, mean CVSS-B 6.88 |
| S2 | Cross-model CWE-pattern homogeneity (novel) | 100% exact-pattern match on SQL/XML tasks |
| S3 | Slopsquatting / package hallucination | Near-zero on security prompts |

## Prompt Categories

| Category | Count | Description |
|----------|-------|-------------|
| OPEN_DESIGN | 5 | "Build X" with many valid designs |
| ALGORITHM | 5 | Multiple valid algorithmic approaches |
| REFACTOR | 5 | Improve given code |
| NAMING | 5 | Tasks where naming is unconstrained |
| CREATIVE_TOOL | 5 | Creative coding tasks |
| SYSTEM_DESIGN | 5 | High-level design decompositions |
| SECURITY_ELICITING | 30 | Neutrally-phrased prompts in security-prone domains (SE-01..30) |

## Key Analyses

1. **Effective-sample-size collapse**: Vendi/N under three kernels across four pools (null, human, inter-LLM, intra-LLM)
2. **Paired ordering test**: Bootstrap-paired delta showing LLM diversity < human diversity
3. **Family ablation**: Leave-one-family-out robustness check across all six model families
4. **Temperature sweep**: Diversity at T=0.0 vs T=0.7 (gap narrows ~15% but does not close)
5. **Cross-model vulnerability convergence**: Same exploitable (CWE, sink) patterns across providers
