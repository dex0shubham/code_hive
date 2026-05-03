# Code Hivemind Research Documentation

## 1) What This Repository Is Trying to Do

This repository studies **code-generation monoculture** across LLMs.

The central research question is:

> When different code LLMs are asked open-ended coding tasks, do they still converge to the same implementation patterns, libraries, architectures, and even vulnerability profiles?

The project extends the "Artificial Hivemind" idea from natural language outputs into **program synthesis** and **software engineering behavior**.

## 2) Core Research Claims and Hypotheses

The repo is structured around these hypotheses:

1. **Cross-model convergence exists**: different model families produce surprisingly similar code.
2. **Similarity is multi-layered**: convergence appears at lexical, structural, semantic, and behavioral levels.
3. **Open-ended tasks still converge**: even where many valid solutions exist, outputs often collapse onto a small set of dominant approaches.
4. **Security risk can converge too**: models can produce overlapping CWE patterns, creating systemic risk.

## 3) End-to-End Research Pipeline

The codebase has two connected pipelines:

- **A) Dataset curation pipeline** (`data_prep/prepare_code_openended_datasets.py`)
- **B) Code-hivemind measurement pipeline** (`run.py` + analysis modules)

### A) Dataset Curation Pipeline (Local Datasets)

Purpose: build a local corpus of coding prompts split by source pool and language.

- **Pool A (closed baseline)**: benchmark/problem-style datasets (LeetCode-like, fixed correctness).
- **Pool B (open-ended candidates)**: instruction-style datasets where solution space is broader.

Two-stage labeling flow:

1. **Stage 1 heuristic**: lexical openness score and closed-ended signals.
2. **Stage 2 validation**:
   - `heuristic`: deterministic only
   - `hybrid` (default): heuristics + LLM only for ambiguous Pool B rows
   - `llm`: LLM validation for each row

Outputs are saved under `local_datasets/` as:

- `data.jsonl`
- `data.csv`
- Hugging Face dataset directory (`hf_dataset/`)
- combined audit file (`combined_audit.jsonl`)

### B) Code-Hivemind Measurement Pipeline

1. **Prompt suite definition** (`prompt_suite.py`)
2. **Multi-provider response collection** (`multi_model_sampler.py`)
3. **Metric computation** (`diversity_metrics.py`, `pipeline.py`)
4. **Approach-level analysis** (`approach_analysis.py`)
5. **Security convergence analysis** (`security_analysis.py`)
6. **Report packaging and plots** (`make_report.py`, plus deep-dive scripts)

## 4) Experimental Design

### Prompt Set

- 30 prompts across 6 categories:
  - `OPEN_DESIGN` (OD)
  - `ALGORITHM` (AL)
  - `REFACTOR` (RF)
  - `NAMING` (NM)
  - `CREATIVE_TOOL` (CT)
  - `SYSTEM_DESIGN` (SD)

Prompts are intentionally open-ended and paired with a minimal system instruction:
"write clean working code, choose your own approach."

### Model Registry

Configured in `config.py` via `MODELS` tuples:

- provider
- model_id
- display name
- family

Current providers include OpenAI, Anthropic, Google, and Together-hosted open models.

### Sampling Setup

Configured through `SamplingConfig`:

- temperatures: default `[0.0, 0.6, 1.0]`
- top_p: `0.9`
- samples per model per temp: default `50`
- max tokens: `2048`

All collected responses are stored as JSONL with metadata:

- prompt/model identifiers
- family/provider
- temperature/sample index
- response text
- token usage and latency

## 5) Metrics Framework

Primary metric code lives in `diversity_metrics.py`.

### Layer 1: Surface Similarity

- token n-gram Jaccard (`mean_ngram_jaccard_3`)
- exact match rate

### Layer 2: Structural Similarity

- AST fingerprint similarity (`mean_ast_similarity`)
- naming overlap (`mean_naming_similarity`)

### Layer 3: Semantic Similarity

- sentence-transformer embedding cosine (`mean_embedding_similarity`)

### Layer 4: Behavioral/Design Signals

- import-set diversity and entropy
- design-pattern detection diversity and entropy

### Composite Hivemind Score

The scalar score (`0` diverse, `1` converged) combines weighted components:

- n-gram similarity
- AST similarity
- naming similarity
- embedding similarity
- inverse normalized pattern entropy

Weights are defined in `compute_diversity()` in `diversity_metrics.py`.

## 6) Analysis Modules and Their Roles

### `pipeline.py`

Primary baseline analysis:

- intra-model diversity
- inter-model homogeneity
- family clustering
- temperature effect

Writes `results/metrics/full_analysis.json`.

### `approach_analysis.py`

Adds a richer architecture-focused lens by extracting:

- AST and style feature vectors
- storage/interface choices
- prompt-specific algorithm tags
- architecture labels (e.g., procedural vs OOP, storage type, interface type)

Computes:

- dominant architecture ratio
- entropy/diversity indices
- import concentration and Jaccard overlap
- feature-space cosine similarity
- intra-model repetition
- cross-model agreement matrices

Writes:

- `results/approach/per_prompt.json`
- `results/approach/per_model.json`
- `results/approach/cross_model.json`
- `results/approach/per_sample.csv`
- `results/approach/summary.csv`

And prompt/global figures in `results/figures/`.

### `security_analysis.py`

Scans generated code for vulnerability patterns and convergence:

- regex detectors mapped to CWEs
- optional Bandit integration
- risky import detection
- unknown package detection (possible hallucination/slopsquatting risk)

Key outputs:

- overall vulnerability rate
- exact CWE-set match rate across model pairs
- overlap rate across model pairs

Writes under `results/security/`, including:

- `security_analysis.json`
- `cwe_pair_matches.csv`
- security figures

### `make_report.py`

Builds publication-ready tables/figures from raw and metric outputs.

Creates:

- response coverage tables
- prompt/category/model summaries
- temperature effect tables
- correlation and distribution plots
- `results/REPORT_README.md`

## 7) CLI Entry Points (What To Run)

### Main runner (`run.py`)

- `python run.py --demo`
- `python run.py --collect --samples 50`
- `python run.py --analyze`
- `python run.py --all`

Optional selectors:

- `--prompts OD-01 CT-04 ...`
- `--models "GPT-4o" "Claude Sonnet 4" ...`
- `--temps 0.0 1.0`

### Additional analyses

- `python approach_analysis.py`
- `python security_analysis.py`
- `python make_report.py`

### Dataset prep

- `python data_prep/prepare_code_openended_datasets.py --stage2_mode hybrid`

## 8) Output Layout and Artifact Semantics

- `results/raw_responses/`: one JSONL per prompt + aggregate file
- `results/metrics/`: baseline multi-layer diversity outputs
- `results/approach/`: architecture/approach homogeneity outputs
- `results/security/`: CWE and vulnerability-convergence outputs
- `results/figures/`: visualization assets for paper/reporting
- `results/tables/`: tabular summaries generated by report builder

Interpretation direction:

- higher **similarity** / **dominant ratio** => stronger convergence
- higher **entropy/diversity indices** => broader solution spread
- higher CWE overlap/exact-set match => stronger security monoculture risk

## 9) Research Narrative Encoded in This Repo

The intended story is:

1. LLMs are asked tasks with many valid implementations.
2. Across model families, outputs still cluster around similar architecture and library choices.
3. Traditional "different strings/ASTs" can overstate diversity.
4. At approach level (and sometimes security level), convergence remains strong.
5. Therefore code-assistant ecosystems may accumulate **systemic monoculture risk**, not just isolated model errors.

## 10) Reproducibility Checklist

1. Install dependencies: `pip install -r requirements.txt`
2. Export API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `TOGETHER_API_KEY`) for collection.
3. Collect responses with fixed seed/sampling settings.
4. Run baseline and approach analyses at the same temperature slice (commonly `1.0`).
5. Run security analysis with optional Bandit validation.
6. Generate summary tables/figures with `make_report.py`.
7. Archive:
   - raw JSONL responses
   - metrics JSON/CSV
   - figure artifacts
   - exact model registry and sampling config used

## 11) Notes on Script Overlap

The repository contains both core and exploratory scripts:

- Core, currently integrated path:
  - `run.py`, `multi_model_sampler.py`, `diversity_metrics.py`, `pipeline.py`,
    `approach_analysis.py`, `security_analysis.py`, `make_report.py`
- Exploratory/alternative analysis scripts:
  - `deep_dive.py`, `ana.py`

`ana.py` and `deep_dive.py` are useful for additional narrative figures and sanity checks but overlap with the newer modular pipeline.

## 12) Suggested Next Documentation Improvements

- Add a strict "paper reproduction" profile (fixed models, temperatures, sample counts).
- Add a data dictionary for all JSONL/CSV output fields.
- Add statistical test documentation (confidence intervals/significance) around key comparisons.
- Add versioned experiment manifests for each reported run.
