# Code Hivemind

**Measuring Output Homogeneity Across Code LLMs**

Extends [Artificial Hivemind](https://arxiv.org/abs/2510.22954) (Jiang et al., NeurIPS 2025 Best Paper) to code generation.

## Research Question

> Do different LLMs produce the same code? When asked to implement a cache, a rate limiter, or a CLI tool, do GPT-4o, Claude, Gemini, DeepSeek, and Qwen converge on the same algorithms, naming conventions, library choices, and architectural patterns?

## Quick Start

```bash
# Run demo with mock data (no API keys needed)
python run.py --demo

# Collect real data (requires API keys)
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export TOGETHER_API_KEY=...
python run.py --collect --samples 50

# Analyze collected data
python run.py --analyze

# Quick test: 2 samples, 1 prompt, 1 temperature
python run.py --collect --samples 2 --temps 1.0 --prompts OD-01
```

## Full Documentation

Detailed research and pipeline documentation is available at:

- `docs/RESEARCH_DOCUMENTATION.md`

## Project Structure

```
code-hivemind/
├── config.py                    # Model registry, sampling params
├── run.py                       # Main CLI entry point
├── requirements.txt
├── prompts/
│   └── prompt_suite.py          # 30 open-ended coding prompts (6 categories)
├── collectors/
│   └── multi_model_sampler.py   # Async multi-provider API collector
├── metrics/
│   └── diversity_metrics.py     # 4-layer measurement framework
├── analysis/
│   └── pipeline.py              # Intra/inter-model analysis pipeline
└── results/                     # Generated outputs
    ├── raw_responses/           # JSONL files per prompt
    ├── metrics/                 # Analysis results
    └── figures/                 # Paper-ready plots
```

## Metrics (4 Layers)

| Layer | Metric | What It Measures |
|-------|--------|-----------------|
| Surface | Token n-gram Jaccard | Raw text overlap |
| Surface | Exact match rate | Identical outputs |
| Structural | AST similarity | Same code structure, ignoring names |
| Structural | Naming similarity | Same variable/function names |
| Semantic | Embedding cosine sim | Same meaning/intent |
| Behavioral | Library choice entropy | Same import decisions |
| Behavioral | Design pattern overlap | Same architectural patterns |
| Composite | **Hivemind Score** | Weighted combination (0=diverse, 1=hivemind) |

## Prompt Categories

| Category | # Prompts | Description | Example |
|----------|-----------|-------------|---------|
| OPEN_DESIGN | 5 | "Build X" with many valid designs | In-memory cache, task queue |
| ALGORITHM | 5 | Multiple valid algorithmic approaches | Anagram groups, cycle detection |
| REFACTOR | 5 | Improve given code (many valid improvements) | Pythonic refactoring |
| NAMING | 5 | Tasks where naming is unconstrained | Card deck, HTTP response class |
| CREATIVE_TOOL | 5 | Fun/creative coding tasks | ASCII art, terminal game |
| SYSTEM_DESIGN | 5 | High-level design decompositions | URL shortener, static site gen |

## Key Analyses

1. **Intra-model repetition**: How diverse is each model's own outputs across samples?
2. **Inter-model homogeneity**: Do different models produce the same code?
3. **Family clustering**: Are models from the same company more similar?
4. **Temperature effects**: Does higher temperature actually increase code diversity?
5. **Category breakdown**: Which task types show the strongest hivemind effect?

## Target Venue

NeurIPS 2026 Evaluations & Datasets Track (deadline: May 6, 2026)
