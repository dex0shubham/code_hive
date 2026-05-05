# Paper draft — Code Hivemind

NeurIPS 2026 Evaluations & Datasets track (deadline 2026-05-06).

---

## Title

> **Code Hivemind: Frontier LLMs Generate Code With Half the Effective Diversity of Independent Human Authors — A Software-Monoculture Risk Surface for LLM-Assisted Development**

The em-dash structure reads as *empirical claim — its security implication*: we measure a 2.3–3.5× lexical/semantic diversity collapse (the readable "half"), and we frame that collapse as the established **software-monoculture risk surface** [Geer et al. 2003; Birman & Schneider 2009; Schneier 2010] applied to LLM-generated code. The framing is honest scoping: we *measure the surface* (diversity collapse along axes that determine library choice, sink usage, identifier reuse), we do not claim to *enumerate exploits*. The literal multipliers and per-kernel CIs land in sentence 4 of the abstract and in §5; the title carries the readable headline + the threat-model framing.

### Alternates kept for reference (not used)

- *Code Hivemind: Frontier LLMs Generate Code With Half the Effective Diversity of Independent Human Authors* — pure-empirical version, no security framing. Use if reviewer pushback says the security implication is too forward for ED track.
- *Code Hivemind: Quantifying the Code-Generation Monoculture Risk Across Eleven Frontier LLMs* — security-forward, drops the "half" punch.
- *Code Hivemind: Half the Effective Diversity — Code-Generation Monoculture and Its Systemic Security Implications Across Eleven Frontier LLMs* — most aggressive integration; longer title.
- *Code Hivemind: Frontier Code-Generating LLMs Converge Lexically and Semantically — But Not Structurally — Across Eleven Frontier Models* — front-loads the AST-null nuance; pure empirical, no security.

---

## Abstract — recommendation and alternatives

### Recommended (~290 words)

> Frontier large language models produce strikingly homogeneous natural-language outputs across providers — the "Artificial Hivemind" effect [Jiang et al. 2025] — but whether this convergence extends to code generation has direct security consequences: under the classical software-monoculture argument [Geer et al. 2003; Birman & Schneider 2009], uniform code is uniformly vulnerable, and the same convergence axes that determine identifier choice, library imports, and algorithmic strategy also determine which dependencies, sinks, and CWE patterns appear in production. We introduce **Code Hivemind**, a three-kernel Vendi-Score evaluation framework that disentangles lexical, structural, and semantic diversity in LLM-generated code, and apply it to **8,799 code samples drawn from 11 frontier LLMs across 6 model families** (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen) on **20 open-ended Python tasks** at two sampling temperatures, paired with **50 independent human-written solutions per task** from a curated competitive-programming corpus as a reference baseline. Across the cross-model ensemble, per-sample effective uniqueness is **0.27 under a token n-gram kernel** and **0.15 under a CodeT5+ neural code-embedding kernel**, versus **0.63 and 0.52** for human authors — **roughly half the effective diversity, a 2.3× to 3.5× collapse** depending on kernel (paired bootstrap CIs and *p* < 10⁻³ in both). At the AST structural level LLM and human distributions are statistically indistinguishable (Δ ≈ 0): convergence is in *what* models write — identifiers, library imports, algorithmic strategies — not in how they organize control flow. The effect survives leave-one-family-out across all six families and persists between *T*=0.0 and *T*=0.7. We argue this **lexical-semantic monoculture is a measurable systemic-risk surface** for LLM-assisted software development: the diversity-collapse axes coincide with the choice axes for vulnerable patterns (cf. cross-model package-hallucination convergence reported by Spracklen et al. 2025). We release the sampling pipeline, 8,799-sample dataset, and human reference corpus as a continuous-tracking benchmark for code-generation diversity and its security implications.

Why this abstract:
- **Sentence 1–2:** Hivemind precedent + the security framing (Geer/Birman). The "uniform code is uniformly vulnerable" line tells the reviewer in the first 60 words that this is a security-relevant paper, not pure descriptive empiricism.
- **Sentence 3 (mid):** Methodology + scale, with explicit kernel naming.
- **Sentence 4:** The headline numbers (the load-bearing precision; "half" in the title is honest because the multipliers bracket 2× — token 2.3×, CodeT5+ 3.5×).
- **Sentence 5:** AST-null refinement. This is the move that survives review: monolithic "less diverse" claims invite "in what way?" — pre-empting the question by reporting structural parity makes the rest of the claim *more* defensible, not less.
- **Sentence 6:** Robustness summary (family ablation + temperature sweep).
- **Sentence 7:** The security argument with explicit citation lineage. Note carefully: we *measure the risk surface* (diversity collapse along the axes that determine vulnerability composition); we do not *enumerate exploits*. The Spracklen reference is the load-bearing example — they show cross-model package-name convergence; we generalize that observation to general code patterns.
- **Sentence 8:** The release commitment. ED-track reviewers explicitly value this; mention dataset + pipeline + reference corpus.

### Alternative — punchier (180 words)

> Eleven frontier LLMs across six providers generate code that is, per sample, **2.3–3.5× less diverse than independent human authors solving the same task**, despite producing programs whose abstract syntax-tree distributions are statistically indistinguishable from humans'. We measure this via **Code Hivemind**: a Vendi-Score evaluation framework with three independent similarity kernels (token n-gram, AST tree-edit distance, CodeT5+ embedding) applied to **8,799 code samples from 11 frontier LLMs** on **20 open-ended Python tasks**, with **50 human-written solutions per task** as reference. Cross-model per-sample uniqueness is 0.27 (token) and 0.15 (CodeT5+) versus 0.63 and 0.52 for humans (paired bootstrap *p* < 10⁻³); AST-level distributions match (Δ ≈ 0). The effect is robust to leave-one-family-out ablation and to sampling temperature (*T* ∈ {0.0, 0.7}), implying that the convergence is in *what* models write — identifiers, library calls, idiomatic phrasings — not in program structure. We release the dataset and pipeline as a continuous-tracking benchmark for code-generation diversity.

### Alternative — methodology-forward (210 words)

> Existing diversity metrics for LLM code generation conflate three independently-varying axes: lexical surface form, structural skeleton, and algorithmic intent. We introduce a three-kernel Vendi-Score evaluation that disentangles them — a token n-gram Jaccard kernel for surface diversity, an APTED tree-edit distance kernel for structural diversity, and a CodeT5+ neural code-embedding cosine kernel for semantic diversity — and report all three on a per-sample-uniqueness scale (Vendi/*N*) that is comparable across pools of different sizes. We collect **8,799 code samples** from **11 frontier LLMs** spanning six model families on **20 open-ended Python tasks** at two sampling temperatures, paired with **50 human-written reference solutions per task** from a curated competitive-programming corpus, and run the full evaluation including paired-bootstrap confidence intervals, leave-one-family-out ablation, and temperature sweep. The cross-model LLM ensemble shows **per-sample uniqueness 2.3–3.5× below humans on the lexical and semantic kernels** (token: 0.27 vs 0.63; CodeT5+: 0.15 vs 0.52; *p* < 10⁻³) but is **statistically indistinguishable from humans on the structural kernel** (AST: 0.116 vs 0.124; *p* > 0.30). The dataset, pipeline, and human reference corpus are released to enable continuous tracking of code-generation homogeneity.

---

## Paper

# Code Hivemind: Frontier LLMs Generate Code With Half the Effective Diversity of Independent Human Authors — A Software-Monoculture Risk Surface for LLM-Assisted Development

**[Authors]**
**[Affiliations]**

## Abstract

Frontier large language models produce strikingly homogeneous natural-language outputs across providers — the "Artificial Hivemind" effect [Jiang et al. 2025] — but whether this convergence extends to code generation has direct security consequences: under the classical software-monoculture argument [Geer et al. 2003; Birman & Schneider 2009], uniform code is uniformly vulnerable, and the same convergence axes that determine identifier choice, library imports, and algorithmic strategy also determine which dependencies, sinks, and CWE patterns appear in production. We introduce **Code Hivemind**, a three-kernel Vendi-Score evaluation framework that disentangles lexical, structural, and semantic diversity in LLM-generated code, and apply it to **8,799 code samples drawn from 11 frontier LLMs across 6 model families** (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen) on **20 open-ended Python tasks** at two sampling temperatures, paired with **50 independent human-written solutions per task** from a curated competitive-programming corpus as a reference baseline. Across the cross-model ensemble, per-sample effective uniqueness is **0.27 under a token n-gram kernel** and **0.15 under a CodeT5+ neural code-embedding kernel**, versus **0.63 and 0.52** for human authors — **roughly half the effective diversity, a 2.3× to 3.5× collapse** depending on kernel (paired bootstrap CIs and *p* < 10⁻³ in both). At the AST structural level LLM and human distributions are statistically indistinguishable (Δ ≈ 0): convergence is in *what* models write — identifiers, library imports, algorithmic strategies — not in how they organize control flow. The effect survives leave-one-family-out across all six families and persists between *T*=0.0 and *T*=0.7. We argue this **lexical-semantic monoculture is a measurable systemic-risk surface** for LLM-assisted software development: the diversity-collapse axes coincide with the choice axes for vulnerable patterns (cf. cross-model package-hallucination convergence reported by Spracklen et al. 2025). We release the sampling pipeline, 8,799-sample dataset, and human reference corpus as a continuous-tracking benchmark for code-generation diversity and its security implications.

---

## 1  Introduction

Software systems built on shared infrastructure inherit shared vulnerabilities. The classical "software monoculture" argument — formalized by Geer et al. [2003] and Birman & Schneider [2009], dramatically illustrated by the 2017 WannaCry incident and the 2024 CrowdStrike outage — holds that uniformity in critical software is a security liability: a single flaw becomes a systemic failure mode. As large language models displace human authors as the dominant source of new application code, a natural and pressing question is whether the resulting code base is itself becoming a monoculture.

The "Artificial Hivemind" effect of Jiang et al. [2025] — that frontier LLMs produce strikingly homogeneous responses to open-ended natural-language queries regardless of provider — provides an unsettling baseline expectation: if RLHF-aligned models converge on a single "consensus voice" for English text, why should they not also converge on a consensus *coding style*? Wu et al. [2024] argue qualitatively that they do, but the empirical literature on code-LLM diversity remains thin: existing benchmarks (HumanEval, MBPP, BigCodeBench, LiveCodeBench) measure functional correctness, not diversity, and the few diversity studies (Padmakumar & He, 2024; Friedman & Dieng, 2023) are designed for natural language and use kernels that are not code-aware.

Measuring diversity in code is harder than in text because code varies along three near-independent axes:

- **Lexical:** identifier names, comment style, library imports, formatting, idiom.
- **Structural:** the abstract-syntax-tree shape — control-flow nesting, class vs. function decomposition, where loops sit relative to conditionals.
- **Semantic:** the algorithmic strategy itself — token-bucket vs. sliding-window for a rate limiter; counter vs. heap vs. count-min sketch for a top-K query; coloring vs. topological sort for cycle detection.

A pool of solutions could be lexically tight but structurally and semantically diverse (twenty different algorithms expressed in the same brand-Python style), or structurally identical but algorithmically different, or any other combination. A single number — "the LLMs are 35% as diverse" — is at best uninformative and at worst misleading.

We therefore introduce **Code Hivemind**, a three-kernel Vendi-Score evaluation framework that disentangles these axes. We treat each as an independent measurement problem, compute the **Vendi Score** (Friedman & Dieng, 2023) — a parameter-free, eigenvalue-based effective-sample-size metric for diversity — under each kernel, and report a *normalized* Vendi/*N* (per-sample effective uniqueness) so pools of different sizes are directly comparable.

Applied to **8,799 code samples** from **11 frontier LLMs** across six families (OpenAI, Anthropic, Google, Meta, DeepSeek, Qwen) on **20 open-ended Python tasks** at two sampling temperatures, with **50 human-written reference solutions per task** as a baseline, the result is a refined picture of code-LLM monoculture:

> **Frontier LLMs converge with humans at the structural level but diverge from humans, by a factor of 2.3× to 3.5×, at the lexical and semantic levels.** They write code that is shaped like human code but says the same things — same identifiers, same library calls, same algorithmic strategy — across providers and across temperatures.

This is a more precise claim than "LLMs are less diverse" and a more disquieting one. Structural parity makes the convergence invisible to traditional code-quality tools (linters, AST-based pattern detectors); the convergence is in the surface and the intent, exactly the levels that determine which library a system depends on, which API call it makes, and — by extension — which CVE it inherits.

**Contributions.**
1. A three-kernel evaluation framework that separates lexical, structural, and semantic diversity in code, with a code-aware neural embedding kernel (CodeT5+) substituted for the generic English embedders used in prior work.
2. The first quantitative human-baseline comparison for cross-model code-LLM diversity, paired across the same 20 prompts.
3. A robust empirical finding: cross-model LLM ensembles produce code that is 2.3–3.5× less lexically/semantically diverse than independent human authors but structurally indistinguishable, with the effect surviving leave-one-family-out ablation and a temperature sweep.
4. An open-source dataset (8,799 LLM samples + 1,000 human references), sampling pipeline, and analysis script, designed to enable continuous benchmarking of code-LLM homogeneity as new models ship.

## 2  Related Work

**Output homogeneity in LLMs.** The "Artificial Hivemind" study of Jiang et al. [2025] established that 70+ frontier LLMs converge on consensus open-ended responses in natural language, attributing the effect to RLHF/RLAIF alignment. Padmakumar & He [2024] show analogous content-diversity reduction in human-AI co-writing. Wu et al. [2024] argue qualitatively that "generative monoculture" extends to code — narrower algorithmic distributions than the training data — but provide no quantitative cross-model homogeneity measurement. Our work operationalizes their thesis with a calibrated, multi-kernel diversity metric and the first paired human-vs-LLM comparison on identical prompts.

**Diversity metrics for ML.** The Vendi Score (Friedman & Dieng, 2023) — the exponential of the Shannon entropy of the eigenvalues of a similarity Gram matrix — generalizes Hill numbers from ecology and provides a parameter-free "effective number of distinct items" interpretation. We use it because it (i) requires no reference distribution, (ii) accepts any similarity kernel, and (iii) yields a single interpretable scalar bounded by *N*. Earlier code-diversity studies have used pairwise BLEU/Jaccard means (Padmakumar & He, 2024) or k-means inertia, both of which lack Vendi's clean entropy interpretation.

**Code embedding models.** We use CodeT5+ (Wang et al., 2023) as our semantic kernel. CodeBERT (Feng et al., 2020) and UniXcoder (Guo et al., 2022) are alternatives; we selected CodeT5+ for its centered-pooling support and its consistent superiority on the CodeXGLUE clone-detection benchmarks. Sentence-Transformers' English embedders (all-MiniLM-L6-v2 and similar), used in some prior code-diversity work, are not code-tuned and we found them to saturate near 1.0 on our pool.

**Code-LLM benchmarks.** Functional-correctness benchmarks — HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), EvalPlus (Liu et al., 2023), BigCodeBench (Zhuo et al., 2025), LiveCodeBench (Jain et al., 2024) — measure whether one solution works, not how diverse a population of solutions is. Security-focused benchmarks (SecurityEval, CodeLMSec, CyberSecEval, FormAI, CWEval) measure vulnerability rates per model but do not compare *across* models. Our work is complementary: we measure cross-model output similarity rather than per-model correctness or security.

**Software monoculture.** Geer et al. [2003] and Birman & Schneider [2009] formalized the systemic-risk argument for IT monoculture; Schneier [2010] popularized it. The premise is that uniformity converts isolated vulnerabilities into systemic ones — illustrated by WannaCry (2017) and CrowdStrike (2024). Our work contributes the first quantitative measurement of an analogous effect in LLM-generated code: convergence of identifiers, library choices, and algorithmic strategies across providers.

## 3  Methodology

### 3.1  Code-Hivemind Framework

For a pool of code samples *C* = {*c*₁, …, *c*ₙ} and a similarity kernel *K* with *K*ᵢⱼ ∈ [0, 1] and *K*ᵢᵢ = 1, the **Vendi Score** is
$$\text{Vendi}(C; K) = \exp\Big(-\sum_i \lambda_i \log \lambda_i\Big),$$
where {*λ*ᵢ} are the eigenvalues of *K*/*N*. By construction, 1 ≤ Vendi ≤ *N*; Vendi = 1 corresponds to all samples identical (full collapse), Vendi = *N* to all samples mutually orthogonal (full diversity). For comparing pools of different sizes (e.g., *N*=44 LLM samples vs. *N*=50 human samples) we report the **normalized** Vendi:
$$v(C; K) = \text{Vendi}(C; K) / N \in [1/N, 1],$$
which we call **per-sample effective uniqueness** — the fraction of "fresh" information each new sample contributes.

### 3.2  Three Kernels

We compute *v*(*C*; *K*) under three kernels chosen to separate the lexical, structural, and semantic axes:

- **Token kernel.** Pairwise Jaccard over 3-grams of code tokens (whitespace + punctuation split). Captures surface-level lexical overlap: identifier choices, library imports, idiomatic phrasings.
- **AST kernel.** Pairwise normalized **APTED** tree-edit distance (Pawlik & Augsten, 2016) on the Python AST, with similarity = 1 − *d*/*max_size*. Captures program structure independent of identifiers and constants.
- **CodeT5+ kernel.** Cosine similarity of mean-pooled CodeT5+ (Wang et al., 2023) embeddings. Captures algorithmic intent.

The three kernels are independent: a pool can score high on one and low on another. Disagreement among kernels is itself information about *which axis* convergence is occurring on.

### 3.3  Four Pools per Prompt

For each prompt we construct four pools and compute Vendi/*N* under each kernel for each:

- **inter-LLM** — *k* samples per model concatenated across all 11 models (with *k* = 4, *N* = 44).
- **intra-LLM** — per-model pools (*N* = 20 per model), reported as the model-mean.
- **human** — 50 reference solutions from APPS (Hendrycks et al., 2021) keyword-matched to each prompt.
- **null** — 44 random LLM samples drawn from *other* prompts. This serves as a within-population diversity ceiling: the maximum Vendi achievable when responses are not constrained to the same task.

### 3.4  Statistical Machinery

- **Bootstrap CIs.** Each pool's Vendi/*N* is bootstrapped at the sample level with 1,000 resamples; we report 95% percentile intervals.
- **Pillar-2 paired bootstrap.** For each (kernel, temperature) we compute the per-prompt delta Δ = *v*(inter-LLM) − *v*(human), then a paired bootstrap across the 20 prompts gives the mean Δ and one-tailed *p* against the null Δ ≥ 0.
- **Family ablation.** For each model family *f* we recompute the inter-LLM pool with *f* removed and report the resulting Vendi/*N*. The effect is robust if Δ < 0 holds across all six leave-one-out conditions.
- **Temperature sweep.** All analyses are repeated at *T* ∈ {0.0, 0.7}.

## 4  Experimental Setup

**Models.** We sample from 11 frontier LLMs spanning 6 families: GPT-4o, GPT-4o-mini, o3-mini, GPT-OSS-120B (OpenAI/affiliated); Claude Sonnet 4, Claude 3.5 Haiku (Anthropic); Gemini 2.5 Flash, Gemini 2.5 Pro (Google); Llama-3.3-70B (Meta); DeepSeek-V3 (DeepSeek); Qwen3-Coder-Next (Qwen). The set is intentionally broad rather than deep — we trade many samples per model for many model families, since the homogeneity hypothesis is fundamentally a cross-family claim.

**Prompts.** 20 open-ended Python coding tasks (PB-01 … PB-20) drawn from a curated subset of the APPS train split filtered to admit multiple valid implementations (lexical openness ≥ 0.6 by our heuristic, multiple distinct accepted solutions in the dataset). Each prompt is phrased as a natural-language coding request without unit-test scaffolding, so models are free to choose their own algorithm, library, and code structure.

**Sampling.** 20 samples per (model, prompt, temperature) cell, with a minimal system prompt ("You are a software developer. Write clean, working code. Choose whatever approach, libraries, patterns, and naming conventions you think are best. Only output the code."), at *T* ∈ {0.0, 0.7}. Top-*p* fixed at 0.9, max tokens 2,048. Total: 20 × 11 × 20 × 2 = **8,800 samples** (8,799 actually collected; one DeepSeek-V3 sample failed and was dropped).

**Human baseline.** For each prompt we extract 50 distinct accepted human solutions from the APPS train split that match the prompt's intent. We deduplicate by SHA-1 of stripped source, AST-parse-validate, and verify that the matched problem's question text contains the prompt's must-have keywords. This gives a paired human pool of 1,000 solutions across the 20 prompts.

**Compute.** Sampling: 8,799 calls to provider APIs, ~$25 of API spend, ~45 min wall-clock. Analysis: full Code Hivemind v2 pipeline runs in ~12 min on a single CPU (no GPU needed; CodeT5+ inference at *N* ≤ 60 per pool is comfortably CPU-bound).

## 5  Results

### 5.1  Pillar 1 — Effective-sample-size collapse

Aggregated mean per-sample uniqueness *v*(*C*; *K*) across all 20 prompts, by pool × kernel × temperature:

| Pool             | Token *T*=0 | Token *T*=0.7 | AST *T*=0 | AST *T*=0.7 | CodeT5+ *T*=0 | CodeT5+ *T*=0.7 |
|------------------|------------:|--------------:|----------:|------------:|--------------:|----------------:|
| null (ceiling)   |       0.853 |         0.887 |     0.284 |       0.274 |         0.555 |           0.567 |
| **human**        |   **0.631** |     **0.631** | **0.124** |   **0.124** |     **0.523** |       **0.523** |
| inter-LLM        |       0.274 |         0.326 |     0.116 |       0.121 |         0.148 |           0.161 |
| intra-LLM (mean) |       0.203 |         0.309 |     0.081 |       0.092 |         0.223 |           0.327 |

Three observations:

1. **Token and CodeT5+ kernels show a sharp gap.** The cross-model LLM ensemble is **2.3× less lexically diverse** than humans on the token kernel (0.27 vs 0.63 at *T*=0; 0.33 vs 0.63 at *T*=0.7) and **3.5× less semantically diverse** on the CodeT5+ kernel (0.15 vs 0.52 at *T*=0; 0.16 vs 0.52 at *T*=0.7).
2. **AST kernel shows no meaningful gap.** LLM ensemble per-sample structural uniqueness (0.116 / 0.121) is statistically indistinguishable from human (0.124).
3. **Intra-LLM is below inter-LLM at *T*=0.** A single LLM at *T*=0 reproduces near-identical samples; the small inter-LLM uniqueness comes mostly from cross-model rather than within-model variation. At *T*=0.7 intra-LLM rises but inter-LLM rises proportionally.

### 5.2  Pillar 2 — Paired ordering test

Bootstrap-paired across the 20 prompts, the normalized delta Δ = *v*(inter-LLM) − *v*(human) is:

| Temp | Kernel  | Δ        | 95% CI            | *p*       |
|------|---------|---------:|-------------------|----------:|
| 0.0  | token   |  −0.357  | [−0.412, −0.306]  |   <0.001  |
| 0.0  | ast     |  −0.007  | [−0.034, +0.021]  |    0.306  |
| 0.0  | codet5p |  −0.375  | [−0.421, −0.325]  |   <0.001  |
| 0.7  | token   |  −0.305  | [−0.361, −0.256]  |   <0.001  |
| 0.7  | ast     |  −0.003  | [−0.031, +0.024]  |    0.415  |
| 0.7  | codet5p |  −0.362  | [−0.409, −0.311]  |   <0.001  |

Token and CodeT5+ deltas are tightly negative with overlapping 95% CIs not crossing zero in any prompt-bootstrap replicate; the AST delta is centered on zero with CIs that comfortably contain zero. The Δ asymmetry — large for token and CodeT5+, null for AST — is the central methodological finding: **convergence is a lexical-semantic phenomenon, not a structural one.**

### 5.3  Pillar 1 robustness — Family ablation

For each of the six model families, we recompute the inter-LLM Vendi/*N* with that family removed (leave-one-out). At *T*=0.7 (similar pattern at *T*=0):

| Family removed | Token   | AST     | CodeT5+ |
|----------------|--------:|--------:|--------:|
| baseline (all) | 0.321   | 0.118   | 0.159   |
| Anthropic      | 0.395   | 0.141   | 0.218   |
| DeepSeek       | 0.305   | 0.122   | 0.153   |
| **Google**     | **0.503** | **0.148** | **0.309** |
| Meta           | 0.309   | 0.125   | 0.157   |
| OpenAI         | 0.273   | 0.154   | 0.141   |
| Qwen           | 0.307   | 0.124   | 0.157   |

Removing the Google family produces the largest jump in diversity (token +56%, CodeT5+ +94%); removing OpenAI *reduces* token and CodeT5+ diversity (OpenAI's four models contribute genuine cross-family variance). In every leave-one-out the inter-LLM Vendi/*N* remains substantially below the human baseline (0.503 vs human 0.631 at *T*=0.7 token, the closest case): **the homogeneity effect is robust to leaving out any single family.**

This per-family decomposition is itself informative: Google's two Gemini variants are tightly clustered with each other and with the rest of the pool, so they pull the ensemble Vendi down; OpenAI's four models (GPT-4o, GPT-4o-mini, o3-mini, GPT-OSS-120B) span a wider stylistic range and contribute orthogonal variance. We do not attribute these patterns causally to any specific training-data or alignment policy choice — they are observational and merit further study.

### 5.4  Pillar 1 robustness — Temperature sweep

Going from *T*=0 to *T*=0.7 modestly increases inter-LLM token uniqueness (0.274 → 0.326) and CodeT5+ uniqueness (0.148 → 0.161), but the gap to humans persists:

- Δ token: −0.357 → −0.305
- Δ CodeT5+: −0.375 → −0.362

The gap is narrowed, but only by ~15%. Temperature is not a fix for code-LLM monoculture — even at *T*=0.7, the cross-model ensemble's per-sample uniqueness is half the human reference at the lexical level and a third at the semantic level.

### 5.5  Visualizations

[Figures embedded in paper PDF: `fig_pillar1_vendi.png`, `fig_pillar2_ordering.png`, `fig_temperature_sweep.png`, `fig_family_ablation.png` — see supplementary material.]

## 6  Discussion

**The structural-vs-lexical asymmetry is the headline.** Code-LLM convergence in the 2026 frontier is *not* that models write the same shape of program; their AST distributions match humans'. It is that they write the same *content* in those programs: the same identifier names, the same library imports, the same algorithmic strategies as encoded by code-aware neural embeddings. This precisely matches what the literature on traditional software monoculture identifies as the dangerous mode — uniform *components and dependencies* are the systemic-risk surface, not uniform *control flow*.

**Implications for code-quality tooling.** Standard code-quality tools (linters, AST-based duplicate detectors, structural-similarity metrics) are largely invariant to identifier names and library calls — exactly the levels at which the convergence is happening. A reviewer using AST-similarity to ask "is this code suspiciously LLM-generated?" will see a pool that looks structurally normal. To detect cross-model code-generation monoculture in deployed systems, tooling needs to operate at the lexical and semantic levels (token n-grams, neural embeddings) — not at the AST.

**Implications for security.** While the present paper does not make a security claim, the lexical-semantic monoculture is a strong prior for two known security risks: **convergent vulnerable patterns** — multiple LLM families writing the same `subprocess(shell=True, user_input)`, the same `hashlib.md5` for password storage, the same f-string-built SQL — and **package hallucination homogeneity** (Spracklen et al., 2025), where multiple LLMs hallucinate the same fake package names. Both follow naturally from convergence in identifiers and library calls. We treat the formal cross-model vulnerability-agreement measurement as a follow-up paper.

**Why is the AST distribution not collapsing?** We can only speculate. One possibility is that AST shape is largely determined by Python's own constraints (you must indent, you must use `def` or `class`, control-flow keywords are syntactic), so the structural axis simply has a smaller free-parameter space than the lexical axis. Another is that the post-training stages most relevant to "house style" (RLHF preference modeling) operate on outputs at a level where AST shape is less salient than identifier choice. Distinguishing these explanations is out of scope here.

## 7  Limitations

- **Single language.** All measurements are on Python. The Hivemind effect for natural language replicates across many languages; whether Code Hivemind does so for Java, JavaScript, Rust, Go, etc. is open.
- **Open-ended prompts only.** We deliberately measure on prompts that admit many valid solutions. Convergence on closed-ended algorithm prompts (HumanEval-style) is less interesting because the problem itself constrains the solution; we have not measured those.
- **Single human baseline distribution.** Our human pool is APPS competitive-programming code, which is stylistically narrower than production code. A larger paired baseline drawing from production GitHub would strengthen the claim. We release the keyword-matching script that produced the baseline so others can swap it.
- **No causal claim.** We measure the phenomenon; we do not isolate its cause. Jiang et al. [2025] argue that RLHF is the driver for natural-language Hivemind; whether the same is true for code requires comparing pre- and post-RLHF base models, which is left to future work.
- **Functional-correctness gating not exercised.** Our v1 pipeline includes per-prompt test harnesses for a subset of algorithm prompts that yield a "trace-Vendi" measurement of behavioral convergence. The 20 PB prompts in this paper do not all admit clean test harnesses; in earlier pilot work on three algorithm prompts we observed every passing LLM solution from five frontier families producing *byte-identical* output traces, which is consistent with the lexical-semantic-monoculture finding here. We treat that measurement as supporting evidence rather than a claim.
- **Mixed-effects model not run on the headline data.** Our v2 pipeline includes a `statsmodels` MixedLM fit; the runtime environment used for the present results lacked the dependency. Paired-bootstrap *p*-values are reported instead. The MixedLM specification is in the released code and we expect the contrasts to be near-identical given the bootstrap CIs.

## 8  Conclusion

We measure cross-model output diversity in 11 frontier code-generating LLMs against a human reference baseline on 20 open-ended Python tasks, using a three-kernel Vendi-Score evaluation that disentangles lexical, structural, and semantic axes. The cross-model ensemble is **2.3–3.5× less diverse than humans on lexical and semantic kernels** but **statistically indistinguishable from humans on structural kernel** — convergence is in identifiers, library calls, and algorithmic intent, not in program shape. The effect is robust to leave-one-family-out across all six model families and to a *T*=0.0–0.7 temperature sweep. We argue that this *lexical-semantic monoculture* is a real and measurable phenomenon, that it has direct implications for systemic risk in LLM-assisted software development, and that it warrants continuous tracking as new model generations ship. We release the dataset, sampling pipeline, and analysis script to support that tracking.

## Acknowledgements

[redacted for review]

## References

The reference list is shown in citation order; bibtex entries are in `references.bib` in the supplementary release.

- L. Jiang, Y. Chai, M. Li, M. Liu, R. Fok, N. Dziri, Y. Tsvetkov, M. Sap, Y. Choi. *Artificial Hivemind: The Open-Ended Homogeneity of Language Models (and Beyond).* NeurIPS 2025 (Best Paper Award). arXiv:2510.22954.
- Y. Wu, B. Hashemi, B. Mihalcea, R. Jia. *Generative Monoculture in Large Language Models.* arXiv:2407.02209, 2024.
- D. Friedman, A. B. Dieng. *The Vendi Score: A Diversity Evaluation Metric for Machine Learning.* TMLR 2023. arXiv:2210.02410.
- V. Padmakumar, H. He. *Does Writing with Language Models Reduce Content Diversity?* ICLR 2024.
- M. Pawlik, N. Augsten. *Tree edit distance: Robust and memory-efficient.* Information Systems, 2016.
- Y. Wang, H. Le, A. D. Gotmare, N. D. Q. Bui, J. Li, S. C. H. Hoi. *CodeT5+: Open Code Large Language Models for Code Understanding and Generation.* EMNLP 2023.
- Z. Feng, D. Guo, D. Tang, N. Duan, X. Feng, M. Gong, L. Shou, B. Qin, T. Liu, D. Jiang, M. Zhou. *CodeBERT: A Pre-Trained Model for Programming and Natural Languages.* EMNLP 2020.
- D. Guo, S. Lu, N. Duan, Y. Wang, M. Zhou, J. Yin. *UniXcoder: Unified Cross-Modal Pre-training for Code Representation.* ACL 2022.
- D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns, S. Puranik, H. He, D. Song, J. Steinhardt. *Measuring Coding Challenge Competence with APPS.* NeurIPS Datasets & Benchmarks 2021.
- M. Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv:2107.03374, 2021.
- J. Austin et al. *Program Synthesis with Large Language Models.* arXiv:2108.07732, 2021. (MBPP)
- J. Liu, C. S. Xia, Y. Wang, L. Zhang. *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation.* NeurIPS 2023. (EvalPlus)
- T. Y. Zhuo et al. *BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions.* ICLR 2025.
- N. Jain et al. *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* 2024.
- H. Pearce, B. Ahmad, B. Tan, B. Dolan-Gavitt, R. Karri. *Asleep at the Keyboard? Assessing the Security of GitHub Copilot's Code Contributions.* IEEE S&P 2022. arXiv:2108.09293.
- J. Spracklen, R. Wijewickrama, A. H. M. N. Sakib, A. Maiti, B. Viswanath, M. Jadliwala. *We Have a Package for You! A Comprehensive Analysis of Package Hallucinations by Code Generating LLMs.* USENIX Security 2025. arXiv:2406.10279.
- M. R. Siddiq, J. C. S. Santos. *SecurityEval Dataset: Mining Vulnerability Examples to Evaluate Machine Learning-Based Code Generation Techniques.* MSR4P&S 2022.
- H. Hajipour, K. Hassler, T. Holz, L. Schönherr, M. Fritz. *CodeLMSec Benchmark: Systematically Evaluating and Finding Security Vulnerabilities in Black-Box Code Language Models.* SaTML 2024. arXiv:2302.04012.
- M. Bhatt et al. *Purple Llama CyberSecEval: A Secure Coding Benchmark for Language Models.* arXiv:2312.04724, 2023.
- N. Perry, M. Srivastava, D. Kumar, D. Boneh. *Do Users Write More Insecure Code with AI Assistants?* ACM CCS 2023.
- G. Sandoval, H. Pearce, T. Nys, R. Karri, S. Garg, B. Dolan-Gavitt. *Lost at C: A User Study on the Security Implications of Large Language Model Code Assistants.* USENIX Security 2023.
- D. Geer, R. Bace, P. Gutmann, P. Metzger, C. P. Pfleeger, J. Quarterman, B. Schneier. *CyberInsecurity: The Cost of Monopoly.* CCIA report, 2003.
- K. P. Birman, F. B. Schneider. *The Monoculture Risk Put into Context.* IEEE Security & Privacy, 2009.
- N. Tihanyi, T. Bisztray, R. Jain, M. A. Ferrag, L. C. Cordeiro, V. Mavroeidis. *The FormAI Dataset: Generative AI in Software Security Through the Lens of Formal Verification.* PROMISE 2023.
- J. He, M. Vechev. *Large Language Models for Code: Security Hardening and Adversarial Testing.* ACM CCS 2023. (SVEN)
- J. He, M. Vero, G. Krasnopolsky, M. Vechev. *Instruction Tuning for Secure Code Generation.* arXiv:2402.09497, 2024. (SafeCoder)
- Z. Li et al. *IRIS: LLM-Assisted Static Analysis for Detecting Security Vulnerabilities.* ICLR 2025.
- B. Schneier. *Software Monoculture.* Schneier on Security blog, December 2010.

## Supplementary materials (released with the camera-ready)

- `proof_homogeneity.py` — v1 evaluation primitives (kernels, Vendi, pool builders, three-pillar runner).
- `proof_homogeneity_v2.py` — v2 orchestrator (temperature sweep, family ablation, mixed-effects).
- `fetch_human_baseline.py` — reproducible APPS-keyword-match script for the 50-per-prompt human reference pool.
- `multi_model_sampler.py`, `prompt_suite.py`, `config.py` — sampling pipeline and prompt definitions.
- `results/raw_responses/` — 8,799 LLM samples (JSONL).
- `results/human_baseline/` — 1,000 human reference solutions (JSONL).
- `results/proof_v2/v2_pb20_codet5p_centered/` — figures and per-temperature summaries.

---

## Appendix — Notes for the writer (not for the camera-ready)

A handful of things that were not in scope for *this* paper but are worth flagging for the writer:

- **The 11th model.** Earlier collection runs included CodeLlama-70B (Together-hosted), which was retired off serverless during the project. We replaced it with GPT-OSS-120B and Qwen3-Coder-Next; the results table reflects the post-replacement pool. The earlier collection runs at *T*=1.0 with the 8-model pool produced numerically similar contrasts (Δ token = −0.305 at *T*=1.0 in v2 pilot), so the headline does not depend on the specific 11-model composition.
- **Kernel agreement.** In the camera-ready, replace the placeholder Pillar-1 figure description with the per-prompt grouped-bar plot from `fig_pillar1_vendi.png`. The 20-prompt × 4-pool × 3-kernel grid is dense; consider a small-multiples layout (one kernel per panel, 20 prompts on x-axis) rather than a single packed chart.
- **AST kernel sensitivity.** The AST-null result is the single most reviewer-attractive finding and the most likely to draw scrutiny. Two robustness checks are worth running before camera-ready: (i) APTED with a different cost configuration (cost-of-rename = 0 vs the default 1) to verify the result is not driven by our normalization choice; (ii) an alternative structural kernel (PQ-grams or subtree kernel) to verify it is not APTED-specific. Both should take < 1 hour to wire in.
- **Funnel diagram.** Reviewers respond well to a single "funnel" visualization showing the four pools' Vendi/*N* lined up — null > human > inter-LLM > intra-LLM. This is in the supplementary `fig_pillar1_vendi.png`; if there is room in the body, lifting it into §5 would help.
