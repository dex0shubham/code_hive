#!/usr/bin/env python3
"""
proof_homogeneity.py
====================
Three-pillar standalone proof of code-LLM output homogeneity.

This script is INDEPENDENT of the rest of the code-hivemind pipeline.
It only reads JSONL files from `--raw-dir` (LLM responses) and optionally
`--human-dir` (human-written reference solutions). It writes everything to
`--out-dir`. It does not import diversity_metrics.py or pipeline.py.

Pillars
-------
1. Effective-sample-size collapse via Vendi Score
   - Token n-gram Jaccard kernel (surface)
   - AST kernel (APTED tree-edit-distance, with AST-node-bag Jaccard fallback)
   - UniXcoder (microsoft/unixcoder-base) embedding cosine kernel
   - Both raw Vendi (effective N) AND normalized Vendi (vendi/N, size-fair) are
     reported.
   - Computed for four pools per prompt:
        shuffled-null  > human  > inter-LLM  > intra-LLM      (expected order)
2. Comparative ordering vs. human baseline
   - Per-prompt paired delta (Sim_inter-LLM - Sim_human) under each kernel
   - Cluster bootstrap CIs (resample at the prompt level)
   - Stouffer's-Z combined p-value across prompts
3. Functional convergence on passing solutions
   - Inline test harnesses for AL-01 (anagram groups), AL-03 (cycle detection
     in directed graphs), AL-05 (balanced parentheses)
   - Subprocess sandbox runner with timeout
   - Trace clustering -> trace-Vendi (perplexity over output-cluster sizes)

Outputs (under --out-dir)
-------------------------
  summary.json                  full numeric results
  summary.md                    human-readable summary
  fig_pillar1_vendi.png         grouped bars: pools x kernels x prompts
  fig_pillar2_ordering.png      paired scatter, LLM vs human Vendi
  fig_pillar3_traces.png        stacked trace-cluster bars per prompt

Usage
-----
  # First test, default settings (AL-01, AL-03, AL-05 at temp=1.0)
  python proof_homogeneity.py

  # Specify everything
  python proof_homogeneity.py \
      --prompts AL-01 AL-03 AL-05 \
      --temperature 1.0 \
      --raw-dir results/raw_responses \
      --human-dir results/human_baseline \
      --out-dir results/proof/v1 \
      --device cpu

Dependencies (in addition to existing requirements.txt)
-------------------------------------------------------
  pip install transformers torch          # for UniXcoder
  pip install apted                       # optional, for tree-edit distance
"""

from __future__ import annotations

import argparse
import ast
import builtins as _builtins
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Suppress noisy warnings from transformers/torch on first load
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

ROOT = Path(__file__).resolve().parent
BUILTIN_NAMES = set(dir(_builtins))


# ─────────────────────────────────────────────────────────────────────
# 1. Code extraction
# ─────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:python|py|javascript|js)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str:
    """Pull code from markdown fences; fall back to raw text."""
    if not text:
        return ""
    m = _FENCE_RE.findall(text)
    return "\n".join(m) if m else text.strip()


# ─────────────────────────────────────────────────────────────────────
# 2. Data loading
# ─────────────────────────────────────────────────────────────────────

def load_llm_responses(raw_dir: Path, prompt_ids: list[str],
                       temperature: float | None) -> dict[str, list[dict]]:
    """Return {prompt_id: [response_dict, ...]} filtered by temperature."""
    by_prompt: dict[str, list[dict]] = {}
    for pid in prompt_ids:
        fp = raw_dir / f"{pid}.jsonl"
        rows: list[dict] = []
        if not fp.exists():
            print(f"  [warn] missing {fp}; skipping {pid}")
            by_prompt[pid] = []
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if temperature is not None:
                    if abs(rec.get("temperature", 0.0) - temperature) > 1e-6:
                        continue
                rows.append(rec)
        by_prompt[pid] = rows
    return by_prompt


def load_human_responses(human_dir: Path | None,
                         prompt_ids: list[str]) -> dict[str, list[dict]]:
    """Return {prompt_id: [human_dict, ...]} or empty dict if not provided.

    Each human record should have at least 'response_text' or 'code' field.
    """
    out: dict[str, list[dict]] = {pid: [] for pid in prompt_ids}
    if human_dir is None or not human_dir.exists():
        return out
    for pid in prompt_ids:
        fp = human_dir / f"{pid}.jsonl"
        if not fp.exists():
            continue
        with open(fp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                out[pid].append(json.loads(line))
    return out


# ─────────────────────────────────────────────────────────────────────
# 3. Kernels
# ─────────────────────────────────────────────────────────────────────

def _token_ngrams(text: str, n: int = 3) -> set[tuple[str, ...]]:
    toks = re.findall(r"[A-Za-z_]\w*|\d+|[^\s\w]", text)
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)} if len(toks) >= n else set()


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def kernel_token(codes: list[str], n: int = 3) -> np.ndarray:
    """Pairwise token n-gram Jaccard similarity matrix (NxN, in [0,1])."""
    grams = [_token_ngrams(c, n) for c in codes]
    N = len(codes)
    K = np.eye(N, dtype=float)
    for i in range(N):
        for j in range(i + 1, N):
            K[i, j] = K[j, i] = _jaccard(grams[i], grams[j])
    return K


def _ast_node_bag(code: str) -> Counter:
    """Multiset of AST node-type names (fallback when APTED unavailable)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return Counter()
    return Counter(type(n).__name__ for n in ast.walk(tree))


def _apted_distance(code_a: str, code_b: str) -> tuple[float, int]:
    """Tree-edit distance via APTED. Returns (distance, max_size)."""
    from apted import APTED, Config  # type: ignore
    from apted.helpers import Tree  # type: ignore

    def to_bracket(node: ast.AST) -> str:
        # APTED's Tree.from_text expects "{label{child1}{child2}...}"
        children = [to_bracket(c) for c in ast.iter_child_nodes(node)]
        return "{" + type(node).__name__ + "".join(children) + "}"

    try:
        ta = Tree.from_text(to_bracket(ast.parse(code_a)))
        tb = Tree.from_text(to_bracket(ast.parse(code_b)))
    except SyntaxError:
        return float("inf"), 1

    def _size(node: ast.AST) -> int:
        return 1 + sum(_size(c) for c in ast.iter_child_nodes(node))

    sa = _size(ast.parse(code_a))
    sb = _size(ast.parse(code_b))
    d = APTED(ta, tb, Config()).compute_edit_distance()
    return float(d), max(sa, sb, 1)


def kernel_ast(codes: list[str]) -> np.ndarray:
    """AST similarity matrix. Tries APTED tree-edit-distance, falls back to
    AST node-bag Jaccard. Sim = 1 - normalized_TED, in [0, 1]."""
    N = len(codes)
    K = np.eye(N, dtype=float)
    try:
        import apted  # noqa: F401
        use_apted = True
    except ImportError:
        use_apted = False
        bags = [_ast_node_bag(c) for c in codes]

    if use_apted:
        for i in range(N):
            for j in range(i + 1, N):
                d, m = _apted_distance(codes[i], codes[j])
                if not math.isfinite(d):
                    K[i, j] = K[j, i] = 0.0
                else:
                    K[i, j] = K[j, i] = max(0.0, 1.0 - d / m)
    else:
        for i in range(N):
            for j in range(i + 1, N):
                a, b = bags[i], bags[j]
                if not a and not b:
                    K[i, j] = K[j, i] = 1.0
                else:
                    inter = sum((a & b).values())
                    union = sum((a | b).values())
                    K[i, j] = K[j, i] = inter / union if union else 0.0
    return K


def kernel_codebert(codes: list[str], device: str = "cpu",
                    max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """Cosine-similarity matrix using microsoft/codebert-base. Thin wrapper
    over kernel_unixcoder with a different model id."""
    return kernel_unixcoder(codes, device=device,
                            model_name="microsoft/codebert-base",
                            max_length=max_length, batch_size=batch_size)


def kernel_tfidf(codes: list[str],
                 ngram_range: tuple = (3, 5),
                 max_features: int = 50000) -> np.ndarray:
    """Cosine-similarity matrix from sklearn TF-IDF features.

    NO transformers / NO torch dependency. Uses character n-grams (3–5) so
    it's robust to identifier choice and tokenization. Tries `char_wb` first
    (character n-grams within word boundaries — usually higher quality), and
    falls back to plain `char` if the documents lack word characters
    (empty/whitespace-only responses can leave char_wb with empty vocab).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
    if not codes:
        return np.zeros((0, 0))

    # Replace empty/tiny docs with a sentinel so they cluster with each other
    # (rather than being silently dropped from the vocabulary).
    safe: list[str] = []
    for c in codes:
        s = (c or "").strip()
        if len(s) < 3:
            s = "EMPTY_OR_TINY_RESPONSE"
        safe.append(s)

    for attempt in ("char_wb", "char"):
        try:
            vec = TfidfVectorizer(
                analyzer=attempt, ngram_range=ngram_range,
                max_features=max_features, lowercase=False, sublinear_tf=True,
            )
            X = vec.fit_transform(safe)
            return np.clip(cosine_similarity(X), 0.0, 1.0)
        except ValueError:
            continue
    # Ultimate fallback — should be unreachable but keeps the run going.
    print(f"  [warn] kernel_tfidf: could not vectorize {len(codes)} docs; "
          f"falling back to identity matrix.")
    return np.eye(len(codes))


def kernel_unixcoder(codes: list[str], device: str = "cpu",
                     model_name: str = "microsoft/unixcoder-base",
                     max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """Cosine-similarity matrix of mean-pooled UniXcoder embeddings.

    UniXcoder (Guo et al. 2022) is a RoBERTa-architecture model fine-tuned on
    code; it consistently outperforms CodeBERT on code-similarity tasks. Used
    here in encoder-only mode: tokenize -> forward -> mean-pool last hidden
    state with attention mask -> L2-normalize -> cosine.

    We import RobertaModel/RobertaTokenizer explicitly because some installs
    of transformers fail to auto-resolve RobertaModel through AutoModel
    (raising "Could not import module 'RobertaModel'"), which is the specific
    issue we hit. Explicit import bypasses the lazy auto-factory.
    """
    import torch  # type: ignore

    tokenizer = None
    model = None
    last_err: Exception | None = None
    last_tb: str = ""

    def _save_err(e):
        import traceback
        nonlocal last_err, last_tb
        last_err = e
        last_tb = traceback.format_exc()

    # Path A: direct submodule import (bypasses the lazy auto-factory entirely;
    # this is what surfaces the *real* underlying error if torch/safetensors is
    # broken, instead of the generic "Could not import module" message).
    try:
        from transformers.models.roberta.modeling_roberta import RobertaModel  # type: ignore
        from transformers.models.roberta.tokenization_roberta import RobertaTokenizer  # type: ignore
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        model = RobertaModel.from_pretrained(model_name).to(device).eval()
    except Exception as e:
        _save_err(e)

    # Path B: explicit top-level Roberta import.
    if model is None:
        try:
            from transformers import RobertaModel, RobertaTokenizer  # type: ignore
            tokenizer = RobertaTokenizer.from_pretrained(model_name)
            model = RobertaModel.from_pretrained(model_name).to(device).eval()
        except Exception as e:
            _save_err(e)

    # Path C: AutoModel.
    if model is None:
        try:
            from transformers import AutoTokenizer, AutoModel  # type: ignore
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device).eval()
        except Exception as e:
            _save_err(e)

    if model is None:
        raise RuntimeError(
            "Failed to load UniXcoder. The underlying traceback is:\n"
            f"{last_tb}\n"
            "Common fixes:\n"
            "  pip install --upgrade transformers tokenizers safetensors\n"
            "  pip install --upgrade torch\n"
            "Or run with --skip-embedding to skip the embedding kernel.\n"
        )

    embs: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            batch = [c if c else " " for c in codes[i:i + batch_size]]
            enc = tokenizer(batch, padding=True, truncation=True,
                            max_length=max_length, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            embs.append(pooled.cpu().numpy())
    E = np.vstack(embs)
    norm = np.linalg.norm(E, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-10, None)
    En = E / norm
    K = En @ En.T
    # Clip any tiny negatives from FP error so the matrix is PSD
    return np.clip(K, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────
# 4. Vendi score
# ─────────────────────────────────────────────────────────────────────

def vendi_score(K: np.ndarray) -> float:
    """Vendi Score = exp(Shannon entropy of eigenvalues of K/N).

    K must be a square PSD similarity matrix with K[i,i]=1 and K[i,j] in [0,1].
    Returns the "effective number of distinct elements" in the pool.
    """
    K = np.asarray(K, dtype=float)
    N = K.shape[0]
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    # Symmetrize for numerical robustness
    K = 0.5 * (K + K.T)
    # eigvalsh requires symmetric; the divide-by-N gives trace 1 when diag=1.
    w = np.linalg.eigvalsh(K / N)
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 1.0
    p = w / s
    # Use natural log so exp(H) is the perplexity; matches the original paper.
    H = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(H))


# ─────────────────────────────────────────────────────────────────────
# 5. Pool builders
# ─────────────────────────────────────────────────────────────────────

@dataclass
class PromptPools:
    prompt_id: str
    inter_llm_codes: list[str]              # 1 sample per (model, ...)
    intra_llm_codes: dict[str, list[str]]   # model_display -> list of codes
    human_codes: list[str]                  # may be empty
    null_codes: list[str]                   # cross-prompt shuffled


def build_pools(prompt_id: str,
                llm_responses: list[dict],
                human_responses: list[dict],
                all_other_llm_codes: list[str],
                rng: np.random.Generator,
                samples_per_model_for_inter: int = 1,
                cap_per_pool: int = 60) -> PromptPools:
    """Construct the four pools used by Pillar 1 for this prompt.

    inter_llm_codes:  for each model, draw `samples_per_model_for_inter` (default 1)
                      randomly. Capped at cap_per_pool overall.
    intra_llm_codes:  full per-model lists (capped at cap_per_pool each).
    human_codes:      from human_responses (capped at cap_per_pool).
    null_codes:       random draw from all_other_llm_codes (other prompts);
                      capped at cap_per_pool.
    """
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in llm_responses:
        by_model[r.get("model_display", "?")].append(r)

    # inter-LLM pool: 1 (or k) per model
    inter: list[str] = []
    for model, lst in by_model.items():
        if not lst:
            continue
        idxs = rng.choice(len(lst), size=min(samples_per_model_for_inter, len(lst)),
                          replace=False)
        for i in idxs:
            inter.append(extract_code(lst[int(i)].get("response_text", "")))
    if len(inter) > cap_per_pool:
        idxs = rng.choice(len(inter), cap_per_pool, replace=False)
        inter = [inter[int(i)] for i in idxs]

    # intra-LLM pools: per-model
    intra: dict[str, list[str]] = {}
    for model, lst in by_model.items():
        codes = [extract_code(r.get("response_text", "")) for r in lst]
        if len(codes) > cap_per_pool:
            idxs = rng.choice(len(codes), cap_per_pool, replace=False)
            codes = [codes[int(i)] for i in idxs]
        intra[model] = codes

    # human pool
    human: list[str] = []
    for r in human_responses:
        c = r.get("code") or extract_code(r.get("response_text", ""))
        if c:
            human.append(c)
    if len(human) > cap_per_pool:
        idxs = rng.choice(len(human), cap_per_pool, replace=False)
        human = [human[int(i)] for i in idxs]

    # null pool: random across-prompt LLM codes, same size as inter
    null_n = min(len(inter), cap_per_pool, len(all_other_llm_codes))
    null: list[str] = []
    if null_n > 0 and all_other_llm_codes:
        idxs = rng.choice(len(all_other_llm_codes), null_n, replace=False)
        null = [all_other_llm_codes[int(i)] for i in idxs]

    return PromptPools(prompt_id, inter, intra, human, null)


# ─────────────────────────────────────────────────────────────────────
# 6. Pillar 1: Vendi over kernels x pools
# ─────────────────────────────────────────────────────────────────────

KERNEL_FNS: dict[str, Callable[..., np.ndarray]] = {}


def _register_kernels(embedder: str, device: str):
    """Register the kernels actually computed for Pillar 1.
    `embedder` ∈ {"unixcoder", "codebert", "tfidf", "none"}."""
    KERNEL_FNS["token"] = lambda codes: kernel_token(codes, n=3)
    KERNEL_FNS["ast"] = lambda codes: kernel_ast(codes)
    if embedder == "unixcoder":
        KERNEL_FNS["unixcoder"] = lambda codes: kernel_unixcoder(codes, device=device)
    elif embedder == "codebert":
        KERNEL_FNS["codebert"] = lambda codes: kernel_codebert(codes, device=device)
    elif embedder == "tfidf":
        KERNEL_FNS["tfidf"] = lambda codes: kernel_tfidf(codes)
    elif embedder == "none":
        pass  # token + ast only
    else:
        raise ValueError(f"unknown --embedder: {embedder!r}")


def _pool_vendi_with_ci(codes: list[str], kernel_name: str,
                        n_boot: int, rng: np.random.Generator) -> dict:
    """Vendi for one pool + per-pool sample-bootstrap CI.

    Bootstrap is sample-level here (resample with replacement WITHIN the pool);
    cluster-bootstrap across prompts is handled separately by the caller.

    Returns both raw Vendi (effective N distinct) AND normalized Vendi (per-
    sample effective uniqueness, in [1/N, 1]). The normalized form is what you
    compare across pools of different sizes.
    """
    if len(codes) < 2:
        v = float(len(codes))
        return {"vendi": v, "ci": [v, v], "n": len(codes),
                "vendi_norm": 1.0 if len(codes) >= 1 else 0.0,
                "ci_norm": [1.0 if len(codes) >= 1 else 0.0] * 2}
    K = KERNEL_FNS[kernel_name](codes)
    point = vendi_score(K)
    boots: list[float] = []
    N = K.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        Kb = K[np.ix_(idx, idx)]
        boots.append(vendi_score(Kb))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot) - 1]
    return {
        "vendi": float(point), "ci": [float(lo), float(hi)], "n": N,
        "vendi_norm": float(point) / N,
        "ci_norm": [float(lo) / N, float(hi) / N],
    }


def run_pillar1(pools_by_prompt: dict[str, PromptPools],
                kernels: list[str], n_boot: int,
                seed: int = 42) -> dict:
    """Per (prompt, kernel, pool) Vendi score and bootstrap CI."""
    rng = np.random.default_rng(seed)
    out: dict = {"per_prompt": {}, "kernels": kernels}
    for pid, pools in pools_by_prompt.items():
        per_kernel: dict = {}
        for kname in kernels:
            print(f"  [P1] {pid} kernel={kname}")
            per_pool: dict = {}
            per_pool["inter_llm"] = _pool_vendi_with_ci(
                pools.inter_llm_codes, kname, n_boot, rng)
            per_pool["null"] = _pool_vendi_with_ci(
                pools.null_codes, kname, n_boot, rng)
            if pools.human_codes:
                per_pool["human"] = _pool_vendi_with_ci(
                    pools.human_codes, kname, n_boot, rng)
            else:
                per_pool["human"] = None
            intra: dict = {}
            for model, codes in pools.intra_llm_codes.items():
                intra[model] = _pool_vendi_with_ci(codes, kname, n_boot, rng)
            # mean-of-models intra value (with model-bootstrap CI)
            if intra:
                vs = [v["vendi"] for v in intra.values()]
                vs_norm = [v["vendi_norm"] for v in intra.values()]
                per_pool["intra_llm_mean"] = {
                    "vendi": float(np.mean(vs)),
                    "ci": [float(np.percentile(vs, 2.5)),
                           float(np.percentile(vs, 97.5))],
                    "vendi_norm": float(np.mean(vs_norm)),
                    "ci_norm": [float(np.percentile(vs_norm, 2.5)),
                                float(np.percentile(vs_norm, 97.5))],
                    "n_models": len(vs),
                }
                per_pool["intra_llm_per_model"] = intra
            else:
                per_pool["intra_llm_mean"] = None
                per_pool["intra_llm_per_model"] = {}
            per_kernel[kname] = per_pool
        out["per_prompt"][pid] = per_kernel
    return out


# ─────────────────────────────────────────────────────────────────────
# 7. Pillar 2: ordering test (LLM vs human)
# ─────────────────────────────────────────────────────────────────────

def _stouffer(z_scores: list[float]) -> float:
    """Stouffer's-Z combiner. Returns combined two-sided p-value."""
    if not z_scores:
        return float("nan")
    z = float(np.sum(z_scores) / np.sqrt(len(z_scores)))
    # two-sided p using normal CDF approximation
    p = math.erfc(abs(z) / math.sqrt(2))
    return p


def _bootstrap_delta(values_a: list[float], values_b: list[float],
                     n_boot: int, rng: np.random.Generator) -> dict:
    """Paired bootstrap on (a_i - b_i). Returns dict with mean, CI, p-value."""
    if not values_a or len(values_a) != len(values_b):
        return {"delta_mean": None, "ci": None, "p": None, "n_pairs": 0}
    deltas = np.array(values_a) - np.array(values_b)
    point = float(deltas.mean())
    boots = []
    n = len(deltas)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(deltas[idx].mean())
    boots.sort()
    lo = float(boots[int(0.025 * n_boot)])
    hi = float(boots[int(0.975 * n_boot) - 1])
    # one-tailed p: H0 = LLM no less diverse than human  ⇒ delta >= 0 (LLM Vendi ≥ human Vendi)
    # We want to show LLM Vendi < human Vendi (delta < 0)
    p_one_tail = float(np.mean(np.array(boots) >= 0)) if point < 0 else 1.0
    return {"delta_mean": point, "ci": [lo, hi], "p": p_one_tail,
            "n_pairs": n}


def run_pillar2(pillar1: dict, n_boot: int, seed: int = 43) -> dict:
    """Ordering: is inter-LLM Vendi < human Vendi across prompts?

    Uses the Vendi point estimates already computed in Pillar 1 (per-prompt,
    per-kernel) and applies a paired bootstrap across prompts.
    """
    rng = np.random.default_rng(seed)
    per_prompt = pillar1["per_prompt"]
    kernels = pillar1["kernels"]

    paired_pids = [pid for pid, kr in per_prompt.items()
                   if all(kr[k].get("human") is not None for k in kernels)]
    if not paired_pids:
        return {"status": "skipped",
                "reason": "no prompt has both LLM and human pools",
                "kernels": kernels}

    out: dict = {"status": "ok", "n_paired_prompts": len(paired_pids),
                 "paired_prompts": paired_pids, "per_kernel": {}}

    for kname in kernels:
        # Raw Vendi vectors (length = n_paired_prompts)
        llm_vals    = [per_prompt[pid][kname]["inter_llm"]["vendi"]      for pid in paired_pids]
        human_vals  = [per_prompt[pid][kname]["human"]["vendi"]          for pid in paired_pids]
        intra_vals  = [per_prompt[pid][kname]["intra_llm_mean"]["vendi"]
                       if per_prompt[pid][kname].get("intra_llm_mean") else float("nan")
                       for pid in paired_pids]
        # Normalized Vendi vectors (per-sample effective uniqueness in [1/N, 1])
        llm_norm    = [per_prompt[pid][kname]["inter_llm"]["vendi_norm"]      for pid in paired_pids]
        human_norm  = [per_prompt[pid][kname]["human"]["vendi_norm"]          for pid in paired_pids]
        intra_norm  = [per_prompt[pid][kname]["intra_llm_mean"]["vendi_norm"]
                       if per_prompt[pid][kname].get("intra_llm_mean") else float("nan")
                       for pid in paired_pids]

        # Raw deltas (sample-size-confounded; useful as a sanity check)
        prim = _bootstrap_delta(llm_vals, human_vals, n_boot, rng)
        sec  = _bootstrap_delta(intra_vals, human_vals, n_boot, rng)
        # Normalized deltas (size-fair; this is the defensible headline number)
        prim_norm = _bootstrap_delta(llm_norm, human_norm, n_boot, rng)
        sec_norm  = _bootstrap_delta(intra_norm, human_norm, n_boot, rng)

        out["per_kernel"][kname] = {
            "llm_vs_human": prim,
            "intra_vs_human": sec,
            "llm_vs_human_norm": prim_norm,
            "intra_vs_human_norm": sec_norm,
            "llm_vendi_mean":    float(np.mean(llm_vals)),
            "human_vendi_mean":  float(np.mean(human_vals)),
            "intra_vendi_mean":  float(np.nanmean(intra_vals)),
            "llm_vendi_norm_mean":   float(np.mean(llm_norm)),
            "human_vendi_norm_mean": float(np.mean(human_norm)),
            "intra_vendi_norm_mean": float(np.nanmean(intra_norm)),
        }
    return out


# ─────────────────────────────────────────────────────────────────────
# 8. Pillar 3: functional convergence
# ─────────────────────────────────────────────────────────────────────

# ── Test harness registry ────────────────────────────────────────────
# Each harness: list of test inputs, entry-point keywords, canonicalizer.
# Inputs must be JSON-serializable; the wrapper applies fn(*input) when the
# input is a list, fn(input) otherwise.

def _canon_anagram_groups(out: Any) -> Any:
    """Canonical form of an anagram-grouping output.

    Accepts list-of-lists, list-of-tuples, list-of-sets, or a dict whose values
    are groups. Returns a frozenset of frozensets of words (sorted-deterministic
    to be JSON-safe in the caller).
    """
    if out is None:
        return None
    if isinstance(out, dict):
        groups = list(out.values())
    elif isinstance(out, (list, tuple)):
        groups = list(out)
    else:
        return repr(out)
    canon = []
    for g in groups:
        if isinstance(g, str):
            canon.append((g,))
        elif isinstance(g, (list, tuple, set, frozenset)):
            canon.append(tuple(sorted(map(str, g))))
        else:
            canon.append((str(g),))
    canon.sort()
    return tuple(canon)


def _canon_bool(out: Any) -> Any:
    if isinstance(out, bool):
        return bool(out)
    if isinstance(out, (int, float)):
        return bool(out)
    if isinstance(out, str):
        return out.strip().lower() in {"true", "1", "yes"}
    return repr(out)


def _canon_string_set(out: Any) -> Any:
    if out is None:
        return None
    if isinstance(out, str):
        return (out,)
    if isinstance(out, (list, tuple, set, frozenset)):
        return tuple(sorted(str(x) for x in out))
    return repr(out)


HARNESSES: dict[str, dict] = {
    "AL-01": {
        "entry_keywords": ["anagram", "group"],
        # input = list-of-words
        "test_inputs": [
            [["eat", "tea", "tan", "ate", "nat", "bat"]],
            [["listen", "silent", "enlist", "tinsel", "abc", "cba", "cab"]],
            [["", ""]],
            [["a"]],
        ],
        "canon": _canon_anagram_groups,
    },
    "AL-03": {
        "entry_keywords": ["cycle", "has_cycle", "detect", "is_cyclic"],
        # input = adjacency dict
        "test_inputs": [
            [{"a": ["b"], "b": ["c"], "c": []}],                      # acyclic
            [{"a": ["b"], "b": ["c"], "c": ["a"]}],                   # cyclic
            [{"1": ["2", "3"], "2": ["4"], "3": ["4"], "4": []}],     # DAG
            [{"x": ["x"]}],                                           # self-loop
            [{}],                                                     # empty
        ],
        "canon": _canon_bool,
    },
    "AL-05": {
        "entry_keywords": ["paren", "generate", "balanced", "valid"],
        # input = integer n (number of pairs)
        "test_inputs": [[0], [1], [2], [3]],
        "canon": _canon_string_set,
    },
}


_WRAPPER_SRC = r'''
import sys, os, json, types, inspect
USER_PATH = sys.argv[1]
TEST_INPUTS = json.loads(sys.argv[2])
ENTRY_KEYWORDS = json.loads(sys.argv[3])

with open(USER_PATH, "r", encoding="utf-8") as f:
    USER_CODE = f.read()

# Mute stdout during exec so any top-level prints in user code don't pollute
# the JSON line we emit at the end.
class _Mute:
    def write(self, *a, **k): pass
    def flush(self, *a, **k): pass
saved_stdout = sys.stdout
sys.stdout = _Mute()

# Use a non-"__main__" module name so any  if __name__ == "__main__":  demo
# blocks the LLM appended will NOT execute (those demos often crash because
# their hard-coded inputs don't match our test cases).
ns = {"__name__": "__sandbox_runner__"}
INITIAL_KEYS = set(ns.keys())

try:
    exec(USER_CODE, ns)
except SystemExit:
    pass
except BaseException as e:
    sys.stdout = saved_stdout
    print(json.dumps({"status": "exec_fail",
                       "error": type(e).__name__ + ": " + str(e)[:200]}))
    sys.exit(0)
sys.stdout = saved_stdout

# Newly-defined names = anything added to the namespace by user code.
new_keys = [k for k in ns.keys()
            if k not in INITIAL_KEYS and not k.startswith("_")]

# Build a candidate list of callables (functions, methods, partials),
# prioritising those whose name matches an entry keyword.
def _score(name):
    nl = name.lower()
    for i, kw in enumerate(ENTRY_KEYWORDS):
        if kw in nl:
            return i
    return len(ENTRY_KEYWORDS) + 1

func_candidates = []
class_candidates = []
for name in new_keys:
    val = ns[name]
    if isinstance(val, type):
        class_candidates.append((name, val))
    elif callable(val):
        func_candidates.append((name, val))

func_candidates.sort(key=lambda nv: _score(nv[0]))
class_candidates.sort(key=lambda nv: _score(nv[0]))

fn = None
fn_label = None
fn_kind = None

# Path A: a free function defined at top level.
if func_candidates:
    fn_label, fn = func_candidates[0]
    fn_kind = "function"

# Path B: a class containing an entry-method (e.g. LeetCode Solution.groupAnagrams).
if fn is None and class_candidates:
    for cname, cls in class_candidates:
        try:
            inst = cls()
        except Exception:
            continue
        method_candidates = []
        for mname, mval in inspect.getmembers(inst, predicate=callable):
            if mname.startswith("_"):
                continue
            method_candidates.append((mname, mval))
        method_candidates.sort(key=lambda nv: _score(nv[0]))
        if method_candidates:
            mname, mval = method_candidates[0]
            fn = mval
            fn_label = cname + "." + mname
            fn_kind = "method"
            break

if fn is None:
    print(json.dumps({"status": "no_entry",
                       "new_names": new_keys,
                       "n_funcs": len(func_candidates),
                       "n_classes": len(class_candidates)}))
    sys.exit(0)

results = []
sys.stdout = _Mute()  # also mute during fn() calls
for inp in TEST_INPUTS:
    try:
        if isinstance(inp, list):
            out = fn(*inp)
        else:
            out = fn(inp)
        try:
            json.dumps(out)
            ok_payload = {"ok": True, "out": out}
        except (TypeError, OverflowError):
            ok_payload = {"ok": True, "out_repr": repr(out)[:1000]}
        results.append(ok_payload)
    except BaseException as e:
        results.append({"ok": False,
                         "error": type(e).__name__ + ": " + str(e)[:200]})
sys.stdout = saved_stdout

print(json.dumps({"status": "ok",
                   "entry_function": fn_label,
                   "entry_kind": fn_kind,
                   "results": results}))
'''


def _write_wrapper_once(out_dir: Path) -> Path:
    p = out_dir / "_run_user_code.py"
    p.write_text(_WRAPPER_SRC, encoding="utf-8")  # always overwrite to avoid stale wrappers
    return p


def run_one_sample(code: str, harness: dict, wrapper_path: Path,
                   timeout: float = 6.0) -> dict:
    """Execute one user code sample against the harness inputs in a subprocess."""
    if not code.strip():
        return {"status": "empty"}
    code_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False,
                                         encoding="utf-8") as f:
            f.write(code)
            code_path = f.name
        try:
            proc = subprocess.run(
                [sys.executable, str(wrapper_path), code_path,
                 json.dumps(harness["test_inputs"]),
                 json.dumps(harness["entry_keywords"])],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout"}
        if proc.returncode != 0:
            return {"status": "crash", "stderr": (proc.stderr or "")[:500]}
        # The wrapper always prints exactly one JSON line on stdout
        line = (proc.stdout or "").strip().splitlines()
        if not line:
            return {"status": "no_output"}
        try:
            return json.loads(line[-1])
        except json.JSONDecodeError:
            return {"status": "parse_fail", "stdout": (proc.stdout or "")[:500]}
    finally:
        if code_path:
            try:
                os.unlink(code_path)
            except OSError:
                pass


def _trace_signature(harness: dict, exec_result: dict) -> tuple | None:
    """Reduce a successful exec_result into a comparable trace tuple, or None
    if the sample didn't pass enough cases to be considered 'working'."""
    if exec_result.get("status") != "ok":
        return None
    canon_fn = harness["canon"]
    sig: list = []
    pass_count = 0
    for r in exec_result.get("results", []):
        if r.get("ok"):
            pass_count += 1
            sig.append(("OK", json.dumps(canon_fn(r.get("out")), default=str)))
        else:
            sig.append(("ERR", str(r.get("error"))[:80]))
    if pass_count == 0:
        return None
    return tuple(sig)


def trace_vendi_from_signatures(sigs: list[tuple | None]) -> dict:
    """Vendi with delta kernel == perplexity over trace clusters (passing only)."""
    passing = [s for s in sigs if s is not None]
    counter = Counter(passing)
    n = len(passing)
    if n == 0:
        return {"vendi": 0.0, "n_passing": 0, "n_total": len(sigs),
                "pass_rate": 0.0, "clusters": {}}
    probs = np.array([c / n for c in counter.values()], dtype=float)
    H = float(-np.sum(probs * np.log(probs + 1e-12)))
    clusters = {hashlib.md5(json.dumps(k, default=str).encode()).hexdigest()[:10]: c
                for k, c in counter.most_common()}
    return {
        "vendi": float(np.exp(H)),
        "n_passing": n,
        "n_total": len(sigs),
        "pass_rate": n / len(sigs) if sigs else 0.0,
        "clusters": clusters,
    }


def run_pillar3(pools_by_prompt: dict[str, PromptPools],
                out_dir: Path, timeout: float = 6.0) -> dict:
    wrapper = _write_wrapper_once(out_dir)
    out: dict = {"per_prompt": {}, "harness_prompts": list(HARNESSES.keys())}
    for pid, pools in pools_by_prompt.items():
        if pid not in HARNESSES:
            out["per_prompt"][pid] = {"status": "no_harness"}
            continue
        h = HARNESSES[pid]
        prompt_out: dict = {"status": "ok"}

        def _process(label: str, codes: list[str]) -> dict:
            sigs: list[tuple | None] = []
            for c in codes:
                r = run_one_sample(c, h, wrapper, timeout=timeout)
                sigs.append(_trace_signature(h, r))
            return trace_vendi_from_signatures(sigs)

        print(f"  [P3] {pid} inter-LLM ({len(pools.inter_llm_codes)})")
        prompt_out["inter_llm"] = _process("inter_llm", pools.inter_llm_codes)
        if pools.human_codes:
            print(f"  [P3] {pid} human ({len(pools.human_codes)})")
            prompt_out["human"] = _process("human", pools.human_codes)
        else:
            prompt_out["human"] = None
        # Also run a small intra-LLM sample (use first model for speed)
        if pools.intra_llm_codes:
            first_model = next(iter(pools.intra_llm_codes))
            codes = pools.intra_llm_codes[first_model][:30]
            print(f"  [P3] {pid} intra-LLM[{first_model}] ({len(codes)})")
            prompt_out["intra_llm_first_model"] = {
                "model": first_model,
                **_process("intra_llm", codes),
            }
        out["per_prompt"][pid] = prompt_out
    return out


# ─────────────────────────────────────────────────────────────────────
# 9. Figures
# ─────────────────────────────────────────────────────────────────────

def make_figures(pillar1: dict, pillar2: dict, pillar3: dict,
                 out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [warn] matplotlib not available; skipping figures")
        return

    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 200,
                         "savefig.bbox": "tight", "font.size": 9})

    # ── Fig 1: Vendi per (prompt, kernel, pool) ──
    kernels = pillar1["kernels"]
    prompts = list(pillar1["per_prompt"].keys())
    try:
        _make_fig1(plt, pillar1, kernels, prompts, out_dir)
    except Exception as e:
        print(f"  [warn] fig_pillar1 failed: {type(e).__name__}: {e}")
    try:
        _make_fig2(plt, pillar1, pillar2, kernels, out_dir)
    except Exception as e:
        print(f"  [warn] fig_pillar2 failed: {type(e).__name__}: {e}")
    try:
        _make_fig3(plt, pillar3, out_dir)
    except Exception as e:
        print(f"  [warn] fig_pillar3 failed: {type(e).__name__}: {e}")


def _make_fig1(plt, pillar1, kernels, prompts, out_dir):
    """Pillar 1 — Vendi per (prompt, kernel, pool) grouped bars.
    Two rows: top = raw Vendi (size-confounded), bottom = normalized Vendi
    (per-sample effective uniqueness, the size-fair view).
    """
    if not prompts:
        return
    fig, axes = plt.subplots(2, len(kernels),
                             figsize=(5 * len(kernels), 7),
                             squeeze=False)
    pool_labels = ["null", "human", "intra_llm_mean", "inter_llm"]
    pool_colors = {"null": "#888", "human": "#1f77b4",
                   "intra_llm_mean": "#2ca02c", "inter_llm": "#d62728"}

    def _plot_row(row_axes, value_key, ci_key, ylabel, hline):
        for ax, kname in zip(row_axes, kernels):
            x = np.arange(len(prompts))
            w = 0.18
            for i, pl in enumerate(pool_labels):
                vals, lo, hi = [], [], []
                for pid in prompts:
                    pk = pillar1["per_prompt"][pid][kname].get(pl)
                    if pk is None:
                        vals.append(np.nan); lo.append(0.0); hi.append(0.0)
                    else:
                        v = pk.get(value_key)
                        if v is None:
                            vals.append(np.nan); lo.append(0.0); hi.append(0.0)
                            continue
                        vals.append(v)
                        ci = pk.get(ci_key, [v, v])
                        lo.append(max(0.0, v - ci[0]))
                        hi.append(max(0.0, ci[1] - v))
                offset = (i - 1.5) * w
                ax.bar(x + offset, vals, w, label=pl,
                       color=pool_colors[pl], alpha=0.85,
                       yerr=[lo, hi], capsize=2)
            ax.set_title(f"kernel = {kname}")
            ax.set_xticks(x)
            ax.set_xticklabels(prompts, rotation=0, fontsize=8)
            ax.set_ylabel(ylabel)
            if hline is not None:
                ax.axhline(hline, color="k", linewidth=0.5, linestyle=":")

    _plot_row(axes[0], "vendi", "ci",
              "Vendi (effective N distinct)", 1.0)
    _plot_row(axes[1], "vendi_norm", "ci_norm",
              "Vendi / N  (per-sample uniqueness)", None)
    axes[0][-1].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Pillar 1 — top: raw Vendi (size-confounded);  "
                 "bottom: normalized (size-fair)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pillar1_vendi.png")
    plt.close(fig)


def _make_fig2(plt, pillar1, pillar2, kernels, out_dir):
    """Pillar 2 — paired LLM vs human Vendi scatter."""
    if pillar2.get("status") != "ok":
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    paired = pillar2.get("paired_prompts", [])
    all_vals: list[float] = []
    for kname in kernels:
        xs = [pillar1["per_prompt"][pid][kname]["human"]["vendi"] for pid in paired]
        ys = [pillar1["per_prompt"][pid][kname]["inter_llm"]["vendi"] for pid in paired]
        ax.scatter(xs, ys, label=kname, alpha=0.85)
        all_vals.extend(xs); all_vals.extend(ys)
    if all_vals:
        mx = max(all_vals) * 1.05
        ax.plot([0, mx], [0, mx], "k:", linewidth=0.6)
        ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Human Vendi (per prompt)")
    ax.set_ylabel("Inter-LLM Vendi (per prompt)")
    ax.set_title("Pillar 2 — points below diagonal = LLMs less diverse than humans")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pillar2_ordering.png")
    plt.close(fig)


def _make_fig3(plt, pillar3, out_dir):
    """Pillar 3 — trace cluster distribution per prompt."""
    if not pillar3.get("per_prompt"):
        return
    import matplotlib
    prompts = [pid for pid, v in pillar3["per_prompt"].items()
               if v.get("status") == "ok"]
    if not prompts:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(prompts))
    w = 0.35
    for offset, source in [(-w / 2, "inter_llm"), (w / 2, "human")]:
        bottoms = np.zeros(len(prompts))
        cmap = matplotlib.colormaps.get_cmap(
            "tab20" if source == "inter_llm" else "Pastel1")
        all_clusters: list[list[tuple[str, int]]] = []
        for pid in prompts:
            src = pillar3["per_prompt"][pid].get(source)
            if src is None:
                all_clusters.append([])
            else:
                items = list(src["clusters"].items())
                all_clusters.append(items)
        max_clusters = max((len(c) for c in all_clusters), default=0)
        for k in range(max_clusters):
            vals = []
            for ci, items in enumerate(all_clusters):
                if k < len(items):
                    n_total = pillar3["per_prompt"][prompts[ci]][source]["n_passing"]
                    vals.append(items[k][1] / n_total if n_total else 0)
                else:
                    vals.append(0)
            ax.bar(x + offset, vals, w, bottom=bottoms,
                   color=cmap(k % cmap.N), edgecolor="white",
                   linewidth=0.3,
                   label=f"{source}-c{k+1}" if k < 3 else None)
            bottoms += np.array(vals)
    ax.set_xticks(x)
    ax.set_xticklabels(prompts)
    ax.set_ylabel("Fraction of passing samples in cluster")
    ax.set_title(
        "Pillar 3 — trace clusters per prompt "
        "(LLM bars left, human right; tall single block = monoculture)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_pillar3_traces.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 10. Summary writer
# ─────────────────────────────────────────────────────────────────────

def write_summary(out_dir: Path, args, pillar1: dict, pillar2: dict, pillar3: dict):
    summary = {
        "config": {
            "prompts": args.prompts,
            "temperature": args.temperature,
            "raw_dir": str(args.raw_dir),
            "human_dir": str(args.human_dir) if args.human_dir else None,
            "kernels": pillar1.get("kernels", []),
            "device": args.device,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "samples_per_model_for_inter": args.samples_per_model_for_inter,
            "cap_per_pool": args.cap_per_pool,
            "ts_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "pillar1_vendi": pillar1,
        "pillar2_ordering": pillar2,
        "pillar3_functional": pillar3,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=str), encoding="utf-8")

    # Markdown summary
    lines: list[str] = []
    lines.append("# Homogeneity Proof — first-test summary\n")
    lines.append(f"- Prompts: {', '.join(args.prompts)}")
    lines.append(f"- Temperature filter: {args.temperature}")
    lines.append(f"- Kernels: {', '.join(pillar1.get('kernels', []))}")
    lines.append(f"- Bootstrap resamples: {args.n_boot}")
    lines.append("")
    lines.append("## Pillar 1 — Vendi per pool\n")
    lines.append(
        "Two views per pool: **raw Vendi** (effective N distinct, bounded by "
        "pool size) and **normalized Vendi** (per-sample effective uniqueness "
        "in [0, 1]; size-fair).\n")
    for pid, kr in pillar1.get("per_prompt", {}).items():
        lines.append(f"### {pid}")
        for kname, pp in kr.items():
            lines.append(f"- kernel={kname}")
            for pool in ["null", "human", "intra_llm_mean", "inter_llm"]:
                v = pp.get(pool)
                if v is None:
                    lines.append(f"  - {pool:<18}  (skipped)")
                else:
                    ci = v.get("ci", [None, None])
                    n = v.get("n", v.get("n_models", "?"))
                    vn = v.get("vendi_norm")
                    cn = v.get("ci_norm", [None, None])
                    if vn is None:
                        lines.append(
                            f"  - {pool:<18}  Vendi={v['vendi']:.2f}"
                            f"   CI95=[{ci[0]:.2f}, {ci[1]:.2f}]   n={n}")
                    else:
                        lines.append(
                            f"  - {pool:<18}  Vendi={v['vendi']:.2f} "
                            f"CI=[{ci[0]:.2f},{ci[1]:.2f}]  "
                            f"norm={vn:.3f} CI=[{cn[0]:.3f},{cn[1]:.3f}]  n={n}")
        lines.append("")

    lines.append("## Pillar 2 — Ordering: LLM vs human Vendi\n")
    if pillar2.get("status") != "ok":
        lines.append(f"- **skipped**: {pillar2.get('reason', 'no human data')}")
    else:
        lines.append("Raw Δ is sample-size-confounded; **normalized Δ is the "
                     "size-fair headline**.\n")
        for kname, res in pillar2["per_kernel"].items():
            d  = res["llm_vs_human"]
            dn = res.get("llm_vs_human_norm")
            di = res.get("intra_vs_human_norm")
            ci = d["ci"]
            lines.append(f"- **{kname}**")
            lines.append(
                f"    raw  Δ(LLM−human)        = {d['delta_mean']:+.3f}  "
                f"CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]   p={d['p']:.4f}   "
                f"n_pairs={d['n_pairs']}")
            if dn is not None and dn.get("delta_mean") is not None:
                cn = dn["ci"]
                lines.append(
                    f"    norm Δ(LLM−human)        = {dn['delta_mean']:+.3f}  "
                    f"CI95=[{cn[0]:+.3f}, {cn[1]:+.3f}]   p={dn['p']:.4f}")
            if di is not None and di.get("delta_mean") is not None:
                cn = di["ci"]
                lines.append(
                    f"    norm Δ(intra-LLM−human)  = {di['delta_mean']:+.3f}  "
                    f"CI95=[{cn[0]:+.3f}, {cn[1]:+.3f}]   p={di['p']:.4f}")
            lines.append(
                f"    means (norm): LLM={res.get('llm_vendi_norm_mean', 0):.3f}  "
                f"human={res.get('human_vendi_norm_mean', 0):.3f}  "
                f"intra-LLM={res.get('intra_vendi_norm_mean', 0):.3f}")
    lines.append("")

    lines.append("## Pillar 3 — Functional convergence (passing solutions only)\n")
    for pid, v in pillar3.get("per_prompt", {}).items():
        if v.get("status") != "ok":
            lines.append(f"- {pid}: {v.get('status')}")
            continue
        inter = v.get("inter_llm")
        human = v.get("human")
        line = f"- **{pid}**: inter-LLM trace-Vendi={inter['vendi']:.2f}  "
        line += f"(passing {inter['n_passing']}/{inter['n_total']})"
        if human:
            line += f"  ;  human trace-Vendi={human['vendi']:.2f} "
            line += f"(passing {human['n_passing']}/{human['n_total']})"
        lines.append(line)
        if inter.get("clusters"):
            lines.append(f"    inter-LLM top clusters: "
                         f"{list(inter['clusters'].values())[:6]}")
    lines.append("")
    lines.append("## Figures\n")
    lines.append("- `fig_pillar1_vendi.png`")
    lines.append("- `fig_pillar2_ordering.png`")
    lines.append("- `fig_pillar3_traces.png`")
    (out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────
# 11. CLI / main
# ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--prompts", nargs="+", default=["AL-01", "AL-03", "AL-05"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--raw-dir", type=Path, default=ROOT / "results" / "raw_responses")
    p.add_argument("--human-dir", type=Path, default=None,
                   help="Optional dir of human reference JSONLs (Pillar 2/3).")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Default: results/proof/<UTC-timestamp>/")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-model-for-inter", type=int, default=4,
                   help="k samples per model for the inter-LLM pool. Default 4 "
                        "x 8 models = ~32 samples, comparable to 30 humans.")
    p.add_argument("--cap-per-pool", type=int, default=60,
                   help="Maximum codes per pool (controls runtime)")
    p.add_argument("--embedder", default="unixcoder",
                   choices=["unixcoder", "codebert", "tfidf", "none"],
                   help="Embedding kernel for Pillar 1. "
                        "'unixcoder' (default) and 'codebert' need a healthy "
                        "transformers/torch install. 'tfidf' is sklearn-only "
                        "and works without GPU/torch. 'none' skips embeddings "
                        "entirely (token + AST kernels only).")
    p.add_argument("--skip-embedding", dest="embedder",
                   action="store_const", const="none",
                   help="Alias for --embedder none.")
    p.add_argument("--skip-codebert", dest="embedder",
                   action="store_const", const="none", help=argparse.SUPPRESS)
    p.add_argument("--skip-pillar3", action="store_true",
                   help="Skip the functional-convergence pillar.")
    p.add_argument("--harness-timeout", type=float, default=6.0)
    p.add_argument("--dry-run", action="store_true",
                   help="Validate setup and exit without computing kernels.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.out_dir is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        args.out_dir = ROOT / "results" / "proof" / f"v1_{ts}"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"== proof_homogeneity ==")
    print(f"  prompts:      {args.prompts}")
    print(f"  temperature:  {args.temperature}")
    print(f"  raw_dir:      {args.raw_dir}")
    print(f"  human_dir:    {args.human_dir}")
    print(f"  out_dir:      {args.out_dir}")
    print(f"  device:       {args.device}")
    print()

    # ── Load data ──
    llm = load_llm_responses(args.raw_dir, args.prompts, args.temperature)
    n_total_llm = sum(len(v) for v in llm.values())
    print(f"Loaded {n_total_llm} LLM responses across {len(llm)} prompts")
    if n_total_llm == 0:
        print(f"\nERROR: no LLM responses found in {args.raw_dir} for "
              f"prompts={args.prompts} at temp={args.temperature}. "
              f"Run multi_model_sampler first.")
        sys.exit(2)

    human = load_human_responses(args.human_dir, args.prompts)
    n_total_human = sum(len(v) for v in human.values())
    print(f"Loaded {n_total_human} human responses across {len(human)} prompts")
    if n_total_human == 0:
        print("  (Pillar 2 will be skipped; pass --human-dir to enable.)")

    if args.dry_run:
        print("\n--dry-run: setup OK, exiting before kernels.")
        return

    # ── Build pools ──
    rng = np.random.default_rng(args.seed)
    pool_codes_for_null: list[str] = []
    for pid, rs in llm.items():
        for r in rs:
            c = extract_code(r.get("response_text", ""))
            if c:
                pool_codes_for_null.append(c)

    pools_by_prompt: dict[str, PromptPools] = {}
    for pid in args.prompts:
        # null draws from OTHER prompts' codes only
        other_codes = []
        for opid, rs in llm.items():
            if opid == pid:
                continue
            for r in rs:
                c = extract_code(r.get("response_text", ""))
                if c:
                    other_codes.append(c)
        pools_by_prompt[pid] = build_pools(
            pid, llm[pid], human.get(pid, []), other_codes,
            rng, samples_per_model_for_inter=args.samples_per_model_for_inter,
            cap_per_pool=args.cap_per_pool,
        )
        p = pools_by_prompt[pid]
        print(f"  pools[{pid}]  inter={len(p.inter_llm_codes)}  "
              f"intra_models={len(p.intra_llm_codes)}  "
              f"human={len(p.human_codes)}  null={len(p.null_codes)}")

    # ── Register kernels ──
    _register_kernels(embedder=args.embedder, device=args.device)
    kernels = list(KERNEL_FNS.keys())
    print(f"  embedder:     {args.embedder}")
    print(f"Active kernels: {kernels}")

    # ── Pillar 1 ──
    print("\n[Pillar 1] Vendi over kernels x pools ...")
    pillar1 = run_pillar1(pools_by_prompt, kernels, args.n_boot, args.seed)

    # ── Pillar 2 ──
    print("\n[Pillar 2] LLM vs human ordering test ...")
    pillar2 = run_pillar2(pillar1, args.n_boot, args.seed + 1)

    # ── Pillar 3 ──
    if args.skip_pillar3:
        pillar3 = {"status": "skipped (--skip-pillar3)"}
    else:
        print("\n[Pillar 3] Functional convergence ...")
        pillar3 = run_pillar3(pools_by_prompt, args.out_dir,
                              timeout=args.harness_timeout)

    # ── Summary first (so plotting bugs never destroy the numbers) ──
    print("\n[Summary] writing ...")
    write_summary(args.out_dir, args, pillar1,
                  pillar2 if isinstance(pillar2, dict) else {"status": "skipped"},
                  pillar3 if isinstance(pillar3, dict) else {"status": "skipped"})

    # ── Figures (each wrapped in try/except inside make_figures) ──
    print("[Figures] writing ...")
    make_figures(pillar1, pillar2 if isinstance(pillar2, dict) else {},
                 pillar3 if isinstance(pillar3, dict) else {},
                 args.out_dir)

    print(f"\nDone. Outputs in: {args.out_dir}")


if __name__ == "__main__":
    main()