#!/usr/bin/env python3
"""
Code Hivemind - Approach / Architecture Homogeneity Analysis
============================================================

Complements the surface/structural/semantic metrics in diversity_metrics.py
with the layer that matters most for open-ended code:

   "Even when generated code differs syntactically, do models converge on
    the same APPROACH -- same algorithms, libraries, architecture patterns,
    storage choices, API designs, and optional features?"

For every (prompt, model, sample) we extract:

  1. A structured feature vector:
       - AST counts:    num_functions, num_classes, num_loops, num_ifs,
                        num_try_blocks, num_decorators, max_nesting_depth
       - Style flags:   uses_oop, uses_dataclass, uses_async, uses_type_hints,
                        uses_logging, uses_context_manager, uses_lambda, ...
       - Imports:       full set of top-level packages
       - Storage:       in-memory / json / sqlite / csv / pickle / file / none
       - Interface:     argparse / click / typer / flask / fastapi / http.server / library
       - Algorithm:     prompt-specific algorithm hints (e.g. ttl/lru/lfu for caches,
                        token-bucket / sliding-window for rate limiters, etc.)

  2. A coarse "architecture label" string built from the above
     (e.g. "class_based|json_storage|argparse_cli|ttl_cache").

Then we compute homogeneity metrics:

  - Dominant-architecture ratio       (largest cluster / total)
  - Shannon entropy                    (over architecture labels)
  - Normalized entropy
  - Simpson diversity index
  - Gini coefficient over imports
  - Top-k library concentration
  - Library Jaccard similarity (mean pairwise)
  - Feature-vector cosine similarity (mean pairwise)
  - Intra-model approach repetition rate
  - Cross-model approach-agreement matrix

Visualisations (saved to results/figures/):
  - approach_distribution_<PROMPT>.png   stacked bars per prompt
  - approach_dominance.png               dominant-ratio per prompt
  - approach_entropy.png                 Shannon entropy per prompt
  - library_frequency_<PROMPT>.png       top libraries bar chart
  - cross_model_agreement.png            heatmap, mean over prompts
  - feature_pca_<PROMPT>.png             PCA scatter coloured by model

Outputs:
  results/approach/per_sample.csv        one row per (prompt, model, sample)
  results/approach/per_prompt.json       homogeneity metrics per prompt
  results/approach/per_model.json        intra-model repetition per model
  results/approach/cross_model.json      pairwise model agreement
  results/approach/summary.csv           publication-friendly summary table

Usage:
  python approach_analysis.py
  python approach_analysis.py --no-figures
  python approach_analysis.py --prompts OD-01 OD-03
  python approach_analysis.py --temperature 1.0
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

# -------- optional plotting --------
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9, "figure.dpi": 150, "savefig.dpi": 200,
        "savefig.bbox": "tight",
    })
except ImportError:
    HAS_MPL = False

try:
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ─────────────────────────────────────────────────────────────────────
# Paths (mirror the rest of the repo)
# ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent
RAW_DIR  = ROOT / "results" / "raw_responses"
FIG_DIR  = ROOT / "results" / "figures"
OUT_DIR  = ROOT / "results" / "approach"

CATEGORY_FROM_ID = {
    "OD": "OPEN_DESIGN", "AL": "ALGORITHM", "RF": "REFACTOR",
    "NM": "NAMING",      "CT": "CREATIVE_TOOL", "SD": "SYSTEM_DESIGN",
}


# ─────────────────────────────────────────────────────────────────────
# 1. Code extraction
# ─────────────────────────────────────────────────────────────────────

_FENCE = re.compile(r"```(?:python|py|javascript|js)?\s*\n(.*?)```", re.DOTALL)

def extract_code(text: str) -> str:
    """Pull code out of markdown fences; fall back to whole text."""
    matches = _FENCE.findall(text or "")
    return "\n".join(matches) if matches else (text or "").strip()


# ─────────────────────────────────────────────────────────────────────
# 2. Structured feature extraction (AST + regex fallback)
# ─────────────────────────────────────────────────────────────────────

@dataclass
class CodeFeatures:
    # AST counts
    num_functions: int = 0
    num_classes: int = 0
    num_methods: int = 0
    num_loops: int = 0
    num_ifs: int = 0
    num_try_blocks: int = 0
    num_decorators: int = 0
    num_imports: int = 0
    max_nesting_depth: int = 0
    loc: int = 0
    parse_ok: bool = True
    # style flags
    uses_oop: bool = False
    uses_dataclass: bool = False
    uses_async: bool = False
    uses_type_hints: bool = False
    uses_logging: bool = False
    uses_context_manager: bool = False
    uses_lambda: bool = False
    uses_comprehensions: bool = False
    uses_generators: bool = False
    uses_threading: bool = False
    uses_multiprocessing: bool = False
    uses_decorator_factory: bool = False
    # behavioural / domain choices
    imports: list[str] = field(default_factory=list)
    storage: str = "none"          # in-memory / json / sqlite / csv / pickle / file
    interface: str = "library"     # library / argparse / click / typer / flask / fastapi / http.server
    cli_present: bool = False
    has_main_block: bool = False


_PY_BUILTINS = set(dir(__builtins__)) if isinstance(__builtins__, dict) is False else set(__builtins__)


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    """Max indentation depth of compound statements inside the tree."""
    BLOCKING = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try,
                ast.With, ast.AsyncWith, ast.FunctionDef,
                ast.AsyncFunctionDef, ast.ClassDef)
    best = depth
    for child in ast.iter_child_nodes(node):
        d = _max_depth(child, depth + 1 if isinstance(child, BLOCKING) else depth)
        if d > best:
            best = d
    return best


def _detect_storage(code: str, imports: list[str]) -> str:
    low = code.lower()
    imps = set(imports)
    if "sqlite3" in imps or "sqlalchemy" in imps:
        return "sqlite"
    if "shelve" in imps:
        return "shelve"
    if "pickle" in imps:
        return "pickle"
    if "csv" in imps:
        return "csv"
    if "json" in imps and re.search(r"\.json['\"]|json\.dump|json\.load", low):
        return "json_file"
    if re.search(r"\bopen\([^\)]+['\"][rwab+]+['\"]", low):
        return "plain_file"
    if "redis" in imps:
        return "redis"
    return "in_memory"


_INTERFACE_PATTERNS = [
    ("fastapi",     r"\bfrom\s+fastapi|import\s+fastapi|FastAPI\("),
    ("flask",       r"\bfrom\s+flask|import\s+flask|Flask\("),
    ("typer",       r"\bimport\s+typer|from\s+typer\b|typer\.Typer\("),
    ("click",       r"\bimport\s+click|@click\."),
    ("argparse",    r"\bargparse\b|ArgumentParser\("),
    ("http.server", r"http\.server|BaseHTTPRequestHandler|HTTPServer"),
    ("aiohttp",     r"\baiohttp\b"),
    ("django",      r"\bfrom\s+django|django\.urls"),
    ("starlette",   r"\bstarlette\b"),
]


def _detect_interface(code: str) -> tuple[str, bool]:
    for label, pat in _INTERFACE_PATTERNS:
        if re.search(pat, code):
            return label, True
    return "library", False


def extract_features(code: str) -> CodeFeatures:
    f = CodeFeatures()
    f.loc = len([ln for ln in code.splitlines() if ln.strip()])

    try:
        tree = ast.parse(code)
    except SyntaxError:
        f.parse_ok = False
        # best-effort regex fallback
        f.imports = sorted(set(re.findall(r"^\s*(?:from|import)\s+([a-zA-Z_][\w\.]*)",
                                          code, flags=re.MULTILINE)))
        f.imports = [i.split(".")[0] for i in f.imports]
        f.num_imports = len(f.imports)
        f.uses_async = bool(re.search(r"\basync\s+def\b", code))
        f.uses_lambda = "lambda " in code
        f.has_main_block = '__name__' in code and '__main__' in code
        f.storage = _detect_storage(code, f.imports)
        f.interface, f.cli_present = _detect_interface(code)
        return f

    imports: list[str] = []
    in_class_stack: list[ast.AST] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                imports.append(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module.split(".")[0])
        elif isinstance(node, ast.ClassDef):
            f.num_classes += 1
            in_class_stack.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Heuristic: methods are FunctionDefs nested inside a ClassDef.
            # ast.walk doesn't carry parent info; we re-detect with parent walk below.
            f.num_functions += 1
            if node.decorator_list:
                f.num_decorators += len(node.decorator_list)
                if any(getattr(d, "id", "") == "dataclass"
                       or (isinstance(d, ast.Attribute) and d.attr == "dataclass")
                       for d in node.decorator_list):
                    f.uses_dataclass = True
            if isinstance(node, ast.AsyncFunctionDef):
                f.uses_async = True
            if node.returns is not None:
                f.uses_type_hints = True
            if any(a.annotation is not None for a in node.args.args):
                f.uses_type_hints = True
            # decorator-factory: nested def inside def
            for inner in ast.walk(node):
                if inner is node:
                    continue
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    f.uses_decorator_factory = True
                    break
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            f.num_loops += 1
        elif isinstance(node, ast.If):
            f.num_ifs += 1
        elif isinstance(node, ast.Try):
            f.num_try_blocks += 1
        elif isinstance(node, ast.With) or isinstance(node, ast.AsyncWith):
            f.uses_context_manager = True
        elif isinstance(node, ast.Lambda):
            f.uses_lambda = True
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp)):
            f.uses_comprehensions = True
        elif isinstance(node, ast.GeneratorExp):
            f.uses_generators = True
        elif isinstance(node, ast.Yield) or isinstance(node, ast.YieldFrom):
            f.uses_generators = True

    # method count = functions defined with a class as direct parent
    methods = 0
    for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
        for child in cls.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods += 1
    f.num_methods = methods
    f.uses_oop = f.num_classes > 0
    f.imports = sorted(set(imports))
    f.num_imports = len(f.imports)
    f.max_nesting_depth = _max_depth(tree)
    f.uses_logging = "logging" in f.imports
    f.uses_threading = "threading" in f.imports
    f.uses_multiprocessing = "multiprocessing" in f.imports or "concurrent" in f.imports
    f.has_main_block = bool(re.search(r"if\s+__name__\s*==\s*['\"]__main__['\"]", code))
    f.uses_dataclass = f.uses_dataclass or "dataclasses" in f.imports
    f.uses_type_hints = f.uses_type_hints or "typing" in f.imports

    f.storage = _detect_storage(code, f.imports)
    f.interface, f.cli_present = _detect_interface(code)
    return f


# ─────────────────────────────────────────────────────────────────────
# 3. Prompt-specific algorithm / architecture detectors
# ─────────────────────────────────────────────────────────────────────
#
# These return a small dict of *categorical* labels for the algorithmic
# choice (the "approach"). A dispatch table picks the right detector
# based on prompt id; unmatched prompts get a generic detector.

def _alg_cache(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    out = {}
    if "ttl" in low or "expire" in low or "time.time" in low:
        out["ttl_logic"] = True
    out["lru"] = "ordereddict" in low or "move_to_end" in low or "lru_cache" in low
    out["lfu"] = "lfu" in low or "frequency" in low and "cache" in low
    out["heap_based"] = "heapq" in low and "cache" in low
    out["lazy_cleanup"] = bool(re.search(r"def\s+(get|access)\b[^)]*\)\s*:\s*\n[^\n]*if[^\n]*expire", code))
    out["eager_cleanup"] = bool(re.search(r"def\s+cleanup\b|def\s+evict\b|def\s+purge\b", code))
    out["thread_safe"] = "threading" in low and "lock" in low
    out["data_struct"] = (
        "ordereddict" if "ordereddict" in low else
        "heap" if "heapq" in low else
        "dict"
    )
    return out


def _alg_task_queue(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "heap_based":    "heapq" in low,
        "queue_module":  "from queue" in low or "import queue" in low,
        "deque":         "deque" in low,
        "sorted_list":   "sort(" in low and "priority" in low,
        "retry_decorator": bool(re.search(r"@retry|def\s+retry\b", code)),
        "max_retries_3": "3" in low and "retry" in low,
    }


def _alg_rate_limiter(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "fixed_window":   "fixed" in low and "window" in low,
        "sliding_window": "sliding" in low and "window" in low,
        "token_bucket":   "token" in low and "bucket" in low,
        "leaky_bucket":   "leak" in low and "bucket" in low,
        "decorator":      bool(re.search(r"def\s+\w+\(.*func.*\):\s*\n", code)) or "@" in code,
        "thread_safe":    "lock" in low,
    }


def _alg_pubsub(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "wildcard":   "wildcard" in low or "*" in code and "subscribe" in low,
        "once":       "once" in low,
        "map_based":  "new map(" in low or " map<" in low,
        "object_based": "{}" in code and "subscribers" in low,
    }


def _alg_kv(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "json_store":   "json" in low and ("dump" in low or "load" in low),
        "pickle":       "pickle" in low,
        "sqlite":       "sqlite3" in low,
        "append_log":   "append" in low and ".log" in low,
        "lock":         "lock" in low,
        "fsync":        "fsync" in low or "flush" in low,
    }


def _alg_anagram(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "sort_key":      "sorted(" in low and ("''.join" in low or '"".join' in low),
        "char_count":    "counter(" in low or "tuple(sorted" in low,
        "prime_hash":    "prime" in low,
        "defaultdict":   "defaultdict" in low,
        "groupby":       "groupby" in low,
    }


def _alg_topk(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "counter":         "counter(" in low,
        "heap":            "heapq" in low,
        "count_min_sketch":"countmin" in low or "count-min" in low or "sketch" in low,
        "space_saving":    "space saving" in low or "space-saving" in low,
        "streaming":       "yield" in low or "iter" in low,
    }


def _alg_cycle(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "dfs":            "dfs" in low or "visit" in low,
        "coloring":       "white" in low and "gray" in low,
        "topological":    "topological" in low or "kahn" in low,
        "iterative":      "stack" in low and "while stack" in low,
        "recursive":      bool(re.search(r"def\s+\w+\([^)]*\)\s*:[\s\S]{0,200}return\s+\w+\(", code)),
    }


def _alg_text_sim(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "tfidf":      "tfidf" in low or "tf-idf" in low or "tfidfvectorizer" in low,
        "cosine":     "cosine" in low,
        "jaccard":    "jaccard" in low,
        "edit_dist":  "levenshtein" in low or "edit_distance" in low,
        "embedding":  "embedding" in low or "sentence_transformers" in low,
        "bm25":       "bm25" in low,
    }


def _alg_parens(code: str, _f: CodeFeatures) -> dict:
    return {
        "recursive":   bool(re.search(r"def\s+\w+\([^)]*\):[\s\S]{0,500}\w+\(", code)),
        "iterative":   "while" in code or "for " in code,
        "backtrack":   "backtrack" in code.lower(),
    }


def _alg_url_shortener(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "hash_md5":  "md5" in low,
        "hash_sha":  "sha" in low,
        "base62":    "base62" in low or "base_62" in low,
        "counter_based": "counter" in low and "increment" in low,
        "random":    "random" in low and "choice" in low,
        "flask":     "flask" in low,
        "fastapi":   "fastapi" in low,
        "http_server": "http.server" in low or "basehttprequest" in low,
    }


def _alg_static_site(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "jinja2":   "jinja2" in low or "jinja" in low,
        "string_format": "format(" in low and "<html" in low,
        "markdown_lib": "markdown" in low or "mistune" in low or "commonmark" in low,
    }


def _alg_rest_framework(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "regex_routes":  "re.compile" in low and "route" in low,
        "trie_routes":   "trie" in low,
        "dict_routes":   "routes[" in low or "self.routes" in low,
        "wsgi":          "wsgi" in low,
        "decorator_register": bool(re.search(r"@\w+\.(get|post|route)\b", code)),
    }


def _alg_migration(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "tracks_in_table": "schema_migrations" in low or "migration" in low and "table" in low,
        "tracks_in_file":  ".json" in low and "migration" in low,
        "rollback":        "rollback" in low or "down(" in low,
        "numbered_files":  bool(re.search(r"\d{3,}_\w+\.sql", code)),
    }


def _alg_scraper_framework(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "requests": "requests" in (_f.imports or []),
        "httpx":    "httpx"    in (_f.imports or []),
        "aiohttp":  "aiohttp"  in (_f.imports or []),
        "bs4":      "bs4" in low or "beautifulsoup" in low,
        "lxml":     "lxml" in low,
        "scrapy":   "scrapy" in low,
        "rate_limit": "rate" in low and "limit" in low,
        "pagination": "page" in low and ("next" in low or "while" in low),
    }


def _alg_card_deck(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "enum_suit": "class suit" in low and "enum" in low,
        "string_suit": '"hearts"' in low or "'hearts'" in low,
        "namedtuple": "namedtuple" in low,
        "dataclass":  "@dataclass" in low,
        "shuffle_random": "random.shuffle" in low,
    }


def _alg_logger(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "wraps_logging": "logging" in (_f.imports or []),
        "custom_levels": "log_level" in low or "loglevel" in low,
        "file_handler":  "filehandler" in low or "writeto" in low,
        "console_handler": "streamhandler" in low or "stdout" in low,
    }


def _alg_config_manager(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "env_vars":   "os.environ" in low or "getenv" in low,
        "json_load":  "json" in (_f.imports or []),
        "yaml_load":  "yaml" in (_f.imports or []),
        "dataclass":  "@dataclass" in low,
        "dict_merge": "update(" in low or "{**" in low,
    }


def _alg_terminal_game(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    txt = low
    return {
        "game_snake":      "snake" in txt,
        "game_hangman":    "hangman" in txt,
        "game_tictactoe":  "tic" in txt and "tac" in txt,
        "game_2048":       "2048" in txt,
        "game_pong":       "pong" in txt,
        "game_minesweeper":"mine" in txt and "sweep" in txt,
        "uses_curses":     "curses" in (_f.imports or []),
        "uses_input":      "input(" in txt,
    }


def _alg_ascii_art(code: str, _f: CodeFeatures) -> dict:
    low = code.lower()
    return {
        "pyfiglet":   "pyfiglet" in low,
        "art_lib":    '"art"' in low or "from art" in low,
        "manual_font": "letters = {" in low or "alphabet = {" in low,
    }


# Dispatch by prompt id, then by (id-prefix, keywords).
_ALG_DISPATCH = {
    "OD-01": _alg_cache,           "OD-02": _alg_task_queue,
    "OD-03": _alg_rate_limiter,    "OD-04": _alg_pubsub,
    "OD-05": _alg_kv,
    "AL-01": _alg_anagram,         "AL-02": _alg_topk,
    "AL-03": _alg_cycle,           "AL-04": _alg_text_sim,
    "AL-05": _alg_parens,
    "SD-01": _alg_url_shortener,   "SD-02": _alg_static_site,
    "SD-03": _alg_rest_framework,  "SD-04": _alg_migration,
    "SD-05": _alg_scraper_framework,
    "NM-01": _alg_card_deck,       "NM-04": _alg_logger,
    "NM-05": _alg_config_manager,
    "CT-01": _alg_ascii_art,       "CT-04": _alg_terminal_game,
}


def algorithmic_choices(prompt_id: str, code: str, feats: CodeFeatures) -> dict:
    detector = _ALG_DISPATCH.get(prompt_id)
    if detector is None:
        # Generic fallback labels for unknown prompts
        return {}
    try:
        return detector(code, feats)
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────────
# 4. Build "architecture label" for each sample
# ─────────────────────────────────────────────────────────────────────

def architecture_label(feats: CodeFeatures, alg: dict) -> str:
    """A short, comparable string summarising the approach."""
    parts = []
    parts.append("oop" if feats.uses_oop else "procedural")
    parts.append(feats.storage)
    parts.append(feats.interface)
    if feats.uses_async:
        parts.append("async")
    if feats.uses_decorator_factory:
        parts.append("decorator-factory")
    # Prompt-specific algorithm tag: pick the FIRST true label as the dominant one
    for k in sorted(alg.keys()):
        v = alg[k]
        if isinstance(v, bool) and v:
            parts.append(f"alg:{k}")
            break
        if isinstance(v, str) and v:
            parts.append(f"alg:{k}={v}")
            break
    return "|".join(parts)


# ─────────────────────────────────────────────────────────────────────
# 5. Diversity / homogeneity metrics
# ─────────────────────────────────────────────────────────────────────

def shannon_entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts if c > 0], dtype=float)
    return float(-(probs * np.log2(probs)).sum())


def normalized_entropy(counts: list[int]) -> float:
    n = len([c for c in counts if c > 0])
    if n <= 1:
        return 0.0
    return shannon_entropy(counts) / np.log2(n)


def simpson_diversity(counts: list[int]) -> float:
    total = sum(counts)
    if total < 2:
        return 0.0
    s = sum(c * (c - 1) for c in counts)
    return 1.0 - s / (total * (total - 1))


def dominant_ratio(counts: list[int]) -> float:
    total = sum(counts)
    return max(counts) / total if total else 0.0


def gini_coefficient(values: list[float]) -> float:
    """Gini over a frequency distribution."""
    if not values:
        return 0.0
    arr = np.sort(np.array(values, dtype=float))
    n = arr.size
    if arr.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(arr) / (n * arr.sum()))


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    u = a | b
    return len(a & b) / len(u) if u else 0.0


def mean_pairwise(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


# ─────────────────────────────────────────────────────────────────────
# 6. Feature-vector encoding for cosine similarity & PCA
# ─────────────────────────────────────────────────────────────────────

NUMERIC_FIELDS = [
    "num_functions", "num_classes", "num_methods", "num_loops", "num_ifs",
    "num_try_blocks", "num_decorators", "num_imports", "max_nesting_depth", "loc",
]
BOOL_FIELDS = [
    "uses_oop", "uses_dataclass", "uses_async", "uses_type_hints",
    "uses_logging", "uses_context_manager", "uses_lambda",
    "uses_comprehensions", "uses_generators", "uses_threading",
    "uses_multiprocessing", "uses_decorator_factory",
    "cli_present", "has_main_block",
]


def featurize(feats: CodeFeatures) -> np.ndarray:
    nums = [getattr(feats, f) for f in NUMERIC_FIELDS]
    bools = [int(getattr(feats, f)) for f in BOOL_FIELDS]
    return np.array(nums + bools, dtype=float)


def normalise_matrix(M: np.ndarray) -> np.ndarray:
    """Standardise columns (zero mean, unit variance) to make cosine meaningful."""
    if M.size == 0:
        return M
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    sd[sd == 0] = 1.0
    return (M - mu) / sd


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(M, axis=1, keepdims=True)
    norm = np.clip(norm, 1e-10, None)
    Mn = M / norm
    return Mn @ Mn.T


# ─────────────────────────────────────────────────────────────────────
# 7. Loading data
# ─────────────────────────────────────────────────────────────────────

def load_responses(prompt_ids: list[str] | None,
                   temperature: float | None) -> dict[str, list[dict]]:
    """Load raw_responses/<PID>.jsonl files, optionally filter by temperature."""
    by_prompt: dict[str, list[dict]] = {}
    files = sorted(RAW_DIR.glob("*.jsonl"))
    for fp in files:
        pid = fp.stem
        if pid == "all_responses":
            continue
        if prompt_ids and pid not in prompt_ids:
            continue
        rows = []
        with open(fp, encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                try:
                    rec = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if temperature is not None:
                    if abs(rec.get("temperature", 0.0) - temperature) > 1e-6:
                        continue
                rows.append(rec)
        if rows:
            by_prompt[pid] = rows
    return by_prompt


# ─────────────────────────────────────────────────────────────────────
# 8. Per-prompt analysis
# ─────────────────────────────────────────────────────────────────────

def analyse_prompt(prompt_id: str, records: list[dict]) -> dict:
    """Run the full approach analysis for a single prompt's responses."""
    samples = []
    for r in records:
        code = extract_code(r.get("response_text", ""))
        feats = extract_features(code)
        alg   = algorithmic_choices(prompt_id, code, feats)
        label = architecture_label(feats, alg)
        samples.append({
            "model":       r.get("model_display"),
            "family":      r.get("model_family"),
            "temperature": r.get("temperature"),
            "sample_index": r.get("sample_index"),
            "code":        code,
            "feats":       feats,
            "alg":         alg,
            "label":       label,
        })

    n = len(samples)
    if n == 0:
        return {"prompt_id": prompt_id, "n": 0}

    # ── Architecture-label distribution ──────────────────────────────
    label_counts = Counter(s["label"] for s in samples)
    counts = list(label_counts.values())
    label_metrics = {
        "n":                 n,
        "n_unique":          len(label_counts),
        "dominant_ratio":    dominant_ratio(counts),
        "shannon_entropy":   shannon_entropy(counts),
        "normalised_entropy": normalized_entropy(counts),
        "simpson_diversity": simpson_diversity(counts),
        "top5":              label_counts.most_common(5),
    }

    # ── Library / import distribution ────────────────────────────────
    all_imports = Counter()
    for s in samples:
        for imp in s["feats"].imports:
            all_imports[imp] += 1

    # samples that share an exact import-set
    importset_counts = Counter(frozenset(s["feats"].imports) for s in samples)
    library_metrics = {
        "unique_libraries":  len(all_imports),
        "top10":             all_imports.most_common(10),
        "library_entropy":   shannon_entropy(list(all_imports.values())),
        "library_gini":      gini_coefficient(list(all_imports.values())),
        "unique_import_sets": len(importset_counts),
        "dominant_import_set_ratio": dominant_ratio(list(importset_counts.values())),
        "top5_import_sets":  [(sorted(s), c) for s, c in importset_counts.most_common(5)],
    }

    # mean pairwise import Jaccard
    sets = [set(s["feats"].imports) for s in samples]
    if len(sets) >= 2:
        sims = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                sims.append(jaccard(sets[i], sets[j]))
        library_metrics["mean_pairwise_jaccard"] = mean_pairwise(sims)
    else:
        library_metrics["mean_pairwise_jaccard"] = 0.0

    # ── Feature-vector cosine similarity ─────────────────────────────
    M = np.vstack([featurize(s["feats"]) for s in samples])
    Mn = normalise_matrix(M)
    cos = cosine_matrix(Mn)
    iu = np.triu_indices_from(cos, k=1)
    feature_metrics = {
        "mean_pairwise_cosine":   float(cos[iu].mean()) if iu[0].size else 0.0,
        "median_pairwise_cosine": float(np.median(cos[iu])) if iu[0].size else 0.0,
    }

    # ── Per-model intra-homogeneity ──────────────────────────────────
    by_model: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(samples):
        by_model[s["model"]].append(i)

    intra = {}
    for model, idxs in by_model.items():
        if len(idxs) < 2:
            continue
        labels = [samples[i]["label"] for i in idxs]
        lc = Counter(labels)
        cs = list(lc.values())
        sub_cos = cos[np.ix_(idxs, idxs)]
        sub_iu = np.triu_indices_from(sub_cos, k=1)
        intra[model] = {
            "n":                  len(idxs),
            "n_unique_labels":    len(lc),
            "dominant_ratio":     dominant_ratio(cs),
            "approach_entropy":   shannon_entropy(cs),
            "feature_cosine_mean": float(sub_cos[sub_iu].mean()) if sub_iu[0].size else 0.0,
            "top_label":          lc.most_common(1)[0][0],
        }

    # ── Cross-model approach agreement ───────────────────────────────
    # For each pair of models, fraction of (i, j) sample pairs (one from each
    # model) that share the same architecture label.
    models = sorted(by_model.keys())
    cross = {}
    for a in models:
        cross[a] = {}
        for b in models:
            if a == b:
                cross[a][b] = 1.0
                continue
            la = [samples[i]["label"] for i in by_model[a]]
            lb = [samples[i]["label"] for i in by_model[b]]
            if not la or not lb:
                cross[a][b] = 0.0
                continue
            # vectorised match rate
            la_arr = np.array(la); lb_arr = np.array(lb)
            same = (la_arr[:, None] == lb_arr[None, :]).mean()
            cross[a][b] = float(same)

    return {
        "prompt_id":     prompt_id,
        "category":      CATEGORY_FROM_ID.get(prompt_id.split("-")[0], "?"),
        "label_metrics": label_metrics,
        "library_metrics": library_metrics,
        "feature_metrics": feature_metrics,
        "intra_model":   intra,
        "cross_model":   cross,
        "samples":       samples,  # used for visualisation / export, dropped from JSON
    }


# ─────────────────────────────────────────────────────────────────────
# 9. Visualisation
# ─────────────────────────────────────────────────────────────────────

def _ensure_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_approach_distribution(prompt_id: str, samples: list[dict]):
    if not HAS_MPL or not samples:
        return
    by_model: dict[str, Counter] = defaultdict(Counter)
    label_universe = Counter()
    for s in samples:
        by_model[s["model"]][s["label"]] += 1
        label_universe[s["label"]] += 1
    top_labels = [lbl for lbl, _ in label_universe.most_common(8)]
    other = "other"
    models = sorted(by_model.keys())

    matrix = []
    for lbl in top_labels + [other]:
        row = []
        for m in models:
            if lbl == other:
                v = sum(c for k, c in by_model[m].items() if k not in top_labels)
            else:
                v = by_model[m].get(lbl, 0)
            row.append(v)
        matrix.append(row)
    matrix = np.array(matrix, dtype=float)
    totals = matrix.sum(axis=0)
    totals[totals == 0] = 1
    pct = matrix / totals

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 0.7), 4.5))
    bottom = np.zeros(len(models))
    cmap = plt.get_cmap("tab20")
    for i, lbl in enumerate(top_labels + [other]):
        ax.bar(models, pct[i], bottom=bottom, label=lbl, color=cmap(i % 20))
        bottom += pct[i]
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of samples")
    ax.set_title(f"{prompt_id}: architecture-label distribution by model")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=35, ha="right")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    plt.savefig(FIG_DIR / f"approach_distribution_{prompt_id}.png")
    plt.close(fig)


def plot_library_frequency(prompt_id: str, samples: list[dict]):
    if not HAS_MPL or not samples:
        return
    counter = Counter()
    for s in samples:
        for imp in s["feats"].imports:
            counter[imp] += 1
    if not counter:
        return
    top = counter.most_common(15)
    libs, counts = zip(*top)
    n = len(samples)
    fracs = [c / n for c in counts]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(libs[::-1], fracs[::-1], color="#3a7ca5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of samples importing this library")
    ax.set_title(f"{prompt_id}: top library choices ({n} samples)")
    plt.savefig(FIG_DIR / f"library_frequency_{prompt_id}.png")
    plt.close(fig)


def plot_pca(prompt_id: str, samples: list[dict]):
    if not HAS_MPL or not HAS_SKLEARN or len(samples) < 3:
        return
    M = np.vstack([featurize(s["feats"]) for s in samples])
    Mn = normalise_matrix(M)
    if Mn.shape[0] < 3 or Mn.shape[1] < 2:
        return
    pca = PCA(n_components=2, random_state=42)
    XY = pca.fit_transform(Mn)
    models = [s["model"] for s in samples]
    uniq = sorted(set(models))
    cmap = plt.get_cmap("tab10")
    colour = {m: cmap(i % 10) for i, m in enumerate(uniq)}

    fig, ax = plt.subplots(figsize=(6, 5))
    for m in uniq:
        mask = [mm == m for mm in models]
        ax.scatter(XY[mask, 0], XY[mask, 1], color=colour[m], label=m,
                   alpha=0.7, s=28, edgecolor="white", linewidth=0.4)
    ax.set_title(f"{prompt_id}: feature-vector PCA by model")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(fontsize=7, loc="best")
    plt.savefig(FIG_DIR / f"feature_pca_{prompt_id}.png")
    plt.close(fig)


def plot_summary_bars(per_prompt: dict[str, dict]):
    if not HAS_MPL or not per_prompt:
        return
    pids = sorted(per_prompt.keys())
    dom  = [per_prompt[p]["label_metrics"]["dominant_ratio"] for p in pids]
    ent  = [per_prompt[p]["label_metrics"]["normalised_entropy"] for p in pids]

    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 0.4), 4))
    ax.bar(pids, dom, color="#c0392b")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Dominant-architecture ratio")
    ax.set_title("Approach-level homogeneity: largest cluster / total per prompt")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(pids, rotation=60, ha="right")
    ax.axhline(0.5, color="grey", lw=0.7, ls="--")
    plt.savefig(FIG_DIR / "approach_dominance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 0.4), 4))
    ax.bar(pids, ent, color="#2c3e50")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Normalised Shannon entropy (1 = max diversity)")
    ax.set_title("Approach-level diversity per prompt")
    ax.set_xticks(range(len(pids)))
    ax.set_xticklabels(pids, rotation=60, ha="right")
    plt.savefig(FIG_DIR / "approach_entropy.png")
    plt.close(fig)


def plot_cross_model_agreement(per_prompt: dict[str, dict]):
    if not HAS_MPL or not per_prompt:
        return
    # Collect union of model names
    all_models: set[str] = set()
    for d in per_prompt.values():
        all_models.update(d["cross_model"].keys())
    models = sorted(all_models)
    if len(models) < 2:
        return

    M = np.zeros((len(models), len(models)), dtype=float)
    counts = np.zeros_like(M)
    idx = {m: i for i, m in enumerate(models)}
    for d in per_prompt.values():
        cm = d["cross_model"]
        for a, row in cm.items():
            for b, v in row.items():
                if a in idx and b in idx:
                    M[idx[a], idx[b]] += v
                    counts[idx[a], idx[b]] += 1
    counts[counts == 0] = 1
    M = M / counts

    fig, ax = plt.subplots(figsize=(0.6 * len(models) + 2, 0.6 * len(models) + 2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(models, fontsize=8)
    for i in range(len(models)):
        for j in range(len(models)):
            ax.text(j, i, f"{M[i, j]:.2f}",
                    ha="center", va="center",
                    color="black" if 0.3 < M[i, j] < 0.75 else "white",
                    fontsize=7)
    ax.set_title("Mean cross-model approach agreement\n(architecture-label match rate, averaged over prompts)")
    fig.colorbar(im, ax=ax, fraction=0.045)
    plt.savefig(FIG_DIR / "cross_model_agreement.png")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# 10. Persistence
# ─────────────────────────────────────────────────────────────────────

def to_json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, frozenset):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, CodeFeatures):
        return asdict(obj)
    return obj


def write_per_sample_csv(per_prompt: dict[str, dict], path: Path):
    import csv
    cols = ["prompt_id", "category", "model", "family", "temperature",
            "sample_index", "label", "loc", "parse_ok"] + NUMERIC_FIELDS + BOOL_FIELDS \
           + ["storage", "interface", "cli_present", "imports"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid, d in per_prompt.items():
            for s in d.get("samples", []):
                feats = s["feats"]
                row = [pid, d.get("category"), s["model"], s["family"],
                       s["temperature"], s["sample_index"], s["label"],
                       feats.loc, feats.parse_ok]
                row += [getattr(feats, k) for k in NUMERIC_FIELDS]
                row += [int(getattr(feats, k)) for k in BOOL_FIELDS]
                row += [feats.storage, feats.interface, int(feats.cli_present),
                        ";".join(feats.imports)]
                w.writerow(row)


def write_summary_csv(per_prompt: dict[str, dict], path: Path):
    import csv
    cols = ["prompt_id", "category", "n_samples", "n_unique_labels",
            "dominant_ratio", "normalised_entropy", "simpson_diversity",
            "library_entropy", "library_gini", "mean_import_jaccard",
            "mean_feature_cosine", "top_label"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for pid in sorted(per_prompt.keys()):
            d = per_prompt[pid]
            lm = d["label_metrics"]; libm = d["library_metrics"]; fm = d["feature_metrics"]
            top_label = lm["top5"][0][0] if lm["top5"] else ""
            w.writerow([pid, d["category"], lm["n"], lm["n_unique"],
                        f"{lm['dominant_ratio']:.3f}",
                        f"{lm['normalised_entropy']:.3f}",
                        f"{lm['simpson_diversity']:.3f}",
                        f"{libm['library_entropy']:.3f}",
                        f"{libm['library_gini']:.3f}",
                        f"{libm['mean_pairwise_jaccard']:.3f}",
                        f"{fm['mean_pairwise_cosine']:.3f}",
                        top_label])


# ─────────────────────────────────────────────────────────────────────
# 11. CLI entry-point
# ─────────────────────────────────────────────────────────────────────

def print_summary(per_prompt: dict[str, dict]):
    print("\n" + "=" * 100)
    print("APPROACH HOMOGENEITY  -  per-prompt summary")
    print("=" * 100)
    header = (f"{'Prompt':<8} {'Cat':<14} {'N':>4} {'unique':>7} "
              f"{'dom%':>6} {'normH':>6} {'simpson':>8} "
              f"{'libH':>6} {'libGini':>8} {'impJac':>7} {'fcos':>6}")
    print(header)
    print("-" * len(header))
    for pid in sorted(per_prompt.keys()):
        d = per_prompt[pid]; lm = d["label_metrics"]; libm = d["library_metrics"]; fm = d["feature_metrics"]
        print(f"{pid:<8} {d['category']:<14} {lm['n']:>4d} {lm['n_unique']:>7d} "
              f"{lm['dominant_ratio']*100:>5.1f}% {lm['normalised_entropy']:>6.3f} "
              f"{lm['simpson_diversity']:>8.3f} "
              f"{libm['library_entropy']:>6.2f} {libm['library_gini']:>8.3f} "
              f"{libm['mean_pairwise_jaccard']:>7.3f} {fm['mean_pairwise_cosine']:>6.3f}")
    # category aggregates
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for pid, d in per_prompt.items():
        by_cat[d["category"]].append(d)
    print("\n" + "-" * 60)
    print("Category averages")
    print("-" * 60)
    print(f"{'Category':<16} {'dom%':>6} {'normH':>6} {'libGini':>8} {'fcos':>6}")
    for cat in sorted(by_cat.keys()):
        dom = np.mean([dd["label_metrics"]["dominant_ratio"] for dd in by_cat[cat]])
        ent = np.mean([dd["label_metrics"]["normalised_entropy"] for dd in by_cat[cat]])
        gin = np.mean([dd["library_metrics"]["library_gini"] for dd in by_cat[cat]])
        fco = np.mean([dd["feature_metrics"]["mean_pairwise_cosine"] for dd in by_cat[cat]])
        print(f"{cat:<16} {dom*100:>5.1f}% {ent:>6.3f} {gin:>8.3f} {fco:>6.3f}")


def main():
    ap = argparse.ArgumentParser(description="Approach / architecture homogeneity analysis")
    ap.add_argument("--prompts", nargs="*", help="Subset of prompt ids (default: all)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Filter to a single temperature (e.g. 1.0). Default: use all.")
    ap.add_argument("--no-figures", action="store_true", help="Skip PNG plotting")
    ap.add_argument("--out-dir", type=str, default=None, help="Override output dir")
    args = ap.parse_args()

    global OUT_DIR
    if args.out_dir:
        OUT_DIR = Path(args.out_dir)

    _ensure_dirs()

    print(f"Loading raw responses from {RAW_DIR}")
    by_prompt = load_responses(args.prompts, args.temperature)
    if not by_prompt:
        print("No responses found. Exiting.")
        sys.exit(1)
    print(f"Loaded {sum(len(v) for v in by_prompt.values())} responses "
          f"across {len(by_prompt)} prompts")

    per_prompt: dict[str, dict] = {}
    for pid, recs in by_prompt.items():
        print(f"  - {pid}: {len(recs)} samples", end="")
        d = analyse_prompt(pid, recs)
        per_prompt[pid] = d
        lm = d["label_metrics"]
        print(f"   dom={lm['dominant_ratio']*100:>4.1f}%  "
              f"normH={lm['normalised_entropy']:.2f}  unique={lm['n_unique']}")

    # ── Persist machine-readable results ─────────────────────────────
    summary_per_prompt = {}
    for pid, d in per_prompt.items():
        summary_per_prompt[pid] = {
            "prompt_id":       pid,
            "category":        d["category"],
            "label_metrics":   to_json_safe(d["label_metrics"]),
            "library_metrics": to_json_safe(d["library_metrics"]),
            "feature_metrics": to_json_safe(d["feature_metrics"]),
            "intra_model":     to_json_safe(d["intra_model"]),
            "cross_model":     to_json_safe(d["cross_model"]),
        }
    with open(OUT_DIR / "per_prompt.json", "w", encoding="utf-8") as f:
        json.dump(summary_per_prompt, f, indent=2)
    print(f"Wrote {OUT_DIR / 'per_prompt.json'}")

    write_per_sample_csv(per_prompt, OUT_DIR / "per_sample.csv")
    print(f"Wrote {OUT_DIR / 'per_sample.csv'}")

    write_summary_csv(per_prompt, OUT_DIR / "summary.csv")
    print(f"Wrote {OUT_DIR / 'summary.csv'}")

    # Aggregate per-model intra-homogeneity across prompts
    agg_model: dict[str, dict] = defaultdict(lambda: {
        "n_prompts": 0, "dom_ratio_sum": 0.0, "feature_cos_sum": 0.0,
        "approach_entropy_sum": 0.0, "n_total_samples": 0,
    })
    for d in per_prompt.values():
        for m, im in d["intra_model"].items():
            agg = agg_model[m]
            agg["n_prompts"] += 1
            agg["dom_ratio_sum"] += im["dominant_ratio"]
            agg["feature_cos_sum"] += im["feature_cosine_mean"]
            agg["approach_entropy_sum"] += im["approach_entropy"]
            agg["n_total_samples"] += im["n"]
    per_model = {}
    for m, agg in agg_model.items():
        n = max(1, agg["n_prompts"])
        per_model[m] = {
            "n_prompts":          agg["n_prompts"],
            "n_total_samples":    agg["n_total_samples"],
            "mean_dominant_ratio": agg["dom_ratio_sum"] / n,
            "mean_feature_cosine": agg["feature_cos_sum"] / n,
            "mean_approach_entropy": agg["approach_entropy_sum"] / n,
        }
    with open(OUT_DIR / "per_model.json", "w", encoding="utf-8") as f:
        json.dump(per_model, f, indent=2)
    print(f"Wrote {OUT_DIR / 'per_model.json'}")

    cross_avg: dict = defaultdict(lambda: defaultdict(list))
    for d in per_prompt.values():
        for a, row in d["cross_model"].items():
            for b, v in row.items():
                cross_avg[a][b].append(v)
    cross_avg_mean = {a: {b: float(np.mean(vs)) for b, vs in row.items()}
                      for a, row in cross_avg.items()}
    with open(OUT_DIR / "cross_model.json", "w", encoding="utf-8") as f:
        json.dump(cross_avg_mean, f, indent=2)
    print(f"Wrote {OUT_DIR / 'cross_model.json'}")

    # ── Stdout summary table ─────────────────────────────────────────
    print_summary(per_prompt)

    # ── Figures ──────────────────────────────────────────────────────
    if not args.no_figures and HAS_MPL:
        print("\nGenerating figures...")
        for pid, d in per_prompt.items():
            plot_approach_distribution(pid, d["samples"])
            plot_library_frequency(pid, d["samples"])
            plot_pca(pid, d["samples"])
        plot_summary_bars(per_prompt)
        plot_cross_model_agreement(per_prompt)
        print(f"Figures -> {FIG_DIR}")


if __name__ == "__main__":
    main()
