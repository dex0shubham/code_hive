"""
Code Hivemind — Diversity Metrics
==================================
Measures for quantifying code homogeneity across and within models.

Metrics are organized in 4 layers:
  1. Surface-level:  token/character overlap, n-gram similarity
  2. Structural:     AST-based similarity after normalization
  3. Semantic:       embedding-based cosine similarity
  4. Behavioral:     functional equivalence testing

We measure both:
  - INTRA-model diversity  (how diverse is one model's own outputs?)
  - INTER-model diversity  (how similar are outputs across different models?)
"""

import ast
import re
import hashlib
import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════
# LAYER 1: Surface-Level Metrics
# ═══════════════════════════════════════════════════════════════════

def extract_code_block(text: str) -> str:
    """Extract code from a markdown-fenced response, or return as-is."""
    # Try to find ```python ... ``` or ```javascript ... ``` blocks
    pattern = r"```(?:python|javascript|js|py)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n".join(matches)
    # If no fenced block, assume entire response is code
    return text.strip()


def character_ngrams(text: str, n: int) -> Counter:
    """Get character n-gram frequency distribution."""
    return Counter(text[i:i+n] for i in range(len(text) - n + 1))


def token_ngrams(text: str, n: int) -> Counter:
    """Get token n-gram frequency distribution (split on whitespace + punctuation)."""
    tokens = re.findall(r"[a-zA-Z_]\w*|[^\s\w]", text)
    return Counter(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Jaccard index between two sets."""
    if not set_a and not set_b:
        return 1.0
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / len(union) if union else 0.0


def ngram_jaccard(text_a: str, text_b: str, n: int = 3) -> float:
    """Jaccard similarity of token n-grams."""
    grams_a = set(token_ngrams(text_a, n).keys())
    grams_b = set(token_ngrams(text_b, n).keys())
    return jaccard_similarity(grams_a, grams_b)


def exact_match_rate(responses: list[str]) -> float:
    """Fraction of response pairs that are exactly identical."""
    if len(responses) < 2:
        return 0.0
    n_pairs = 0
    n_match = 0
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            n_pairs += 1
            if responses[i].strip() == responses[j].strip():
                n_match += 1
    return n_match / n_pairs if n_pairs else 0.0


# ═══════════════════════════════════════════════════════════════════
# LAYER 2: Structural (AST) Metrics
# ═══════════════════════════════════════════════════════════════════

def normalize_identifiers(code: str) -> str:
    """
    Normalize all user-defined identifiers to generic names.
    This lets us compare STRUCTURE while ignoring NAMING.
    e.g., 'def calculate_total(items):' → 'def func_0(var_0):'
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code  # can't parse, return as-is

    # Collect all user-defined names
    name_map = {}
    counters = {"func": 0, "var": 0, "cls": 0, "arg": 0}

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name not in name_map:
                name_map[node.name] = f"func_{counters['func']}"
                counters["func"] += 1
        elif isinstance(node, ast.ClassDef):
            if node.name not in name_map:
                name_map[node.name] = f"cls_{counters['cls']}"
                counters["cls"] += 1
        elif isinstance(node, ast.arg):
            if node.arg not in name_map and node.arg != "self":
                name_map[node.arg] = f"arg_{counters['arg']}"
                counters["arg"] += 1
        elif isinstance(node, ast.Name):
            if node.id not in name_map and node.id not in dir(__builtins__):
                name_map[node.id] = f"var_{counters['var']}"
                counters["var"] += 1

    # Apply renaming
    result = code
    # Sort by length (longest first) to avoid partial replacements
    for old_name, new_name in sorted(name_map.items(), key=lambda x: -len(x[0])):
        result = re.sub(rf"\b{re.escape(old_name)}\b", new_name, result)

    return result


def ast_fingerprint(code: str) -> str:
    """
    Create a structural fingerprint of Python code by serializing its AST.
    Ignores names, string literals, and number values — captures only structure.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return hashlib.md5(code.encode()).hexdigest()

    def _serialize(node, depth=0):
        if depth > 30:
            return "..."
        parts = [type(node).__name__]
        for child in ast.iter_child_nodes(node):
            parts.append(_serialize(child, depth + 1))
        return f"({' '.join(parts)})"

    return _serialize(tree)


def ast_structural_similarity(code_a: str, code_b: str) -> float:
    """
    Structural similarity based on normalized AST fingerprints.
    Uses Jaccard on AST node sequences.
    """
    fp_a = ast_fingerprint(code_a)
    fp_b = ast_fingerprint(code_b)

    # Tokenize fingerprints
    tokens_a = set(re.findall(r"\w+", fp_a))
    tokens_b = set(re.findall(r"\w+", fp_b))

    return jaccard_similarity(tokens_a, tokens_b)


def naming_similarity(code_a: str, code_b: str) -> float:
    """
    Measure how similar the NAMING CHOICES are between two code samples.
    Extracts all user-defined identifiers and computes overlap.
    """
    def extract_names(code: str) -> set:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(re.findall(r"[a-zA-Z_]\w{2,}", code))

        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
            elif isinstance(node, ast.arg) and node.arg != "self":
                names.add(node.arg)
            elif isinstance(node, ast.Name):
                if node.id not in dir(__builtins__):
                    names.add(node.id)
        return names

    names_a = extract_names(code_a)
    names_b = extract_names(code_b)
    return jaccard_similarity(names_a, names_b)


# ═══════════════════════════════════════════════════════════════════
# LAYER 3: Semantic (Embedding) Metrics
# ═══════════════════════════════════════════════════════════════════

class EmbeddingComputer:
    """
    Computes semantic similarity using sentence-transformers.
    Lazy-loads the model on first use.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts. Returns (N, D) array."""
        self._load()
        return self._model.encode(texts, show_progress_bar=False)

    def cosine_sim_matrix(self, texts: list[str]) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        embeddings = self.embed(texts)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-8, None)
        normed = embeddings / norms
        return normed @ normed.T

    def mean_pairwise_similarity(self, texts: list[str]) -> float:
        """Mean of upper-triangle pairwise cosine similarities."""
        if len(texts) < 2:
            return 0.0
        sim_matrix = self.cosine_sim_matrix(texts)
        # Extract upper triangle (excluding diagonal)
        mask = np.triu(np.ones_like(sim_matrix, dtype=bool), k=1)
        return float(sim_matrix[mask].mean())


# ═══════════════════════════════════════════════════════════════════
# LAYER 4: Behavioral Metrics
# ═══════════════════════════════════════════════════════════════════

def extract_imports(code: str) -> set:
    """Extract all imported module names from code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set(re.findall(r"import\s+(\w+)|from\s+(\w+)", code))

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def library_choice_diversity(code_samples: list[str]) -> dict:
    """
    Measure diversity in library/import choices across samples.
    Returns distribution of import sets and entropy.
    """
    import_sets = []
    for code in code_samples:
        imports = frozenset(extract_imports(code))
        import_sets.append(imports)

    # Count unique import sets
    counter = Counter(import_sets)
    total = len(import_sets)

    # Shannon entropy of import-set distribution
    probs = np.array([c / total for c in counter.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return {
        "unique_import_sets": len(counter),
        "total_samples": total,
        "entropy": float(entropy),
        "most_common": [
            (list(imports), count)
            for imports, count in counter.most_common(5)
        ],
    }


def design_pattern_detection(code: str) -> set:
    """
    Detect common design patterns/approaches in code.
    Returns a set of pattern tags.
    """
    patterns = set()

    # Detect class vs functional approach
    if re.search(r"\bclass\s+\w+", code):
        patterns.add("class_based")
    if re.search(r"\bdef\s+\w+", code) and "class_based" not in patterns:
        patterns.add("function_based")

    # Detect specific patterns
    if re.search(r"@\w+", code):
        patterns.add("uses_decorators")
    if re.search(r"@dataclass", code):
        patterns.add("uses_dataclass")
    if re.search(r"\basync\s+(def|for|with)", code):
        patterns.add("async_pattern")
    if re.search(r"typing\.|Optional|Union|List\[|Dict\[", code):
        patterns.add("type_hints")
    if re.search(r"\bEnum\b|\benum\b", code):
        patterns.add("uses_enum")
    if re.search(r"__init__.*\*\*kwargs", code):
        patterns.add("kwargs_pattern")
    if re.search(r"try:.*except.*:", code, re.DOTALL):
        patterns.add("error_handling")
    if re.search(r"logging\.|logger\.", code):
        patterns.add("uses_logging")
    if re.search(r"with\s+\w+.*:", code):
        patterns.add("context_manager")
    if re.search(r"\[.*for\s+\w+\s+in\s+", code):
        patterns.add("list_comprehension")
    if re.search(r"lambda\s+", code):
        patterns.add("uses_lambda")
    if re.search(r"namedtuple|NamedTuple", code):
        patterns.add("uses_namedtuple")
    if re.search(r"__slots__", code):
        patterns.add("uses_slots")

    return patterns


def pattern_diversity(code_samples: list[str]) -> dict:
    """Measure diversity in design patterns across samples."""
    all_patterns = [design_pattern_detection(c) for c in code_samples]
    pattern_counter = Counter(frozenset(p) for p in all_patterns)
    total = len(all_patterns)

    probs = np.array([c / total for c in pattern_counter.values()])
    entropy = -np.sum(probs * np.log2(probs + 1e-10))

    return {
        "unique_pattern_sets": len(pattern_counter),
        "total_samples": total,
        "entropy": float(entropy),
        "most_common": [
            (list(patterns), count)
            for patterns, count in pattern_counter.most_common(5)
        ],
    }


# ═══════════════════════════════════════════════════════════════════
# COMPOSITE SCORER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DiversityReport:
    """Full diversity report for a set of code responses."""
    prompt_id: str
    group_label: str           # e.g., model name or "all_models"
    n_samples: int
    # Surface
    exact_match_rate: float
    mean_ngram_jaccard_3: float
    # Structural
    mean_ast_similarity: float
    mean_naming_similarity: float
    # Semantic
    mean_embedding_similarity: float
    # Behavioral
    library_diversity: dict
    pattern_diversity: dict
    # Composite
    hivemind_score: float      # 0=fully diverse, 1=full hivemind


def compute_diversity(
    code_samples: list[str],
    prompt_id: str,
    group_label: str,
    embedder: Optional[EmbeddingComputer] = None,
    max_pairs: int = 500,
) -> DiversityReport:
    """
    Compute all diversity metrics for a set of code samples.

    Args:
        code_samples: List of code strings (already extracted from responses).
        prompt_id: ID of the prompt these respond to.
        group_label: Label for this group (model name, "intra", "inter", etc.).
        embedder: Optional pre-initialized embedding computer.
        max_pairs: Max pairwise comparisons (for efficiency).
    """
    n = len(code_samples)
    if n < 2:
        return DiversityReport(
            prompt_id=prompt_id, group_label=group_label, n_samples=n,
            exact_match_rate=0, mean_ngram_jaccard_3=0, mean_ast_similarity=0,
            mean_naming_similarity=0, mean_embedding_similarity=0,
            library_diversity={}, pattern_diversity={},
            hivemind_score=0,
        )

    # Subsample pairs if needed
    all_pairs = list(itertools.combinations(range(n), 2))
    if len(all_pairs) > max_pairs:
        rng = np.random.RandomState(42)
        pair_indices = rng.choice(len(all_pairs), max_pairs, replace=False)
        pairs = [all_pairs[i] for i in pair_indices]
    else:
        pairs = all_pairs

    # ── Surface metrics ──
    em_rate = exact_match_rate(code_samples)

    ngram_sims = [
        ngram_jaccard(code_samples[i], code_samples[j], n=3)
        for i, j in pairs
    ]
    mean_ngram = float(np.mean(ngram_sims))

    # ── Structural metrics ──
    ast_sims = [
        ast_structural_similarity(code_samples[i], code_samples[j])
        for i, j in pairs
    ]
    mean_ast = float(np.mean(ast_sims))

    naming_sims = [
        naming_similarity(code_samples[i], code_samples[j])
        for i, j in pairs
    ]
    mean_naming = float(np.mean(naming_sims))

    # ── Semantic metrics ──
    if embedder is None:
        embedder = EmbeddingComputer()
    mean_embed = embedder.mean_pairwise_similarity(code_samples)

    # ── Behavioral metrics ──
    lib_div = library_choice_diversity(code_samples)
    pat_div = pattern_diversity(code_samples)

    # ── Composite hivemind score ──
    # Weighted average of similarity metrics (higher = more hivemind)
    hivemind = (
        0.15 * mean_ngram
        + 0.25 * mean_ast
        + 0.20 * mean_naming
        + 0.30 * mean_embed
        + 0.10 * (1.0 - min(pat_div["entropy"] / 3.0, 1.0))  # normalize entropy
    )

    return DiversityReport(
        prompt_id=prompt_id,
        group_label=group_label,
        n_samples=n,
        exact_match_rate=em_rate,
        mean_ngram_jaccard_3=mean_ngram,
        mean_ast_similarity=mean_ast,
        mean_naming_similarity=mean_naming,
        mean_embedding_similarity=mean_embed,
        library_diversity=lib_div,
        pattern_diversity=pat_div,
        hivemind_score=float(hivemind),
    )