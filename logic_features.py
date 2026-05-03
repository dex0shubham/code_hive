"""
Logic-level feature extraction for code samples.

Captures lightweight control-flow and data-flow signals that approximate the
"core logic / strategy" used in a snippet, complementing the surface-level
metrics in diversity_metrics.py and the architecture metrics in
approach_analysis.py.

All features are deterministic and AST-based (no execution). Each snippet
produces:
  - a fixed-length numeric feature vector for cosine similarity
  - a short, comparable "logic label" (e.g. "loop_asc|bitwise")

These let us answer questions like: "did two models pick the same algorithmic
strategy?" even when their text/AST shapes differ.
"""

from __future__ import annotations

import ast
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np


_BITWISE_BINOPS = (ast.BitAnd, ast.BitOr, ast.BitXor, ast.LShift, ast.RShift)
_BITWISE_UNARYOPS = (ast.Invert,)
_ARITH_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow)


_OP_KEYS = [
    "BitAnd", "BitOr", "BitXor", "LShift", "RShift", "Invert",
    "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow",
]
_PROV_KEYS = [
    "const", "name", "loop_var", "loop_arith", "arith",
    "bitwise", "call", "comp", "fstring", "subscript", "other",
]


def _is_call_to(node, name: str) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == name
    if isinstance(func, ast.Attribute):
        return func.attr == name
    return False


def _classify_for_iter(node: ast.For) -> dict:
    """For 'for x in <iter>' loops, classify direction and step where possible."""
    out = {"iter_kind": "other", "direction": "unknown", "step": None}
    it = node.iter
    if isinstance(it, ast.Call):
        fname = ""
        if isinstance(it.func, ast.Name):
            fname = it.func.id
        elif isinstance(it.func, ast.Attribute):
            fname = it.func.attr

        if fname == "range":
            out["iter_kind"] = "range"
            args = it.args
            if len(args) <= 2:
                out.update(direction="asc", step=1)
            elif len(args) == 3:
                step = args[2]
                if isinstance(step, ast.Constant) and isinstance(step.value, int):
                    out["step"] = step.value
                    if step.value < 0:
                        out["direction"] = "desc"
                    elif step.value > 0:
                        out["direction"] = "asc"
                    else:
                        out["direction"] = "zero"
                elif isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
                    out["direction"] = "desc"
        elif fname == "reversed":
            out["iter_kind"] = "iter"
            out["direction"] = "desc"
        elif fname in {"enumerate"}:
            out["iter_kind"] = "enumerate"
        elif fname in {"zip", "map", "filter"}:
            out["iter_kind"] = "iter"
    elif isinstance(it, (ast.List, ast.Tuple, ast.Set)):
        out["iter_kind"] = "iter"
    elif isinstance(it, ast.Name):
        out["iter_kind"] = "iter"
    return out


def _classify_value_provenance(expr: ast.AST, loop_vars: set[str]) -> str:
    """Coarse origin classifier for a value expression. Always returns a key in _PROV_KEYS."""
    if isinstance(expr, ast.Constant):
        return "const"
    if isinstance(expr, ast.JoinedStr):
        for v in expr.values:
            if isinstance(v, ast.FormattedValue):
                kind = _classify_value_provenance(v.value, loop_vars)
                if kind != "other":
                    return kind
        return "fstring"
    if isinstance(expr, ast.Name):
        return "loop_var" if expr.id in loop_vars else "name"
    if isinstance(expr, ast.UnaryOp):
        if isinstance(expr.op, _BITWISE_UNARYOPS):
            return "bitwise"
        return "other"
    if isinstance(expr, ast.BinOp):
        if isinstance(expr.op, _BITWISE_BINOPS):
            return "bitwise"
        if isinstance(expr.op, _ARITH_BINOPS):
            for child in ast.walk(expr):
                if isinstance(child, ast.Name) and child.id in loop_vars:
                    return "loop_arith"
            return "arith"
        return "other"
    if isinstance(expr, ast.Call):
        return "call"
    if isinstance(expr, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        return "comp"
    if isinstance(expr, ast.Subscript):
        return "subscript"
    return "other"


def _max_depth(node: ast.AST, depth: int = 0) -> int:
    BLOCKING = (ast.For, ast.AsyncFor, ast.While, ast.If, ast.Try,
                ast.With, ast.AsyncWith, ast.FunctionDef,
                ast.AsyncFunctionDef, ast.ClassDef)
    best = depth
    for child in ast.iter_child_nodes(node):
        d = _max_depth(child, depth + 1 if isinstance(child, BLOCKING) else depth)
        if d > best:
            best = d
    return best


@dataclass
class LogicFeatures:
    parse_ok: bool = True

    # Control-flow counts
    num_for: int = 0
    num_while: int = 0
    num_if: int = 0
    num_try: int = 0
    num_break: int = 0
    num_continue: int = 0
    num_return: int = 0
    num_yield: int = 0
    num_function_calls: int = 0
    num_print_calls: int = 0
    max_nesting_depth: int = 0

    # Loop semantics
    loop_directions: list[str] = field(default_factory=list)
    loop_steps: list = field(default_factory=list)
    has_ascending_loop: bool = False
    has_descending_loop: bool = False
    uses_comprehension: bool = False
    has_recursion: bool = False
    has_generator: bool = False

    # Operators
    op_counts: dict[str, int] = field(default_factory=dict)
    has_bitwise: bool = False
    has_arith: bool = False

    # Output-sink data flow
    sink_provenance_counts: dict[str, int] = field(default_factory=dict)
    has_unrolled_outputs: bool = False

    def vector(self) -> np.ndarray:
        scalar_fields = [
            "num_for", "num_while", "num_if", "num_try", "num_break", "num_continue",
            "num_return", "num_yield", "num_function_calls", "num_print_calls",
            "max_nesting_depth",
        ]
        bool_fields = [
            "has_ascending_loop", "has_descending_loop", "uses_comprehension",
            "has_recursion", "has_generator", "has_bitwise", "has_arith",
            "has_unrolled_outputs",
        ]
        nums = [getattr(self, f) for f in scalar_fields]
        bools = [int(getattr(self, f)) for f in bool_fields]
        ops = [self.op_counts.get(k, 0) for k in _OP_KEYS]
        prov = [self.sink_provenance_counts.get(k, 0) for k in _PROV_KEYS]
        return np.array(nums + bools + ops + prov, dtype=float)


def extract_logic_features(code: str) -> LogicFeatures:
    f = LogicFeatures()
    try:
        tree = ast.parse(code or "")
    except SyntaxError:
        f.parse_ok = False
        return f

    f.max_nesting_depth = _max_depth(tree)

    # Loop variable names (cheap union across all for-loops)
    loop_vars: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            tgt = node.target
            if isinstance(tgt, ast.Name):
                loop_vars.add(tgt.id)
            elif isinstance(tgt, (ast.Tuple, ast.List)):
                for el in tgt.elts:
                    if isinstance(el, ast.Name):
                        loop_vars.add(el.id)

    func_defs: dict[str, ast.AST] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_defs[node.name] = node

    op_counter: Counter = Counter()
    prov_counter: Counter = Counter()

    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            f.num_for += 1
            info = _classify_for_iter(node)
            f.loop_directions.append(info["direction"])
            f.loop_steps.append(info["step"])
            if info["direction"] == "asc":
                f.has_ascending_loop = True
            elif info["direction"] == "desc":
                f.has_descending_loop = True
        elif isinstance(node, ast.While):
            f.num_while += 1
            f.loop_directions.append("unknown")
            f.loop_steps.append(None)
        elif isinstance(node, ast.If):
            f.num_if += 1
        elif isinstance(node, ast.Try):
            f.num_try += 1
        elif isinstance(node, ast.Break):
            f.num_break += 1
        elif isinstance(node, ast.Continue):
            f.num_continue += 1
        elif isinstance(node, ast.Return):
            f.num_return += 1
            if node.value is not None:
                prov_counter[_classify_value_provenance(node.value, loop_vars)] += 1
        elif isinstance(node, (ast.Yield, ast.YieldFrom)):
            f.num_yield += 1
            f.has_generator = True
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            f.uses_comprehension = True
        elif isinstance(node, ast.Call):
            f.num_function_calls += 1
            if _is_call_to(node, "print") and node.args:
                f.num_print_calls += 1
                prov_counter[_classify_value_provenance(node.args[0], loop_vars)] += 1
        elif isinstance(node, ast.BinOp):
            op_counter[type(node.op).__name__] += 1
            if isinstance(node.op, _BITWISE_BINOPS):
                f.has_bitwise = True
            if isinstance(node.op, _ARITH_BINOPS):
                f.has_arith = True
        elif isinstance(node, ast.UnaryOp):
            op_counter[type(node.op).__name__] += 1
            if isinstance(node.op, _BITWISE_UNARYOPS):
                f.has_bitwise = True

    # Naive recursion: function body contains a Call to its own name.
    for name, fnode in func_defs.items():
        for inner in ast.walk(fnode):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name) and inner.func.id == name:
                f.has_recursion = True
                break
        if f.has_recursion:
            break

    if f.num_for + f.num_while == 0 and f.num_print_calls >= 3:
        f.has_unrolled_outputs = True

    f.op_counts = dict(op_counter)
    f.sink_provenance_counts = dict(prov_counter)
    return f


def derive_logic_label(f: LogicFeatures) -> str:
    """Short, comparable label summarising core logic strategy."""
    if not f.parse_ok:
        return "unparseable"
    parts: list[str] = []
    if f.has_recursion:
        parts.append("recursive")
    if f.has_generator:
        parts.append("generator")
    if f.has_unrolled_outputs:
        parts.append("unrolled")
    elif f.has_ascending_loop and f.has_descending_loop:
        parts.append("loop_mixed")
    elif f.has_ascending_loop:
        parts.append("loop_asc")
    elif f.has_descending_loop:
        parts.append("loop_desc")
    elif f.num_for + f.num_while > 0:
        parts.append("loop_other")
    if f.uses_comprehension:
        parts.append("comprehension")
    if f.has_bitwise:
        parts.append("bitwise")
    if not parts:
        parts.append("straight_line")
    return "|".join(parts)


def cosine_matrix(M: np.ndarray) -> np.ndarray:
    if M.size == 0:
        return np.zeros((M.shape[0], M.shape[0]))
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    Mn = M / norms
    return Mn @ Mn.T


def features_as_dict(f: LogicFeatures) -> dict:
    return asdict(f)
