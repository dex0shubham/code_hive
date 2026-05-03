"""
Trace-based behavioral equivalence for generated code samples.

Each runnable snippet is executed in a separate Python subprocess with a
hard timeout. We capture stdout/stderr, then compute pairwise similarity of
the observable outputs. This gives an "are these doing the same thing?"
signal that is independent of source-text or AST similarity.

WARNING: This module *executes* untrusted LLM-generated Python locally.
Network and filesystem are NOT sandboxed here. Run inside a VM/container if
your prompts may produce dangerous code.
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass


_MAIN_HINTS = ("if __name__", '__main__')


def is_runnable_as_script(code: str) -> bool:
    """Heuristic: does this snippet do something observable when run directly?"""
    if not code:
        return False
    if any(h in code for h in _MAIN_HINTS):
        return True
    for raw in code.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw.startswith((" ", "\t")):  # indented (inside def/class)
            continue
        if raw.startswith(("def ", "class ", "import ", "from ", "@")):
            continue
        if stripped.startswith(("print(", "for ", "while ", "if ", "try:", "input(")):
            return True
    return False


@dataclass
class RunResult:
    success: bool
    stdout: str
    stderr: str
    returncode: int
    elapsed_ms: float
    timed_out: bool = False


def run_code(code: str, timeout: float = 8.0) -> RunResult:
    """Run `code` in a subprocess, capture stdout/stderr, enforce timeout."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(code)
            tmp_path = f.name
        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            elapsed = (time.monotonic() - t0) * 1000
            return RunResult(
                success=(proc.returncode == 0),
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                returncode=proc.returncode,
                elapsed_ms=elapsed,
            )
        except subprocess.TimeoutExpired:
            elapsed = (time.monotonic() - t0) * 1000
            return RunResult(
                success=False, stdout="", stderr="TIMEOUT",
                returncode=-1, elapsed_ms=elapsed, timed_out=True,
            )
        except Exception as e:  # noqa: BLE001
            elapsed = (time.monotonic() - t0) * 1000
            return RunResult(
                success=False, stdout="", stderr=f"EXEC_ERROR: {type(e).__name__}: {e}",
                returncode=-2, elapsed_ms=elapsed,
            )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def normalize_output(s: str) -> tuple[str, ...]:
    return tuple(line.strip() for line in (s or "").splitlines() if line.strip())


def compare_outputs(a: str, b: str) -> dict:
    la = normalize_output(a)
    lb = normalize_output(b)
    sa, sb = set(la), set(lb)
    union = sa | sb
    return {
        "exact_match": la == lb and len(la) > 0,
        "set_match":   sa == sb and len(sa) > 0,
        "jaccard":     len(sa & sb) / len(union) if union else 0.0,
        "seq_ratio":   difflib.SequenceMatcher(None, la, lb).ratio() if la and lb else 0.0,
        "len_a":       len(la),
        "len_b":       len(lb),
    }
