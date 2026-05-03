#!/usr/bin/env python3

"""
Download code-generation datasets from Hugging Face and segregate prompts by
source pool (A closed baseline vs B open-ended candidates), task language, and
open vs closed-ended labels.

Two-stage pipeline:

Stage 1:
    Rule-based candidate extraction (lexical openness score).

Stage 2:
    - heuristic: fast rules only (default for speed).
    - hybrid: heuristics for all rows; LLM only on Pool B rows that look ambiguous.
    - llm: one API call per row (slow; strongest labels).

Usage:
    python prepare_code_openended_datasets.py --stage2_mode hybrid

Optional:
    python prepare_code_openended_datasets.py --max_rows_per_dataset 5000
    python prepare_code_openended_datasets.py --output_dir ./local_datasets
    python prepare_code_openended_datasets.py --llm_cache_path ./local_datasets/stage2_llm_cache.jsonl
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from openai import OpenAI
from tqdm import tqdm


def _get_openai_client() -> OpenAI:
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# -----------------------------
# Dataset pools
# -----------------------------

POOL_A_CLOSED_BASELINE = [
    "greengerong/leetcode",
    "newfacade/LeetCodeDataset",
    "Alishohadaee/leetcode-problems-dataset",
    "kaysss/leetcode-problem-set",
    "whiskwhite/leetcode-complete",
    "codeparrot/apps",
    "BAAI/TACO",
    "livecodebench/code_generation",
    "livecodebench/code_generation_lite",
    "nvidia/LiveCodeBench-CPP",
]

POOL_B_OPEN_ENDED_CANDIDATES = [
    "HuggingFaceH4/CodeAlpaca_20K",
    "iamtarun/python_code_instructions_18k_alpaca",
    "iamtarun/code_instructions_120k_alpaca",
    "ise-uiuc/Magicoder-OSS-Instruct-75K",
    "nickrosh/Evol-Instruct-Code-80k-v1",
    "codefuse-ai/Evol-instruction-66k",
    "m-a-p/CodeFeedback-Filtered-Instruction",
    "HuggingFaceH4/Code-Feedback",
    "fxmeng/CodeFeedback-Python105K",
    "theblackcat102/evol-codealpaca-v1",
]


# -----------------------------
# Stage 1 rule-based filter
# -----------------------------

OPEN_ENDED_POSITIVE = [
    "create an application",
    "create a python application",
    "build an application",
    "build a web app",
    "create a web app",
    "build a cli",
    "command-line tool",
    "dashboard",
    "pipeline",
    "data augmentation",
    "web crawler",
    "crawler",
    "scraper",
    "text-based adventure game",
    "adventure game",
    "simulation",
    "design an api",
    "create an api",
    "rest api",
    "graphql api",
    "library",
    "framework",
    "plugin",
    "refactor",
    "improve",
    "optimize",
    "make it modular",
    "rewrite",
    "frontend",
    "backend",
    "full-stack",
    "react component",
    "vue component",
    "javascript code to",
    "generate a url path",
    "write a program that",
    "write a script that",
    "build a tool",
    "create a tool",
    "design a system",
    "implement a system",
    "mini project",
    "small project",
    "application should support",
    "user input",
    "provide textual feedback",
    "interactive",
    "visualization",
    "plot",
    "gui",
    "tkinter",
    "pygame",
    "flask app",
    "fastapi app",
    "django app",
    "streamlit",
    "chatbot",
    "recommendation system",
    "authentication system",
    "crud",
    "todo app",
    "blog app",
    "portfolio website",
    "landing page",
]

CLOSED_ENDED_NEGATIVE = [
    "given an array",
    "given a string",
    "given the root",
    "given a binary tree",
    "given a linked list",
    "given an integer",
    "given two integers",
    "return the",
    "return a",
    "return an",
    "constraints:",
    "example 1:",
    "example:",
    "input:",
    "output:",
    "explanation:",
    "you must solve",
    "time complexity",
    "space complexity",
    "leetcode",
    "atcoder",
    "codeforces",
    "hidden test cases",
    "passes all tests",
    "pass all tests",
    "implement the function",
    "complete the function",
    "function signature",
    "class solution",
    "def solve",
    "stdin",
    "stdout",
    "sample input",
    "sample output",
    "the answer is guaranteed",
    "modulo",
    "1 <= ",
    "0 <= ",
    "n <=",
    "constraints",
]

# Stronger regex signals for closed-ended competitive-programming problems.
CLOSED_ENDED_REGEXES = [
    r"\bexample\s*\d+\s*:",
    r"\binput\s*:",
    r"\boutput\s*:",
    r"\bconstraints\s*:",
    r"\breturn\s+(the|a|an)\b",
    r"\bgiven\s+(an?|the)\s+(array|string|integer|linked list|binary tree|matrix|graph)\b",
    r"\bclass\s+Solution\b",
    r"\bdef\s+\w+\s*\(",
    r"\bpublic\s+\w+\s+\w+\s*\(",
]


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    if isinstance(text, str):
        return re.sub(r"\s+", " ", text).strip()
    if isinstance(text, (list, tuple)):
        return " ".join(normalize_text(x) for x in text)
    if isinstance(text, dict):
        return json.dumps(text, ensure_ascii=False)
    return str(text)


def openness_score(prompt: str) -> int:
    p = prompt.lower()

    pos = sum(term in p for term in OPEN_ENDED_POSITIVE)
    neg = sum(term in p for term in CLOSED_ENDED_NEGATIVE)

    regex_neg = sum(bool(re.search(pattern, prompt, flags=re.IGNORECASE)) for pattern in CLOSED_ENDED_REGEXES)

    # Positive signals are weighted higher because Pool B contains many instruction-style prompts.
    # Regex negatives are weighted higher because they strongly indicate algorithmic benchmark format.
    return 2 * pos - neg - 2 * regex_neg


def stage1_is_open_ended_candidate(prompt: str) -> bool:
    return openness_score(prompt) >= 2


# -----------------------------
# Stage 2 validation
# -----------------------------

OPEN_ENDED_DIMENSION_PATTERNS = {
    "architecture": [
        "application",
        "system",
        "framework",
        "library",
        "module",
        "backend",
        "frontend",
        "full-stack",
        "api",
    ],
    "library choice": [
        "scraper",
        "crawler",
        "pipeline",
        "visualization",
        "plot",
        "web app",
        "flask",
        "fastapi",
        "django",
        "react",
        "streamlit",
    ],
    "UI": [
        "ui",
        "gui",
        "frontend",
        "react",
        "vue",
        "component",
        "dashboard",
        "landing page",
        "website",
        "button",
        "popup",
    ],
    "API design": [
        "api",
        "endpoint",
        "rest",
        "graphql",
        "crud",
        "authentication",
        "route",
    ],
    "style": [
        "refactor",
        "improve",
        "rewrite",
        "clean",
        "readable",
        "modular",
        "optimize",
    ],
    "algorithm choice": [
        "recommendation",
        "simulation",
        "game",
        "search",
        "ranking",
        "scheduler",
        "planner",
    ],
}


def detect_open_ended_dimensions(prompt: str) -> List[str]:
    p = prompt.lower()
    dimensions = []

    for dimension, patterns in OPEN_ENDED_DIMENSION_PATTERNS.items():
        if any(pattern in p for pattern in patterns):
            dimensions.append(dimension)

    return sorted(set(dimensions))


NON_PYTHON_LANGUAGE_HINTS = [
    "javascript",
    "typescript",
    "java",
    "c++",
    "cpp",
    "c#",
    "csharp",
    "golang",
    "go ",
    "rust",
    "kotlin",
    "swift",
    "ruby",
    "php",
    "scala",
    "r language",
    "matlab",
]


PYTHON_LANGUAGE_HINTS = [
    "python",
    "python3",
    "py ",
    "django",
    "flask",
    "fastapi",
    "streamlit",
    "pandas",
    "numpy",
    "pytest",
]


def infer_task_language(prompt: str) -> str:
    p = prompt.lower()

    if any(hint in p for hint in NON_PYTHON_LANGUAGE_HINTS):
        return "NON_PYTHON"
    if any(hint in p for hint in PYTHON_LANGUAGE_HINTS):
        return "PYTHON"
    return "NON_PYTHON"


def strong_closed_signals(prompt: str) -> bool:
    p = prompt.lower()
    return any(
        signal in p
        for signal in [
            "constraints:",
            "example 1:",
            "sample input",
            "sample output",
            "class solution",
            "return the",
            "given an array",
            "given a string",
            "given the root",
        ]
    )


def stage2_validate_open_ended_heuristic(prompt: str, source_pool: str) -> Dict[str, Any]:
    """
    Deterministic Stage 2 (no LLM). Same output keys as the LLM path for downstream code.
    """
    score = openness_score(prompt)
    dimensions = detect_open_ended_dimensions(prompt)
    strong_closed = strong_closed_signals(prompt)
    candidate = stage1_is_open_ended_candidate(prompt)

    if source_pool == "A_closed_baseline":
        label = (
            "OPEN_ENDED"
            if candidate and len(dimensions) >= 2 and not strong_closed and score >= 4
            else "CLOSED_ENDED"
        )
    else:
        label = "OPEN_ENDED" if candidate and len(dimensions) >= 1 and not strong_closed else "CLOSED_ENDED"

    reason = (
        f"heuristic; score={score}; dimensions={dimensions}; "
        f"strong_closed={strong_closed}; source_pool={source_pool}"
    )

    return {
        "label": label,
        "reason": reason,
        "open_ended_dimensions": dimensions,
        "task_language": infer_task_language(prompt),
        "stage1_candidate": candidate,
        "openness_score": score,
        "stage2_backend": "heuristic",
    }


def stage2_needs_llm(prompt: str, source_pool: str, heuristic: Dict[str, Any]) -> bool:
    """
    True only for ambiguous Pool B rows: conflicting or weak heuristic evidence.
    Pool A skips LLM entirely (saves most calls; baseline is overwhelmingly closed).
    """
    if source_pool != "B_open_ended_candidates":
        return False

    candidate = heuristic["stage1_candidate"]
    label = heuristic["label"]
    score = heuristic["openness_score"]
    dimensions = heuristic["open_ended_dimensions"]
    strong_closed = strong_closed_signals(prompt)

    if strong_closed and candidate:
        return True
    if candidate and label == "CLOSED_ENDED":
        return True
    if (not candidate) and label == "OPEN_ENDED":
        return True
    if label == "OPEN_ENDED" and len(dimensions) == 0:
        return True
    return False


def llm_cache_key(prompt: str, source_pool: str) -> str:
    h = hashlib.sha256()
    h.update(source_pool.encode("utf-8"))
    h.update(b"\n")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def load_llm_cache_jsonl(path: Path) -> Dict[str, Dict[str, Any]]:
    cache: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cache[obj["key"]] = obj["value"]
    return cache


def append_llm_cache_jsonl(path: Path, key: str, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps({"key": key, "value": value}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(record + "\n")


def stage2_validate_open_ended_llm(prompt: str, source_pool: str) -> Dict[str, Any]:
    system_msg = """You are classifying coding prompts for a research dataset.

OPEN_ENDED means many substantially different programs, architectures,
libraries, interfaces, or implementation designs could all satisfy the request,
and correctness is not fully captured by deterministic tests.

CLOSED_ENDED means the prompt specifies a fixed algorithmic/function behavior
where unit tests or exact input-output behavior define correctness.

Return only valid JSON:
{
  "label": "OPEN_ENDED" or "CLOSED_ENDED",
  "reason": "...",
  "open_ended_dimensions": ["architecture", "library choice", "UI", "API design", "style", "algorithm choice"],
  "task_language": "PYTHON" or "NON_PYTHON"
}
"""

    user_msg = f"""Classify this coding prompt.

Source pool: {source_pool}

Prompt:
{prompt}
"""

    response = _get_openai_client().chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)

    if result["label"] not in {"OPEN_ENDED", "CLOSED_ENDED"}:
        raise ValueError(f"Invalid label: {result}")
    task_language = result.get("task_language", "").strip().upper()
    if task_language not in {"PYTHON", "NON_PYTHON"}:
        task_language = infer_task_language(prompt)

    return {
        "label": result["label"],
        "reason": result.get("reason", ""),
        "open_ended_dimensions": result.get("open_ended_dimensions", []),
        "task_language": task_language,
        "stage1_candidate": stage1_is_open_ended_candidate(prompt),
        "openness_score": openness_score(prompt),
        "stage2_backend": "llm",
    }


def stage2_validate_open_ended(
    prompt: str,
    source_pool: str,
    *,
    mode: str = "hybrid",
    llm_cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    llm_cache_path: Optional[Path] = None,
) -> Dict[str, Any]:
    mode_l = mode.lower()
    if mode_l == "heuristic":
        return stage2_validate_open_ended_heuristic(prompt, source_pool)

    cache_key = llm_cache_key(prompt, source_pool)

    def _from_cache(cached: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(cached)
        out["stage2_backend"] = "llm_cached"
        return out

    def _llm_and_store() -> Dict[str, Any]:
        out = stage2_validate_open_ended_llm(prompt, source_pool)
        if llm_cache is not None:
            llm_cache[cache_key] = dict(out)
        if llm_cache_path is not None:
            append_llm_cache_jsonl(llm_cache_path, cache_key, dict(out))
        return out

    if mode_l == "llm":
        if llm_cache is not None and cache_key in llm_cache:
            return _from_cache(llm_cache[cache_key])
        return _llm_and_store()

    if mode_l != "hybrid":
        raise ValueError(f"Unknown stage2 mode: {mode}")

    heuristic = stage2_validate_open_ended_heuristic(prompt, source_pool)
    if not stage2_needs_llm(prompt, source_pool, heuristic):
        return heuristic

    if llm_cache is not None and cache_key in llm_cache:
        return _from_cache(llm_cache[cache_key])

    return _llm_and_store()


# -----------------------------
# Prompt extraction
# -----------------------------

LIKELY_PROMPT_COLUMNS = [
    "instruction",
    "prompt",
    "question",
    "problem",
    "description",
    "content",
    "query",
    "input",
    "text",
    "title",
]

LIKELY_RESPONSE_COLUMNS = [
    "output",
    "response",
    "answer",
    "solution",
    "completion",
    "code",
    "canonical_solution",
    "solutions",
]


def choose_prompt_column(column_names: List[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in column_names}

    for name in LIKELY_PROMPT_COLUMNS:
        if name in lower_to_original:
            return lower_to_original[name]

    # Fallback: choose first column containing a likely keyword.
    for c in column_names:
        lc = c.lower()
        if any(name in lc for name in LIKELY_PROMPT_COLUMNS):
            return c

    return None


def choose_response_column(column_names: List[str]) -> Optional[str]:
    lower_to_original = {c.lower(): c for c in column_names}

    for name in LIKELY_RESPONSE_COLUMNS:
        if name in lower_to_original:
            return lower_to_original[name]

    for c in column_names:
        lc = c.lower()
        if any(name in lc for name in LIKELY_RESPONSE_COLUMNS):
            return c

    return None


def combine_instruction_input(row: Dict[str, Any]) -> Optional[str]:
    """
    Some Alpaca-style datasets use:
        instruction + input + output

    In those cases, combine instruction and input as the prompt.
    """
    keys = {k.lower(): k for k in row.keys()}

    if "instruction" in keys:
        instruction = normalize_text(row.get(keys["instruction"]))
        extra_input = normalize_text(row.get(keys["input"])) if "input" in keys else ""

        if extra_input:
            return f"{instruction}\n\nInput:\n{extra_input}".strip()

        return instruction.strip()

    return None


def extract_prompt_and_response(row: Dict[str, Any], column_names: List[str]) -> Tuple[str, str, str, str]:
    """
    Returns:
        prompt, response, prompt_column_used, response_column_used
    """
    combined = combine_instruction_input(row)
    response_col = choose_response_column(column_names)
    response = normalize_text(row.get(response_col)) if response_col else ""

    if combined:
        return combined, response, "instruction(+input)", response_col or ""

    prompt_col = choose_prompt_column(column_names)
    prompt = normalize_text(row.get(prompt_col)) if prompt_col else ""

    return prompt, response, prompt_col or "", response_col or ""


# -----------------------------
# Dataset loading
# -----------------------------

def iter_splits(dataset_obj: Any) -> Iterable[Tuple[str, Dataset]]:
    if isinstance(dataset_obj, DatasetDict):
        for split_name, split_ds in dataset_obj.items():
            yield split_name, split_ds
    elif isinstance(dataset_obj, Dataset):
        yield "train", dataset_obj
    else:
        raise TypeError(f"Unsupported dataset object type: {type(dataset_obj)}")


def safe_load_dataset(dataset_name: str):
    """
    Some datasets may require trust_remote_code or have unusual configs.
    This function tries a few safe loading paths.
    """
    errors = []

    for kwargs in [
        {},
        {"trust_remote_code": True},
    ]:
        try:
            return load_dataset(dataset_name, **kwargs)
        except Exception as exc:
            errors.append(f"{kwargs}: {repr(exc)}")

    raise RuntimeError(f"Could not load dataset {dataset_name}. Errors: {errors}")


def process_dataset(
    dataset_name: str,
    source_pool: str,
    max_rows_per_dataset: Optional[int] = None,
    stage2_mode: str = "hybrid",
    llm_cache: Optional[MutableMapping[str, Dict[str, Any]]] = None,
    llm_cache_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    print(f"\nDownloading/loading: {dataset_name} [{source_pool}]")

    try:
        ds_obj = safe_load_dataset(dataset_name)
    except Exception as exc:
        print(f"WARNING: failed to load {dataset_name}: {exc}")
        return []

    rows_out: List[Dict[str, Any]] = []
    total_seen_for_dataset = 0

    for split_name, split_ds in iter_splits(ds_obj):
        column_names = list(split_ds.column_names)

        if max_rows_per_dataset is not None:
            remaining = max_rows_per_dataset - total_seen_for_dataset
            if remaining <= 0:
                break
            split_ds = split_ds.select(range(min(len(split_ds), remaining)))

        print(f"  Split: {split_name}, rows: {len(split_ds)}, columns: {column_names}")

        for idx, row in enumerate(tqdm(split_ds, desc=f"{dataset_name}:{split_name}")):
            prompt, response, prompt_col, response_col = extract_prompt_and_response(row, column_names)
            prompt = normalize_text(prompt)

            if not prompt or len(prompt) < 20:
                continue

            validation = stage2_validate_open_ended(
                prompt,
                source_pool,
                mode=stage2_mode,
                llm_cache=llm_cache,
                llm_cache_path=llm_cache_path,
            )

            rows_out.append(
                {
                    "dataset": dataset_name,
                    "source_pool": source_pool,
                    "split": split_name,
                    "row_index": idx,
                    "prompt": prompt,
                    "response": response,
                    "label": validation["label"],
                    "reason": validation["reason"],
                    "open_ended_dimensions": validation["open_ended_dimensions"],
                    "task_language": validation["task_language"],
                    "stage2_backend": validation.get("stage2_backend", ""),
                    "stage1_candidate": validation["stage1_candidate"],
                    "openness_score": validation["openness_score"],
                    "prompt_column": prompt_col,
                    "response_column": response_col,
                }
            )

            total_seen_for_dataset += 1

    return rows_out


# -----------------------------
# Save outputs
# -----------------------------

def save_local_dataset(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.mkdir(parents=True, exist_ok=True)

    if not rows:
        print(f"WARNING: no rows to save for {output_path}")
        return

    df = pd.DataFrame(rows)

    # Save as JSONL for easy inspection.
    jsonl_path = output_path / "data.jsonl"
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    # Save as CSV for quick manual review.
    csv_path = output_path / "data.csv"
    df.to_csv(csv_path, index=False)

    # Save as Hugging Face Dataset.
    hf_ds = Dataset.from_pandas(df, preserve_index=False)
    hf_ds.save_to_disk(str(output_path / "hf_dataset"))

    print(f"Saved {len(df):,} rows to:")
    print(f"  {jsonl_path}")
    print(f"  {csv_path}")
    print(f"  {output_path / 'hf_dataset'}")


def summarize(rows: List[Dict[str, Any]], name: str) -> None:
    if not rows:
        print(f"\n{name}: no rows")
        return

    df = pd.DataFrame(rows)

    print(f"\n{name} summary")
    print("=" * (len(name) + 8))
    print(f"Rows: {len(df):,}")

    print("\nBy source pool:")
    print(df["source_pool"].value_counts(dropna=False).to_string())

    print("\nBy dataset:")
    print(df["dataset"].value_counts(dropna=False).head(30).to_string())

    if "openness_score" in df.columns:
        print("\nOpenness score stats:")
        print(df["openness_score"].describe().to_string())

    if "task_language" in df.columns:
        print("\nBy task language:")
        print(df["task_language"].value_counts(dropna=False).to_string())

    if "stage2_backend" in df.columns:
        print("\nBy stage2 backend:")
        print(df["stage2_backend"].value_counts(dropna=False).to_string())


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        default="local_datasets",
        help="Directory where local datasets will be saved.",
    )
    parser.add_argument(
        "--max_rows_per_dataset",
        type=int,
        default=None,
        help="Optional cap per dataset for testing/debugging.",
    )
    parser.add_argument(
        "--skip_pool_a",
        action="store_true",
        help="Skip closed-ended baseline pool.",
    )
    parser.add_argument(
        "--skip_pool_b",
        action="store_true",
        help="Skip open-ended candidate pool.",
    )
    parser.add_argument(
        "--stage2_mode",
        type=str,
        choices=["heuristic", "hybrid", "llm"],
        default="hybrid",
        help=(
            "Stage-2 labeling: 'heuristic' (rules only, fastest), "
            "'hybrid' (rules + LLM only on ambiguous Pool B rows), "
            "'llm' (one API call per row, slowest)."
        ),
    )
    parser.add_argument(
        "--llm_cache_path",
        type=str,
        default=None,
        help="Optional JSONL file to load/store LLM Stage-2 outputs (dedup across runs).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    llm_cache: Dict[str, Dict[str, Any]] = {}
    llm_cache_path: Optional[Path] = Path(args.llm_cache_path) if args.llm_cache_path else None
    if args.stage2_mode in ("hybrid", "llm") and llm_cache_path is not None:
        llm_cache = load_llm_cache_jsonl(llm_cache_path)

    all_rows: List[Dict[str, Any]] = []

    if not args.skip_pool_a:
        for dataset_name in POOL_A_CLOSED_BASELINE:
            all_rows.extend(
                process_dataset(
                    dataset_name=dataset_name,
                    source_pool="A_closed_baseline",
                    max_rows_per_dataset=args.max_rows_per_dataset,
                    stage2_mode=args.stage2_mode,
                    llm_cache=llm_cache,
                    llm_cache_path=llm_cache_path,
                )
            )

    if not args.skip_pool_b:
        for dataset_name in POOL_B_OPEN_ENDED_CANDIDATES:
            all_rows.extend(
                process_dataset(
                    dataset_name=dataset_name,
                    source_pool="B_open_ended_candidates",
                    max_rows_per_dataset=args.max_rows_per_dataset,
                    stage2_mode=args.stage2_mode,
                    llm_cache=llm_cache,
                    llm_cache_path=llm_cache_path,
                )
            )

    if not all_rows:
        raise RuntimeError("No rows were loaded. Check dataset names, internet access, or Hugging Face authentication.")

    summarize(all_rows, "All code prompts")

    pool_language_groups = [
        ("A_closed_baseline", "PYTHON", "pool_a_closed_baseline/python_tasks"),
        ("A_closed_baseline", "NON_PYTHON", "pool_a_closed_baseline/non_python_tasks"),
        ("B_open_ended_candidates", "PYTHON", "pool_b_open_ended_candidates/python_tasks"),
        ("B_open_ended_candidates", "NON_PYTHON", "pool_b_open_ended_candidates/non_python_tasks"),
    ]

    for source_pool, task_language, relative_path in pool_language_groups:
        grouped_rows = [
            r for r in all_rows if r["source_pool"] == source_pool and r.get("task_language") == task_language
        ]
        summarize(grouped_rows, f"{source_pool} :: {task_language}")
        save_local_dataset(grouped_rows, output_dir / relative_path)

    # Also save a combined audit file.
    audit_path = output_dir / "combined_audit.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_json(audit_path, orient="records", lines=True, force_ascii=False)

    print(f"\nSaved combined audit file to: {audit_path}")


if __name__ == "__main__":
    main()