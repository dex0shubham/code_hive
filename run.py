#!/usr/bin/env python3
"""
Code Hivemind - Main Pipeline Runner
======================================

Usage:
  python run.py --demo                           # mock data, no API keys
  python run.py --collect --samples 50           # collect from all LLMs
  python run.py --collect --samples 2 --prompts OD-01 CT-04
  python run.py --analyze                        # analyze collected data
  python run.py --all                            # collect + analyze
"""

import argparse
import asyncio
import json
import os
import sys
import random
from pathlib import Path

# ─────────────────────────────────────────────────────────────────
# PATH BOOTSTRAP — must run BEFORE any project imports.
# Adds the project root (the folder containing this file) to
# sys.path so that "import config", "import prompt_suite",
# etc. resolve on Windows, macOS, and Linux no matter what
# working directory the user starts from.
# ─────────────────────────────────────────────────────────────────
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)
# ─────────────────────────────────────────────────────────────────

from config import SamplingConfig, RAW_RESPONSES_DIR, METRICS_DIR
from prompt_suite import PROMPTS


def cmd_collect(args):
    from multi_model_sampler import collect_all
    cfg = SamplingConfig(
        temperatures=args.temps,
        samples_per_model_per_temp=args.samples,
    )
    asyncio.run(collect_all(cfg, args.prompts, args.models))


def cmd_analyze(args):
    from pipeline import run_full_analysis
    run_full_analysis()


def cmd_demo(args):
    print("=" * 60)
    print("CODE HIVEMIND - DEMO MODE")
    print("Generating synthetic responses to test the pipeline")
    print("=" * 60)

    os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)

    cache_implementations = {
        "GPT-4o": '''
import time
from collections import OrderedDict
import threading

class TTLCache:
    def __init__(self, max_size=100, ttl=300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    self.cache.move_to_end(key)
                    return value
                else:
                    del self.cache[key]
            return None

    def set(self, key, value):
        with self.lock:
            if key in self.cache:
                del self.cache[key]
            elif len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())

    def cleanup(self):
        with self.lock:
            now = time.time()
            expired = [k for k, (v, t) in self.cache.items() if now - t >= self.ttl]
            for k in expired:
                del self.cache[k]
''',
        "Claude Sonnet 4": '''
import time
from threading import Lock
from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class CacheEntry:
    value: Any
    expires_at: float

class ExpiringCache:
    def __init__(self, default_ttl: float = 300.0):
        self._store: dict[str, CacheEntry] = {}
        self._default_ttl = default_ttl
        self._lock = Lock()

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            if time.monotonic() > entry.expires_at:
                del self._store[key]
                return None
            return entry.value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            self._store[key] = CacheEntry(
                value=value,
                expires_at=time.monotonic() + ttl,
            )

    def cleanup(self) -> int:
        with self._lock:
            now = time.monotonic()
            expired_keys = [
                k for k, entry in self._store.items()
                if now > entry.expires_at
            ]
            for k in expired_keys:
                del self._store[k]
            return len(expired_keys)
''',
        "Gemini 2.5 Pro": '''
import time
import threading
from collections import OrderedDict

class InMemoryCache:
    """A thread-safe in-memory cache with TTL-based expiration."""

    def __init__(self, max_entries=1000, default_ttl_seconds=300):
        self._cache = OrderedDict()
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        self._lock = threading.RLock()

    def get(self, key):
        with self._lock:
            if key not in self._cache:
                return None
            value, expiry_time = self._cache[key]
            if time.time() > expiry_time:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    def set(self, key, value, ttl=None):
        ttl = ttl or self._default_ttl
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            if len(self._cache) >= self._max_entries:
                self._cache.popitem(last=False)
            self._cache[key] = (value, time.time() + ttl)

    def cleanup(self):
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, (_, expiry) in self._cache.items()
                if current_time > expiry
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)
''',
        "DeepSeek-V3": '''
import time
import threading

class Cache:
    def __init__(self, ttl=300):
        self.ttl = ttl
        self.data = {}
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.data:
                val, ts = self.data[key]
                if time.time() - ts < self.ttl:
                    return val
                del self.data[key]
        return None

    def set(self, key, value):
        with self.lock:
            self.data[key] = (value, time.time())

    def cleanup(self):
        with self.lock:
            now = time.time()
            self.data = {
                k: (v, t) for k, (v, t) in self.data.items()
                if now - t < self.ttl
            }
''',
        "Qwen2.5-Coder": '''
import time
from collections import OrderedDict
from threading import Lock

class LRUCacheWithTTL:
    def __init__(self, capacity: int = 128, ttl: int = 300):
        self.capacity = capacity
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self._lock = Lock()

    def get(self, key: str):
        with self._lock:
            if key not in self.cache:
                return None
            value, timestamp = self.cache[key]
            if time.time() - timestamp > self.ttl:
                self.cache.pop(key)
                return None
            self.cache.move_to_end(key)
            return value

    def set(self, key: str, value) -> None:
        with self._lock:
            if key in self.cache:
                self.cache.pop(key)
            elif len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
            self.cache[key] = (value, time.time())

    def cleanup(self) -> None:
        with self._lock:
            current = time.time()
            keys_to_remove = [
                k for k, (_, ts) in self.cache.items()
                if current - ts > self.ttl
            ]
            for k in keys_to_remove:
                del self.cache[k]
''',
    }

    mock_responses = []
    families = {
        "GPT-4o":          ("openai",    "gpt-4o",          "OpenAI"),
        "Claude Sonnet 4": ("anthropic", "claude-sonnet-4",  "Anthropic"),
        "Gemini 2.5 Pro":  ("google",    "gemini-2.5-pro",   "Google"),
        "DeepSeek-V3":     ("together",  "deepseek-v3",      "DeepSeek"),
        "Qwen2.5-Coder":   ("together",  "qwen-coder",       "Qwen"),
    }

    for model_name, code in cache_implementations.items():
        provider, model_id, family = families[model_name]
        for temp in [0.0, 1.0]:
            for sample_idx in range(10):
                varied_code = code
                if temp > 0 and sample_idx > 0:
                    tweaks = [
                        ("max_size", random.choice(["max_size", "capacity", "max_entries", "limit"])),
                        ("ttl", random.choice(["ttl", "ttl_seconds", "expiry", "timeout"])),
                    ]
                    for old, new in tweaks:
                        if random.random() > 0.7:
                            varied_code = varied_code.replace(old, new, 1)

                mock_responses.append({
                    "prompt_id": "OD-01",
                    "model_provider": provider,
                    "model_id": model_id,
                    "model_display": model_name,
                    "model_family": family,
                    "temperature": temp,
                    "top_p": 0.9,
                    "sample_index": sample_idx,
                    "response_text": f"```python\n{varied_code}\n```",
                    "finish_reason": "stop",
                    "latency_ms": random.uniform(500, 3000),
                    "input_tokens": random.randint(50, 100),
                    "output_tokens": random.randint(200, 500),
                    "timestamp": "2026-04-27T12:00:00Z",
                })

    outpath = Path(RAW_RESPONSES_DIR) / "OD-01.jsonl"
    with open(outpath, "w") as f:
        for r in mock_responses:
            f.write(json.dumps(r) + "\n")
    print(f"Generated {len(mock_responses)} mock responses -> {outpath}")

    print("\nRunning analysis on mock data...\n")
    from diversity_metrics import (
        extract_code_block, naming_similarity,
        library_choice_diversity, design_pattern_detection,
    )

    print("-- Naming Similarity (pairwise between models) --")
    unique_models = list(cache_implementations.keys())
    for i in range(len(unique_models)):
        for j in range(i + 1, len(unique_models)):
            code_i = cache_implementations[unique_models[i]]
            code_j = cache_implementations[unique_models[j]]
            sim = naming_similarity(code_i, code_j)
            print(f"  {unique_models[i]:20s} <-> {unique_models[j]:20s}: {sim:.3f}")

    print("\n-- Library Choices --")
    lib_div = library_choice_diversity(list(cache_implementations.values()))
    print(f"  Unique import sets: {lib_div['unique_import_sets']}")
    print(f"  Entropy: {lib_div['entropy']:.3f}")
    for imports, count in lib_div['most_common']:
        print(f"    {imports}: {count}")

    print("\n-- Design Patterns --")
    for name, code in cache_implementations.items():
        patterns = design_pattern_detection(code)
        print(f"  {name:20s}: {sorted(patterns)}")

    print("\n-- Embedding-Based Similarity --")
    try:
        from diversity_metrics import EmbeddingComputer
        embedder = EmbeddingComputer()
        inter_sim = embedder.mean_pairwise_similarity(list(cache_implementations.values()))
        print(f"  Mean inter-model cosine similarity: {inter_sim:.3f}")
        print(f"  (Hivemind paper threshold for concern: > 0.8)")

        codes_list = list(cache_implementations.values())
        sim_matrix = embedder.cosine_sim_matrix(codes_list)
        print("\n  Pairwise similarity matrix:")
        print(f"  {'':20s}", end="")
        for name in unique_models:
            print(f"{name[:10]:>12s}", end="")
        print()
        for i, name_i in enumerate(unique_models):
            print(f"  {name_i:20s}", end="")
            for j in range(len(unique_models)):
                print(f"{sim_matrix[i][j]:>12.3f}", end="")
            print()
    except Exception as e:
        print(f"  Skipping embedding metrics: {type(e).__name__}")
        print("  (Optional: pip install sentence-transformers torch)")
        print("  All other metrics ran fine above.")

    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("The mock data shows how 5 different models all produce")
    print("remarkably similar cache implementations -- same libraries,")
    print("same patterns, similar naming. This is the Code Hivemind.")
    print("=" * 60)


def main():
    from dotenv import load_dotenv
    load_dotenv()


    parser = argparse.ArgumentParser(
        description="Code Hivemind: Measuring LLM Code Output Homogeneity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all", action="store_true", help="Run collect + analyze")
    parser.add_argument("--collect", action="store_true", help="Collect LLM responses")
    parser.add_argument("--analyze", action="store_true", help="Analyze collected data")
    parser.add_argument("--demo", action="store_true", help="Run demo with mock data")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--temps", nargs="*", type=float, default=[0.0, 1.0])
    parser.add_argument("--prompts", nargs="*", help="Prompt IDs to process")
    parser.add_argument("--models", nargs="*", help="Model names to use")
    args = parser.parse_args()

    if args.demo:
        cmd_demo(args)
    elif args.collect or args.all:
        cmd_collect(args)
        if args.all:
            cmd_analyze(args)
    elif args.analyze:
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()