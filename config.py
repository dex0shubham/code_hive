"""
Code Hivemind: Measuring Output Homogeneity Across Code LLMs
=============================================================
Configuration for the experimental pipeline.

Paper: Extends "Artificial Hivemind" (Jiang et al., NeurIPS 2025)
       to code generation across 10+ LLMs.
"""

import os
from dataclasses import dataclass, field
from typing import Optional

# -- Model registry --
# Each entry: (provider, model_id, display_name, family)

MODELS = [
    ("openai",    "gpt-4o",                              "GPT-4o",             "OpenAI"),
    ("openai",    "gpt-4o-mini",                         "GPT-4o-mini",        "OpenAI"),
    ("openai",    "o3-mini",                              "o3-mini",            "OpenAI"),
    ("anthropic", "claude-sonnet-4-20250514",             "Claude Sonnet 4",    "Anthropic"),
    ("anthropic", "claude-haiku-3-5-20241022",            "Claude 3.5 Haiku",   "Anthropic"),
    ("google",    "gemini-2.5-flash",                     "Gemini 2.5 Flash",   "Google"),
    ("google",    "gemini-2.5-pro",                       "Gemini 2.5 Pro",     "Google"),
    ("together",  "deepseek-ai/DeepSeek-V3",             "DeepSeek-V3",        "DeepSeek"),
    ("together",  "Qwen/Qwen2.5-Coder-32B-Instruct",    "Qwen2.5-Coder",     "Qwen"),
    ("together",  "meta-llama/Llama-3.3-70B-Instruct",   "Llama-3.3-70B",     "Meta"),
    ("together",  "codellama/CodeLlama-70b-Instruct-hf", "CodeLlama-70B",     "Meta"),
]

# -- Sampling parameters --

@dataclass
class SamplingConfig:
    temperatures: list = field(default_factory=lambda: [0.0, 1.0])
    top_p: float = 0.9
    samples_per_model_per_temp: int = 50
    max_tokens: int = 2048
    seed: Optional[int] = 42

# -- Prompt categories --

PROMPT_CATEGORIES = [
    "OPEN_DESIGN",
    "ALGORITHM",
    "REFACTOR",
    "NAMING",
    "CREATIVE_TOOL",
    "SYSTEM_DESIGN",
]

# -- Output paths --

OUTPUT_DIR = "results"
RAW_RESPONSES_DIR = os.path.join(OUTPUT_DIR, "raw_responses")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# -- API keys --

def get_api_key(provider: str) -> str:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
    }
    env_var = key_map.get(provider, f"{provider.upper()}_API_KEY")
    key = os.environ.get(env_var)
    if not key:
        raise ValueError(f"Missing API key: set {env_var} environment variable")
    return key