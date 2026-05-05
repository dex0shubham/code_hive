"""
Code Hivemind — Multi-Model Collector
======================================
Samples code responses from multiple LLM providers in parallel.
"""

# -- Path bootstrap (so cross-folder imports work on Windows) --
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
# ---------------------------------------------------------------

import asyncio
import json
import os
import random
import re
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import Optional

import httpx

from config import MODELS, SamplingConfig, RAW_RESPONSES_DIR, get_api_key
from prompt_suite import PROMPTS, SYSTEM_PROMPT, CodePrompt


@dataclass
class Response:
    """A single code generation response with full metadata."""
    prompt_id: str
    model_provider: str
    model_id: str
    model_display: str
    model_family: str
    temperature: float
    top_p: float
    sample_index: int
    response_text: str
    finish_reason: str
    latency_ms: float
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    timestamp: str

    @property
    def uid(self) -> str:
        raw = f"{self.prompt_id}:{self.model_id}:t{self.temperature}:s{self.sample_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# -- Provider-specific API callers --

async def call_openai(client, model, prompt, cfg, temp):
    headers = {
        "Authorization": f"Bearer {get_api_key('openai')}",
        "Content-Type": "application/json",
    }
    is_reasoning_model = model.startswith(("o1", "o3", "o4"))
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    }

    if is_reasoning_model:
        body["max_completion_tokens"] = cfg.max_tokens
    else:
        body["temperature"] = temp
        body["top_p"] = cfg.top_p
        body["max_tokens"] = cfg.max_tokens

    if not is_reasoning_model and cfg.seed is not None and temp > 0:
        body["seed"] = cfg.seed

    t0 = time.monotonic()
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers=headers, json=body, timeout=120,
    )
    latency = (time.monotonic() - t0) * 1000
    data = resp.json()
    if resp.status_code >= 400 or "choices" not in data:
        error = data.get("error", data)
        message = error.get("message", error) if isinstance(error, dict) else error
        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {message}")
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason", "unknown"),
        "latency_ms": latency,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


async def call_anthropic(client, model, prompt, cfg, temp):
    headers = {
        "x-api-key": get_api_key("anthropic"),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": cfg.max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temp,
        "top_p": cfg.top_p,
    }
    t0 = time.monotonic()
    resp = await client.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers, json=body, timeout=120,
    )
    latency = (time.monotonic() - t0) * 1000
    data = resp.json()
    text = "".join(
        block["text"] for block in data.get("content", [])
        if block.get("type") == "text"
    )
    usage = data.get("usage", {})
    return {
        "text": text,
        "finish_reason": data.get("stop_reason", "unknown"),
        "latency_ms": latency,
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


async def call_google(client, model, prompt, cfg, temp):
    api_key = get_api_key("google")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temp,
            "topP": cfg.top_p,
            "maxOutputTokens": cfg.max_tokens,
        },
    }
    t0 = time.monotonic()
    resp = await client.post(url, json=body, timeout=120)
    latency = (time.monotonic() - t0) * 1000
    data = resp.json()
    candidates = data.get("candidates", [{}])
    text = ""
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return {
        "text": text,
        "finish_reason": candidates[0].get("finishReason", "unknown") if candidates else "error",
        "latency_ms": latency,
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
    }


async def call_together(client, model, prompt, cfg, temp):
    headers = {
        "Authorization": f"Bearer {get_api_key('together')}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": max(temp, 0.01),
        "top_p": cfg.top_p,
        "max_tokens": cfg.max_tokens,
    }
    t0 = time.monotonic()
    resp = await client.post(
        "https://api.together.ai/v1/chat/completions",
        headers=headers, json=body, timeout=120,
    )
    latency = (time.monotonic() - t0) * 1000
    data = resp.json()
    if resp.status_code >= 400 or "choices" not in data or not data["choices"]:
        error = data.get("error", data) if isinstance(data, dict) else data
        message = error.get("message", error) if isinstance(error, dict) else error
        raise RuntimeError(f"Together API error ({resp.status_code}): {message}")
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["message"]["content"],
        "finish_reason": choice.get("finish_reason", "unknown"),
        "latency_ms": latency,
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
    }


PROVIDER_CALLERS = {
    "openai": call_openai,
    "anthropic": call_anthropic,
    "google": call_google,
    "together": call_together,
}

# Providers that have known strict rate-limits and benefit from aggressive retry
_RETRY_PROVIDERS = {"together"}
_MAX_RETRIES = 6


async def _call_with_retry(caller, client, model, prompt_text, cfg, temp,
                            *, provider: str = "", label: str = ""):
    """Call a provider API with exponential backoff on 429 / 503 responses.

    429 (rate-limit): waits based on the dynamic RPM stated in the error
    message if present, otherwise doubles the delay each attempt, capped at
    120 s. Jitter ±5 s is always added to spread concurrent retries.

    503 (service unavailable): shorter fixed-base backoff (10 s, 15 s, …).

    All other errors are re-raised immediately.
    """
    delay_base = 10.0
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return await caller(client, model, prompt_text, cfg, temp)
        except RuntimeError as exc:
            msg = str(exc)
            is_429 = "429" in msg
            is_503 = "503" in msg
            retriable = is_429 or is_503
            if not retriable or attempt == _MAX_RETRIES:
                raise
            if is_429:
                # Parse "dynamic" rate limit from Together's message, e.g.
                # "Your dynamic rate limit is 0.666… RPM"
                rpm_m = re.search(r"dynamic.*?rate limit is ([\d.]+) RPM",
                                   msg, re.IGNORECASE)
                if rpm_m:
                    rpm = float(rpm_m.group(1))
                    wait = (60.0 / max(rpm, 0.1)) + random.uniform(2, 8)
                else:
                    wait = min(delay_base * (2 ** attempt), 120.0) + random.uniform(0, 5)
            else:  # 503
                wait = min(delay_base * (1.5 ** attempt), 60.0) + random.uniform(0, 3)
            wait = min(wait, 150.0)
            reason = "rate-limit" if is_429 else "service-unavail"
            print(f"    [{label} retry {attempt + 1}/{_MAX_RETRIES}] "
                  f"{reason}: waiting {wait:.0f}s …")
            await asyncio.sleep(wait)
    raise RuntimeError("unreachable")  # never reached


async def sample_one(client, provider, model_id, model_display, model_family,
                     prompt, cfg, temp, sample_idx):
    caller = PROVIDER_CALLERS[provider]
    label = f"{model_display} {prompt.id} t={temp} s={sample_idx}"
    try:
        if provider in _RETRY_PROVIDERS:
            result = await _call_with_retry(
                caller, client, model_id, prompt.prompt, cfg, temp,
                provider=provider, label=label,
            )
        else:
            result = await caller(client, model_id, prompt.prompt, cfg, temp)
        return Response(
            prompt_id=prompt.id,
            model_provider=provider,
            model_id=model_id,
            model_display=model_display,
            model_family=model_family,
            temperature=temp,
            top_p=cfg.top_p,
            sample_index=sample_idx,
            response_text=result["text"],
            finish_reason=result["finish_reason"],
            latency_ms=result["latency_ms"],
            input_tokens=result.get("input_tokens"),
            output_tokens=result.get("output_tokens"),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
    except Exception as e:
        print(f"  ERROR [{model_display}] {prompt.id} t={temp} s={sample_idx}: {e}")
        return None


def _append_responses(responses: list, out_dir: Path) -> None:
    """Append a batch of Response objects to per-prompt JSONL files immediately.

    Uses append mode so partial runs accumulate across re-runs and Ctrl+C
    never destroys data already written in this session.
    """
    by_prompt: dict[str, list] = {}
    for r in responses:
        by_prompt.setdefault(r.prompt_id, []).append(r)
    for pid, rs in by_prompt.items():
        outpath = out_dir / f"{pid}.jsonl"
        with open(outpath, "a", encoding="utf-8") as fh:
            for r in rs:
                fh.write(json.dumps(asdict(r)) + "\n")


async def collect_all(cfg, prompt_ids=None, model_filter=None):
    os.makedirs(RAW_RESPONSES_DIR, exist_ok=True)
    out_dir = Path(RAW_RESPONSES_DIR)

    prompts = PROMPTS
    if prompt_ids:
        prompts = [p for p in PROMPTS if p.id in prompt_ids]

    models = MODELS
    if model_filter:
        models = [m for m in MODELS if m[2] in model_filter]

    total = len(prompts) * len(models) * len(cfg.temperatures) * cfg.samples_per_model_per_temp
    print(f"Code Hivemind Collector")
    print(f"  Prompts:      {len(prompts)}")
    print(f"  Models:       {len(models)}")
    print(f"  Temperatures: {cfg.temperatures}")
    print(f"  Samples/temp: {cfg.samples_per_model_per_temp}")
    print(f"  Total calls:  {total}")
    print()

    all_responses: list[Response] = []
    sem = asyncio.Semaphore(10)
    interrupted = False

    try:
        async with httpx.AsyncClient() as client:
            for prompt in prompts:
                print(f"[{prompt.id}] {prompt.category}: {prompt.prompt[:60]}...")
                for provider, model_id, display, family in models:
                    for temp in cfg.temperatures:
                        tasks = []
                        for s in range(cfg.samples_per_model_per_temp):
                            async def _task(p=provider, m=model_id, d=display,
                                            f=family, t=temp, si=s):
                                async with sem:
                                    return await sample_one(
                                        client, p, m, d, f, prompt, cfg, t, si)
                            tasks.append(_task())

                        results = await asyncio.gather(*tasks)
                        valid = [r for r in results if r is not None]
                        all_responses.extend(valid)
                        # ── Incremental save: flush this batch to disk right away ──
                        _append_responses(valid, out_dir)
                        print(f"  {display} t={temp}: {len(valid)}/{len(tasks)} OK")

                prompt_responses = [r for r in all_responses if r.prompt_id == prompt.id]
                print(f"  Saved {len(prompt_responses)} responses -> "
                      f"{out_dir / (prompt.id + '.jsonl')}\n")

    except (KeyboardInterrupt, asyncio.CancelledError):
        interrupted = True
        print(f"\n[!] Interrupted — {len(all_responses)} responses already saved to disk.")

    # Always write the merged all_responses.jsonl so downstream tools can read it
    full_path = out_dir / "all_responses.jsonl"
    with open(full_path, "w", encoding="utf-8") as f:
        for r in all_responses:
            f.write(json.dumps(asdict(r)) + "\n")

    status = "interrupted" if interrupted else "Done"
    print(f"\n{status}! {len(all_responses)} total responses -> {full_path}")
    return all_responses


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", nargs="*")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--temps", nargs="*", type=float, default=[0.0, 1.0])
    args = parser.parse_args()

    cfg = SamplingConfig(temperatures=args.temps, samples_per_model_per_temp=args.samples)
    asyncio.run(collect_all(cfg, args.prompts, args.models))