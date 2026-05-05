"""Quick diagnostic for Together model IDs. Hits each model with one tiny call
and prints status + truncated body. Doesn't depend on the rest of the repo."""
import os
import httpx

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MODELS = [
    # known-good control
    "deepseek-ai/DeepSeek-V3",

    # Serverless (default API key) — use *-Turbo / *-FP8 / gpt-oss; bare
    # Llama-3.3-70B-Instruct and Qwen2.5-Coder-32B-Instruct are dedicated-only.
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen3-Coder-Next-FP8",
    "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    "openai/gpt-oss-120b",

    # Llama-3.3 / Llama-3.1 (may be dedicated-only on some accounts)
    "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",

    # Qwen Coder (32B / 30B often dedicated-only)
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",

    # Legacy CodeLlama (typically dedicated-only if listed)
    "codellama/CodeLlama-70b-Instruct-hf",

    # Other reasonable additions
    "deepseek-ai/DeepSeek-R1",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mistral-Small-24B-Instruct-2501",
]


def main():
    key = os.environ.get("TOGETHER_API_KEY")
    if not key:
        print("ERROR: set TOGETHER_API_KEY")
        return
    headers = {"Authorization": f"Bearer {key}"}
    for m in MODELS:
        body = {"model": m,
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 5}
        try:
            r = httpx.post("https://api.together.ai/v1/chat/completions",
                           headers=headers, json=body, timeout=15)
            ok = (r.status_code == 200) and ("choices" in r.text)
            mark = "OK " if ok else "BAD"
            print(f"{mark}  {m:55s}  status={r.status_code}  body={r.text[:140]!r}",
                  flush=True)
        except Exception as e:
            print(f"EXC {m:55s}  {type(e).__name__}: {str(e)[:120]}",
                  flush=True)


if __name__ == "__main__":
    main()
