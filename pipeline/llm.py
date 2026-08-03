"""
Provider-agnostic chat call.

Two backends:
  api    — any OpenAI-compatible endpoint (DeepSeek today, others via config)
  ollama — local models over Ollama's native /api/chat

Ollama gets its own path rather than its OpenAI-compat shim because we need
`num_ctx`. Ollama's default context is 4096 tokens; our retrieved context runs
12-20K chars (~4-6K tokens) plus a ~900-token schema prompt, so the default
silently truncates the contract out of the prompt and the model looks far worse
than it is. The native endpoint also returns real token counts and timings.
"""

import re
import time
import httpx
from typing import NamedTuple

from config import (
    DEEPSEEK_MODEL, DEEPSEEK_BASE_URL, MAX_TOKENS_PER_CALL,
    OLLAMA_BASE_URL, OLLAMA_NUM_CTX, OLLAMA_TIMEOUT,
)

_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


class LLMResponse(NamedTuple):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    load_s: float          # ollama only: weight-load time, 0.0 for api
    num_ctx: int = 0       # window actually used (may be below the request after backoff)


def strip_reasoning(text: str) -> str:
    """Reasoning models wrap chain-of-thought in <think> tags before the JSON."""
    return _THINK_RE.sub("", text or "").strip()


def call_llm(
    system: str,
    user: str,
    provider: str = "api",
    model: str | None = None,
    max_tokens: int = MAX_TOKENS_PER_CALL,
    json_mode: bool = False,
    num_ctx: int | None = None,
) -> LLMResponse:
    if provider == "ollama":
        return _call_ollama(system, user, model, max_tokens, json_mode,
                            num_ctx or OLLAMA_NUM_CTX)
    return _call_api(system, user, model, max_tokens, json_mode)


def _call_api(system, user, model, max_tokens, json_mode) -> LLMResponse:
    import os
    from openai import OpenAI

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=DEEPSEEK_BASE_URL,
    )
    kwargs = {}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    # V4 thinks by default at high effort and burns the whole max_tokens budget
    # on reasoning, returning empty content. reasoning_effort alone does not
    # switch it off — the thinking block has to be set explicitly.
    if (model or DEEPSEEK_MODEL).startswith("deepseek-v4"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model or DEEPSEEK_MODEL,
        max_tokens=max_tokens,
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **kwargs,
    )
    latency = time.perf_counter() - t0
    usage = resp.usage
    return LLMResponse(
        text=strip_reasoning(resp.choices[0].message.content),
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        latency_s=latency,
        load_s=0.0,
        num_ctx=0,
    )


def _call_ollama(system, user, model, max_tokens, json_mode, num_ctx) -> LLMResponse:
    """
    Retries with a halved context on HTTP 500. A 23K-token prefill can OOM the
    GPU even when the same num_ctx loads fine for a short prompt, so the failure
    only appears on the largest contracts. Halving is a real deployment
    constraint, not a fudge — the effective window is reported.
    """
    last = None
    ctx = num_ctx
    while ctx >= 4096:
        try:
            return _ollama_once(system, user, model, max_tokens, json_mode, ctx)
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 500:
                raise
            last = e
            ctx //= 2
    raise last


def _ollama_once(system, user, model, max_tokens, json_mode, num_ctx) -> LLMResponse:
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "num_ctx": num_ctx,
            "num_predict": max_tokens,
            "temperature": 0,
        },
    }
    if json_mode:
        payload["format"] = "json"

    t0 = time.perf_counter()
    r = httpx.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    latency = time.perf_counter() - t0
    data = r.json()

    msg = data.get("message", {})
    # Newer Ollama splits reasoning into its own field; older inlines <think> tags.
    content = strip_reasoning(msg.get("content", ""))
    if not content and msg.get("thinking"):
        # Reasoning model spent its whole num_predict budget on chain-of-thought
        # and never emitted the answer. Distinct from malformed JSON.
        raise RuntimeError(
            f"reasoning-only response: {data.get('eval_count', 0)} tokens of thinking, "
            f"no content (num_predict={max_tokens}, done={data.get('done_reason')})"
        )

    return LLMResponse(
        text=content,
        prompt_tokens=data.get("prompt_eval_count", 0),
        completion_tokens=data.get("eval_count", 0),
        latency_s=latency,
        load_s=data.get("load_duration", 0) / 1e9,
        num_ctx=num_ctx,
    )


def list_ollama_models() -> list[dict]:
    """Generation-capable local models, smallest first."""
    r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    r.raise_for_status()
    models = []
    for m in r.json().get("models", []):
        caps = m.get("capabilities", [])
        if "completion" not in caps:
            continue          # embedding-only, e.g. nomic-embed-text
        models.append({
            "name": m["name"],
            "size_gb": round(m["size"] / 1e9, 2),
            "params": m.get("details", {}).get("parameter_size", "?"),
            "quant": m.get("details", {}).get("quantization_level", "?"),
            "family": m.get("details", {}).get("family", "?"),
            "context_length": m.get("details", {}).get("context_length"),
            "thinking": "thinking" in caps,
            "vision": "vision" in caps,
            # vision-first models (minicpm-v) vs general models that also see
            # (gemma4) — only the former are useless for text extraction
            "vision_only": "vision" in caps and "tools" not in caps,
        })
    return sorted(models, key=lambda m: m["size_gb"])
