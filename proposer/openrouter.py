"""Rung 1 — OpenRouter free models (OpenAI-compatible chat endpoint)."""
from __future__ import annotations

import os

import httpx

from harness import config
from proposer.errors import RungError, RungQuotaError, RungUnavailable, looks_like_quota
from proposer.prompt import extract_strategy
from proposer.result import RungResult

NAME = "openrouter"


def available() -> str | None:
    if not os.environ.get("OPENROUTER_API_KEY"):
        return "OPENROUTER_API_KEY not set"
    return None


def propose(static_context: str, dynamic_context: str, attempt: int = 0) -> RungResult:
    reason = available()
    if reason:
        raise RungUnavailable(reason)
    models = config.OPENROUTER_MODELS
    model = models[attempt % len(models)]
    base = os.environ.get("CAR_OPENROUTER_BASE_URL", config.OPENROUTER_BASE_URL).rstrip("/")
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "HTTP-Referer": "https://github.com/crypto-autoresearch",
        "X-Title": "crypto-autoresearch",
    }
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": static_context},
            {"role": "user", "content": dynamic_context},
        ],
        "max_tokens": config.PROPOSER_MAX_OUTPUT_TOKENS,
        "temperature": 0.7,
    }
    try:
        r = httpx.post(f"{base}/chat/completions", headers=headers, json=body,
                       timeout=config.OPENROUTER_TIMEOUT_SEC)
    except httpx.HTTPError as e:
        raise RungError(f"openrouter network error: {type(e).__name__}: {e}") from e
    if r.status_code == 429 or r.status_code == 402:
        raise RungQuotaError(f"openrouter {r.status_code} on {model}: {r.text[:200]}")
    if r.status_code >= 400:
        if looks_like_quota(r.text):
            raise RungQuotaError(f"openrouter {r.status_code} on {model}: {r.text[:200]}")
        raise RungError(f"openrouter {r.status_code} on {model}: {r.text[:200]}")
    try:
        data = r.json()
    except ValueError as e:
        raise RungError(f"openrouter non-JSON response: {r.text[:200]}") from e
    if "error" in data:
        msg = str(data["error"])
        if looks_like_quota(msg):
            raise RungQuotaError(f"openrouter error on {model}: {msg[:200]}")
        raise RungError(f"openrouter error on {model}: {msg[:200]}")
    try:
        text = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as e:
        raise RungError(f"openrouter malformed response: {str(data)[:200]}") from e
    source = extract_strategy(text)
    if not source:
        raise RungError(f"openrouter {model}: no strategy.py in response")
    usage = data.get("usage") or {}
    return RungResult(
        source=source, rung=NAME, model=data.get("model") or model,
        input_tokens=int(usage.get("prompt_tokens") or 0),
        output_tokens=int(usage.get("completion_tokens") or 0),
        cost_usd=0.0 if model.endswith(":free") else None,
    )
