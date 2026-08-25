"""Rung 3 — Anthropic API direct, with prompt caching on the static context."""
from __future__ import annotations

import os

from harness import config
from harness.budget import estimate_cost_usd
from proposer.errors import RungError, RungQuotaError, RungUnavailable
from proposer.prompt import extract_strategy
from proposer.result import RungResult

NAME = "anthropic"
_THINKING_PREFIXES = ("claude-sonnet-5", "claude-opus-5", "claude-opus-4", "claude-sonnet-4-6", "claude-fable")


def available() -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return "ANTHROPIC_API_KEY not set"
    return None


def propose(static_context: str, dynamic_context: str, attempt: int = 0,
            model: str | None = None) -> RungResult:
    reason = available()
    if reason:
        raise RungUnavailable(reason)
    import anthropic  # imported lazily so the loop works without the SDK configured

    model = model or os.environ.get("CAR_ANTHROPIC_MODEL", config.ANTHROPIC_MODEL)
    client = anthropic.Anthropic(max_retries=0)  # the ladder owns retry/backoff policy
    kwargs = dict(
        model=model,
        max_tokens=config.PROPOSER_MAX_OUTPUT_TOKENS,
        system=[{"type": "text", "text": static_context, "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": dynamic_context}],
    )
    if model.startswith(_THINKING_PREFIXES):
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": "medium"}
    try:
        resp = client.messages.create(**kwargs)
    except anthropic.RateLimitError as e:
        raise RungQuotaError(f"anthropic 429: {e.message}") from e
    except anthropic.APIStatusError as e:
        if e.status_code in (402, 529) or "credit" in str(e).lower() or "quota" in str(e).lower():
            raise RungQuotaError(f"anthropic {e.status_code}: {e.message}") from e
        raise RungError(f"anthropic {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RungError(f"anthropic connection error: {e}") from e

    if resp.stop_reason == "refusal":
        raise RungError("anthropic refusal")
    text = "".join(b.text for b in resp.content if b.type == "text")
    source = extract_strategy(text)
    if not source:
        raise RungError(f"anthropic {model}: no strategy.py in response")
    u = resp.usage
    cache_read = int(getattr(u, "cache_read_input_tokens", 0) or 0)
    cache_write = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
    cost = estimate_cost_usd(model, u.input_tokens, u.output_tokens, cache_write, cache_read)
    return RungResult(
        source=source, rung=NAME, model=resp.model or model,
        input_tokens=int(u.input_tokens), output_tokens=int(u.output_tokens),
        cache_read_tokens=cache_read, cache_write_tokens=cache_write, cost_usd=cost,
        extra={"stop_reason": resp.stop_reason},
    )
