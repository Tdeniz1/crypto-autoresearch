"""proposer.propose(program, current_strategy, log_summary) -> (strategy_source, usage_record)"""
from __future__ import annotations

from proposer.ladder import AllRungsFailed, Ladder
from proposer.prompt import build_dynamic, build_static, extract_strategy, summarize_log

_default_ladder: Ladder | None = None


def propose(program: str, current_strategy: str, log_summary: str,
            ladder: Ladder | None = None) -> tuple[str, dict]:
    """The one interface. Returns a full replacement strategy.py plus a usage record."""
    global _default_ladder
    if ladder is None:
        if _default_ladder is None:
            _default_ladder = Ladder()
        ladder = _default_ladder
    static = build_static(program)
    dynamic = build_dynamic(current_strategy, log_summary)
    res, usage = ladder.propose(static, dynamic)
    usage.update({
        "input_tokens": res.input_tokens, "output_tokens": res.output_tokens,
        "cache_read_tokens": res.cache_read_tokens, "cache_write_tokens": res.cache_write_tokens,
        "cost_usd": res.cost_usd, "extra": res.extra,
    })
    return res.source, usage


__all__ = ["propose", "Ladder", "AllRungsFailed", "build_static", "build_dynamic",
           "extract_strategy", "summarize_log"]
