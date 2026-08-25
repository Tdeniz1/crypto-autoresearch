from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RungResult:
    source: str
    rung: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float | None = None
    extra: dict = field(default_factory=dict)
