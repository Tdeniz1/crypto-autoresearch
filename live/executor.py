"""Paper-trading executor stub. Same position interface as the backtest.

A strategy returns a position series in [-1, 1]; the LAST value is the target
position to hold from the next bar's open. The executor turns that target
into an order intent against Alpaca paper. Real order routing is NOT wired:
`submit=False` always logs intents; `submit=True` raises NotImplementedError.

This module depends only on pandas so it can be deployed (Railway) without the
research harness. It never imports harness/ or the research loop.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol

import pandas as pd

log = logging.getLogger("live.executor")

Strategy = Callable[[pd.DataFrame], "pd.Series"]
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"


@dataclass
class OrderIntent:
    ts: str
    symbol: str
    side: str            # buy | sell
    qty: float
    target_position: float
    current_position: float
    price_hint: float | None
    submitted: bool = False
    broker_order_id: str | None = None


class Executor(Protocol):
    def current_position(self, symbol: str) -> float: ...
    def set_target_position(self, symbol: str, target: float, price_hint: float | None = None) -> OrderIntent | None: ...


@dataclass
class AlpacaPaperExecutor:
    """Stub against Alpaca paper. Positions are fractions of `notional_per_unit`."""
    api_key: str | None = field(default_factory=lambda: os.environ.get("ALPACA_API_KEY"))
    api_secret: str | None = field(default_factory=lambda: os.environ.get("ALPACA_API_SECRET"))
    base_url: str = field(default_factory=lambda: os.environ.get("ALPACA_BASE_URL", ALPACA_PAPER_URL))
    notional_per_unit: float = 1000.0     # USD notional for |position| == 1
    min_delta: float = 0.05               # ignore position changes smaller than this
    submit: bool = False                  # real routing not wired; True raises
    positions: dict[str, float] = field(default_factory=dict)
    intents: list[OrderIntent] = field(default_factory=list)

    def current_position(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0)

    def set_target_position(self, symbol: str, target: float, price_hint: float | None = None) -> OrderIntent | None:
        target = max(-1.0, min(1.0, float(target) if target == target else 0.0))  # NaN -> flat
        cur = self.current_position(symbol)
        delta = target - cur
        if abs(delta) < self.min_delta:
            return None
        qty = abs(delta) * self.notional_per_unit / price_hint if price_hint else abs(delta) * self.notional_per_unit
        intent = OrderIntent(ts=datetime.now(timezone.utc).isoformat(timespec="seconds"), symbol=symbol,
                             side="buy" if delta > 0 else "sell", qty=round(qty, 6), target_position=target,
                             current_position=cur, price_hint=price_hint)
        if self.submit:
            self._submit(intent)   # raises: not wired
        log.info("order intent %s", json.dumps(asdict(intent)))
        self.intents.append(intent)
        self.positions[symbol] = target
        return intent

    def _submit(self, intent: OrderIntent) -> None:
        raise NotImplementedError(
            "Real Alpaca order routing is deliberately not wired. When it is: POST "
            f"{self.base_url}/v2/orders with APCA-API-KEY-ID/APCA-API-SECRET-KEY headers, "
            "notional or qty, side, type=market, time_in_force=gtc; set intent.broker_order_id.")


def run_bar(strategy: Strategy, bars: pd.DataFrame, executor: Executor, symbol: str) -> OrderIntent | None:
    """Apply a strategy to the latest bar window and push the target to the executor.

    `bars` must be the same shape the backtest saw: RangeIndex, columns
    open/high/low/close/volume, oldest first, most recent bar last (closed).
    """
    frame = bars[["open", "high", "low", "close", "volume"]].reset_index(drop=True)
    sig = strategy(frame)
    target = float(pd.Series(sig).iloc[-1])
    return executor.set_target_position(symbol, target, price_hint=float(frame["close"].iloc[-1]))


def load_strategy_file(path: str = "strategy.py") -> Strategy:
    """Import strategy.py exactly as the harness would, without the harness."""
    ns: dict = {}
    with open(path, "r", encoding="utf-8") as f:
        exec(compile(f.read(), path, "exec"), ns)  # noqa: S102 — file was scanned by the loop before commit
    return ns["strategy"]
