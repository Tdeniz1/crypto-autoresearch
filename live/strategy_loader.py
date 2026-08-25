"""Load the promoted strategy snapshot and capture the indicator values behind a signal."""
from __future__ import annotations

import json
import numbers
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from live import config

COLUMNS = ["open", "high", "low", "close", "volume"]


def load_promoted() -> dict:
    if not config.PROMOTED_FILE.exists():
        raise SystemExit("no promoted strategy: run `uv run python -m live.promote <sha> --symbol BTC/USDT --timeframe 1h`")
    return json.loads(config.PROMOTED_FILE.read_text())


def load_strategy(source_path: Path):
    ns: dict = {"__name__": "promoted_strategy"}
    exec(compile(source_path.read_text(), str(source_path), "exec"), ns)  # noqa: S102 — snapshot of a scanned commit
    fn = ns.get("strategy")
    if not callable(fn):
        raise RuntimeError(f"{source_path}: no strategy()")
    doc = (ns.get("__doc__") or "").strip().splitlines()
    return fn, (doc[0] if doc else "")


def _last_value(v):
    try:
        if isinstance(v, pd.Series):
            x = v.iloc[-1]
        elif isinstance(v, np.ndarray) and v.ndim == 1 and len(v):
            x = v[-1]
        elif isinstance(v, (bool, np.bool_)):
            return bool(v)
        elif isinstance(v, numbers.Number):
            x = v
        else:
            return None
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        x = float(x)
        return None if not np.isfinite(x) else round(x, 8)
    except (TypeError, ValueError, IndexError):
        return None


def run_with_indicators(fn, bars: pd.DataFrame, max_keys: int = 30) -> tuple[float, dict]:
    """Run strategy(bars); return (signal at last bar, {local name: value at last bar}).

    Indicator capture uses a return-event trace on the strategy's own frame: every local that is a
    Series/array/number is read at the final bar. Works for any strategy without a convention.
    """
    frame = bars[COLUMNS].reset_index(drop=True).astype("float64")
    captured: dict = {}
    code = getattr(fn, "__code__", None)

    def tracer(f, event, arg):
        if event == "call":
            return tracer if f.f_code is code else None
        if event == "return" and f.f_code is code:
            for k, v in list(f.f_locals.items()):
                if k == "df" or isinstance(v, pd.DataFrame) or k.startswith("_"):
                    continue
                val = _last_value(v)
                if val is not None:
                    captured[k] = val
        return tracer

    old = sys.gettrace()
    sys.settrace(tracer)
    try:
        out = fn(frame)
    finally:
        sys.settrace(old)
    sig = pd.Series(np.asarray(out, dtype="float64")).iloc[-1]
    sig = 0.0 if not np.isfinite(sig) else float(np.clip(sig, -1.0, 1.0))
    indicators = dict(list(captured.items())[:max_keys])
    indicators["close"] = float(frame["close"].iloc[-1])
    return sig, indicators
