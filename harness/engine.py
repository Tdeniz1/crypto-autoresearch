"""Vectorized backtest engine.

Execution model (structural, a strategy cannot opt out):
  * strategy(df) returns signal[t], the desired position decided AFTER bar t
    closes, using bars 0..t only.
  * The engine holds pos[t+1] = signal[t]: the position is entered at the OPEN
    of bar t+1, so the position held during bar t is decided from data through
    bar t-1. Slippage and fees are charged on every unit of position change at
    that t+1 open.
  * Bar return for the holding period is open[t+1] / open[t] - 1 (open-to-open),
    because that is the price at which positions actually change hands.

Guards:
  * assert_no_lookahead(): truncation test — signals on bars [0, k) must be
    bit-identical whether or not bars >= k exist. Catches shift(-n),
    center=True, full-series normalisation, rank(), idxmax(), etc.
  * guard_full_series_reductions(): runtime monkeypatch that raises when a
    strategy calls a pandas/numpy reduction over an object whose length equals
    the full input length (fit-on-full-series). Rolling/expanding are exempt
    because they never call these entry points.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from harness import config

StrategyFn = Callable[[pd.DataFrame], "pd.Series | np.ndarray"]


class LookaheadError(RuntimeError):
    pass


class FullSeriesReductionError(RuntimeError):
    pass


class StrategyOutputError(RuntimeError):
    pass


@dataclass
class BacktestResult:
    signal: np.ndarray          # raw strategy output (clipped, NaN->0), decided at close of t
    position: np.ndarray        # position held during bar t (= signal[t-1])
    bar_returns: np.ndarray     # open-to-open return of bar t
    gross: np.ndarray
    cost: np.ndarray
    net: np.ndarray
    equity: np.ndarray          # compounded net equity, starts at 1
    buy_hold: np.ndarray        # compounded open-to-open equity of holding the asset
    trades: pd.DataFrame        # one row per trade: start, end, pnl, side
    fee_bps: float
    slippage_bps: float
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------
_PD_REDUCTIONS = ["mean", "std", "var", "median", "min", "max", "sum", "quantile",
                  "skew", "kurt", "sem", "prod", "mode", "idxmax", "idxmin", "argmax", "argmin"]
_NP_REDUCTIONS = ["mean", "std", "var", "median", "average", "percentile", "quantile",
                  "nanmean", "nanstd", "nanvar", "nanmedian", "nanpercentile", "nanquantile",
                  "ptp", "argmax", "argmin", "nanargmax", "nanargmin"]


@contextlib.contextmanager
def guard_full_series_reductions(n: int):
    """Raise if a reduction is called over an object of exactly n elements."""
    saved = []

    def wrap_pd(cls, name):
        orig = getattr(cls, name)

        def wrapped(self, *a, **k):
            axis = k.get("axis", a[0] if a and name in ("mean", "std", "var", "sum", "min", "max") and isinstance(a[0], int) else None)
            if len(self) == n and axis in (None, 0, "index"):
                raise FullSeriesReductionError(
                    f"{cls.__name__}.{name}() over the full {n}-bar input. Statistics must be "
                    f"causal: use .rolling(w) or .expanding(), never the whole series.")
            return orig(self, *a, **k)
        wrapped.__name__ = name
        setattr(cls, name, wrapped)
        saved.append((cls, name, orig))

    def wrap_np(name):
        orig = getattr(np, name)

        def wrapped(a, *args, **k):
            try:
                is_full = np.ndim(a) == 1 and len(a) == n and k.get("axis") is None
            except TypeError:
                is_full = False
            if is_full:
                raise FullSeriesReductionError(
                    f"np.{name}() over the full {n}-bar input. Statistics must be causal: "
                    f"use rolling/expanding windows, never the whole series.")
            return orig(a, *args, **k)
        wrapped.__name__ = name
        setattr(np, name, wrapped)
        saved.append((np, name, orig))

    for name in _PD_REDUCTIONS:
        for cls in (pd.Series, pd.DataFrame):
            if hasattr(cls, name):
                wrap_pd(cls, name)
    for name in _NP_REDUCTIONS:
        if hasattr(np, name):
            wrap_np(name)
    try:
        yield
    finally:
        for obj, name, orig in reversed(saved):
            setattr(obj, name, orig)


def _coerce_signal(raw, n: int) -> np.ndarray:
    if isinstance(raw, pd.DataFrame):
        raise StrategyOutputError("strategy returned a DataFrame; return a Series of positions")
    if isinstance(raw, pd.Series):
        arr = raw.to_numpy(dtype="float64", na_value=np.nan)
    else:
        try:
            arr = np.asarray(raw, dtype="float64")
        except (TypeError, ValueError) as e:
            raise StrategyOutputError(f"strategy output not numeric: {e}") from e
    if arr.ndim != 1 or len(arr) != n:
        raise StrategyOutputError(f"strategy output shape {arr.shape}, expected ({n},)")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return np.clip(arr, -config.MAX_ABS_POSITION, config.MAX_ABS_POSITION)


def run_strategy(strategy: StrategyFn, df: pd.DataFrame) -> np.ndarray:
    """Execute a strategy under the full-series guard and coerce its output."""
    n = len(df)
    frame = df[["open", "high", "low", "close", "volume"]].copy()
    frame.index = pd.RangeIndex(n)
    with guard_full_series_reductions(n):
        raw = strategy(frame)
    return _coerce_signal(raw, n)


def assert_no_lookahead(strategy: StrategyFn, df: pd.DataFrame, full_signal: np.ndarray,
                        cuts: tuple[float, ...] = (0.13, 0.22, 0.31, 0.40, 0.49, 0.58, 0.67, 0.76, 0.85, 0.94)) -> None:
    """Truncation test: signal[:k] must not depend on bars >= k."""
    n = len(df)
    for frac in cuts:
        k = max(10, int(n * frac))
        trunc = run_strategy(strategy, df.iloc[:k].reset_index(drop=True))
        if not np.array_equal(trunc, full_signal[:k]):
            bad = int(np.flatnonzero(trunc != full_signal[:k])[0])
            raise LookaheadError(
                f"signal at bar {bad} changed when bars >= {k} were removed: the strategy "
                f"uses future data (shift(-n), center=True, full-series stats, rank, ...)")


# --------------------------------------------------------------------------
# Core backtest
# --------------------------------------------------------------------------
def _segment_trades(position: np.ndarray, net: np.ndarray) -> pd.DataFrame:
    n = len(position)
    if n == 0:
        return pd.DataFrame(columns=["start", "end", "side", "pnl", "bars"])
    sign = np.sign(position)
    prev = np.concatenate([[0.0], sign[:-1]])
    starts = (sign != 0) & (sign != prev)
    trade_id = np.cumsum(starts)
    in_trade = sign != 0
    if trade_id[in_trade].size == 0:
        return pd.DataFrame(columns=["start", "end", "side", "pnl", "bars"])
    ids = trade_id[in_trade]
    idx = np.arange(n)[in_trade]
    pnl = np.bincount(ids, weights=net[in_trade])[1:]
    bars = np.bincount(ids)[1:]
    # first/last bar index per trade id
    first = pd.Series(idx).groupby(ids).min().to_numpy()
    last = pd.Series(idx).groupby(ids).max().to_numpy()
    side = sign[first]
    return pd.DataFrame({"start": first, "end": last, "side": side, "pnl": pnl, "bars": bars})


def run_backtest(df: pd.DataFrame, signal: np.ndarray, bars_per_year: int,
                 fee_bps: float | None = None, slippage_bps: float | None = None,
                 extra_delay: int = 0) -> BacktestResult:
    """Structural t+1-open execution. `signal[t]` is decided after bar t closes."""
    fee_bps = config.FEE_BPS if fee_bps is None else fee_bps
    slippage_bps = config.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    n = len(df)
    open_ = df["open"].to_numpy(dtype="float64")
    sig = np.asarray(signal, dtype="float64")
    assert sig.shape == (n,), sig.shape

    shift = 1 + int(extra_delay)               # structural one-bar delay (+ canary delay)
    position = np.zeros(n)
    position[shift:] = sig[:-shift]            # pos[t] = signal[t-shift]

    bar_ret = np.zeros(n)
    bar_ret[:-1] = open_[1:] / open_[:-1] - 1.0   # open[t] -> open[t+1]; last bar flat
    gross = position * bar_ret
    turnover = np.abs(np.diff(position, prepend=0.0))
    cost = turnover * (fee_bps + slippage_bps) / 1e4
    net = gross - cost
    equity = np.cumprod(1.0 + net)
    buy_hold = np.cumprod(1.0 + bar_ret)

    trades = _segment_trades(position, net)
    res = BacktestResult(signal=sig, position=position, bar_returns=bar_ret, gross=gross,
                         cost=cost, net=net, equity=equity, buy_hold=buy_hold, trades=trades,
                         fee_bps=fee_bps, slippage_bps=slippage_bps)
    res.stats = compute_stats(res, bars_per_year)
    return res


def max_drawdown(equity: np.ndarray) -> float:
    if len(equity) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = 1.0 - equity / peak
    return float(dd.max())


def sharpe(net: np.ndarray, bars_per_year: int) -> float:
    if len(net) < 2:
        return 0.0
    sd = net.std(ddof=1)
    if not np.isfinite(sd) or sd == 0.0:
        return 0.0
    return float(net.mean() / sd * np.sqrt(bars_per_year))


def compute_stats(res: BacktestResult, bars_per_year: int) -> dict:
    n = len(res.net)
    tr = res.trades
    wins = tr[tr.pnl > 0].pnl if len(tr) else pd.Series(dtype=float)
    losses = tr[tr.pnl <= 0].pnl if len(tr) else pd.Series(dtype=float)
    mean_win = float(wins.mean()) if len(wins) else 0.0
    mean_loss = float(-losses.mean()) if len(losses) else 0.0
    turnover_total = float(np.abs(np.diff(res.position, prepend=0.0)).sum())
    net = res.net
    m = net.mean() if n else 0.0
    s = net.std(ddof=1) if n > 1 else 0.0
    skew = float(((net - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    kurt = float(((net - m) ** 4).mean() / s ** 4) if s > 0 else 3.0
    return {
        "bars": int(n),
        "bars_per_year": int(bars_per_year),
        "sharpe": sharpe(net, bars_per_year),
        "sharpe_gross": sharpe(res.gross, bars_per_year),
        "total_return": float(res.equity[-1] - 1.0) if n else 0.0,
        "buy_hold_return": float(res.buy_hold[-1] - 1.0) if n else 0.0,
        "max_drawdown": max_drawdown(res.equity),
        "trades": int(len(tr)),
        "win_rate": float((tr.pnl > 0).mean()) if len(tr) else 0.0,
        "payoff_ratio": float(mean_win / mean_loss) if mean_loss > 0 else (float("inf") if mean_win > 0 else 0.0),
        "time_in_market": float((res.position != 0).mean()) if n else 0.0,
        "turnover_annual": float(turnover_total / n * bars_per_year) if n else 0.0,
        "total_cost": float(res.cost.sum()),
        "net_sum": float(net.sum()),
        "skew": skew,
        "kurt": kurt,
    }
