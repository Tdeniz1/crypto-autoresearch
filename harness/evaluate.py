"""Glue: source text -> strategy fn -> per-series backtests -> one metric."""
from __future__ import annotations

import types
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from harness import config, metric as metric_mod
from harness.data import Series, load_series
from harness.engine import (BacktestResult, LookaheadError, StrategyOutputError,
                            assert_no_lookahead, run_backtest, run_strategy)
from harness.scan import ScanError, scan_source


class StrategyLoadError(RuntimeError):
    pass


@dataclass
class Evaluation:
    metric: float
    per_series: dict[str, dict]
    results: dict[str, BacktestResult] = field(default_factory=dict)
    signals: dict[str, np.ndarray] = field(default_factory=dict)
    description: str = ""

    def aggregate(self) -> dict:
        ss = list(self.per_series.values())
        if not ss:
            return {}
        n_tr = sum(s["trades"] for s in ss)
        wins = sum(s["win_rate"] * s["trades"] for s in ss)
        payoffs = [s["payoff_ratio"] for s in ss if np.isfinite(s["payoff_ratio"])]
        return {
            "sharpe": float(np.mean([s["sharpe"] for s in ss])),
            "sharpe_gross": float(np.mean([s["sharpe_gross"] for s in ss])),
            "max_drawdown": float(max(s["max_drawdown"] for s in ss)),
            "trades": int(n_tr),
            "min_trades": int(min(s["trades"] for s in ss)),
            "win_rate": float(wins / n_tr) if n_tr else 0.0,
            "payoff_ratio": float(np.mean(payoffs)) if payoffs else 0.0,
            "time_in_market": float(np.mean([s["time_in_market"] for s in ss])),
            "turnover_annual": float(np.mean([s["turnover_annual"] for s in ss])),
            "total_return": float(np.mean([s["total_return"] for s in ss])),
            "buy_hold_return": float(np.mean([s["buy_hold_return"] for s in ss])),
            "bars": int(sum(s["bars"] for s in ss)),
            "bars_per_year": float(np.mean([s["bars_per_year"] for s in ss])),
            "skew": float(np.mean([s["skew"] for s in ss])),
            "kurt": float(np.mean([s["kurt"] for s in ss])),
        }


def load_strategy(source: str, *, scan: bool = True):
    """Compile strategy.py source into a callable. Scans first unless told not to."""
    if scan:
        scan_source(source)  # raises ScanError
    mod = types.ModuleType("strategy_candidate")
    try:
        exec(compile(source, "strategy.py", "exec"), mod.__dict__)  # noqa: S102 — scanned first
    except Exception as e:  # noqa: BLE001
        raise StrategyLoadError(f"{type(e).__name__}: {e}") from e
    fn = getattr(mod, "strategy", None)
    if not callable(fn):
        raise StrategyLoadError("no callable `strategy` in module")
    doc = (mod.__doc__ or "").strip().splitlines()
    return fn, (doc[0].strip() if doc else "")


def evaluate_fn(fn, series: list[Series], split: str = "val", run_checks: bool = True,
                fee_bps: float | None = None, slippage_bps: float | None = None) -> Evaluation:
    per, results, signals = {}, {}, {}
    for s in series:
        df = getattr(s, split)
        sig = run_strategy(fn, df)
        if run_checks:
            assert_no_lookahead(fn, df, sig)
        res = run_backtest(df, sig, s.bars_per_year, fee_bps, slippage_bps)
        per[s.id] = res.stats
        results[s.id] = res
        signals[s.id] = sig
    return Evaluation(metric=metric_mod.score(list(per.values())), per_series=per,
                      results=results, signals=signals)


def evaluate_source(source: str, series: list[Series] | None = None, split: str = "val",
                    run_checks: bool = True, scan: bool = True, **kw) -> Evaluation:
    fn, desc = load_strategy(source, scan=scan)
    series = series or load_series()
    ev = evaluate_fn(fn, series, split=split, run_checks=run_checks, **kw)
    ev.description = desc
    return ev


__all__ = ["Evaluation", "evaluate_fn", "evaluate_source", "load_strategy", "StrategyLoadError",
           "ScanError", "LookaheadError", "StrategyOutputError"]
