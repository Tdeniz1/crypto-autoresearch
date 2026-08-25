"""Three automated leakage canaries, run on every candidate that beats the best.

1. shuffled returns  -> LEAK_SUSPECTED   (still scores well on permuted bars)
2. extra delay       -> TIMING_FRAGILE   (edge vanishes with one more bar of delay)
3. sign flip         -> COST_MODEL_BROKEN(inverted strategy also profits, or the
                        cost identity net_orig + net_inv == -2*cost is violated)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from harness import config
from harness.engine import StrategyFn, run_backtest, run_strategy

LEAK_SUSPECTED = "LEAK_SUSPECTED"
TIMING_FRAGILE = "TIMING_FRAGILE"
COST_MODEL_BROKEN = "COST_MODEL_BROKEN"


def shuffle_bars(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Permute bar order while preserving each bar's intra-bar shape.

    Every original bar j is described by its ratios to the previous close
    (o/c_prev, h/c_prev, l/c_prev, c/c_prev). The permuted series rebuilds a
    price path from the shuffled close ratios and re-applies each bar's own
    ratios, so returns are permuted but bar structure is intact.
    """
    o = df["open"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy()
    c = df["close"].to_numpy(); v = df["volume"].to_numpy()
    n = len(df)
    if n < 3:
        return df.copy()
    c_prev = c[:-1]
    ratios = np.stack([o[1:] / c_prev, h[1:] / c_prev, l[1:] / c_prev, c[1:] / c_prev, v[1:]], axis=1)
    perm = rng.permutation(n - 1)
    r = ratios[perm]
    new_c = np.empty(n); new_c[0] = c[0]
    new_c[1:] = c[0] * np.cumprod(r[:, 3])
    prev = new_c[:-1]
    out = pd.DataFrame({
        "open": np.concatenate([[o[0]], prev * r[:, 0]]),
        "high": np.concatenate([[h[0]], prev * r[:, 1]]),
        "low": np.concatenate([[l[0]], prev * r[:, 2]]),
        "close": new_c,
        "volume": np.concatenate([[v[0]], r[:, 4]]),
    })
    out.index = pd.RangeIndex(n)
    return out


def _mean(xs):
    xs = [x for x in xs if np.isfinite(x)]
    return float(np.mean(xs)) if xs else 0.0


def run_canaries(strategy: StrategyFn, series_frames: list[tuple[str, pd.DataFrame, int]],
                 signals: dict[str, np.ndarray], real_sharpe: float,
                 fee_bps: float | None = None, slippage_bps: float | None = None,
                 seed: int = config.SEED) -> dict:
    """series_frames: [(id, val_df, bars_per_year)], signals: id -> real signal."""
    fee = config.FEE_BPS if fee_bps is None else fee_bps
    slip = config.SLIPPAGE_BPS if slippage_bps is None else slippage_bps
    rng = np.random.default_rng(seed)

    shuffled, delayed, inverted = [], [], []
    identity_violation = 0.0
    cost_paid = 0.0
    trades_total = 0
    for sid, df, bpy in series_frames:
        sig = signals[sid]
        # 1 — shuffled returns
        for _ in range(config.CANARY_SHUFFLES):
            sdf = shuffle_bars(df, rng)
            s_sig = run_strategy(strategy, sdf)
            shuffled.append(run_backtest(sdf, s_sig, bpy, fee, slip).stats["sharpe"])
        # 2 — extra delay
        delayed.append(run_backtest(df, sig, bpy, fee, slip, extra_delay=1).stats["sharpe"])
        # 3 — sign flip + cost identity
        base = run_backtest(df, sig, bpy, fee, slip)
        inv = run_backtest(df, -sig, bpy, fee, slip)
        inverted.append(inv.stats["sharpe"])
        identity_violation = max(identity_violation,
                                 abs(base.net.sum() + inv.net.sum() + 2.0 * base.cost.sum()))
        cost_paid += base.stats["total_cost"]
        trades_total += base.stats["trades"]

    sh, de, iv = _mean(shuffled), _mean(delayed), _mean(inverted)
    flags = []
    leak = sh > config.LEAK_SHUFFLE_ABS_SHARPE and sh > config.LEAK_SHUFFLE_FRAC * max(real_sharpe, 0.0)
    if leak:
        flags.append(LEAK_SUSPECTED)
    timing = real_sharpe > 0 and de < config.TIMING_RETAIN_FRAC * real_sharpe
    if timing:
        flags.append(TIMING_FRAGILE)
    cost_broken = (real_sharpe > 0 and iv > config.COST_INVERTED_SHARPE) \
        or (trades_total > 0 and cost_paid <= 0.0) \
        or identity_violation > config.COST_IDENTITY_TOL
    if cost_broken:
        flags.append(COST_MODEL_BROKEN)
    return {
        "shuffled": {"sharpe": sh, "flag": bool(leak), "n": len(shuffled)},
        "delay": {"sharpe": de, "retained": (de / real_sharpe) if real_sharpe else None, "flag": bool(timing)},
        "signflip": {"sharpe": iv, "cost_paid": cost_paid, "identity_violation": identity_violation,
                     "flag": bool(cost_broken)},
        "flags": flags,
    }
