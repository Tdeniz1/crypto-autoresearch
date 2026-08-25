import numpy as np
import pandas as pd

from harness.canaries import COST_MODEL_BROKEN, LEAK_SUSPECTED, TIMING_FRAGILE, run_canaries, shuffle_bars
from harness.evaluate import evaluate_fn
from tests.conftest import synthetic_bars


def _frames(series):
    return [(s.id, s.val, s.bars_per_year) for s in series]


def test_shuffle_preserves_bar_shapes_and_return_multiset():
    df = synthetic_bars(500)
    out = shuffle_bars(df, np.random.default_rng(1))
    assert len(out) == len(df)
    a = np.sort((df["close"] / df["close"].shift(1)).dropna().to_numpy())
    b = np.sort((out["close"] / out["close"].shift(1)).dropna().to_numpy())
    assert np.allclose(a, b)
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-9).all()


def test_lookahead_strategy_flagged_leak(synth_series):
    def leaky(d):
        o = d["open"].to_numpy()
        f = np.zeros(len(d)); f[:-2] = np.sign(o[2:] - o[1:-1])
        return pd.Series(f, index=d.index)
    ev = evaluate_fn(leaky, synth_series, "val", run_checks=False)
    c = run_canaries(leaky, _frames(synth_series), ev.signals, ev.aggregate()["sharpe"])
    assert LEAK_SUSPECTED in c["flags"]


def test_clean_strategy_not_flagged_leak(synth_series):
    def clean(d):
        m = d["close"].rolling(50).mean()
        return (d["close"] > m).astype(float)
    ev = evaluate_fn(clean, synth_series, "val")
    c = run_canaries(clean, _frames(synth_series), ev.signals, ev.aggregate()["sharpe"])
    assert LEAK_SUSPECTED not in c["flags"]
    assert COST_MODEL_BROKEN not in c["flags"]


def test_zeroed_cost_model_flagged(synth_series):
    def churn(d):
        return pd.Series(np.where(np.arange(len(d)) % 2 == 0, 1.0, -1.0), index=d.index)
    ev = evaluate_fn(churn, synth_series, "val", fee_bps=0.0, slippage_bps=0.0)
    c = run_canaries(churn, _frames(synth_series), ev.signals, ev.aggregate()["sharpe"], fee_bps=0.0, slippage_bps=0.0)
    assert COST_MODEL_BROKEN in c["flags"]


def test_rebate_cost_model_flagged(synth_series):
    def churn(d):
        return pd.Series(np.where(np.arange(len(d)) % 2 == 0, 1.0, -1.0), index=d.index)
    ev = evaluate_fn(churn, synth_series, "val", fee_bps=-10.0, slippage_bps=0.0)
    c = run_canaries(churn, _frames(synth_series), ev.signals, ev.aggregate()["sharpe"], fee_bps=-10.0, slippage_bps=0.0)
    assert COST_MODEL_BROKEN in c["flags"]
    assert c["signflip"]["sharpe"] > 0  # inverted strategy also "profits"


def test_timing_fragile_on_engineered_one_bar_edge():
    # Engineer a series where the open[t+1]->open[t+2] move follows the sign of bar t's body,
    # but is independent of bar t-1's body: tradable at the t+1 open, gone with one more bar of delay.
    rng = np.random.default_rng(3)
    n = 4000
    body = rng.normal(0, 0.01, n)
    open_ = np.empty(n); open_[0] = 100.0
    for t in range(1, n):
        prev_body = body[t - 2] if t >= 2 else 0.0
        open_[t] = open_[t - 1] * (1 + 0.004 * np.sign(prev_body) + rng.normal(0, 0.002))
    close = open_ * (1 + body)
    high = np.maximum(open_, close) * 1.001; low = np.minimum(open_, close) * 0.999
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": np.ones(n)})
    from harness.data import Series
    s = [Series(id="e", bars_per_year=8760, train=df, val=df)]

    def one_bar_edge(d):
        return pd.Series(np.sign(d["close"] - d["open"]), index=d.index).fillna(0.0)
    ev = evaluate_fn(one_bar_edge, s, "val", fee_bps=1.0, slippage_bps=1.0)
    c = run_canaries(one_bar_edge, _frames(s), ev.signals, ev.aggregate()["sharpe"], fee_bps=1.0, slippage_bps=1.0)
    assert ev.aggregate()["sharpe"] > 0
    assert TIMING_FRAGILE in c["flags"]
    assert LEAK_SUSPECTED not in c["flags"]   # no lookahead here: pure timing fragility
