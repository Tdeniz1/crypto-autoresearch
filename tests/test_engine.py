import numpy as np
import pandas as pd
import pytest

from harness import config
from harness.engine import (FullSeriesReductionError, LookaheadError, assert_no_lookahead,
                            run_backtest, run_strategy)
from tests.conftest import synthetic_bars


def test_execution_is_t_plus_one_open():
    df = synthetic_bars(50)
    sig = np.zeros(50); sig[10] = 1.0  # decide long after bar 10 closes
    res = run_backtest(df, sig, 8760, fee_bps=0, slippage_bps=0)
    assert res.position[10] == 0.0 and res.position[11] == 1.0 and res.position[12] == 0.0
    o = df["open"].to_numpy()
    assert np.isclose(res.gross[11], o[12] / o[11] - 1)
    assert res.gross[10] == 0.0


def test_costs_charged_per_side_on_position_change():
    df = synthetic_bars(20)
    sig = np.zeros(20); sig[3] = 1.0; sig[4] = -1.0  # long then flip to short
    res = run_backtest(df, sig, 8760, fee_bps=10, slippage_bps=10)
    per_unit = 20 / 1e4
    assert np.isclose(res.cost[4], per_unit)          # 0 -> +1
    assert np.isclose(res.cost[5], 2 * per_unit)      # +1 -> -1
    assert np.isclose(res.cost[6], per_unit)          # -1 -> 0
    assert np.isclose(res.cost.sum(), 4 * per_unit)


def test_cost_identity_sign_flip():
    df = synthetic_bars(500)
    rng = np.random.default_rng(0)
    sig = rng.choice([-1.0, 0.0, 1.0], 500)
    a = run_backtest(df, sig, 8760)
    b = run_backtest(df, -sig, 8760)
    assert abs(a.net.sum() + b.net.sum() + 2 * a.cost.sum()) < 1e-9


def test_extra_delay_shifts_one_more_bar():
    df = synthetic_bars(30)
    sig = np.zeros(30); sig[5] = 1.0
    res = run_backtest(df, sig, 8760, extra_delay=1)
    assert res.position[6] == 0.0 and res.position[7] == 1.0


def test_lookahead_truncation_catches_shift_minus_one():
    df = synthetic_bars(3000)

    def leaky(d):
        return np.sign(d["close"].shift(-1) - d["close"]).fillna(0.0)

    sig = run_strategy(leaky, df)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky, df, sig)


def test_lookahead_truncation_catches_full_series_rank():
    df = synthetic_bars(3000)

    def leaky(d):
        r = d["close"].rank(pct=True)
        return (r > 0.5).astype(float)

    sig = run_strategy(leaky, df)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(leaky, df, sig)


def test_full_series_reduction_guard():
    df = synthetic_bars(1000)

    def bad(d):
        z = (d["close"] - d["close"].mean()) / d["close"].std()
        return (z > 0).astype(float)

    with pytest.raises(FullSeriesReductionError):
        run_strategy(bad, df)

    def bad_np(d):
        return (d["close"] > np.mean(d["close"].to_numpy())).astype(float)

    with pytest.raises(FullSeriesReductionError):
        run_strategy(bad_np, df)

    def good(d):
        z = (d["close"] - d["close"].rolling(50).mean()) / d["close"].rolling(50).std()
        return (z > 0).astype(float)

    sig = run_strategy(good, df)
    assert_no_lookahead(good, df, sig)
    assert np.all(np.abs(sig) <= 1)


def test_guard_is_removed_after_run():
    df = synthetic_bars(100)
    run_strategy(lambda d: d["close"] * 0, df)
    assert float(df["close"].mean()) > 0  # would raise if the patch leaked


def test_clipping_and_nan_handling():
    df = synthetic_bars(10)
    sig = run_strategy(lambda d: pd.Series([np.nan, 5, -5, 0.5] + [0] * 6, index=d.index), df)
    assert sig[0] == 0 and sig[1] == 1 and sig[2] == -1 and sig[3] == 0.5


def test_trade_segmentation_and_stats():
    df = synthetic_bars(100)
    sig = np.zeros(100); sig[5:15] = 1.0; sig[30:40] = -1.0; sig[40:50] = 1.0
    res = run_backtest(df, sig, 8760)
    assert res.stats["trades"] == 3
    assert res.stats["time_in_market"] == pytest.approx(30 / 100)
