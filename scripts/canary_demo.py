"""Prove the three canaries catch their bugs, on the real validation data.

  bug A  obvious lookahead        -> LEAK_SUSPECTED  (also caught earlier by scan + truncation test)
  bug B  same-bar execution       -> TIMING_FRAGILE
  bug C  profits only at zero cost-> COST_MODEL_BROKEN (cost model deliberately zeroed / rebated)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from harness import config  # noqa: E402
from harness.canaries import run_canaries  # noqa: E402
from harness.data import load_series  # noqa: E402
from harness.engine import LookaheadError, run_backtest, run_strategy  # noqa: E402
from harness.evaluate import evaluate_fn  # noqa: E402
from harness.scan import ScanError, scan_source  # noqa: E402

LOOKAHEAD_SRC = '''"""BUG A: peeks at next bar's close."""
import numpy as np
import pandas as pd

def strategy(df):
    nxt = df["close"].shift(-1)
    return pd.Series(np.sign(nxt - df["close"]), index=df.index).fillna(0.0)
'''


def bug_a_lookahead(df):
    # trade in the direction of the NEXT bar's open-to-open move (what the engine actually pays)
    o = df["open"].to_numpy()
    fut = np.zeros(len(df)); fut[:-2] = np.sign(o[2:] - o[1:-1])
    return pd.Series(fut, index=df.index)


def honest_momentum(df):
    return pd.Series(np.sign(df["close"] - df["close"].shift(1)), index=df.index).fillna(0.0)


def bug_b_same_bar(df):
    # Classic same-bar execution bug: the 1-bar momentum signal is aligned one bar too early,
    # so the position "executes" on the bar whose close it was computed from. One extra bar of
    # delay turns it back into the honest rule - and the edge vanishes.
    return honest_momentum(df).shift(-1).fillna(0.0)


def bug_c_churn(df):
    # alternates long/short every bar: only 'profitable' if the cost model pays you to trade
    n = len(df)
    return pd.Series(np.where(np.arange(n) % 2 == 0, 1.0, -1.0), index=df.index)


def show(name, canaries, ev):
    agg = ev.aggregate()
    print(f"\n=== {name} ===")
    print(f"  metric {ev.metric:+.3f}  sharpe {agg['sharpe']:+.2f}  trades {agg['trades']}")
    print(f"  shuffled-returns sharpe {canaries['shuffled']['sharpe']:+.2f}  -> flag={canaries['shuffled']['flag']}")
    print(f"  extra-delay sharpe      {canaries['delay']['sharpe']:+.2f}  retained={canaries['delay']['retained']}  -> flag={canaries['delay']['flag']}")
    print(f"  sign-flip sharpe        {canaries['signflip']['sharpe']:+.2f}  cost_paid={canaries['signflip']['cost_paid']:.4f}  identity_violation={canaries['signflip']['identity_violation']:.2e}  -> flag={canaries['signflip']['flag']}")
    print(f"  FLAGS: {canaries['flags'] or 'none'}")


def main():
    series = load_series(["s0", "s2"])   # the two hourly series keep this quick
    frames = [(s.id, s.val, s.bars_per_year) for s in series]

    # ---------------------------------------------------------------- A ---
    print("=== bug A: obvious lookahead ===")
    try:
        scan_source(LOOKAHEAD_SRC)
    except ScanError as e:
        print(f"  layer 1 (static scan) rejects shift(-1): {e}")
    try:
        evaluate_fn(bug_a_lookahead, series, "val", run_checks=True)
        print("  layer 2 (truncation test): NOT caught")
    except LookaheadError as e:
        print(f"  layer 2 (truncation test) rejects: {str(e)[:100]}")
    ev = evaluate_fn(bug_a_lookahead, series, "val", run_checks=False)
    can = run_canaries(bug_a_lookahead, frames, ev.signals, ev.aggregate()["sharpe"])
    show("bug A canaries (checks bypassed to reach them)", can, ev)
    assert "LEAK_SUSPECTED" in can["flags"], "canary 1 failed to flag lookahead"

    # ---------------------------------------------------------------- B ---
    print("\n=== bug B: same-bar execution (off-by-one alignment) ===")
    try:
        evaluate_fn(bug_b_same_bar, series, "val", run_checks=True)
        print("  truncation test: NOT caught")
    except LookaheadError as e:
        print(f"  truncation test rejects first: {str(e)[:90]}")
    ev_h = evaluate_fn(honest_momentum, series, "val", run_checks=True)
    print(f"  (honest, correctly aligned version of the same rule: sharpe {ev_h.aggregate()['sharpe']:+.2f})")
    ev = evaluate_fn(bug_b_same_bar, series, "val", run_checks=False)
    can = run_canaries(bug_b_same_bar, frames, ev.signals, ev.aggregate()["sharpe"])
    show("bug B canaries (checks bypassed to reach them)", can, ev)
    assert "TIMING_FRAGILE" in can["flags"], "canary 2 did not flag same-bar dependence"

    # ---------------------------------------------------------------- C ---
    for label, fee, slip in [("frozen cost model (fee 10 / slip 10 bps)", None, None),
                             ("ZEROED cost model (fee 0 / slip 0)", 0.0, 0.0),
                             ("REBATE cost model (fee -10 / slip 0)", -10.0, 0.0)]:
        ev = evaluate_fn(bug_c_churn, series, "val", run_checks=True, fee_bps=fee, slippage_bps=slip)
        can = run_canaries(bug_c_churn, frames, ev.signals, ev.aggregate()["sharpe"], fee_bps=fee, slippage_bps=slip)
        show(f"bug C: churner under {label}", can, ev)
        if fee is None:
            assert "COST_MODEL_BROKEN" not in can["flags"], "false positive on a correct cost model"
        else:
            assert "COST_MODEL_BROKEN" in can["flags"], f"canary 3 missed broken cost model ({label})"
    print("\nall three canaries caught their bug; the correct cost model produced no false positive.")


if __name__ == "__main__":
    main()
