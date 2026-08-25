import json

import numpy as np
import pandas as pd
import pytest

from live import alerter as al
from live import config as lc
from live import notify
from live.alerter_report import build as build_report
from live.strategy_loader import run_with_indicators


def mk_bars(n, start_ts, tf_sec, seed=0, trend=0.0):
    rng = np.random.default_rng(seed)
    r = rng.normal(trend, 0.005, n)
    close = 100 * np.cumprod(1 + r)
    open_ = np.concatenate([[100.0], close[:-1]])
    return pd.DataFrame({"ts": start_ts + np.arange(n) * tf_sec * 1000, "open": open_, "high": np.maximum(open_, close) * 1.001,
                         "low": np.minimum(open_, close) * 0.999, "close": close, "volume": np.ones(n)})


class FakeFeed:
    def __init__(self, bars):
        self.bars = bars; self.n_visible = 400; self.fail = 0

    def fetch_closed_bars(self, symbol, tf, limit):
        if self.fail:
            self.fail -= 1
            raise ConnectionError("exchange down")
        return self.bars.iloc[:self.n_visible].tail(limit).reset_index(drop=True)


def ma_strategy(df):
    fast = df["close"].rolling(5).mean()
    slow = df["close"].rolling(20).mean()
    regime = (fast > slow).astype(float)
    return regime


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(lc, "LOG_DIR", tmp_path)
    monkeypatch.setattr(lc, "ALERTS_LOG", tmp_path / "alerts.jsonl")
    monkeypatch.setattr(lc, "OUTCOMES_LOG", tmp_path / "alert_outcomes.jsonl")
    monkeypatch.setattr(lc, "LIVE_BARS_DIR", tmp_path / "live_bars")
    monkeypatch.setattr(lc, "STATE_FILE", tmp_path / "state.json")
    monkeypatch.setattr(lc, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(lc, "PROMOTED_FILE", tmp_path / "promoted.json")
    monkeypatch.setattr(lc, "LOOKBACK_BARS", 300)
    monkeypatch.setattr(lc, "ERROR_STREAK_HALT", 3)
    pushes = []
    monkeypatch.setattr(notify, "CHANNELS", {"fake": lambda t, b, k: (pushes.append((k, t, b)) or (True, "ok"))})
    monkeypatch.setattr(lc, "NOTIFY_CHANNELS", ["fake"])
    prom = {"sha": "abc1234", "symbol": "BTC/USDT", "timeframe": "1h", "description": "test ma", "source": "x",
            "backtest": {"max_drawdown": 0.30}, "holdout_checked": True}
    (tmp_path / "promoted.json").write_text(json.dumps(prom))
    tf = 3600
    bars = mk_bars(600, 1_700_000_000_000, tf, trend=0.001)
    feed = FakeFeed(bars)
    clock = {"now": bars["ts"].iloc[feed.n_visible - 1] / 1000 + tf + 30}
    a = al.Alerter(mode="paper", feed=feed, strategy_fn=ma_strategy, promoted=prom, now_fn=lambda: clock["now"])
    return a, feed, clock, pushes, tf


def advance(a, feed, clock, tf, k=1):
    feed.n_visible += k
    clock["now"] = feed.bars["ts"].iloc[feed.n_visible - 1] / 1000 + tf + 30


def test_first_tick_alerts_on_position_change_with_full_contents(env):
    a, feed, clock, pushes, tf = env
    rec = a.tick()
    assert rec is not None and rec["kind"] == "signal"
    for key in ("ts", "symbol", "direction", "current_position", "target_position", "indicators", "size", "strategy_sha", "bar_time"):
        assert key in rec
    assert rec["strategy_sha"] == "abc1234" and rec["symbol"] == "BTC/USDT"
    assert {"fast", "slow", "regime", "close"} <= set(rec["indicators"])           # captured strategy locals
    assert rec["size"]["trade_notional_usd"] == pytest.approx(lc.ACCOUNT_EQUITY_USD * lc.RISK_FRACTION * abs(rec["delta"]))
    assert pushes and pushes[-1][0] == "signal" and "ACK REQUIRED" in pushes[-1][2]
    assert (lc.ALERTS_LOG).exists() and rec["delivered"]


def test_no_alert_without_position_change_or_new_bar(env):
    a, feed, clock, pushes, tf = env
    a.tick(); n = len(pushes)
    assert a.tick() is None and len(pushes) == n           # same bar again → nothing
    # find a bar where the signal does not change
    for _ in range(50):
        advance(a, feed, clock, tf)
        before = a.state["designed_position"]
        r = a.tick()
        if r is None:
            assert a.state["designed_position"] == before
            break
    else:
        pytest.fail("never saw an unchanged bar")


def test_ack_required_reminder_and_outcomes(env):
    a, feed, clock, pushes, tf = env
    rec = a.tick(); assert rec
    advance(a, feed, clock, tf, 3)
    a.tick()
    kinds = [p[0] for p in pushes]
    assert "reminder" in kinds
    # ack it
    class A: pass
    x = A(); x.alert_id = rec["id"]; x.outcome = "skipped"; x.reason = "asleep"; x.price = None
    al.cmd_ack(x)
    assert al.pending_alerts() == [] or all(p["id"] != rec["id"] for p in al.pending_alerts())
    with pytest.raises(SystemExit):
        al.cmd_ack(x)   # no double outcomes


def test_stale_data_halts_loudly(env):
    a, feed, clock, pushes, tf = env
    a.tick()
    clock["now"] += tf * 5          # five hours pass, feed still shows the old bar
    rec = a.tick()
    assert rec["kind"] == "halt" and "stale" in rec["reason"]
    assert pushes[-1][0] == "halt"
    assert a.state["halted"] and a.tick() is None    # sticky


def test_error_streak_halts(env):
    a, feed, clock, pushes, tf = env
    a.tick()
    feed.fail = 5
    advance(a, feed, clock, tf)
    assert a.tick() is None and a.tick() is None
    rec = a.tick()
    assert rec["kind"] == "halt" and "error streak" in rec["reason"]


def test_drawdown_halt(env, monkeypatch):
    a, feed, clock, pushes, tf = env
    a.tick()
    a.state["designed_equity"] = 0.6; a.state["designed_peak"] = 1.0   # 40% dd > 30% backtested
    advance(a, feed, clock, tf)
    rec = a.tick()
    assert rec["kind"] == "halt" and "drawdown" in rec["reason"]


def test_resume_clears_halt(env):
    a, feed, clock, pushes, tf = env
    a.halt("test")
    class A: pass
    x = A(); x.reason = "checked"; x.reset_drawdown = True
    al.cmd_resume(x)
    assert not al.load_state()["halted"] and pushes[-1][0] == "resume"


def test_report_gap_when_alerts_skipped(env):
    a, feed, clock, pushes, tf = env
    ids = []
    for _ in range(120):
        r = a.tick()
        if r:
            ids.append(r["id"])
        advance(a, feed, clock, tf)
    assert len(ids) >= 2
    class A: pass
    for i, aid in enumerate(ids):
        x = A(); x.alert_id = aid; x.outcome = "taken" if i % 2 == 0 else "skipped"; x.reason = "test"; x.price = None
        al.cmd_ack(x)
    md, s = build_report("paper")
    assert s["alerts"] == len(ids) and s["skipped"] == len(ids) // 2 and s["unlogged"] == 0
    assert "gap" in s and "as designed" in md and "actually taken" in md
    assert abs(s["designed_return"] - s["taken_return"]) == pytest.approx(abs(s["gap"]))


def test_unlogged_counted_as_skipped(env):
    a, feed, clock, pushes, tf = env
    a.tick()
    md, s = build_report("paper")
    assert s["unlogged"] == 1 and "WITHOUT A LOGGED OUTCOME" in md


def test_indicator_capture_generic():
    df = mk_bars(100, 0, 3600)
    def strat(d):
        w = 10
        m = d["close"].rolling(w).mean()
        ratio = d["close"] / m
        return (ratio > 1).astype(float)
    sig, ind = run_with_indicators(strat, df)
    assert sig in (0.0, 1.0) and ind["w"] == 10 and "m" in ind and "ratio" in ind and "close" in ind


def test_ntfy_headers_are_ascii_and_never_raise(monkeypatch):
    import httpx
    seen = {}
    def fake_post(url, content=None, headers=None, timeout=None):
        seen.update(headers); headers["Title"].encode("ascii")
        class R: status_code = 200
        return R()
    monkeypatch.setattr(httpx, "post", fake_post)
    monkeypatch.setattr(lc, "NTFY_TOPIC", "t")
    ok, msg = notify.push_ntfy("\U0001f4c8 BTC/USDT OPEN LONG (paper)", "body \u2192 ok", "signal")
    assert ok and seen["Title"] == "BTC/USDT OPEN LONG (paper)" and seen["Priority"] == "high"
    monkeypatch.setattr(httpx, "post", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    ok, msg = notify.push_ntfy("x", "y"); assert not ok and "boom" in msg
    monkeypatch.setattr(notify, "CHANNELS", {"bad": lambda t, b, k: (_ for _ in ()).throw(ValueError("x"))})
    assert notify.push("t", "b", "signal", ["bad"])["bad"].startswith("FAILED")
