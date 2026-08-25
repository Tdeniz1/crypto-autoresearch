"""Signal alerter. Runs the promoted strategy on live bars and alerts when the position should change.

NEVER places orders. Same code path in paper and live mode; only credentials/tags differ.

    uv run python -m live.alerter run --once [--mode paper]      # one tick (cron / Railway cron)
    uv run python -m live.alerter run --loop [--mode paper]      # daemon: wakes after every bar close
    uv run python -m live.alerter ack <alert_id> taken|skipped --reason "..." [--price 61234.5]
    uv run python -m live.alerter status
    uv run python -m live.alerter resume --reason "..."          # clear a halt (halts are sticky)
    uv run python -m live.alerter test-push

Every signal alert requires an outcome (`ack`). Un-acked alerts are re-notified every tick.
Health halts (stale data, API error streak, live drawdown > backtested max DD) are alerts too —
priority urgent — and stop signal alerts until you `resume`.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from live import config
from live.notify import push
from live.strategy_loader import load_promoted, load_strategy, run_with_indicators

log = logging.getLogger("alerter")


# --------------------------------------------------------------------------- io
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).isoformat(timespec="seconds")


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def append_jsonl(p: Path, rec: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, separators=(",", ":"), default=str) + "\n")
        f.flush()


def load_state() -> dict:
    if config.STATE_FILE.exists():
        try:
            return json.loads(config.STATE_FILE.read_text())
        except json.JSONDecodeError:
            pass
    return {"designed_position": 0.0, "actual_position": 0.0, "last_bar_ts": None, "error_streak": 0,
            "halted": False, "halt_reason": None, "designed_equity": 1.0, "designed_peak": 1.0,
            "actual_equity": 1.0, "actual_peak": 1.0, "alerts_emitted": 0}


def save_state(st: dict) -> None:
    config.STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = config.STATE_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=2))
    os.replace(tmp, config.STATE_FILE)


def bars_log_path(symbol: str, timeframe: str) -> Path:
    return config.LIVE_BARS_DIR / f"{symbol.replace('/', '_')}_{timeframe}.jsonl"


# --------------------------------------------------------------------------- data
def timeframe_seconds(tf: str) -> int:
    unit = tf[-1]; n = int(tf[:-1])
    return n * {"m": 60, "h": 3600, "d": 86400, "w": 604800}[unit]


class ExchangeFeed:
    """ccxt-backed closed-bar feed. Only the alerter's fetch path touches the network."""

    def __init__(self, exchange_id: str = config.EXCHANGE):
        import ccxt
        self.ex = getattr(ccxt, exchange_id)({"enableRateLimit": True})

    def fetch_closed_bars(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        tf_ms = timeframe_seconds(timeframe) * 1000
        rows = self.ex.fetch_ohlcv(symbol, timeframe, limit=limit + 1)
        now_ms = self.ex.milliseconds()
        rows = [r for r in rows if r[0] + tf_ms <= now_ms]   # drop the in-progress candle
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts").sort_values("ts")
        return df.tail(limit).reset_index(drop=True)


# --------------------------------------------------------------------------- alerts
def direction_words(cur: float, tgt: float) -> str:
    side = lambda x: "LONG" if x > 0 else ("SHORT" if x < 0 else "FLAT")  # noqa: E731
    if tgt == 0:
        return f"CLOSE {side(cur)} → FLAT"
    if cur == 0:
        return f"OPEN {side(tgt)}"
    if (cur > 0) != (tgt > 0):
        return f"FLIP {side(cur)} → {side(tgt)}"
    return f"{'INCREASE' if abs(tgt) > abs(cur) else 'REDUCE'} {side(tgt)}"


def make_alert_id(symbol: str, bar_ts: int, kind: str) -> str:
    return f"{kind[:3]}-{datetime.fromtimestamp(bar_ts / 1000, tz=timezone.utc).strftime('%Y%m%d%H%M')}-{symbol.replace('/', '')}"


def emit(rec: dict, title: str, body: str, channels: list[str] | None = None) -> dict:
    rec["delivery"] = push(title, body, rec["kind"], channels)
    rec["delivered"] = any(v.startswith("ok") for v in rec["delivery"].values())
    append_jsonl(config.ALERTS_LOG, rec)
    log.warning("%s %s | %s", rec["kind"].upper(), title, body.replace("\n", " · ")[:300])
    return rec


def outcomes_by_alert() -> dict[str, dict]:
    out = {}
    for o in read_jsonl(config.OUTCOMES_LOG):
        out[o["alert_id"]] = o
    return out


def pending_alerts() -> list[dict]:
    acked = outcomes_by_alert()
    return [a for a in read_jsonl(config.ALERTS_LOG) if a.get("kind") == "signal" and a["id"] not in acked]


# --------------------------------------------------------------------------- core tick
class Alerter:
    def __init__(self, mode: str = "paper", feed=None, notify_channels: list[str] | None = None,
                 strategy_fn=None, promoted: dict | None = None, now_fn=None):
        if mode not in config.MODES:
            raise ValueError(f"mode must be one of {list(config.MODES)}")
        self.mode = mode
        self.creds = config.MODES[mode]
        self.promoted = promoted or load_promoted()
        self.symbol = self.promoted["symbol"]; self.timeframe = self.promoted["timeframe"]
        self.tf_sec = timeframe_seconds(self.timeframe)
        self.feed = feed or ExchangeFeed()
        self.channels = notify_channels
        self.now_fn = now_fn or time.time
        if strategy_fn is None:
            strategy_fn, _ = load_strategy(config.ROOT / self.promoted["source"])
        self.strategy = strategy_fn
        self.state = load_state()
        self.dd_limit = float(self.promoted["backtest"]["max_drawdown"]) * config.DD_HALT_MULT

    # ---- halts ---------------------------------------------------------------
    def halt(self, reason: str, detail: dict | None = None) -> dict:
        self.state["halted"] = True; self.state["halt_reason"] = reason
        save_state(self.state)
        rec = {"id": f"halt-{utcnow().strftime('%Y%m%d%H%M%S')}", "ts": iso(), "mode": self.mode, "kind": "halt",
               "symbol": self.symbol, "timeframe": self.timeframe, "reason": reason, "detail": detail or {},
               "strategy_sha": self.promoted["sha"]}
        return emit(rec, f"🚨 ALERTER HALTED ({self.mode}) — {self.symbol}",
                    f"{reason}\nNo further signal alerts until you run: python -m live.alerter resume --reason ...\n"
                    f"strategy {self.promoted['sha']}", self.channels)

    def check_stale(self, last_closed_bar_ts_ms: int | None) -> str | None:
        if last_closed_bar_ts_ms is None:
            return None
        closed_at = last_closed_bar_ts_ms / 1000 + self.tf_sec
        age = self.now_fn() - closed_at
        limit = self.tf_sec * (1 + config.STALE_GRACE_MULT)
        if age > limit:
            return f"stale data: newest closed {self.timeframe} bar is {age / 60:.0f} min old (limit {limit / 60:.0f} min)"
        return None

    # ---- one tick ------------------------------------------------------------
    def tick(self) -> dict | None:
        """Fetch bars, run strategy, alert on a position change. Returns the alert record or None."""
        st = self.state
        self.remind_pending()
        if st.get("halted"):
            log.warning("halted (%s); not evaluating signals. `resume` to clear.", st.get("halt_reason"))
            return None
        try:
            bars = self.feed.fetch_closed_bars(self.symbol, self.timeframe, config.LOOKBACK_BARS)
            st["error_streak"] = 0
        except Exception as e:  # noqa: BLE001
            st["error_streak"] = st.get("error_streak", 0) + 1
            save_state(st)
            log.error("fetch error %d/%d: %s", st["error_streak"], config.ERROR_STREAK_HALT, e)
            if st["error_streak"] >= config.ERROR_STREAK_HALT:
                return self.halt(f"API error streak: {st['error_streak']} consecutive fetch failures ({type(e).__name__}: {str(e)[:120]})")
            return None
        if bars.empty:
            return self.halt("exchange returned no closed bars")
        last_ts = int(bars["ts"].iloc[-1])
        stale = self.check_stale(last_ts)
        if stale:
            return self.halt(stale, {"last_bar_ts": last_ts})
        if st.get("last_bar_ts") == last_ts:
            log.info("no new bar since %s; nothing to do", datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc))
            save_state(st)
            return None

        # mark-to-market the previous bar with the positions held through it (open->open like the engine)
        self._mark_to_market(bars, last_ts)

        try:
            target, indicators = run_with_indicators(self.strategy, bars)
        except Exception as e:  # noqa: BLE001
            return self.halt(f"strategy raised {type(e).__name__}: {str(e)[:160]}", {"trace": traceback.format_exc()[-800:]})

        current = float(st.get("designed_position", 0.0))
        price = float(bars["close"].iloc[-1])
        append_jsonl(bars_log_path(self.symbol, self.timeframe), {
            "ts": last_ts, "open": float(bars["open"].iloc[-1]), "high": float(bars["high"].iloc[-1]),
            "low": float(bars["low"].iloc[-1]), "close": price, "volume": float(bars["volume"].iloc[-1]),
            "designed_position": target, "mode": self.mode})
        st["last_bar_ts"] = last_ts

        # drawdown halt (designed path; actual path when acks are current)
        dd_designed = 1 - st["designed_equity"] / st["designed_peak"]
        dd_actual = 1 - st["actual_equity"] / st["actual_peak"]
        if max(dd_designed, dd_actual) > self.dd_limit:
            st["designed_position"] = target
            save_state(st)
            return self.halt(f"live drawdown {max(dd_designed, dd_actual):.1%} exceeds backtested max drawdown "
                             f"{self.promoted['backtest']['max_drawdown']:.1%} × {config.DD_HALT_MULT}",
                             {"dd_designed": dd_designed, "dd_actual": dd_actual})

        alert = None
        if abs(target - current) >= config.MIN_POSITION_DELTA:
            equity = config.ACCOUNT_EQUITY_USD
            notional = equity * config.RISK_FRACTION * abs(target)
            delta_notional = equity * config.RISK_FRACTION * abs(target - current)
            rec = {
                "id": make_alert_id(self.symbol, last_ts, "signal"), "ts": iso(), "mode": self.mode, "kind": "signal",
                "symbol": self.symbol, "timeframe": self.timeframe,
                "bar_ts": last_ts, "bar_time": datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc).isoformat(timespec="minutes"),
                "direction": direction_words(current, target),
                "current_position": current, "target_position": target, "delta": target - current,
                "price": price, "indicators": indicators,
                "size": {"risk_fraction": config.RISK_FRACTION, "account_equity_usd": equity,
                         "target_notional_usd": round(notional, 2), "trade_notional_usd": round(delta_notional, 2),
                         "trade_qty": round(delta_notional / price, 6) if price else None},
                "strategy_sha": self.promoted["sha"], "strategy_desc": self.promoted.get("description", ""),
                "holdout_checked": bool(self.promoted.get("holdout_checked")),
                "execution_note": "execute at the NEXT bar open — that is what the backtest assumed",
                "drawdown": {"designed": round(dd_designed, 4), "actual": round(dd_actual, 4), "halt_at": round(self.dd_limit, 4)},
            }
            ind = ", ".join(f"{k}={v}" for k, v in list(indicators.items())[:8])
            body = (f"{rec['direction']}  {current:+.2f} → {target:+.2f}\n"
                    f"bar {rec['bar_time']} close {price:,.2f}\n"
                    f"trade ≈ ${delta_notional:,.0f} ({rec['size']['trade_qty']} units) · target ${notional:,.0f} "
                    f"({config.RISK_FRACTION:.0%} risk × {equity:,.0f})\n"
                    f"{ind}\nstrategy {self.promoted['sha']} · {self.mode}\n"
                    f"ACK REQUIRED: alerter ack {rec['id']} taken|skipped --reason ...")
            hold = "" if rec["holdout_checked"] else " [holdout NOT checked]"
            alert = emit(rec, f"📈 {self.symbol} {rec['direction']} ({self.mode}){hold}", body, self.channels)
            st["alerts_emitted"] = st.get("alerts_emitted", 0) + 1
        st["designed_position"] = target
        save_state(st)
        return alert

    def _mark_to_market(self, bars: pd.DataFrame, last_ts: int) -> None:
        """Apply the open->open return of the just-closed bar to designed and actual equity."""
        st = self.state
        prev_ts = st.get("last_bar_ts")
        if prev_ts is None:
            return
        idx = bars.index[bars["ts"] == prev_ts]
        if len(idx) == 0 or idx[0] + 1 >= len(bars):
            return
        i = idx[0]
        # position decided at prev bar close is held from open[i+1] to open[i+2]; if the newest bar is i+1 we
        # approximate its holding return with open[i+1] -> close[i+1] until the next open is known.
        o1 = float(bars["open"].iloc[i + 1]); nxt = float(bars["open"].iloc[i + 2]) if i + 2 < len(bars) else float(bars["close"].iloc[i + 1])
        r = nxt / o1 - 1.0
        cost = (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4
        for key in ("designed", "actual"):
            pos = float(st.get(f"{key}_position", 0.0))
            prev_pos = float(st.get(f"{key}_prev_position", 0.0))
            net = pos * r - abs(pos - prev_pos) * cost
            st[f"{key}_equity"] = st.get(f"{key}_equity", 1.0) * (1 + net)
            st[f"{key}_peak"] = max(st.get(f"{key}_peak", 1.0), st[f"{key}_equity"])
            st[f"{key}_prev_position"] = pos
        self._sync_actual_position()

    def _sync_actual_position(self) -> None:
        """Actual position = designed path with skipped/unlogged alerts leaving the position unchanged."""
        acked = outcomes_by_alert()
        pos = 0.0
        for a in read_jsonl(config.ALERTS_LOG):
            if a.get("kind") != "signal":
                continue
            o = acked.get(a["id"])
            if o and o["outcome"] == "taken":
                pos = float(a["target_position"])
        self.state["actual_position"] = pos

    def remind_pending(self) -> None:
        pend = [a for a in pending_alerts() if self.now_fn() - a["bar_ts"] / 1000 > self.tf_sec * (1 + config.REMIND_AFTER_BARS)]
        if not pend:
            return
        ids = ", ".join(a["id"] for a in pend[-5:])
        rec = {"id": f"rem-{utcnow().strftime('%Y%m%d%H%M%S')}", "ts": iso(), "mode": self.mode, "kind": "reminder",
               "pending": [a["id"] for a in pend], "strategy_sha": self.promoted["sha"]}
        emit(rec, f"⏳ {len(pend)} alert(s) need an outcome", f"Log taken/skipped + reason:\n{ids}\n"
             f"python -m live.alerter ack <id> taken|skipped --reason \"...\"", self.channels)

    # ---- loop ----------------------------------------------------------------
    def run_loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as e:  # noqa: BLE001 — never die silently
                log.error("tick crashed: %s", e)
                self.halt(f"alerter crashed: {type(e).__name__}: {str(e)[:160]}")
            now = self.now_fn()
            wake = (int(now // self.tf_sec) + 1) * self.tf_sec + 20   # 20 s after the next bar close
            time.sleep(max(5.0, wake - now))


# --------------------------------------------------------------------------- cli
def cmd_ack(a) -> None:
    alerts = {x["id"]: x for x in read_jsonl(config.ALERTS_LOG) if x.get("kind") == "signal"}
    if a.alert_id not in alerts:
        sys.exit(f"unknown alert id {a.alert_id}; pending: {[p['id'] for p in pending_alerts()]}")
    if not a.reason.strip():
        sys.exit("a reason is required")
    if a.alert_id in outcomes_by_alert():
        sys.exit(f"{a.alert_id} already has an outcome")
    rec = {"alert_id": a.alert_id, "outcome": a.outcome, "reason": a.reason.strip(), "price": a.price, "ts": iso(),
           "target_position": alerts[a.alert_id]["target_position"], "mode": alerts[a.alert_id]["mode"]}
    append_jsonl(config.OUTCOMES_LOG, rec)
    st = load_state()
    if a.outcome == "taken":
        st["actual_position"] = float(alerts[a.alert_id]["target_position"])
        save_state(st)
    print(f"logged {a.outcome} for {a.alert_id}: {a.reason}")
    left = pending_alerts()
    if left:
        print(f"still pending: {[p['id'] for p in left]}")


def cmd_status(a) -> None:
    st = load_state()
    prom = load_promoted() if config.PROMOTED_FILE.exists() else {}
    print(json.dumps({"promoted": {k: prom.get(k) for k in ("sha", "symbol", "timeframe", "description", "holdout_checked")},
                      "state": st, "pending_alerts": [p["id"] for p in pending_alerts()],
                      "alerts_total": len(read_jsonl(config.ALERTS_LOG)),
                      "ntfy_topic": config.NTFY_TOPIC or "(unset)"}, indent=2, default=str))


def cmd_resume(a) -> None:
    st = load_state()
    if not st.get("halted"):
        print("not halted"); return
    reason = st.get("halt_reason")
    st["halted"] = False; st["halt_reason"] = None; st["error_streak"] = 0
    if a.reset_drawdown:
        st["designed_peak"] = st["designed_equity"]; st["actual_peak"] = st["actual_equity"]
    save_state(st)
    rec = {"id": f"res-{utcnow().strftime('%Y%m%d%H%M%S')}", "ts": iso(), "kind": "resume", "reason": a.reason,
           "cleared_halt": reason}
    emit(rec, "✅ alerter resumed", f"cleared: {reason}\nby: {a.reason}")


def cmd_run(a) -> None:
    al = Alerter(mode=a.mode)
    if not al.promoted.get("holdout_checked"):
        log.warning("promoted strategy %s has holdout_checked=false — run scripts/run_holdout.py and flip it in live/promoted.json", al.promoted["sha"])
    if a.loop:
        al.run_loop()
    else:
        out = al.tick()
        print(json.dumps(out, indent=2, default=str) if out else "no position change")


def cmd_test_push(a) -> None:
    print(json.dumps(push("crypto-autoresearch test", "If you can read this overnight, delivery works.", "report"), indent=2))


def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s alerter: %(message)s", datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--mode", choices=list(config.MODES), default="paper")
    g = r.add_mutually_exclusive_group(); g.add_argument("--once", action="store_true"); g.add_argument("--loop", action="store_true")
    r.set_defaults(fn=cmd_run)
    k = sub.add_parser("ack"); k.add_argument("alert_id"); k.add_argument("outcome", choices=["taken", "skipped"])
    k.add_argument("--reason", required=True); k.add_argument("--price", type=float, default=None); k.set_defaults(fn=cmd_ack)
    s = sub.add_parser("status"); s.set_defaults(fn=cmd_status)
    z = sub.add_parser("resume"); z.add_argument("--reason", required=True); z.add_argument("--reset-drawdown", action="store_true"); z.set_defaults(fn=cmd_resume)
    t = sub.add_parser("test-push"); t.set_defaults(fn=cmd_test_push)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
