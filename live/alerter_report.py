"""Designed-vs-taken report: what the promoted strategy did vs what you actually did.

    uv run python -m live.alerter_report [--mode paper] [--notify]

Reads logs/alerts.jsonl, logs/alert_outcomes.jsonl, logs/live_bars/*.jsonl. Writes
reports/alerter-<date>.md. Never places orders, never calls a model.
"""
from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from live import config
from live.alerter import bars_log_path, outcomes_by_alert, read_jsonl
from live.notify import push
from live.strategy_loader import load_promoted


def _sharpe(net: np.ndarray, bars_per_year: float) -> float:
    if len(net) < 2 or net.std(ddof=1) == 0:
        return 0.0
    return float(net.mean() / net.std(ddof=1) * math.sqrt(bars_per_year))


def _mdd(eq: np.ndarray) -> float:
    return float((1 - eq / np.maximum.accumulate(eq)).max()) if len(eq) else 0.0


def path_stats(bars: pd.DataFrame, pos: np.ndarray, bars_per_year: float) -> dict:
    """Engine-consistent: signal at bar t is held from open[t+1] to open[t+2]; costs per side in bps."""
    o = bars["open"].to_numpy(dtype="float64")
    n = len(o)
    held = np.zeros(n); held[1:] = pos[:-1]
    r = np.zeros(n); r[:-1] = o[1:] / o[:-1] - 1.0
    cost = np.abs(np.diff(held, prepend=0.0)) * (config.FEE_BPS + config.SLIPPAGE_BPS) / 1e4
    net = held * r - cost
    eq = np.cumprod(1 + net)
    return {"return": float(eq[-1] - 1) if n else 0.0, "sharpe": _sharpe(net, bars_per_year), "max_drawdown": _mdd(eq),
            "bars": int(n), "time_in_market": float((held != 0).mean()) if n else 0.0, "equity": eq, "net": net}


def build(mode: str | None) -> tuple[str, dict]:
    prom = load_promoted()
    symbol, tf = prom["symbol"], prom["timeframe"]
    alerts = [a for a in read_jsonl(config.ALERTS_LOG) if (mode is None or a.get("mode") == mode)]
    signals = [a for a in alerts if a.get("kind") == "signal"]
    halts = [a for a in alerts if a.get("kind") == "halt"]
    outcomes = outcomes_by_alert()
    bars = pd.DataFrame(read_jsonl(bars_log_path(symbol, tf)))
    if mode is not None and not bars.empty and "mode" in bars:
        bars = bars[bars["mode"] == mode]
    bars = bars.drop_duplicates("ts").sort_values("ts").reset_index(drop=True) if not bars.empty else bars

    taken = [a for a in signals if outcomes.get(a["id"], {}).get("outcome") == "taken"]
    skipped = [a for a in signals if outcomes.get(a["id"], {}).get("outcome") == "skipped"]
    unlogged = [a for a in signals if a["id"] not in outcomes]

    lines = [f"# Alerter report — {prom['symbol']} {prom['timeframe']} · strategy {prom['sha']} · mode {mode or 'all'}",
             f"generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}", ""]
    if unlogged:
        lines += [f"## ⚠️  {len(unlogged)} ALERT(S) WITHOUT A LOGGED OUTCOME — treated as SKIPPED below", ""]
        lines += [f"- `{a['id']}` {a['bar_time']} {a['direction']} — `python -m live.alerter ack {a['id']} taken|skipped --reason ...`" for a in unlogged]
        lines.append("")

    n_sig = len(signals)
    skip_rate = (len(skipped) + len(unlogged)) / n_sig if n_sig else 0.0
    lines += ["## Discipline", "",
              f"| alerts | taken | skipped | unlogged | skip rate |", "|---:|---:|---:|---:|---:|",
              f"| {n_sig} | {len(taken)} | {len(skipped)} | {len(unlogged)} | **{skip_rate:.0%}** |", ""]

    summary = {"alerts": n_sig, "taken": len(taken), "skipped": len(skipped), "unlogged": len(unlogged), "skip_rate": skip_rate}
    if not bars.empty and "designed_position" in bars:
        bpy = 365 * 86400 / (int(prom.get("tf_seconds") or {"m": 60, "h": 3600, "d": 86400}[tf[-1]] * int(tf[:-1])))
        designed = bars["designed_position"].to_numpy(dtype="float64")
        # taken path: position changes only at alerts whose outcome is taken
        actual = np.zeros(len(bars)); pos = 0.0
        taken_ts = {a["bar_ts"]: float(a["target_position"]) for a in taken}
        for i, ts in enumerate(bars["ts"].to_numpy()):
            if ts in taken_ts:
                pos = taken_ts[ts]
            actual[i] = pos
        d = path_stats(bars, designed, bpy); t = path_stats(bars, actual, bpy)
        gap = d["return"] - t["return"]
        summary.update({"designed_return": d["return"], "taken_return": t["return"], "gap": gap,
                        "designed_sharpe": d["sharpe"], "taken_sharpe": t["sharpe"], "bars": len(bars)})
        lines += ["## Strategy-as-designed vs actually-taken", "",
                  f"Forward window: {len(bars)} closed bars "
                  f"({datetime.fromtimestamp(int(bars['ts'].iloc[0]) / 1000, tz=timezone.utc):%Y-%m-%d %H:%M} → "
                  f"{datetime.fromtimestamp(int(bars['ts'].iloc[-1]) / 1000, tz=timezone.utc):%Y-%m-%d %H:%M} UTC), "
                  f"costs {config.FEE_BPS:.0f}+{config.SLIPPAGE_BPS:.0f} bps/side, execution at next open.", "",
                  "| path | return | sharpe | max dd | time in market |", "|---|---:|---:|---:|---:|",
                  f"| as designed | {d['return']:+.2%} | {d['sharpe']:+.2f} | {d['max_drawdown']:.1%} | {d['time_in_market']:.0%} |",
                  f"| actually taken | {t['return']:+.2%} | {t['sharpe']:+.2f} | {t['max_drawdown']:.1%} | {t['time_in_market']:.0%} |",
                  f"| **gap (designed − taken)** | **{gap:+.2%}** | {d['sharpe'] - t['sharpe']:+.2f} | | |", ""]
        if gap > 0:
            lines.append(f"**You left {gap:+.2%} on the table by not following the strategy.** Either follow it or stop running it.")
        elif gap < 0:
            lines.append(f"Your deviations beat the strategy by {-gap:+.2%} this window. That is discretion, not the strategy — the backtest says nothing about it.")
        else:
            lines.append("No gap: every alert was taken (or no alerts fired).")
        lines.append("")

    if skipped or unlogged:
        lines += ["## Skipped / unlogged alerts", "", "| id | bar | direction | outcome | reason |", "|---|---|---|---|---|"]
        for a in signals:
            o = outcomes.get(a["id"])
            if o and o["outcome"] == "taken":
                continue
            lines.append(f"| `{a['id']}` | {a['bar_time']} | {a['direction']} | {o['outcome'] if o else 'UNLOGGED'} | {(o or {}).get('reason', '')} |")
        lines.append("")
    if halts:
        lines += ["## Halts", ""] + [f"- {h['ts']}: **{h['reason']}**" for h in halts] + [""]
    lines += ["## All alerts", "", "| id | bar | direction | cur→tgt | price | outcome | sha |", "|---|---|---|---|---:|---|---|"]
    for a in signals:
        o = outcomes.get(a["id"], {})
        lines.append(f"| `{a['id']}` | {a['bar_time']} | {a['direction']} | {a['current_position']:+.2f}→{a['target_position']:+.2f} | "
                     f"{a['price']:,.2f} | {o.get('outcome', 'UNLOGGED')} | `{a['strategy_sha']}` |")
    if not signals:
        lines.append("| — | no signal alerts yet | | | | | |")
    lines += ["", f"> Promoted strategy holdout_checked = {prom.get('holdout_checked')}. Paper mode is the full forward test; "
                  "nothing here touches real money."]
    return "\n".join(lines) + "\n", summary


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=list(config.MODES) + ["all"], default="paper")
    ap.add_argument("--notify", action="store_true", help="push the one-line summary")
    a = ap.parse_args(argv)
    md, s = build(None if a.mode == "all" else a.mode)
    config.REPORT_DIR.mkdir(exist_ok=True)
    p = config.REPORT_DIR / f"alerter-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.md"
    p.write_text(md)
    print(md)
    print(f"written {p}")
    if a.notify:
        gap = s.get("gap")
        body = (f"{s['alerts']} alerts · taken {s['taken']} · skipped {s['skipped']} · unlogged {s['unlogged']} · "
                f"skip rate {s['skip_rate']:.0%}" + (f" · gap {gap:+.2%}" if gap is not None else ""))
        print(push("Alerter discipline report", body, "report"))


if __name__ == "__main__":
    main()
