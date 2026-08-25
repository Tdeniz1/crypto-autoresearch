"""Promote a research commit to the live alerter.

    uv run python -m live.promote <sha> --symbol BTC/USDT --timeframe 1h

Snapshots strategy.py from that commit into live/promoted/<sha>.py, records its
validation stats (max drawdown is the live DD-halt reference), and writes
live/promoted.json. This is the ONLY live/ module that imports the harness; it
runs in the research checkout, by hand, after you have looked at the holdout.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone

from live import config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sha")
    ap.add_argument("--symbol", default="BTC/USDT")
    ap.add_argument("--timeframe", default="1h")
    ap.add_argument("--mode", choices=list(config.MODES), default="paper")
    a = ap.parse_args()

    sha = subprocess.run(["git", "rev-parse", "--short", a.sha], cwd=config.ROOT, capture_output=True, text=True, check=True).stdout.strip()
    src = subprocess.run(["git", "show", f"{sha}:strategy.py"], cwd=config.ROOT, capture_output=True, text=True, check=True).stdout
    subject = subprocess.run(["git", "log", "-1", "--format=%s", sha], cwd=config.ROOT, capture_output=True, text=True).stdout.strip()

    from harness.data import load_series
    from harness.evaluate import evaluate_source
    from harness.scan import scan_source
    scan_source(src)
    ev = evaluate_source(src, series=load_series(), split="val", run_checks=True)
    agg = ev.aggregate()

    config.PROMOTED_DIR.mkdir(parents=True, exist_ok=True)
    snap = config.PROMOTED_DIR / f"{sha}.py"
    snap.write_text(src)
    rec = {
        "sha": sha, "subject": subject, "description": ev.description, "source": str(snap.relative_to(config.ROOT)),
        "symbol": a.symbol, "timeframe": a.timeframe, "mode": a.mode,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backtest": {"metric": ev.metric, "sharpe": agg["sharpe"], "max_drawdown": agg["max_drawdown"],
                     "trades": agg["trades"], "per_series": {k: {"sharpe": v["sharpe"], "max_drawdown": v["max_drawdown"]}
                                                             for k, v in ev.per_series.items()}},
        "holdout_checked": False,   # flip by hand after scripts/run_holdout.py; the alerter warns while False
    }
    config.PROMOTED_FILE.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print(f"\npromoted {sha} → {snap}. Live DD halt reference = {agg['max_drawdown']:.1%} × DD_HALT_MULT.")


if __name__ == "__main__":
    main()
