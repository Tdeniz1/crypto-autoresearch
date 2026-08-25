"""THE ONLY FILE THAT READS THE HOLDOUT. Run by hand, never by the loop.

    CRYPTO_HOLDOUT_DIR=~/crypto-holdout uv run python scripts/run_holdout.py [--sha <commit>]

Evaluates strategy.py (or the strategy.py at a given commit) on the holdout
slice and writes reports/holdout-<date>.md (gitignored). Refuses to run inside
the runner's process (CAR_RUNNER=1).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from harness import config  # noqa: E402
from harness.data import Series, load_manifest, load_series  # noqa: E402
from harness.evaluate import evaluate_fn, load_strategy  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", default=None, help="git commit whose strategy.py to evaluate (default: working tree)")
    a = ap.parse_args()
    if os.environ.get("CAR_RUNNER"):
        sys.exit("refusing: holdout evaluation may not run inside the research loop")
    hd = os.environ.get(config.HOLDOUT_ENV_VAR)
    if not hd:
        sys.exit(f"set {config.HOLDOUT_ENV_VAR} (the runner must NOT have it)")
    hd_path = Path(hd).expanduser()
    private = json.loads((hd_path / "manifest_private.json").read_text())
    label = {s["id"]: f"{s['symbol']} {s['timeframe']} holdout {s['holdout'][0][:10]}..{s['holdout'][1][:10]}" for s in private["series"]}

    if a.sha:
        src = subprocess.run(["git", "show", f"{a.sha}:strategy.py"], cwd=config.ROOT, capture_output=True, text=True, check=True).stdout
    else:
        src = config.STRATEGY_FILE.read_text()
    fn, desc = load_strategy(src, scan=True)

    manifest = load_manifest()
    ids = config.eval_series_ids(manifest)
    val_series = load_series(ids)
    hold = []
    for s in val_series:
        df = pd.read_parquet(hd_path / f"{s.id}_holdout.parquet").astype("float64").reset_index(drop=True)
        hold.append(Series(id=s.id, bars_per_year=s.bars_per_year, train=s.train, val=df))

    ev_val = evaluate_fn(fn, val_series, "val", run_checks=True)
    ev_hold = evaluate_fn(fn, hold, "val", run_checks=True)   # .val slot holds the holdout frame

    lines = [f"# HOLDOUT evaluation — {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"strategy: {desc} (commit {a.sha or 'working tree'})", "",
             f"validation metric: {ev_val.metric:+.4f}", f"**holdout metric:   {ev_hold.metric:+.4f}**", "",
             "| series | split | sharpe | mdd | trades | win | payoff | tim | return | B&H |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for ev, split in ((ev_val, "validation"), (ev_hold, "HOLDOUT")):
        for sid, s in ev.per_series.items():
            lines.append(f"| {label.get(sid, sid)} | {split} | {s['sharpe']:+.2f} | {s['max_drawdown']:.0%} | {s['trades']} | "
                         f"{s['win_rate']:.0%} | {s['payoff_ratio']:.2f} | {s['time_in_market']:.0%} | {s['total_return']:+.1%} | {s['buy_hold_return']:+.1%} |")
    out = "\n".join(lines)
    print(out)
    config.REPORT_DIR.mkdir(exist_ok=True)
    p = config.REPORT_DIR / f"holdout-{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M')}.md"
    p.write_text(out + "\n")
    print(f"\nwritten {p} (gitignored)")


if __name__ == "__main__":
    main()
