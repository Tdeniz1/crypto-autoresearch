"""One-time data fetch. The ONLY file that touches the network for market data.

Pulls OHLCV via ccxt from the first reachable public exchange, splits each
series chronologically into train / validation / holdout, and writes:

  data/frozen/<id>_train.parquet, <id>_val.parquet   (time-blind: no timestamps)
  data/frozen/manifest.json                          (opaque ids + bars_per_year only)
  data/frozen/CHECKSUMS.sha256

  $CRYPTO_HOLDOUT_DIR/<id>_holdout.parquet           (OUTSIDE the project tree)
  $CRYPTO_HOLDOUT_DIR/manifest_private.json          (symbols, timeframes, date ranges)

Run:  CRYPTO_HOLDOUT_DIR=~/crypto-holdout uv run python prepare.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import ccxt
import pandas as pd

from harness import config

COLUMNS = ["open", "high", "low", "close", "volume"]


def fetch_all(exchange, symbol: str, timeframe: str, since_ms: int | None, limit: int) -> pd.DataFrame:
    rows: list[list] = []
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    cursor = since_ms
    last_ts = None
    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=limit)
        if not batch:
            break
        if last_ts is not None:
            batch = [b for b in batch if b[0] > last_ts]
            if not batch:
                break
        rows.extend(batch)
        last_ts = batch[-1][0]
        cursor = last_ts + tf_ms
        print(f"  {symbol} {timeframe}: {len(rows):>7} bars  (last {pd.Timestamp(last_ts, unit='ms')})", end="\r", flush=True)
        if len(batch) < limit * 0.5 or last_ts + tf_ms > exchange.milliseconds():
            break
        time.sleep(exchange.rateLimit / 1000)
    print()
    df = pd.DataFrame(rows, columns=["ts"] + COLUMNS).drop_duplicates("ts").sort_values("ts")
    df = df[(df[COLUMNS] > 0).all(axis=1)].reset_index(drop=True)
    return df


def pick_exchange() -> ccxt.Exchange:
    for name in config.EXCHANGES:
        try:
            ex = getattr(ccxt, name)({"enableRateLimit": True})
            ex.load_markets()
            missing = [s for s in config.SYMBOLS if s not in ex.markets]
            if missing:
                print(f"{name}: missing {missing}, skipping")
                continue
            print(f"using exchange: {name}")
            return ex
        except Exception as e:  # noqa: BLE001
            print(f"{name}: unreachable ({type(e).__name__}: {str(e)[:80]}), skipping")
    sys.exit("no usable exchange")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2015-01-01", help="earliest date to request; endpoint decides what it has")
    ap.add_argument("--limit", type=int, default=1000)
    args = ap.parse_args()

    holdout_dir = os.environ.get(config.HOLDOUT_ENV_VAR)
    if not holdout_dir:
        sys.exit(f"set {config.HOLDOUT_ENV_VAR} to a directory OUTSIDE the repo (e.g. ~/crypto-holdout)")
    holdout_path = Path(holdout_dir).expanduser().resolve()
    if config.ROOT in holdout_path.parents or holdout_path == config.ROOT:
        sys.exit(f"{config.HOLDOUT_ENV_VAR} must point outside the project tree")
    holdout_path.mkdir(parents=True, exist_ok=True)
    config.FROZEN_DIR.mkdir(parents=True, exist_ok=True)

    ex = pick_exchange()
    since_ms = ex.parse8601(args.since + "T00:00:00Z")
    limit = min(args.limit, getattr(ex, "options", {}).get("fetchOHLCVLimit", args.limit) or args.limit)

    manifest = {"series": []}
    private = {"exchange": ex.id, "split": config.SPLIT, "series": []}
    checksums = {}
    sid = 0
    for symbol in config.SYMBOLS:
        for tf in config.TIMEFRAMES:
            df = fetch_all(ex, symbol, tf, since_ms, limit)
            n = len(df)
            if n < 5000:
                sys.exit(f"{symbol} {tf}: only {n} bars; refusing to build a split that thin")
            n_train = int(n * config.SPLIT[0])
            n_val = int(n * config.SPLIT[1])
            parts = {
                "train": df.iloc[:n_train],
                "val": df.iloc[n_train:n_train + n_val],
                "holdout": df.iloc[n_train + n_val:],
            }
            series_id = f"s{sid}"
            sid += 1
            for part, sub in parts.items():
                blind = sub[COLUMNS].reset_index(drop=True).astype("float64")
                if part == "holdout":
                    out = holdout_path / f"{series_id}_holdout.parquet"
                else:
                    out = config.FROZEN_DIR / f"{series_id}_{part}.parquet"
                blind.to_parquet(out, index=False)
                if part != "holdout":
                    checksums[out.name] = sha256_file(out)
            manifest["series"].append({
                "id": series_id,
                "bars_per_year": config.BARS_PER_YEAR[tf],
                "n_train": n_train,
                "n_val": n_val,
            })
            private["series"].append({
                "id": series_id, "symbol": symbol, "timeframe": tf, "n_total": n,
                "train": [str(pd.Timestamp(parts["train"].ts.iloc[0], unit="ms")), str(pd.Timestamp(parts["train"].ts.iloc[-1], unit="ms"))],
                "val": [str(pd.Timestamp(parts["val"].ts.iloc[0], unit="ms")), str(pd.Timestamp(parts["val"].ts.iloc[-1], unit="ms"))],
                "holdout": [str(pd.Timestamp(parts["holdout"].ts.iloc[0], unit="ms")), str(pd.Timestamp(parts["holdout"].ts.iloc[-1], unit="ms"))],
                "n_holdout": len(parts["holdout"]),
            })
            print(f"  {series_id}: {symbol} {tf}  train={n_train} val={n_val} holdout={len(parts['holdout'])}")

    (config.FROZEN_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    checksums["manifest.json"] = sha256_file(config.FROZEN_DIR / "manifest.json")
    (config.FROZEN_DIR / "CHECKSUMS.sha256").write_text(
        "".join(f"{v}  {k}\n" for k, v in sorted(checksums.items()))
    )
    (holdout_path / "manifest_private.json").write_text(json.dumps(private, indent=2))
    print(f"\nfrozen  -> {config.FROZEN_DIR}\nholdout -> {holdout_path}  (never set {config.HOLDOUT_ENV_VAR} for the runner)")


if __name__ == "__main__":
    main()
