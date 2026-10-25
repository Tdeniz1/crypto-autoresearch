# crypto-autoresearch

A ratchet-pattern research harness for crypto bar strategies, re-implemented from
scratch for backtesting. A proposer (free OpenRouter models → Claude Code headless →
Anthropic API, with fallback) rewrites one file, `strategy.py`; a frozen harness
scores it on a validation slice; better metrics are committed, worse ones reverted.
Leakage is treated as the main adversary: physical holdout isolation, a time-blind
proposer, a static scan, causal-statistics guards, t+1-open execution, three
canaries on every winner, and deflated-Sharpe accounting for the number of trials.

## Results so far (honest)

Two sessions, 40 experiments total, run on 1h and 15m BTC/ETH bars with 10 bps fees and 10 bps slippage per side.

| session | experiments | kept | validation metric | Sharpe (net) | deflated Sharpe | vs buy and hold |
|---|---:|---:|---:|---:|---:|---|
| seed strategy | n/a | n/a | -26.50 | n/a | n/a | n/a |
| 2026-08-25 (Claude Haiku) | 10 | 4 | -9.82 | -0.49 | -7.39 (16 trials) | -31.6% vs +71.2% |
| 2026-10-25 (Claude Sonnet) | 30 | 10 | **-4.07** | -0.20 | -5.21 (51 trials) | -20.7% vs +71.2% |

**The loop improves the metric, and the strategy still loses to buy and hold.** That is the
expected result and the reason the harness exists: after fees, a 2-coin bar strategy rarely beats
holding, and the deflated Sharpe (which charges for every trial run) stays far below zero. Every
kept candidate passed all three canaries. The second session also shows the failure mode the
deflated Sharpe guards against: five kept experiments in a row just lengthened one smoothing
window (100 to 150 to 250 to 400 to 800 bars), which is tuning to the validation slice, not a new
idea. The holdout has not been opened.

`make canary-demo` plants three bugs (look-ahead leak, timing leak, broken cost model) and shows
each is caught by its canary, with no false positive on the correct cost model.

## Layout

```
prepare.py            one-time ccxt fetch → data/frozen/ (+ holdout OUTSIDE the tree)
harness/              FROZEN. config, data, engine, metric, scan, canaries, budget, determinism, evaluate
strategy.py           the ONLY file the loop edits
proposer/             propose(program, current_strategy, log_summary) + 3-rung ladder + offline mock
runner.py             the loop (pure Python, no model calls)
report.py             morning report → reports/YYYY-MM-DD.md
analyze_providers.py  keep rate / improvement by rung+model
dashboard/            read-only FastAPI + vanilla JS dashboard
scripts/run_holdout.py   the only reader of the holdout, run by hand
scripts/canary_demo.py   proves the canaries catch three planted bugs
live/                 signal alerter (promote → alert on position change → mandatory ack → designed-vs-taken report); see live/README.md
program.md            research directions + rules given to the proposer
logs/session.jsonl    one record per experiment      logs/ledger.jsonl  one record per proposer call
```

## Setup

```bash
uv sync --extra dev
cp .env.example .env        # fill in OPENROUTER_API_KEY / CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY
claude setup-token          # once; paste the token into .env as CLAUDE_CODE_OAUTH_TOKEN (rung 2 unattended)

# one-time data fetch. CRYPTO_HOLDOUT_DIR must be OUTSIDE the repo and must NOT be in .env
CRYPTO_HOLDOUT_DIR=~/crypto-holdout uv run python prepare.py
```

`prepare.py` tries `binanceus`, then `coinbase`, then `kraken` (binance.com is geo-blocked in the US),
fetches BTC/USDT and ETH/USDT at 1h and 15m as far back as the endpoint allows, and splits 70/15/15
chronologically. The frozen files carry no timestamps and opaque ids (`s0..s3`); the mapping to
symbols/timeframes/dates lives only in `$CRYPTO_HOLDOUT_DIR/manifest_private.json`.

## Run

```bash
make determinism      # bit-identical metric test (the runner also runs it at startup and refuses to start if it fails)
make test             # scan / engine / canary / ladder / holdout-guard unit tests
make smoke            # 10 iterations through the real ladder
make smoke-mock       # 10 iterations offline with the deterministic mock proposer
make overnight        # runs until 07:00 local (STOP_AT=06:30 make overnight to change)
make fallback-test    # rung 1 forced to 429 → retries → demotes to rung 2 → keeps going
make canary-demo      # three planted bugs, each caught by its canary
make dashboard        # http://localhost:8765 — read-only, works on last session's files
make report           # regenerate reports/YYYY-MM-DD.md for the latest session
make providers        # attribution table across sessions
CRYPTO_HOLDOUT_DIR=~/crypto-holdout make holdout   # manual holdout evaluation, writes reports/holdout-*.md (gitignored)
```

Direct invocations:

```bash
uv run python runner.py --max-iters 10 --no-llm-report
uv run python runner.py --stop-at 07:00
uv run python runner.py --rungs claude_code,anthropic --rung3-cap 0.50
uv run python report.py --session latest [--no-llm]
uv run python scripts/run_holdout.py --sha <commit>
```

## Execution and metric semantics

- `strategy(df) -> Series` returns the desired position in [-1, 1] decided after bar *t* closes.
- The engine enters it at the **open of bar t+1** and holds to the open of t+2 (open-to-open returns).
  Fees (`FEE_BPS=10`) and slippage (`SLIPPAGE_BPS=10`) are charged per side on every unit of position change.
- Metric = mean over series of annualised net Sharpe × drawdown penalty; `-inf` if any series has fewer than
  `MIN_TRADES=30` trades. One float, nothing else.
- Guards: truncation test (signal on bars `[0,k)` must not change when bars `>= k` are removed, 10 cut points),
  runtime patch that raises on any pandas/numpy reduction over the full input, and the static scan in `harness/scan.py`.
- Canaries on every candidate that beats the best: shuffled returns (`LEAK_SUSPECTED`), extra delay (`TIMING_FRAGILE`),
  sign flip + cost identity (`COST_MODEL_BROKEN`). `LEAK_SUSPECTED` candidates are reverted, never kept.
- Deflated Sharpe (Bailey & López de Prado) is computed against the cumulative trial count in the ledger and is the
  dashboard's sort key.

## Fallback ladder

`openrouter → claude_code → anthropic`. On 429/quota: two retries with exponential backoff, then demote and log
(WARN on the first demotion of each rung per session). Demoted rungs are retried after `RUNG_COOLDOWN_SEC`.
Rung 3 refuses to exceed `RUNG3_SESSION_USD_CAP` (default $1.00): the session stops cleanly and the report is written.
If every rung fails `MAX_CONSECUTIVE_PROPOSER_FAILURES` times in a row, the session also stops cleanly.

## Holdout discipline

`data/holdout/` does not exist in this repo. The holdout parquet files live wherever `CRYPTO_HOLDOUT_DIR` points,
outside the tree. The runner aborts at startup if that variable is present in its environment, rung 2's subprocess
gets a scrubbed environment, and `scripts/run_holdout.py` refuses to run inside the runner. Run it yourself, rarely.

## Signal alerting (no auto-execution)

`live/alerter.py` runs a promoted commit's `strategy.py` on live bars and pushes an alert (ntfy.sh + `logs/alerts.jsonl`)
when the position should change — timestamp, symbol, direction, current→target, the strategy's own indicator values
at that bar, trade size from `RISK_FRACTION`, git sha. Every alert needs `ack <id> taken|skipped --reason`; un-acked
alerts are re-pushed each tick and `live/alerter_report.py` reports the skip rate and the designed-vs-taken gap.
Stale bars, API error streaks and live drawdown beyond the backtested max DD halt the alerter with an urgent alert;
`resume` clears it. Run paper mode for the full forward period first — `--mode live` only swaps credentials/tags and
still never places orders. Details: `live/README.md`.
