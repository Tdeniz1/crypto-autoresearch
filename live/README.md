# live/ — signal alerting (no auto-execution)

`live/` is importable without the research harness (pandas, numpy, ccxt, httpx only) so it can
run on Railway. `live/promote.py` is the one exception: it imports `harness/` and runs in the
research checkout, by hand.

```
promote.py         git sha → live/promoted/<sha>.py snapshot + live/promoted.json (backtest max DD = halt reference)
alerter.py         run --once | --loop · ack · status · resume · test-push
alerter_report.py  strategy-as-designed vs actually-taken, skip rate, gap
notify.py          ntfy.sh push (+ macOS banner when local)
executor.py        paper executor stub (kept from the harness; still no real orders)
```

## First run (paper)

```bash
# 1. subscribe to a topic in the ntfy app (iOS/Android), then:
echo NTFY_TOPIC=car-<something-unguessable> >> .env      # topics are public by name — make it long
uv run python -m live.alerter test-push

# 2. promote the commit you validated on the holdout
CRYPTO_HOLDOUT_DIR=~/crypto-holdout uv run python scripts/run_holdout.py --sha 971bf57
uv run python -m live.promote 971bf57 --symbol BTC/USDT --timeframe 1h
#    then set "holdout_checked": true in live/promoted.json by hand

# 3. run it — paper for the full forward period first
uv run python -m live.alerter run --loop --mode paper       # or `run --once` from cron every bar
```

Every signal alert must be acknowledged: `uv run python -m live.alerter ack <id> taken --reason "filled 61,230"` or
`... skipped --reason "asleep"`. Un-acked alerts are re-pushed every tick. `uv run python -m live.alerter_report`
prints the designed-vs-taken gap and the skip rate; `--notify` pushes the one-liner.

Halts (stale bar, API error streak, live drawdown > backtested max DD) are urgent-priority alerts and stop
signal alerts until `uv run python -m live.alerter resume --reason "..."`.

Paper → live is one flag: `--mode live` selects `ALPACA_LIVE_*` credentials and tags alerts `live`.
Orders are never placed in either mode.

## Railway

Cron service, command `python -m live.alerter run --once --mode paper`, schedule `1 * * * *` for 1h bars
(`*/15 * * * *` for 15m). Mount a volume at `/app/logs` so alerts/outcomes/state persist. Variables:
`NTFY_TOPIC`, `ALERTER_EQUITY_USD`, `ALERTER_RISK_FRACTION`, `ALERTER_CHANNELS=ntfy`. Requirements in
`live/requirements.txt`.
