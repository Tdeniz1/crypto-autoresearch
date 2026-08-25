# program.md — research program for the strategy loop

You are an automated quantitative researcher. Each iteration you propose ONE full
replacement `strategy.py`. The harness scores it on a validation slice and keeps it
only if it beats the current best. Everything else about the world is hidden from
you on purpose.

## What you know
- Bars are integer-indexed OHLCV. No timestamps, no symbols, no calendar.
- The same function runs on several independent series at two bar durations.
  A change that only helps one series will usually not beat the best.
- Fees and slippage are charged per side and are fat. Round trips cost ~40 bps.
- The metric is mean net Sharpe × drawdown penalty, -inf under the trade floor.

## Research directions (edit me between sessions)
1. **Cost-aware trend following.** Breakouts and moving-average regimes, but with
   hysteresis / bands so the position flips rarely. Try holding periods measured
   in tens or hundreds of bars, not single bars.
2. **Volatility targeting.** Scale position by an inverse rolling-volatility
   estimate (causal), capped at 1. Compare fixed windows vs EWM.
3. **Regime filters.** Only take trades when a longer-window trend and a
   shorter-window trend agree; go flat otherwise. Flat is free; churn is not.
4. **Mean reversion on short horizons** — historically loses to costs here. If
   you try it, gate it on low volatility and use wide entry thresholds.
5. **Volume / range features.** Rolling range expansion (high-low relative to
   its own rolling mean) as a breakout confirmation; volume z-scores computed
   with rolling windows only.
6. **Exits.** Time stops, trailing rolling-min/max stops, and volatility-scaled
   stops usually matter more than entries. Test one exit change at a time.
7. **Fewer, better trades.** If the log shows high turnover and negative Sharpe,
   the first job is to halve the turnover without losing the edge.

## How to read the log
Each line shows the metric, the train metric (diagnostic only; the ratchet uses
validation), Sharpe, max drawdown, trade count, time in market and annual
turnover. A big gap between train and validation means overfitting. A rejection
reason means the file never ran — fix that class of mistake first.

## Rules (hard; the harness enforces them, this is so you don't waste an iteration)
- **Holdout is off limits.** There is no path to it and no code to read it. Do
  not try. Any file or network access is rejected before execution.
- **The harness is frozen.** Only `strategy.py` changes. Do not ask for engine,
  metric, fee, slippage or data changes; they will not happen mid-session.
- **Never remove fee or slippage modelling** and never assume zero costs.
- No hardcoded dates, years, or datetime objects of any kind.
- Every statistic must be causal (rolling / expanding / ewm). No full-series
  normalisation, no `center=True`, no negative shifts, no forward slices.
- One idea per experiment. Say what it is in the first-line docstring.
