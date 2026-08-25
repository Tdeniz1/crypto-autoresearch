"""Prompt assembly for every rung. Time-blind: no dates, symbols or years."""
from __future__ import annotations

import re

from harness import config

HARNESS_DESCRIPTION = f"""
# Harness contract (frozen — you cannot change any of this)

You are writing ONE file, strategy.py. It must define:

    def strategy(df: pd.DataFrame) -> pd.Series

`df` has a plain integer RangeIndex (bar 0, 1, 2, ...) and float columns
open, high, low, close, volume. There are NO timestamps. Do not try to infer
calendar time, asset identity or market regime from the data; refer to bars by
integer index only. The same function is run on several independent bar
series of two different bar durations; it must work on all of them unchanged.

Return a Series of the same length: the desired position after bar t closes,
in [-1, 1] (1 = fully long, -1 = fully short, 0 = flat). NaN is treated as 0.

Execution is structural: your signal at bar t is entered at the OPEN of bar t+1
and held until the open of t+2. Fees ({config.FEE_BPS:.0f} bps) and slippage
({config.SLIPPAGE_BPS:.0f} bps) are charged per side on every unit of position
change. Round trip ≈ {2 * (config.FEE_BPS + config.SLIPPAGE_BPS):.0f} bps. High turnover is
expensive; the seed strategy loses money mainly through costs.

Metric (one float, higher is better):
  mean over series of annualised net Sharpe × drawdown penalty
  (penalty falls linearly to a floor as max drawdown approaches {config.DD_FULL_PENALTY:.0%}),
  and -inf if ANY series has fewer than {config.MIN_TRADES} trades.

Hard rules — violations are rejected BEFORE running and count as a failed experiment:
  * imports: only numpy (np), pandas (pd), math
  * no file/network/env/process access, no eval/exec/getattr, no dunder attributes
  * no dates, years, datetime, Timestamp or time-related names of any kind
  * no free variables: functions may only use their arguments, locals, module-level
    literal constants, helper functions defined in the file, and basic builtins
  * module level may contain only imports, literal constants, functions, docstring
  * no rolling(..., center=True); no shift/diff/pct_change with negative or
    non-literal periods; no [k:] / [:-k] slices on iloc/loc/values; no bfill,
    interpolate, rank, sort, resample, np.roll, convolve
  * every statistic must be causal: rolling(w) / expanding() / ewm() only.
    Calling .mean()/.std()/.max()/... (pandas or numpy) over the whole input
    raises at runtime. A truncation test also runs: the signal on bars [0, k)
    must be identical whether or not bars >= k exist.
  * keep the file under {config.SCAN_MAX_LINES} lines

Your file MUST start with a one-line module docstring describing the idea; it is
what shows up in the experiment log.
"""


def build_static(program_md: str) -> str:
    return program_md.strip() + "\n\n" + HARNESS_DESCRIPTION.strip() + "\n"


def build_dynamic(current_strategy: str, log_summary: str) -> str:
    return (
        "## Experiment log (most recent last; metric = validation score)\n"
        f"{log_summary.strip() or '(no experiments yet)'}\n\n"
        "## Current best strategy.py\n"
        "```python\n" + current_strategy.rstrip() + "\n```\n\n"
        "Propose ONE new full replacement strategy.py that you expect to beat the current best. "
        "Change one idea at a time so the log stays interpretable. Prefer fewer, larger, "
        "cost-aware trades. Reply with the complete file in a single ```python fenced block "
        "and nothing else after it."
    )


_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_strategy(text: str) -> str | None:
    """Pull the last python fenced block; fall back to raw text if it looks like a module."""
    if not text:
        return None
    blocks = _FENCE_RE.findall(text)
    if blocks:
        cands = [b for b in blocks if "def strategy" in b] or blocks
        return cands[-1].strip() + "\n"
    if "def strategy" in text and ("import numpy" in text or "import pandas" in text):
        return text.strip() + "\n"
    return None


def summarize_log(records: list[dict], best_metric: float, rows: int = config.LOG_SUMMARY_ROWS) -> str:
    """Time-blind text summary of recent session records for the proposer."""
    lines = []
    for r in records[-rows:]:
        m = r.get("metric")
        ms = "-inf" if m is None else f"{m:+.3f}"
        st = r.get("status", "?")
        desc = (r.get("description") or "")[:90]
        extra = ""
        s = r.get("stats") or {}
        if s:
            extra = (f" sharpe={s.get('sharpe', 0):+.2f} mdd={s.get('max_drawdown', 0):.0%} "
                     f"trades={s.get('trades', 0)} tim={s.get('time_in_market', 0):.0%} "
                     f"turn/yr={s.get('turnover_annual', 0):.0f}")
        reason = f" reason: {r['reason'][:140]}" if r.get("reason") else ""
        tm = r.get("train_metric")
        tms = f" train={tm:+.2f}" if isinstance(tm, (int, float)) else ""
        lines.append(f"exp {r.get('exp')}: {st.upper():<18} metric={ms}{tms}{extra} — {desc}{reason}")
    lines.append(f"current best metric: {best_metric:+.4f}")
    return "\n".join(lines)
