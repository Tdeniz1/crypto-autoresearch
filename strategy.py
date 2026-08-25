"""Range-expansion confirmation: dual-EMA regime + 80-bar breakout entries are only taken when the entry bar's true range is expanded vs its own rolling average, filtering out low-conviction breakouts that whipsaw."""
import numpy as np
import pandas as pd

FAST = 50
SLOW = 200
BREAKOUT = 80
RANGE_WINDOW = 20
RANGE_MULT = 1.1


def strategy(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]

    fast = close.ewm(span=FAST, min_periods=SLOW).mean()
    slow = close.ewm(span=SLOW, min_periods=SLOW).mean()
    hi = close.rolling(BREAKOUT, min_periods=BREAKOUT).max().shift(1)
    lo = close.rolling(BREAKOUT, min_periods=BREAKOUT).min().shift(1)

    bar_range = high - low
    range_avg = bar_range.rolling(RANGE_WINDOW, min_periods=RANGE_WINDOW).mean().shift(1)
    range_expanded = bar_range > (range_avg * RANGE_MULT)

    n = len(close)
    pos = np.zeros(n)
    state = 0.0
    for i in range(n):
        f = fast.iat[i]
        s = slow.iat[i]
        h = hi.iat[i]
        l = lo.iat[i]
        c = close.iat[i]
        ra = range_avg.iat[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(h) or np.isnan(l) or np.isnan(ra):
            pos[i] = state
            continue
        regime = 1.0 if f > s else (-1.0 if f < s else 0.0)
        if state == 1.0 and regime != 1.0:
            state = 0.0
        elif state == -1.0 and regime != -1.0:
            state = 0.0
        if state == 0.0:
            confirmed = bool(range_expanded.iat[i])
            if c > h and regime == 1.0 and confirmed:
                state = 1.0
            elif c < l and regime == -1.0 and confirmed:
                state = -1.0
        pos[i] = state
    return pd.Series(pos, index=df.index)
