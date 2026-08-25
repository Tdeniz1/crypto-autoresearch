"""Regime filter: dual EMA trend agreement gates breakout entries; go flat on disagreement instead of flipping directly, to cut turnover."""
import numpy as np
import pandas as pd

FAST = 50
SLOW = 200
BREAKOUT = 80


def strategy(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    fast = close.ewm(span=FAST, min_periods=SLOW).mean()
    slow = close.ewm(span=SLOW, min_periods=SLOW).mean()
    hi = close.rolling(BREAKOUT, min_periods=BREAKOUT).max().shift(1)
    lo = close.rolling(BREAKOUT, min_periods=BREAKOUT).min().shift(1)

    n = len(close)
    pos = np.zeros(n)
    state = 0.0
    for i in range(n):
        f = fast.iat[i]
        s = slow.iat[i]
        h = hi.iat[i]
        l = lo.iat[i]
        c = close.iat[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(h) or np.isnan(l):
            pos[i] = state
            continue
        regime = 1.0 if f > s else (-1.0 if f < s else 0.0)
        if state == 1.0 and regime != 1.0:
            state = 0.0
        elif state == -1.0 and regime != -1.0:
            state = 0.0
        if state == 0.0:
            if c > h and regime == 1.0:
                state = 1.0
            elif c < l and regime == -1.0:
                state = -1.0
        pos[i] = state
    return pd.Series(pos, index=df.index)
