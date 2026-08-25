"""Volume confirmation: add a rolling volume z-score gate on top of the range-expansion breakout filter, requiring participation not clearly below average before taking a signal."""
import numpy as np
import pandas as pd

FAST = 50
SLOW = 200
BREAKOUT = 80
RANGE_WINDOW = 20
RANGE_MULT = 1.1
VOL_WINDOW = 20
VOL_Z_MIN = -0.5


def strategy(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    fast = close.ewm(span=FAST, min_periods=SLOW).mean()
    slow = close.ewm(span=SLOW, min_periods=SLOW).mean()
    hi = close.rolling(BREAKOUT, min_periods=BREAKOUT).max().shift(1)
    lo = close.rolling(BREAKOUT, min_periods=BREAKOUT).min().shift(1)

    bar_range = high - low
    range_avg = bar_range.rolling(RANGE_WINDOW, min_periods=RANGE_WINDOW).mean().shift(1)
    range_expanded = bar_range > (range_avg * RANGE_MULT)

    vol_mean = volume.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).mean().shift(1)
    vol_std = volume.rolling(VOL_WINDOW, min_periods=VOL_WINDOW).std().shift(1)
    vol_std_safe = vol_std.where(vol_std > 1e-9)
    vol_z = (volume - vol_mean) / vol_std_safe
    volume_confirmed = vol_z > VOL_Z_MIN

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
        vz = vol_z.iat[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(h) or np.isnan(l) or np.isnan(ra) or np.isnan(vz):
            pos[i] = state
            continue
        regime = 1.0 if f > s else (-1.0 if f < s else 0.0)
        if state == 1.0 and regime != 1.0:
            state = 0.0
        elif state == -1.0 and regime != -1.0:
            state = 0.0
        if state == 0.0:
            confirmed = bool(range_expanded.iat[i]) and bool(volume_confirmed.iat[i])
            if c > h and regime == 1.0 and confirmed:
                state = 1.0
            elif c < l and regime == -1.0 and confirmed:
                state = -1.0
        pos[i] = state
    return pd.Series(pos, index=df.index)
