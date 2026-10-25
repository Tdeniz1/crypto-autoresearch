"""Even slower vol-sizing EWM span (400 vs 250, same 250-bar warmup) for smoother position sizes, on top of 0.1 size floor, 0.5% regime-exit hysteresis, 20% stop, dual-EMA regime + 80-bar breakout + range/volume confirmed entries."""
import numpy as np
import pandas as pd

FAST = 50
SLOW = 200
BREAKOUT = 80
RANGE_WINDOW = 20
RANGE_MULT = 1.1
VOL_WINDOW = 20
VOL_Z_MIN = -0.5
STOP_PCT = 0.20
VOL_SPAN = 400
VOL_WARMUP = 250
MIN_SIZE = 0.1
EXIT_BAND = 0.005


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

    ret = close.pct_change(1)
    rvol = ret.ewm(span=VOL_SPAN, min_periods=VOL_WARMUP).std()
    rvol_ref = rvol.expanding(min_periods=VOL_WARMUP).mean()
    size_raw = (rvol_ref / rvol.where(rvol > 1e-12)).clip(lower=MIN_SIZE, upper=1.0)

    n = len(close)
    pos = np.zeros(n)
    state = 0.0
    size = 1.0
    entry_price = np.nan
    for i in range(n):
        f = fast.iat[i]
        s = slow.iat[i]
        h = hi.iat[i]
        l = lo.iat[i]
        c = close.iat[i]
        ra = range_avg.iat[i]
        vz = vol_z.iat[i]
        if np.isnan(f) or np.isnan(s) or np.isnan(h) or np.isnan(l) or np.isnan(ra) or np.isnan(vz):
            pos[i] = state * size
            continue
        regime = 1.0 if f > s else (-1.0 if f < s else 0.0)
        if state == 1.0 and f < s * (1.0 - EXIT_BAND):
            state = 0.0
            entry_price = np.nan
        elif state == -1.0 and f > s * (1.0 + EXIT_BAND):
            state = 0.0
            entry_price = np.nan

        if state == 1.0 and not np.isnan(entry_price):
            if c <= entry_price * (1.0 - STOP_PCT):
                state = 0.0
                entry_price = np.nan
        elif state == -1.0 and not np.isnan(entry_price):
            if c >= entry_price * (1.0 + STOP_PCT):
                state = 0.0
                entry_price = np.nan

        if state == 0.0:
            confirmed = bool(range_expanded.iat[i]) and bool(volume_confirmed.iat[i])
            sz = size_raw.iat[i]
            if np.isnan(sz):
                sz = 1.0
            if c > h and regime == 1.0 and confirmed:
                state = 1.0
                size = float(sz)
                entry_price = c
            elif c < l and regime == -1.0 and confirmed:
                state = -1.0
                size = float(sz)
                entry_price = c
        pos[i] = state * size
    return pd.Series(pos, index=df.index)
