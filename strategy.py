"""Seed: 48-bar breakout with a slow-trend filter, long/short, flat inside the band."""
import numpy as np
import pandas as pd

BREAKOUT = 48
TREND = 200


def strategy(df: pd.DataFrame) -> pd.Series:
    close = df["close"]
    hi = close.rolling(BREAKOUT).max().shift(1)
    lo = close.rolling(BREAKOUT).min().shift(1)
    trend = close.rolling(TREND).mean()
    long_sig = (close > hi) & (close > trend)
    short_sig = (close < lo) & (close < trend)
    pos = pd.Series(np.where(long_sig, 1.0, np.where(short_sig, -1.0, np.nan)), index=df.index)
    pos = pos.ffill().fillna(0.0)
    return pos
