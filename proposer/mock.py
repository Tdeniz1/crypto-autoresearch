"""Offline proposer for smoke/unit tests. Deterministic (seeded), no network.

Cycles through parameterised strategy templates. Every Nth proposal is
deliberately malformed so the scan/lookahead rejection paths get exercised.
"""
from __future__ import annotations

import random

from proposer.result import RungResult

NAME = "mock"

_TEMPLATES = [
    '''"""Mock: {a}-bar breakout with {b}-bar trend filter, exit on {c}-bar reversal."""
import numpy as np
import pandas as pd

A = {a}
B = {b}
C = {c}


def strategy(df):
    close = df["close"]
    hi = close.rolling(A).max().shift(1)
    lo = close.rolling(A).min().shift(1)
    trend = close.rolling(B).mean()
    ex_hi = close.rolling(C).max().shift(1)
    ex_lo = close.rolling(C).min().shift(1)
    pos = pd.Series(np.nan, index=df.index)
    pos[(close > hi) & (close > trend)] = 1.0
    pos[(close < lo) & (close < trend)] = -1.0
    pos = pos.ffill().fillna(0.0)
    flat_long = (pos == 1.0) & (close < ex_lo)
    flat_short = (pos == -1.0) & (close > ex_hi)
    pos[flat_long | flat_short] = 0.0
    return pos
''',
    '''"""Mock: EMA {a}/{b} crossover, long only above slow EMA, vol-target {c}bp."""
import numpy as np
import pandas as pd

FAST = {a}
SLOW = {b}
VOL_BP = {c}


def strategy(df):
    close = df["close"]
    fast = close.ewm(span=FAST, adjust=False).mean()
    slow = close.ewm(span=SLOW, adjust=False).mean()
    ret = close.pct_change(1)
    vol = ret.rolling(SLOW).std()
    target = (VOL_BP / 1e4) / vol.replace(0.0, np.nan)
    raw = np.where(fast > slow, 1.0, 0.0)
    pos = pd.Series(raw, index=df.index) * target.clip(upper=1.0)
    return pos.fillna(0.0)
''',
    '''"""Mock: mean reversion — fade {a}-bar z-score beyond {b}, hold max {c} bars."""
import numpy as np
import pandas as pd

W = {a}
Z = {b}
HOLD = {c}


def strategy(df):
    close = df["close"]
    mu = close.rolling(W).mean()
    sd = close.rolling(W).std()
    z = (close - mu) / sd.replace(0.0, np.nan)
    sig = pd.Series(np.where(z > Z, -1.0, np.where(z < -Z, 1.0, 0.0)), index=df.index)
    held = sig.rolling(HOLD).sum()
    pos = held.clip(-1.0, 1.0)
    return pos.fillna(0.0)
''',
    '''"""Mock: {a}-bar momentum sign with {b}-bar trend confirm and {c}% ATR stop-out."""
import numpy as np
import pandas as pd

MOM = {a}
TREND = {b}
ATR_MULT = {c}


def strategy(df):
    close = df["close"]
    mom = close.pct_change(MOM)
    trend = close.rolling(TREND).mean()
    tr = (df["high"] - df["low"]).rolling(14).mean()
    band = tr * ATR_MULT / 100.0
    up = (mom > 0) & (close > trend + band)
    dn = (mom < 0) & (close < trend - band)
    pos = pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=df.index)
    return pos
''',
]

_BAD = [
    # forbidden: negative shift (scan)
    '''"""Mock BAD: peeks one bar ahead."""
import numpy as np
import pandas as pd


def strategy(df):
    return np.sign(df["close"].shift(-1) - df["close"]).fillna(0.0)
''',
    # forbidden: file access (scan)
    '''"""Mock BAD: tries to read a file."""
import pandas as pd


def strategy(df):
    extra = pd.read_parquet("data/holdout/x.parquet")
    return df["close"] * 0
''',
    # passes scan, fails at runtime: full-series z-score
    '''"""Mock BAD: full-series z-score (non-causal normalisation)."""
import numpy as np
import pandas as pd


def strategy(df):
    close = df["close"]
    z = (close - close.mean()) / close.std()
    return pd.Series(np.where(z < -1, 1.0, np.where(z > 1, -1.0, 0.0)), index=df.index)
''',
]


class MockProposer:
    def __init__(self, seed: int = 0, bad_every: int = 4):
        self.rng = random.Random(seed)
        self.calls = 0
        self.bad_every = bad_every

    def propose(self, static_context: str, dynamic_context: str, attempt: int = 0) -> RungResult:
        self.calls += 1
        if self.bad_every and self.calls % self.bad_every == 0:
            src = _BAD[(self.calls // self.bad_every - 1) % len(_BAD)]
        else:
            t = _TEMPLATES[(self.calls - 1) % len(_TEMPLATES)]
            src = t.format(a=self.rng.choice([12, 24, 48, 96, 168]),
                           b=self.rng.choice([96, 200, 400, 800]),
                           c=self.rng.choice([8, 16, 24, 48, 100]))
        return RungResult(source=src, rung=NAME, model="mock-template",
                         input_tokens=len(static_context + dynamic_context) // 4,
                         output_tokens=len(src) // 4, cost_usd=0.0)


_default: MockProposer | None = None


def propose(static_context: str, dynamic_context: str, attempt: int = 0) -> RungResult:
    global _default
    if _default is None:
        _default = MockProposer()
    return _default.propose(static_context, dynamic_context, attempt)
