import pytest

from harness.scan import ScanError, scan_source

GOOD = '''"""ok"""
import numpy as np
import pandas as pd
W = 20
def strategy(df):
    m = df["close"].rolling(W).mean()
    return pd.Series(np.where(df["close"] > m, 1.0, 0.0), index=df.index)
'''


def test_good_passes():
    scan_source(GOOD)


@pytest.mark.parametrize("snippet,needle", [
    ('start = "2021-03-01"', "date-like"),
    ("YEAR = 2020", "year-like"),
    ("import datetime", "not allowed"),
    ("from pandas import Timestamp", "forbidden"),
    ('x = pd.read_parquet("data/holdout/s0.parquet")', "forbidden"),
    ('import os', "not allowed"),
    ('e = __import__("os").environ', "forbidden"),
    ('r = df["close"].rolling(5, center=True).mean()', "center=True"),
    ('r = df["close"].shift(-1)', "reaches forward"),
    ('r = df["close"].shift(periods=-2)', "reaches forward"),
    ('r = df["close"].iloc[1:].values', "offset slice"),
    ('r = df["close"].values[:-1]', "offset slice"),
    ('r = df["close"].bfill()', "forbidden"),
    ('r = eval("1")', "forbidden"),
    ('r = df.__class__', "dunder"),
    ('r = undefined_name * 2', "free variable"),
    ('r = np.roll(df["close"].values, -1)', "forbidden"),
])
def test_rejections(snippet, needle):
    src = GOOD.replace("    m = df", f"    {snippet}\n    m = df")
    with pytest.raises(ScanError) as ei:
        scan_source(src)
    assert needle in str(ei.value)


def test_module_level_code_rejected():
    with pytest.raises(ScanError):
        scan_source(GOOD + "\nprint(1)\n")


def test_missing_strategy():
    with pytest.raises(ScanError, match="no `def strategy"):
        scan_source("import numpy as np\n")


def test_module_constant_shift_allowed():
    src = GOOD.replace("W = 20", "W = 20\nK = 3").replace("rolling(W).mean()", "rolling(W).mean().shift(K)")
    scan_source(src)


def test_negative_module_constant_shift_rejected():
    src = GOOD.replace("W = 20", "W = 20\nK = -3").replace("rolling(W).mean()", "rolling(W).mean().shift(K)")
    with pytest.raises(ScanError, match="reaches forward"):
        scan_source(src)
