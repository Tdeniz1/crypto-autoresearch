import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness import config  # noqa: E402
from harness.data import Series  # noqa: E402


def synthetic_bars(n: int = 4000, seed: int = 7, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(drift, 0.01, n)
    close = 100 * np.cumprod(1 + r)
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    vol = rng.lognormal(10, 0.5, n)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol})


@pytest.fixture
def synth_series():
    return [Series(id="t0", bars_per_year=8760, train=synthetic_bars(3000, 1), val=synthetic_bars(2000, 2)),
            Series(id="t1", bars_per_year=8760, train=synthetic_bars(3000, 3), val=synthetic_bars(2000, 4))]


@pytest.fixture
def has_frozen_data():
    if not (config.FROZEN_DIR / "manifest.json").exists():
        pytest.skip("frozen data not prepared")
    return True


@pytest.fixture(autouse=True)
def _no_holdout_env(monkeypatch):
    monkeypatch.delenv(config.HOLDOUT_ENV_VAR, raising=False)
