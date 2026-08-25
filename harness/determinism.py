"""Seed everything and prove the metric is bit-identical across two runs."""
from __future__ import annotations

import os
import random
import struct

import numpy as np

from harness import config


def seed_everything(seed: int = config.SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def metric_bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


def check_determinism(source: str, series=None) -> tuple[bool, float, float]:
    """Evaluate the same strategy source twice; compare the metric bit-for-bit."""
    from harness.evaluate import evaluate_source  # late import: avoids a cycle

    seed_everything()
    a = evaluate_source(source, series=series, split="val", run_checks=False)
    seed_everything()
    b = evaluate_source(source, series=series, split="val", run_checks=False)
    return metric_bits(a.metric) == metric_bits(b.metric), a.metric, b.metric
