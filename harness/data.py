"""Frozen-data access. Time-blind: bars are integer-indexed, no timestamps."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from harness import config

COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class Series:
    id: str
    bars_per_year: int
    train: pd.DataFrame
    val: pd.DataFrame


def load_manifest() -> dict:
    p = config.FROZEN_DIR / "manifest.json"
    if not p.exists():
        raise FileNotFoundError("data/frozen/manifest.json missing — run prepare.py once (see README)")
    return json.loads(p.read_text())


def _read(p: Path) -> pd.DataFrame:
    df = pd.read_parquet(p)
    assert list(df.columns) == COLUMNS, f"{p}: unexpected columns {list(df.columns)}"
    df = df.astype("float64").reset_index(drop=True)
    df.index = pd.RangeIndex(len(df))
    return df


def load_series(ids: list[str] | None = None) -> list[Series]:
    manifest = load_manifest()
    ids = ids or config.eval_series_ids(manifest)
    out = []
    for s in manifest["series"]:
        if s["id"] not in ids:
            continue
        out.append(Series(
            id=s["id"],
            bars_per_year=int(s["bars_per_year"]),
            train=_read(config.FROZEN_DIR / f"{s['id']}_train.parquet"),
            val=_read(config.FROZEN_DIR / f"{s['id']}_val.parquet"),
        ))
    if not out:
        raise RuntimeError("no series loaded")
    return out


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_checksums() -> dict[str, str]:
    p = config.FROZEN_DIR / "CHECKSUMS.sha256"
    if not p.exists():
        raise FileNotFoundError("data/frozen/CHECKSUMS.sha256 missing — run prepare.py")
    out = {}
    for line in p.read_text().splitlines():
        digest, name = line.split("  ", 1)
        out[name.strip()] = digest.strip()
    return out


def verify_frozen(strict: bool = True) -> dict[str, str]:
    """Recompute frozen-data checksums; raise loudly on any mismatch."""
    expected = read_checksums()
    actual = {name: sha256_file(config.FROZEN_DIR / name) for name in expected}
    bad = {k: (expected[k], actual[k]) for k in expected if expected[k] != actual[k]}
    if bad and strict:
        raise RuntimeError(f"FROZEN DATA MODIFIED during session: {list(bad)}")
    return actual
