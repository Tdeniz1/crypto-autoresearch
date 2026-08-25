"""Budget ledger: logs/ledger.jsonl. Arithmetic only, no model calls.

One record per proposer call: timestamp, rung, model, tokens, dollar cost (where
known), wall-clock seconds, and whether the experiment was kept or reverted.
"""
from __future__ import annotations

import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

from harness import config


class BudgetExhausted(RuntimeError):
    pass


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int,
                      cache_write: int = 0, cache_read: int = 0) -> float | None:
    key = next((k for k in config.PRICES_PER_MTOK if model.startswith(k)), None)
    if key is None:
        return None
    pi, po, pw, pr = config.PRICES_PER_MTOK[key]
    return (input_tokens * pi + output_tokens * po + cache_write * pw + cache_read * pr) / 1e6


def read_jsonl(path: Path, tolerate_partial: bool = True) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                if not tolerate_partial:
                    raise
    return out


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=_json_default, separators=(",", ":"))
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def _json_default(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return None
    try:
        import numpy as np  # local import keeps this module dependency-light
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (math.isinf(v) or math.isnan(v)) else v
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:
        pass
    return str(o)


class Ledger:
    def __init__(self, path: Path = config.LEDGER_LOG, session_id: str | None = None,
                 rung3_cap_usd: float = config.RUNG3_SESSION_USD_CAP):
        self.path = path
        self.session_id = session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.rung3_cap_usd = rung3_cap_usd
        self.records: list[dict] = []          # this session's records (in-memory copy)
        self.session_start = time.time()
        self._all_time_count = len(read_jsonl(path))

    # ---- write ----------------------------------------------------------
    def record(self, *, exp: int, rung: str, model: str, input_tokens: int, output_tokens: int,
               cost_usd: float | None, proposer_seconds: float, experiment_seconds: float,
               outcome: str, cache_read_tokens: int = 0, cache_write_tokens: int = 0,
               extra: dict | None = None) -> dict:
        self._all_time_count += 1
        rec = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "exp": exp,
            "rung": rung,
            "model": model,
            "input_tokens": int(input_tokens or 0),
            "output_tokens": int(output_tokens or 0),
            "cache_read_tokens": int(cache_read_tokens or 0),
            "cache_write_tokens": int(cache_write_tokens or 0),
            "cost_usd": None if cost_usd is None else float(cost_usd),
            "proposer_seconds": float(proposer_seconds),
            "experiment_seconds": float(experiment_seconds),
            "outcome": outcome,
            "cumulative_experiments": self._all_time_count,
            "session_experiments": len(self.records) + 1,
        }
        if extra:
            rec.update(extra)
        self.records.append(rec)
        append_jsonl(self.path, rec)
        return rec

    # ---- arithmetic -----------------------------------------------------
    @property
    def cumulative_experiments(self) -> int:
        return self._all_time_count

    def spend_by_rung(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for r in self.records:
            out[r["rung"]] = out.get(r["rung"], 0.0) + (r["cost_usd"] or 0.0)
        return out

    def rung3_spend(self) -> float:
        return self.spend_by_rung().get("anthropic", 0.0)

    def rung3_remaining(self) -> float:
        return max(0.0, self.rung3_cap_usd - self.rung3_spend())

    def assert_rung3_budget(self, projected_cost: float = 0.0) -> None:
        if self.rung3_spend() + projected_cost > self.rung3_cap_usd + 1e-9:
            raise BudgetExhausted(
                f"rung 3 session spend ${self.rung3_spend():.4f} + ${projected_cost:.4f} would exceed "
                f"cap ${self.rung3_cap_usd:.2f}")

    def rolling_mean_seconds(self, window: int = 10) -> float | None:
        xs = [r["experiment_seconds"] for r in self.records[-window:]]
        return (sum(xs) / len(xs)) if xs else None

    def projection(self, stop_time: float | None, window: int = 10) -> dict:
        mean_s = self.rolling_mean_seconds(window)
        remaining_s = None if stop_time is None else max(0.0, stop_time - time.time())
        fits = None
        if mean_s and remaining_s is not None:
            fits = int(remaining_s // mean_s)
        return {
            "mean_seconds_per_experiment": mean_s,
            "seconds_remaining": remaining_s,
            "experiments_that_fit": fits,
            "spend_by_rung": self.spend_by_rung(),
            "rung3_cap_usd": self.rung3_cap_usd,
            "rung3_remaining_usd": self.rung3_remaining(),
        }
