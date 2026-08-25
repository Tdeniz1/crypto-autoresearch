"""Provider attribution: keep rate and mean metric improvement grouped by proposer."""
from __future__ import annotations

import argparse
import math
from collections import defaultdict

from harness import config
from harness.budget import read_jsonl


def attribution(records: list[dict], ledger: list[dict] | None = None) -> list[dict]:
    groups: dict[tuple, dict] = defaultdict(lambda: {"calls": 0, "kept": 0, "improvements": [], "metrics": [],
                                                    "costs": [], "latencies": [], "rejected": 0})
    for r in records:
        if r.get("kind") != "experiment" or not r.get("rung"):
            continue
        g = groups[(r["rung"], r.get("model") or "?")]
        g["calls"] += 1
        if r["status"] == "kept":
            g["kept"] += 1
            if r.get("improvement") is not None:
                g["improvements"].append(r["improvement"])
        if r["status"].startswith("rejected"):
            g["rejected"] += 1
        if r.get("metric") is not None and math.isfinite(r["metric"]):
            g["metrics"].append(r["metric"])
        if r.get("cost_usd") is not None:
            g["costs"].append(r["cost_usd"])
        if r.get("proposer_seconds"):
            g["latencies"].append(r["proposer_seconds"])
    rows = []
    for (rung, model), g in groups.items():
        total_cost = sum(g["costs"])
        rows.append({
            "rung": rung, "model": model, "calls": g["calls"], "kept": g["kept"],
            "rejected": g["rejected"],
            "keep_rate": g["kept"] / g["calls"] if g["calls"] else 0.0,
            "mean_improvement_kept": (sum(g["improvements"]) / len(g["improvements"])) if g["improvements"] else None,
            "mean_metric": (sum(g["metrics"]) / len(g["metrics"])) if g["metrics"] else None,
            "total_cost_usd": total_cost,
            "cost_per_kept_usd": (total_cost / g["kept"]) if g["kept"] else None,
            "mean_latency_s": (sum(g["latencies"]) / len(g["latencies"])) if g["latencies"] else None,
        })
    rows.sort(key=lambda r: (-r["kept"], -r["keep_rate"], r["rung"]))
    return rows


def fmt(v, spec="{:.3f}", none="—"):
    return none if v is None else spec.format(v)


def render_table(rows: list[dict]) -> str:
    head = "| rung | model | calls | kept | keep rate | mean Δmetric (kept) | mean metric | cost | $/kept | mean latency |"
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    body = [f"| {r['rung']} | {r['model']} | {r['calls']} | {r['kept']} | {r['keep_rate']:.0%} | "
            f"{fmt(r['mean_improvement_kept'], '{:+.3f}')} | {fmt(r['mean_metric'], '{:+.3f}')} | "
            f"${r['total_cost_usd']:.3f} | {fmt(r['cost_per_kept_usd'], '${:.3f}')} | {fmt(r['mean_latency_s'], '{:.0f}s')} |"
            for r in rows]
    return "\n".join([head, sep, *body]) if rows else "(no proposer calls)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default=None, help="session id, or 'latest' (default: all sessions)")
    a = ap.parse_args()
    recs = read_jsonl(config.SESSION_LOG)
    if a.session:
        sid = a.session
        if sid == "latest":
            sids = [r["session_id"] for r in recs]
            sid = sids[-1] if sids else None
        recs = [r for r in recs if r.get("session_id") == sid]
    print(render_table(attribution(recs)))


if __name__ == "__main__":
    main()
