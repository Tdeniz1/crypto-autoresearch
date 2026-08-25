"""Read-only dashboard backend.

HARD RULE: this process reads logs/session.jsonl, logs/ledger.jsonl, logs/curves/*.json
and git history. It never writes to them, never triggers experiments, never
calls a model, and never touches holdout data. Zero coupling with runner.py:
if the runner is not running, it serves the last session's files.
"""
from __future__ import annotations

import json
import math
import statistics
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse

from analyze_providers import attribution
from harness import config
from harness.metric import deflated_sharpe

STATIC = Path(__file__).parent / "static"
app = FastAPI(title="crypto-autoresearch dashboard (read-only)")


class MidWrite(Exception):
    pass


def _read_jsonl_strict_tail(path: Path) -> list[dict]:
    """Parse a jsonl file. A partial LAST line means a write is in progress -> MidWrite.
    Any other unparseable line is skipped."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    if not raw:
        return []
    text = raw.decode("utf-8", errors="replace")
    lines = text.split("\n")
    complete = text.endswith("\n")
    out = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not complete:
                raise MidWrite(path.name)
            continue
    return out


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _session_filter(records: list[dict], session: str) -> tuple[str | None, list[dict]]:
    if not records:
        return None, []
    sid = session
    if session in ("latest", "", None):
        sid = records[-1].get("session_id")
    return sid, [r for r in records if r.get("session_id") == sid]


def _parse_ts(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts).timestamp()
    except ValueError:
        return None


def build_state(session: str = "latest") -> dict:
    all_sess = _read_jsonl_strict_tail(config.SESSION_LOG)
    all_led = _read_jsonl_strict_tail(config.LEDGER_LOG)
    sid, recs = _session_filter(all_sess, session)
    led = [r for r in all_led if r.get("session_id") == sid]
    exps = [r for r in recs if r.get("kind") == "experiment"]
    base = next((r for r in recs if r.get("kind") == "baseline"), None)
    end = next((r for r in recs if r.get("kind") == "session_end"), None)
    now = time.time()

    # ------------------------------------------------------------ health --
    kept = [r for r in exps if r.get("status") == "kept"]
    start_ts = _parse_ts(recs[0]["ts"]) if recs else None
    last_ts = _parse_ts(recs[-1]["ts"]) if recs else None
    running = end is None and last_ts is not None and (now - last_ts) < 1800
    elapsed = ((last_ts if end else now) - start_ts) if start_ts else 0.0
    secs = [r.get("experiment_seconds") for r in exps[-10:] if r.get("experiment_seconds")]
    mean_s = (sum(secs) / len(secs)) if secs else None
    stop_time = (end or {}).get("stop_time") or (exps[-1].get("stop_time") if exps else None)
    projected = None
    if mean_s and stop_time and running:
        projected = max(0, int((stop_time - now) // mean_s))
    spend = defaultdict(float)
    calls = defaultdict(int)
    for r in led:
        spend[r["rung"]] += r.get("cost_usd") or 0.0
        calls[r["rung"]] += 1
    demotions = (end or {}).get("demotions") or (exps[-1].get("demotions") if exps else []) or []
    active_rung = (exps[-1].get("active_rung") if exps else None)
    health = {
        "experiments": len(exps), "kept": len(kept),
        "keep_rate": (len(kept) / len(exps)) if exps else 0.0,
        "elapsed_s": elapsed, "mean_seconds_per_experiment": mean_s,
        "projected_remaining": projected, "stop_time": stop_time,
        "active_rung": active_rung if running else None, "last_rung": (exps[-1].get("rung") if exps else None),
        "demotions": demotions,
        "spend_by_rung": {k: {"usd": spend[k], "calls": calls[k]} for k in sorted(set(spend) | set(calls))},
        "rung3_cap_usd": config.RUNG3_SESSION_USD_CAP, "rung3_spend_usd": spend.get("anthropic", 0.0),
        "rejections": sum(1 for r in exps if str(r.get("status", "")).startswith("rejected")),
        "proposer_failures": sum(1 for r in exps if r.get("status") == "proposer_failed"),
    }

    # ----------------------------------------------------------- ratchet --
    points = []
    best = base["metric"] if base and _finite(base.get("metric")) else None
    best_series = []
    last_improve_exp = 0
    for r in exps:
        m = r.get("metric") if _finite(r.get("metric")) else None
        if r.get("status") == "kept" and m is not None and (best is None or m > best):
            best = m
            last_improve_exp = r["exp"]
        points.append({"exp": r["exp"], "metric": m, "status": r.get("status"), "best": best,
                       "neg_inf": bool(r.get("metric_is_neg_inf"))})
    flat_for = (exps[-1]["exp"] - last_improve_exp) if exps else 0
    ratchet = {"points": points, "baseline": base["metric"] if base else None,
               "flat_for": flat_for, "exhausted": flat_for >= config.DASHBOARD_PLATEAU_EXPERIMENTS,
               "plateau_threshold": config.DASHBOARD_PLATEAU_EXPERIMENTS}

    # ------------------------------------------------------- leaderboard --
    # multiple-testing count = every experiment ever recorded (all sessions)
    n_trials = max(1, sum(1 for r in all_sess if r.get("kind") == "experiment"))
    trial_sharpes = [r["stats"]["sharpe"] for r in all_sess if r.get("kind") == "experiment" and r.get("stats")]
    recent = [r["metric"] for r in exps[-config.FLAG_IMPROVEMENT_STD_WINDOW:] if _finite(r.get("metric"))]
    recent_std = statistics.pstdev(recent) if len(recent) >= 2 else None
    rows = []
    for r in exps:
        s = r.get("stats") or {}
        if not s or not _finite(r.get("metric")):
            continue
        d = deflated_sharpe(s["sharpe"], s["bars"], s["bars_per_year"], n_trials, trial_sharpes, s.get("skew", 0), s.get("kurt", 3))
        flags = []
        c = r.get("canaries") or {}
        cflags = list(c.get("flags") or [])
        if s["min_trades"] <= config.MIN_TRADES * (1 + config.FLAG_TRADE_FLOOR_PROXIMITY):
            flags.append("NEAR_TRADE_FLOOR")
        if s["time_in_market"] < config.FLAG_TIME_IN_MARKET_MIN:
            flags.append("LOW_TIME_IN_MARKET")
        if s["turnover_annual"] > config.FLAG_TURNOVER_MAX:
            flags.append("HIGH_TURNOVER")
        if s["max_drawdown"] > config.FLAG_MDD_MAX:
            flags.append("DEEP_DRAWDOWN")
        if r.get("status") == "kept" and recent_std and r.get("improvement") is not None and r["improvement"] < recent_std:
            flags.append("IMPROVEMENT_WITHIN_NOISE")
        rows.append({
            "exp": r["exp"], "metric": r["metric"], "status": r.get("status"),
            "raw_sharpe": s["sharpe"], "deflated_sharpe": d["deflated"], "dsr_prob": d["dsr_prob"],
            "max_drawdown": s["max_drawdown"], "trades": s["trades"], "min_trades": s["min_trades"],
            "win_rate": s["win_rate"], "payoff_ratio": s["payoff_ratio"],
            "time_in_market": s["time_in_market"], "turnover_annual": s["turnover_annual"],
            "rung": r.get("rung"), "model": r.get("model"), "sha": r.get("git_sha"),
            "description": r.get("description"), "flags": flags, "canary_flags": cflags,
            "canaries": c, "excluded": "LEAK_SUSPECTED" in cflags,
            "improvement": r.get("improvement"), "train_metric": r.get("train_metric"),
        })
    excluded = [x for x in rows if x["excluded"]]
    ranked = sorted([x for x in rows if not x["excluded"]],
                    key=lambda x: (-(x["deflated_sharpe"] if _finite(x["deflated_sharpe"]) else -1e9), -x["metric"]))
    for i, x in enumerate(ranked, 1):
        x["rank"] = i
    leaderboard = {"rows": ranked[:config.DASHBOARD_TOP_N], "excluded": excluded, "n_trials": n_trials,
                   "recent_metric_std": recent_std}

    # ------------------------------------------------------------ equity --
    best_kept = max(kept, key=lambda r: r["metric"] if _finite(r.get("metric")) else -1e18, default=None)
    equity_exp = best_kept["exp"] if best_kept else (0 if base else None)
    equity = None
    if equity_exp is not None:
        p = config.CURVE_DIR / f"exp_{equity_exp}.json"
        if p.exists():
            try:
                cur = json.loads(p.read_text())
            except json.JSONDecodeError as e:
                raise MidWrite(p.name) from e
            src = best_kept or base
            st = src.get("stats") or {}
            equity = {"exp": equity_exp, "description": src.get("description"), "series": cur["series"],
                      "strategy_return": st.get("total_return"), "buy_hold_return": st.get("buy_hold_return"),
                      "beats_buy_hold": (st.get("total_return") or 0) > (st.get("buy_hold_return") or 0),
                      "spread": (st.get("total_return") or 0) - (st.get("buy_hold_return") or 0),
                      "per_series": {k: {"strategy": v.get("total_return"), "buy_hold": v.get("buy_hold_return")}
                                     for k, v in (src.get("per_series") or {}).items()}}

    # --------------------------------------------------------- proposers --
    proposers = attribution(exps)

    sessions = []
    seen = set()
    for r in all_sess:
        s_id = r.get("session_id")
        if s_id and s_id not in seen:
            seen.add(s_id); sessions.append(s_id)

    return {
        "ok": True, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "session": {"id": sid, "running": running, "stop_reason": (end or {}).get("stop_reason"),
                    "start_ts": recs[0]["ts"] if recs else None, "last_ts": recs[-1]["ts"] if recs else None,
                    "sessions": sessions[-20:]},
        "health": health, "ratchet": ratchet, "leaderboard": leaderboard, "equity": equity,
        "proposers": proposers,
        "config": {"min_trades": config.MIN_TRADES, "turnover_max": config.FLAG_TURNOVER_MAX,
                   "mdd_max": config.FLAG_MDD_MAX, "tim_min": config.FLAG_TIME_IN_MARKET_MIN,
                   "fee_bps": config.FEE_BPS, "slippage_bps": config.SLIPPAGE_BPS,
                   "rung3_cap_usd": config.RUNG3_SESSION_USD_CAP},
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/state")
def api_state(session: str = Query("latest")):
    try:
        return JSONResponse(build_state(session))
    except MidWrite as e:
        return JSONResponse({"ok": False, "busy": True, "file": str(e)}, status_code=503)


@app.get("/api/diff/{sha}")
def api_diff(sha: str):
    if not sha.replace("-", "").isalnum() or len(sha) > 40:
        raise HTTPException(400, "bad sha")
    if not _git("cat-file", "-e", f"{sha}^{{commit}}") and _git("rev-parse", "--verify", sha).strip() == "":
        raise HTTPException(404, "unknown commit")
    diff = _git("diff", f"{sha}^!", "--", "strategy.py")
    if not diff.strip():
        diff = _git("show", f"{sha}:strategy.py")
        diff = "(initial commit — full file)\n" + diff
    subject = _git("log", "-1", "--format=%s", sha).strip()
    return {"sha": sha, "subject": subject, "diff": diff}
