"""Morning report: reads session.jsonl + ledger.jsonl, writes reports/YYYY-MM-DD.md.

The only place a strong model is worth using: an optional narrative section,
model set by config.REPORT_MODEL (independent of the proposer).
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone

from analyze_providers import attribution, render_table
from harness import config
from harness.budget import read_jsonl


def _git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def _fmt(x, spec="{:+.4f}"):
    return "-inf" if x is None or (isinstance(x, float) and not math.isfinite(x)) else spec.format(x)


def session_records(session_id: str | None) -> tuple[str | None, list[dict], list[dict]]:
    recs = read_jsonl(config.SESSION_LOG)
    led = read_jsonl(config.LEDGER_LOG)
    if not recs:
        return None, [], []
    if session_id in (None, "latest"):
        session_id = recs[-1]["session_id"]
    return session_id, [r for r in recs if r.get("session_id") == session_id], \
        [r for r in led if r.get("session_id") == session_id]


def build_markdown(session_id: str, recs: list[dict], led: list[dict]) -> str:
    exps = [r for r in recs if r.get("kind") == "experiment"]
    end = next((r for r in recs if r.get("kind") == "session_end"), None)
    base = next((r for r in recs if r.get("kind") == "baseline"), None)
    kept = [r for r in exps if r["status"] == "kept"]
    statuses = Counter(r["status"] for r in exps)
    start_ts, end_ts = (recs[0]["ts"], recs[-1]["ts"]) if recs else ("?", "?")
    best = max(kept, key=lambda r: r["metric"], default=None)
    best_metric = best["metric"] if best else (base["metric"] if base else None)
    best_sha = best["git_sha"] if best else (base["git_sha"] if base else None)
    start_sha = base["git_sha"] if base else None

    # throughput
    secs = [r.get("experiment_seconds") or 0 for r in exps]
    mean_s = sum(secs) / len(secs) if secs else 0.0
    try:
        t0 = datetime.fromisoformat(start_ts); t1 = datetime.fromisoformat(end_ts)
        elapsed_h = max((t1 - t0).total_seconds(), 1) / 3600
    except ValueError:
        elapsed_h = 0.0
    per_hour = len(exps) / elapsed_h if elapsed_h else 0.0

    # spend
    spend = Counter()
    for r in led:
        spend[r["rung"]] += r.get("cost_usd") or 0.0
    demotions = (end or {}).get("demotions") or (exps[-1].get("demotions") if exps else []) or []

    diff = ""
    if best_sha and start_sha and best_sha != start_sha:
        diff = _git("diff", f"{start_sha}..{best_sha}", "--", "strategy.py")

    lines = [
        f"# Research session report — {session_id}",
        "",
        f"- Window: {start_ts} → {end_ts} (UTC), elapsed ≈ {elapsed_h:.1f} h",
        f"- Stop reason: **{(end or {}).get('stop_reason', 'unknown')}**",
        f"- Experiments run: **{len(exps)}**   kept: **{len(kept)}**   keep rate: **{(len(kept) / len(exps)) if exps else 0:.0%}**",
        f"- Outcomes: " + ", ".join(f"{k}={v}" for k, v in sorted(statuses.items())),
        f"- Baseline metric: {_fmt(base['metric'] if base else None)} → best metric: **{_fmt(best_metric)}** "
        f"(commit `{best_sha}`)",
        f"- Throughput: mean {mean_s:.0f} s/experiment, ≈ {per_hour:.1f} experiments/hour",
        "",
        "## Spend by rung",
        "",
        "| rung | calls | spend |", "|---|---:|---:|",
    ]
    calls = Counter(r["rung"] for r in led)
    for rung in sorted(set(list(calls) + list(spend))):
        lines.append(f"| {rung} | {calls[rung]} | ${spend[rung]:.4f} |")
    lines += [f"| **total** | {sum(calls.values())} | **${sum(spend.values()):.4f}** |",
              f"\nRung 3 cap: ${config.RUNG3_SESSION_USD_CAP:.2f}", ""]

    lines += ["## Demotions", ""]
    if demotions:
        lines += [f"- {d['ts']}: {d['from_rung']} → {d['to_rung'] or '(none)'} — {d['reason']}" for d in demotions]
    else:
        lines.append("- none")

    lines += ["", "## Provider attribution", "", render_table(attribution(exps)), ""]

    lines += ["## Kept experiments", ""]
    if kept:
        lines += ["| exp | metric | Δ | sharpe | mdd | trades | canaries | rung/model | sha | idea |",
                  "|---:|---:|---:|---:|---:|---:|---|---|---|---|"]
        for r in kept:
            s = r.get("stats") or {}
            c = (r.get("canaries") or {}).get("flags") or []
            lines.append(f"| {r['exp']} | {_fmt(r['metric'])} | {_fmt(r.get('improvement'))} | {s.get('sharpe', 0):+.2f} | "
                         f"{s.get('max_drawdown', 0):.0%} | {s.get('trades', 0)} | {', '.join(c) or 'clean'} | "
                         f"{r['rung']}/{r['model']} | `{r['git_sha']}` | {(r.get('description') or '')[:80]} |")
    else:
        lines.append("- none kept this session")

    if best:
        s = best.get("stats") or {}
        d = best.get("deflated") or {}
        lines += ["", "## Best strategy", "",
                  f"- exp {best['exp']}: {best.get('description', '')}",
                  f"- validation metric {_fmt(best['metric'])}, train metric {_fmt(best.get('train_metric'))}",
                  f"- Sharpe (net) {s.get('sharpe', 0):+.2f}, deflated {_fmt(d.get('deflated'), '{:+.2f}')} "
                  f"(DSR p={d.get('dsr_prob', 0):.2f}, {d.get('n_trials', 0)} trials), max DD {s.get('max_drawdown', 0):.0%}, "
                  f"trades {s.get('trades', 0)}, win rate {s.get('win_rate', 0):.0%} / payoff {s.get('payoff_ratio', 0):.2f}, "
                  f"time in market {s.get('time_in_market', 0):.0%}, turnover {s.get('turnover_annual', 0):.0f}/yr",
                  f"- vs buy & hold (validation): strategy {s.get('total_return', 0):+.1%} vs B&H {s.get('buy_hold_return', 0):+.1%}",
                  f"- canaries: {best.get('canaries')}",
                  "", "### Diff that produced it", "", "```diff", diff.strip() or "(no diff: best is the session baseline)", "```"]

    lines += ["", "## Rejections (last 10)", ""]
    rej = [r for r in exps if r["status"].startswith("rejected") or r["status"] == "proposer_failed"][-10:]
    lines += [f"- exp {r['exp']} {r['status']} [{r.get('rung')}]: {(r.get('reason') or '')[:200]}" for r in rej] or ["- none"]
    lines += ["", "> Leaderboard rank is validation performance. It is not evidence until the holdout is run "
              "(scripts/run_holdout.py, by hand, never by the loop).", ""]
    return "\n".join(lines)


def narrative(markdown: str, model: str = config.REPORT_MODEL) -> str | None:
    """Optional narrative from a strong model. Never required; failures are swallowed."""
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
        client = anthropic.Anthropic()
        prompt = ("You are reviewing an overnight automated trading-strategy research session. Below is the "
                  "machine-generated report. Write a concise (<= 250 words) analyst note: what worked, what the "
                  "rejections and canary flags suggest, whether the improvements look real or like validation "
                  "overfitting (use the deflated Sharpe and trial count), and 3 concrete directions for the next "
                  "session. Do not mention dates, assets or market events.\n\n" + markdown)
        kwargs = dict(model=model, max_tokens=config.REPORT_MAX_TOKENS,
                      messages=[{"role": "user", "content": prompt}])
        if model.startswith(("claude-fable-5", "claude-opus-5")):
            try:
                resp = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"],
                                                   fallbacks="default", **kwargs)
            except Exception:  # noqa: BLE001 — fall back to the plain endpoint
                resp = client.messages.create(**kwargs)
        else:
            resp = client.messages.create(**kwargs)
        if resp.stop_reason == "refusal":
            return None
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:  # noqa: BLE001
        return f"_(narrative unavailable: {type(e).__name__}: {str(e)[:120]})_"


def write_report(session_id: str | None = None, llm: bool = True) -> str:
    sid, recs, led = session_records(session_id)
    if sid is None:
        raise SystemExit("no session records in logs/session.jsonl")
    md = build_markdown(sid, recs, led)
    if llm:
        note = narrative(md)
        if note:
            md += f"\n## Analyst note ({config.REPORT_MODEL})\n\n{note}\n"
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = config.REPORT_DIR / f"{day}.md"
    if path.exists() and sid not in path.read_text():
        k = 2
        while (config.REPORT_DIR / f"{day}_{k}.md").exists() and sid not in (config.REPORT_DIR / f"{day}_{k}.md").read_text():
            k += 1
        path = config.REPORT_DIR / f"{day}_{k}.md"
    path.write_text(md)
    return str(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="latest")
    ap.add_argument("--no-llm", action="store_true")
    a = ap.parse_args()
    print(write_report(a.session, llm=not a.no_llm))


if __name__ == "__main__":
    main()
