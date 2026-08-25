"""runner.py — the ratchet loop. Pure Python, no model calls of its own.

Each iteration: ask the proposer for a new strategy.py -> static scan ->
backtest on validation -> compare to best -> canaries on winners -> git commit
(kept) or git checkout (reverted) -> one record in logs/session.jsonl and one in
logs/ledger.jsonl -> print a throughput projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from harness import config
from harness.budget import BudgetExhausted, Ledger, append_jsonl, read_jsonl
from harness.canaries import LEAK_SUSPECTED, run_canaries
from harness.data import load_series, verify_frozen
from harness.determinism import check_determinism, seed_everything
from harness.engine import LookaheadError, FullSeriesReductionError, StrategyOutputError
from harness.evaluate import StrategyLoadError, evaluate_fn, load_strategy
from harness.metric import NEG_INF, deflated_sharpe
from harness.scan import ScanError, scan_source
from proposer import AllRungsFailed, Ladder, build_dynamic, build_static, summarize_log

log = logging.getLogger("runner")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_stop_at(hhmm: str) -> float:
    """Next local occurrence of HH:MM as an epoch timestamp."""
    h, m = (int(x) for x in hhmm.split(":"))
    now = datetime.now()
    target = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
    return target.timestamp()


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:10]


def fmt_metric(x) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-inf"
    return f"{x:+.4f}"


class Runner:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.session_id = a.session_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.stop_time = parse_stop_at(a.stop_at) if a.stop_at else None
        if a.stop_in:
            self.stop_time = time.time() + a.stop_in
        self.ledger = Ledger(session_id=self.session_id, rung3_cap_usd=a.rung3_cap)
        rungs = ["mock"] if a.proposer == "mock" else (a.rungs.split(",") if a.rungs else list(config.PROPOSER_RUNGS))
        self.ladder = Ladder(rungs=rungs, ledger=self.ledger)
        self.series = None
        self.best_metric = NEG_INF
        self.best_before = NEG_INF
        self.best_sha = None
        self.exp = 0
        self.records: list[dict] = []
        self.consecutive_failures = 0
        self.stop_reason = "max_iters"
        self.frozen_checksums = None
        self.start_sha = None

    # ------------------------------------------------------------ git ----
    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True)
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)} failed: {r.stderr.strip()}")
        return r.stdout.strip()

    # ------------------------------------------------------- startup -----
    def startup_checks(self) -> None:
        if os.environ.get(config.HOLDOUT_ENV_VAR):
            sys.exit(f"ABORT: {config.HOLDOUT_ENV_VAR} is set in the runner's environment. "
                     f"The loop must never be able to locate the holdout. Unset it and relaunch.")
        os.environ["CAR_RUNNER"] = "1"
        seed_everything(self.a.seed)
        self.frozen_checksums = verify_frozen()
        log.info("frozen data checksums OK (%d files)", len(self.frozen_checksums))
        self.series = load_series()
        log.info("series loaded: %s", ", ".join(f"{s.id}(val={len(s.val)})" for s in self.series))

        src = config.STRATEGY_FILE.read_text()
        ok, m1, m2 = check_determinism(src, self.series)
        if not ok:
            sys.exit(f"ABORT: determinism test FAILED on strategy.py ({m1!r} != {m2!r}). Refusing to start.")
        log.info("determinism test passed (metric %s bit-identical twice)", fmt_metric(m1))

        self.git("rev-parse", "--is-inside-work-tree")
        if self.git("status", "--porcelain", "--", "strategy.py"):
            self.git("add", "strategy.py")
            self.git("commit", "-q", "-m", f"chore: strategy.py state at session {self.session_id} start")
            log.info("committed dirty strategy.py before starting")
        self.start_sha = self.git("rev-parse", "--short", "HEAD")

    def baseline(self) -> None:
        src = config.STRATEGY_FILE.read_text()
        fn, desc = load_strategy(src, scan=False)
        ev = evaluate_fn(fn, self.series, "val", run_checks=False)
        self.best_metric = ev.metric
        self.best_sha = self.start_sha
        rec = self._record(status="baseline", kind="baseline", metric=ev.metric, ev=ev, description=desc,
                           source=src, rung=None, model=None, git_sha=self.start_sha, reason=None)
        self._save_curves(0, ev)
        log.info("baseline metric %s (%s)", fmt_metric(ev.metric), desc)

    # ------------------------------------------------------- records -----
    def _record(self, *, status: str, kind: str = "experiment", metric, ev, description, source,
                rung, model, git_sha, reason, train_metric=None, canaries=None,
                proposer_seconds=0.0, experiment_seconds=0.0, cost_usd=None, usage=None) -> dict:
        stats = ev.aggregate() if ev is not None else {}
        per = ev.per_series if ev is not None else {}
        n_trials = self.ledger.cumulative_experiments
        trial_sharpes = [r["stats"]["sharpe"] for r in self.all_records if r.get("stats")]
        if stats:
            trial_sharpes.append(stats["sharpe"])
        deflated = None
        if stats and metric is not None and math.isfinite(metric):
            deflated = deflated_sharpe(stats["sharpe"], stats["bars"], stats["bars_per_year"],
                                       max(n_trials, 1), trial_sharpes, stats["skew"], stats["kurt"])
        improvement = None
        if metric is not None and math.isfinite(metric) and math.isfinite(self.best_before):
            improvement = metric - self.best_before
        rec = {
            "ts": utcnow(), "session_id": self.session_id, "kind": kind, "exp": self.exp,
            "status": status, "reason": reason,
            "metric": None if metric is None or not math.isfinite(metric) else float(metric),
            "metric_is_neg_inf": bool(metric is not None and metric == NEG_INF),
            "train_metric": None if train_metric is None or not math.isfinite(train_metric) else float(train_metric),
            "best_before": None if not math.isfinite(self.best_before) else float(self.best_before),
            "best_after": None if not math.isfinite(self.best_metric) else float(self.best_metric),
            "improvement": improvement,
            "rung": rung, "model": model, "git_sha": git_sha,
            "description": description, "strategy_sha": short_hash(source) if source else None,
            "stats": stats, "per_series": per, "canaries": canaries, "deflated": deflated,
            "n_trials": n_trials,
            "proposer_seconds": proposer_seconds, "experiment_seconds": experiment_seconds,
            "cost_usd": cost_usd,
            "demotions": [d.__dict__ for d in self.ladder.demotions],
            "active_rung": self.ladder.active_rung(),
            "usage": usage,
        }
        append_jsonl(config.SESSION_LOG, rec)
        self.records.append(rec)
        return rec

    @property
    def all_records(self) -> list[dict]:
        if not hasattr(self, "_prior"):
            self._prior = [r for r in read_jsonl(config.SESSION_LOG) if r.get("kind") in ("experiment", "baseline")]
        return self._prior + self.records

    def _save_curves(self, exp: int, ev) -> None:
        config.CURVE_DIR.mkdir(parents=True, exist_ok=True)
        out = {"exp": exp, "series": {}}
        for sid, res in ev.results.items():
            n = len(res.equity)
            step = max(1, n // 1500)
            idx = np.arange(0, n, step)
            out["series"][sid] = {
                "bar": idx.tolist(),
                "equity": res.equity[idx].round(6).tolist(),
                "buy_hold": res.buy_hold[idx].round(6).tolist(),
                "bars_per_year": res.stats["bars_per_year"],
            }
        (config.CURVE_DIR / f"exp_{exp}.json").write_text(json.dumps(out, separators=(",", ":")))

    # ----------------------------------------------------- iteration -----
    def iteration(self) -> bool:
        """Returns False when the session must stop."""
        self.exp += 1
        self.best_before = self.best_metric
        t_start = time.time()
        program = config.PROGRAM_FILE.read_text()
        current = config.STRATEGY_FILE.read_text()
        summary = summarize_log(self.records, self.best_metric)

        # ---- propose ----
        try:
            res, usage = self.ladder.propose(build_static(program), build_dynamic(current, summary))
        except BudgetExhausted as e:
            self.stop_reason = f"rung3 budget cap: {e}"
            log.warning("STOP: %s", e)
            self._record(status="stopped_budget", kind="event", metric=None, ev=None, description="",
                         source=None, rung="anthropic", model=None, git_sha=None, reason=str(e))
            self.exp -= 1
            return False
        except AllRungsFailed as e:
            self.consecutive_failures += 1
            log.error("proposer failed (%d/%d consecutive): %s", self.consecutive_failures,
                      config.MAX_CONSECUTIVE_PROPOSER_FAILURES, e)
            self._record(status="proposer_failed", metric=None, ev=None, description="", source=None,
                         rung=None, model=None, git_sha=None, reason=str(e)[:400],
                         experiment_seconds=time.time() - t_start)
            if self.consecutive_failures >= config.MAX_CONSECUTIVE_PROPOSER_FAILURES:
                self.stop_reason = f"{self.consecutive_failures} consecutive proposer failures"
                return False
            self.ladder.sleep(min(300, 30 * self.consecutive_failures))
            return True
        self.consecutive_failures = 0
        proposer_seconds = usage["seconds"]
        rung, model = res.rung, res.model
        source = res.source
        cost = res.cost_usd

        # ---- write + scan + evaluate ----
        config.STRATEGY_FILE.write_text(source)
        status, reason, ev, train_metric, canaries, sha, desc = "reverted", None, None, None, None, None, ""
        try:
            scan_source(source)
            fn, desc = load_strategy(source, scan=False)
            ev = evaluate_fn(fn, self.series, "val", run_checks=True)
            try:
                train_metric = evaluate_fn(fn, self.series, "train", run_checks=False).metric
            except Exception as e:  # noqa: BLE001 — train metric is a diagnostic only
                log.info("train eval failed: %s", e)
            if ev.metric > self.best_metric:
                canaries = run_canaries(fn, [(s.id, s.val, s.bars_per_year) for s in self.series],
                                        ev.signals, ev.aggregate()["sharpe"], seed=self.a.seed)
                if LEAK_SUSPECTED in canaries["flags"] and config.REJECT_ON_LEAK:
                    status, reason = "rejected_leak", f"canary flags: {canaries['flags']}"
                else:
                    status = "kept"
        except ScanError as e:
            status, reason = "rejected_scan", str(e)[:600]
        except LookaheadError as e:
            status, reason = "rejected_lookahead", str(e)[:600]
        except FullSeriesReductionError as e:
            status, reason = "rejected_full_series", str(e)[:600]
        except (StrategyLoadError, StrategyOutputError) as e:
            status, reason = "rejected_runtime", str(e)[:600]
        except Exception as e:  # noqa: BLE001 — a crashing strategy is a failed experiment, not a crashed loop
            status, reason = "rejected_runtime", f"{type(e).__name__}: {str(e)[:500]}"

        metric = ev.metric if ev is not None else None
        if status == "kept":
            self.best_metric = metric
            self.git("add", "strategy.py")
            msg = f"exp {self.exp}: metric {metric:.4f} [{rung}/{model}] — {desc[:100]}"
            self.git("commit", "-q", "-m", msg)
            sha = self.git("rev-parse", "--short", "HEAD")
            self.best_sha = sha
            self._save_curves(self.exp, ev)
        else:
            self.git("checkout", "--", "strategy.py")
            if reason is None:
                reason = f"metric {fmt_metric(metric)} <= best {fmt_metric(self.best_metric)}"

        exp_seconds = time.time() - t_start
        self.ledger.record(exp=self.exp, rung=rung, model=model, input_tokens=res.input_tokens,
                           output_tokens=res.output_tokens, cost_usd=cost,
                           cache_read_tokens=res.cache_read_tokens, cache_write_tokens=res.cache_write_tokens,
                           proposer_seconds=proposer_seconds, experiment_seconds=exp_seconds, outcome=status)
        self._record(status=status, metric=metric, ev=ev, description=desc, source=source, rung=rung,
                     model=model, git_sha=sha, reason=reason, train_metric=train_metric, canaries=canaries,
                     proposer_seconds=proposer_seconds, experiment_seconds=exp_seconds, cost_usd=cost,
                     usage={k: usage.get(k) for k in ("attempts",)})

        # ---- projection ----
        p = self.ledger.projection(self.stop_time)
        spend = " ".join(f"{k}=${v:.3f}" for k, v in p["spend_by_rung"].items()) or "$0"
        eta = ""
        if p["seconds_remaining"] is not None:
            eta = f" | stop in {p['seconds_remaining'] / 3600:.1f}h → ~{p['experiments_that_fit']} more fit"
        flags = f" canaries={canaries['flags'] or 'clean'}" if canaries else ""
        log.info("[exp %d] %-16s metric=%s best=%s rung=%s/%s %.0fs (proposer %.0fs)%s | mean %.0fs/exp%s | spend %s r3cap=$%.2f%s",
                 self.exp, status.upper(), fmt_metric(metric), fmt_metric(self.best_metric), rung, model,
                 exp_seconds, proposer_seconds, flags, p["mean_seconds_per_experiment"] or 0, eta, spend,
                 p["rung3_cap_usd"], (" | " + reason[:120]) if reason and status != "kept" else "")

        verify_frozen()  # fail loudly if anything touched the frozen data mid-session
        if res.extra.get("budget_exhausted"):
            self.stop_reason = "rung3 budget cap reached"
            return False
        return True

    # ------------------------------------------------------------ run ----
    def run(self) -> None:
        self.startup_checks()
        self.baseline()
        try:
            while True:
                if self.a.max_iters is not None and self.exp >= self.a.max_iters:
                    self.stop_reason = "max_iters"
                    break
                if self.stop_time is not None and time.time() >= self.stop_time:
                    self.stop_reason = "wall-clock stop time"
                    break
                if not self.iteration():
                    break
        except KeyboardInterrupt:
            self.stop_reason = "interrupted"
            log.warning("interrupted — finishing cleanly")
        finally:
            self.finish()

    def finish(self) -> None:
        try:
            verify_frozen()
        except Exception as e:  # noqa: BLE001
            log.error("FROZEN DATA CHECK FAILED AT SESSION END: %s", e)
        n_kept = sum(1 for r in self.records if r["status"] == "kept")
        summary = {
            "ts": utcnow(), "session_id": self.session_id, "kind": "session_end", "exp": self.exp,
            "status": "session_end", "stop_reason": self.stop_reason, "experiments": self.exp,
            "kept": n_kept, "best_metric": None if not math.isfinite(self.best_metric) else self.best_metric,
            "best_sha": self.best_sha, "start_sha": self.start_sha,
            "demotions": [d.__dict__ for d in self.ladder.demotions],
            "spend_by_rung": self.ledger.spend_by_rung(),
            "stop_time": self.stop_time,
        }
        append_jsonl(config.SESSION_LOG, summary)
        log.info("session %s ended: %s — %d experiments, %d kept, best %s",
                 self.session_id, self.stop_reason, self.exp, n_kept, fmt_metric(self.best_metric))
        if not self.a.skip_report:
            try:
                from report import write_report
                path = write_report(self.session_id, llm=not self.a.no_llm_report)
                log.info("report written: %s", path)
            except Exception as e:  # noqa: BLE001
                log.error("report failed: %s", e)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-iters", type=int, default=None)
    ap.add_argument("--stop-at", default=None, help="local HH:MM wall-clock stop, e.g. 07:00")
    ap.add_argument("--stop-in", type=float, default=None, help="seconds from now (testing)")
    ap.add_argument("--proposer", choices=["ladder", "mock"], default="ladder")
    ap.add_argument("--rungs", default=None, help="comma list overriding config.PROPOSER_RUNGS")
    ap.add_argument("--rung3-cap", type=float, default=config.RUNG3_SESSION_USD_CAP)
    ap.add_argument("--session-id", default=None)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--skip-report", action="store_true")
    ap.add_argument("--no-llm-report", action="store_true", help="write the report without the narrative model call")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    Runner(a).run()


if __name__ == "__main__":
    main()
