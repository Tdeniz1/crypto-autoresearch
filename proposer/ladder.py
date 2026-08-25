"""Three-rung fallback ladder.

Rules:
  * rungs are tried in configured order, starting from the highest rung that is
    not cooling down;
  * on a quota error (429 etc.) the current rung is retried RETRIES_PER_RUNG
    times with exponential backoff, then demoted;
  * non-quota errors are retried once, then demoted;
  * a demotion is logged with its reason; the first demotion of each rung in a
    session is logged at WARN level;
  * a demoted rung is retried after RUNG_COOLDOWN_SEC (rung 1 may recover);
  * rung 3 refuses to exceed the session dollar cap: BudgetExhausted propagates
    so the runner stops cleanly and writes the report;
  * if every rung fails, AllRungsFailed is raised — the runner counts these and
    stops cleanly after MAX_CONSECUTIVE_PROPOSER_FAILURES.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from harness import config
from harness.budget import BudgetExhausted, Ledger, estimate_cost_usd
from proposer import anthropic_api, claude_code, mock, openrouter
from proposer.errors import RungError, RungQuotaError, RungUnavailable
from proposer.result import RungResult

log = logging.getLogger("proposer")

RUNGS = {
    openrouter.NAME: openrouter,
    claude_code.NAME: claude_code,
    anthropic_api.NAME: anthropic_api,
    mock.NAME: mock,
}


class AllRungsFailed(RuntimeError):
    pass


@dataclass
class Demotion:
    ts: str
    from_rung: str
    to_rung: str | None
    reason: str


@dataclass
class Ladder:
    rungs: list[str] = field(default_factory=lambda: list(config.PROPOSER_RUNGS))
    ledger: Ledger | None = None
    retries: int = config.RETRIES_PER_RUNG
    backoff_base: float = config.BACKOFF_BASE_SEC
    cooldown: float = config.RUNG_COOLDOWN_SEC
    sleep = staticmethod(time.sleep)
    cooling: dict[str, float] = field(default_factory=dict)
    demotions: list[Demotion] = field(default_factory=list)
    warned: set[str] = field(default_factory=set)
    unavailable: dict[str, str] = field(default_factory=dict)

    # -- helpers -----------------------------------------------------------
    def active_rung(self, exclude: set[str] | None = None) -> str | None:
        now = time.time()
        for r in self.rungs:
            if r in self.unavailable or (exclude and r in exclude):
                continue
            if self.cooling.get(r, 0.0) > now:
                continue
            return r
        return None

    def _demote(self, rung: str, reason: str, sticky: bool) -> None:
        idx = self.rungs.index(rung)
        nxt = self.rungs[idx + 1] if idx + 1 < len(self.rungs) else None
        if sticky:
            self.unavailable[rung] = reason
        else:
            self.cooling[rung] = time.time() + self.cooldown
        d = Demotion(ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), from_rung=rung, to_rung=nxt, reason=reason)
        self.demotions.append(d)
        level = logging.WARNING if rung not in self.warned else logging.INFO
        self.warned.add(rung)
        log.log(level, "DEMOTION %s -> %s: %s", rung, nxt or "(none left)", reason)

    def _budget_guard(self, rung: str) -> None:
        if rung == anthropic_api.NAME and self.ledger is not None:
            # projected cost of one call at the configured model, ~8k in / 3k out worst case
            model = config.ANTHROPIC_MODEL
            proj = estimate_cost_usd(model, 8000, config.PROPOSER_MAX_OUTPUT_TOKENS // 2) or 0.0
            self.ledger.assert_rung3_budget(proj)

    # -- main --------------------------------------------------------------
    def propose(self, static_context: str, dynamic_context: str) -> tuple[RungResult, dict]:
        attempts: list[dict] = []
        tried: set[str] = set()   # a rung demoted during THIS call is never retried in the same call
        t0 = time.time()
        while True:
            rung = self.active_rung(exclude=tried)
            if rung is None:
                raise AllRungsFailed("every rung is unavailable or cooling down: " +
                                     "; ".join(f"{a['rung']}: {a['error']}" for a in attempts[-6:]))
            mod = RUNGS[rung]
            tried.add(rung)
            tries = 0
            while True:
                try:
                    self._budget_guard(rung)
                    res = mod.propose(static_context, dynamic_context, attempt=tries)
                    if rung == anthropic_api.NAME and self.ledger is not None and res.cost_usd:
                        # the call happened; if it blew the cap, stop the session after it is recorded
                        if self.ledger.rung3_spend() + res.cost_usd > self.ledger.rung3_cap_usd + 1e-9:
                            res.extra["budget_exhausted"] = True
                    usage = {
                        "rung": res.rung, "model": res.model, "attempts": attempts + [{"rung": rung, "ok": True, "tries": tries + 1}],
                        "demotions": [d.__dict__ for d in self.demotions], "seconds": time.time() - t0,
                    }
                    return res, usage
                except BudgetExhausted:
                    raise
                except RungUnavailable as e:
                    attempts.append({"rung": rung, "error": str(e)})
                    self._demote(rung, f"unavailable: {e}", sticky=True)
                    break
                except RungQuotaError as e:
                    attempts.append({"rung": rung, "error": str(e), "kind": "quota"})
                    if tries < self.retries:
                        delay = self.backoff_base * (2 ** tries)
                        log.info("rung %s quota error (%s); retry %d/%d in %.1fs", rung, str(e)[:80], tries + 1, self.retries, delay)
                        self.sleep(delay)
                        tries += 1
                        continue
                    self._demote(rung, f"quota after {self.retries} retries: {str(e)[:160]}", sticky=False)
                    break
                except RungError as e:
                    attempts.append({"rung": rung, "error": str(e), "kind": e.kind})
                    if tries < 1:
                        self.sleep(self.backoff_base)
                        tries += 1
                        continue
                    self._demote(rung, f"error: {str(e)[:160]}", sticky=False)
                    break
                except Exception as e:  # noqa: BLE001 — never crash the loop
                    attempts.append({"rung": rung, "error": f"{type(e).__name__}: {e}", "kind": "unexpected"})
                    self._demote(rung, f"unexpected {type(e).__name__}: {str(e)[:160]}", sticky=False)
                    break
