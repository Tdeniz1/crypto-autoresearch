import pytest

from harness.budget import BudgetExhausted, Ledger
from proposer import ladder as ladder_mod
from proposer.errors import RungError, RungQuotaError
from proposer.ladder import AllRungsFailed, Ladder
from proposer.result import RungResult


class FakeRung:
    def __init__(self, name, behaviour):
        self.NAME = name
        self.behaviour = behaviour   # list of exceptions/results consumed in order, last repeats
        self.calls = 0

    def propose(self, static, dynamic, attempt=0):
        self.calls += 1
        b = self.behaviour[min(self.calls - 1, len(self.behaviour) - 1)]
        if isinstance(b, Exception):
            raise b
        return b


@pytest.fixture
def fake_rungs(monkeypatch):
    r1 = FakeRung("r1", [RungQuotaError("429 rate limit")])
    r2 = FakeRung("r2", [RungResult(source="def strategy(df): return df.close*0", rung="r2", model="m2")])
    r3 = FakeRung("r3", [RungResult(source="def strategy(df): return df.close*0", rung="r3", model="m3")])
    monkeypatch.setattr(ladder_mod, "RUNGS", {"r1": r1, "r2": r2, "r3": r3})
    return r1, r2, r3


def test_quota_retries_twice_then_demotes(fake_rungs, caplog):
    r1, r2, r3 = fake_rungs
    sleeps = []
    lad = Ladder(rungs=["r1", "r2", "r3"], retries=2, backoff_base=1.0, cooldown=999)
    lad.sleep = sleeps.append
    with caplog.at_level("WARNING"):
        res, usage = lad.propose("s", "d")
    assert res.rung == "r2"
    assert r1.calls == 3                      # initial + 2 retries
    assert sleeps == [1.0, 2.0]               # exponential backoff
    assert len(lad.demotions) == 1 and lad.demotions[0].from_rung == "r1" and lad.demotions[0].to_rung == "r2"
    assert any("DEMOTION r1 -> r2" in m for m in caplog.messages)
    # second call: r1 is cooling down, goes straight to r2, no second WARN
    res2, _ = lad.propose("s", "d")
    assert res2.rung == "r2" and r1.calls == 3


def test_all_rungs_fail(fake_rungs):
    r1, r2, r3 = fake_rungs
    r2.behaviour = [RungError("boom")]
    r3.behaviour = [RungError("boom3")]
    lad = Ladder(rungs=["r1", "r2", "r3"], retries=2, backoff_base=0.0, cooldown=999)
    lad.sleep = lambda s: None
    with pytest.raises(AllRungsFailed):
        lad.propose("s", "d")
    assert [d.from_rung for d in lad.demotions] == ["r1", "r2", "r3"]


def test_rung3_budget_cap_stops_cleanly(tmp_path, monkeypatch):
    fake = FakeRung("anthropic", [RungResult(source="x", rung="anthropic", model="claude-haiku-4-5", cost_usd=0.5)])
    monkeypatch.setattr(ladder_mod, "RUNGS", {"anthropic": fake})
    led = Ledger(path=tmp_path / "ledger.jsonl", session_id="t", rung3_cap_usd=0.60)
    lad = Ladder(rungs=["anthropic"], ledger=led)
    res, _ = lad.propose("s", "d")
    assert not res.extra.get("budget_exhausted")
    led.record(exp=1, rung="anthropic", model="claude-haiku-4-5", input_tokens=1, output_tokens=1, cost_usd=0.5,
               proposer_seconds=1, experiment_seconds=1, outcome="kept")
    res2, _ = lad.propose("s", "d")   # pre-call projection ($0.02) fits; the actual $0.50 blows the cap
    assert res2.extra.get("budget_exhausted") is True  # runner stops cleanly after recording it
    led.record(exp=2, rung="anthropic", model="claude-haiku-4-5", input_tokens=1, output_tokens=1, cost_usd=0.5,
               proposer_seconds=1, experiment_seconds=1, outcome="kept")
    with pytest.raises(BudgetExhausted):
        lad.propose("s", "d")   # spend $1.00 > cap: refused BEFORE any call is made
    assert fake.calls == 2


def test_cooldown_expires(fake_rungs):
    r1, r2, r3 = fake_rungs
    lad = Ladder(rungs=["r1", "r2", "r3"], retries=0, backoff_base=0.0, cooldown=0.0)
    lad.sleep = lambda s: None
    lad.propose("s", "d")
    r1.behaviour = [RungResult(source="x", rung="r1", model="m1")]
    res, _ = lad.propose("s", "d")
    assert res.rung == "r1"  # cooldown of 0s => rung 1 retried and recovered
