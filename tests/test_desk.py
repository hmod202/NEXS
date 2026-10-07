import asyncio
import time
from types import SimpleNamespace

import anthropic
import pytest

from nexs.broker import SimBroker
from nexs.bus import Bus
from nexs.desk import Decision, Desk, Signal
from nexs.llm import LLM

RISK = {"trading_enabled": True, "max_position_value": 2000, "max_total_exposure": 8000, "max_orders_per_hour": 10,
        "daily_loss_limit": 300, "min_confidence": 0.6, "stop_loss_pct": 1.0, "take_profit_pct": 2.0,
        "allow_short": False}


class NoLLM(LLM):
    def __init__(self):
        super().__init__()
        self.available, self.status = False, "off"


def make_desk(tmp_path, symbols=("AAPL", "SPY")):
    broker = SimBroker(list(symbols), seed=1)
    desk = Desk(broker, Bus(tmp_path / "t.db"), NoLLM())
    desk.closes = {s: list(broker.hist[s]) for s in symbols}
    desk.account = {"equity": 100_000, "daily_pnl": 0}
    return desk, broker


def run(coro):
    return asyncio.run(coro)


def test_risk_sizes_order_to_position_limit_with_stop_and_target(tmp_path):
    desk, broker = make_desk(tmp_path)
    desk.closes["AAPL"] = [100.0]
    [o] = run(desk.risk_check([Decision(symbol="AAPL", action="buy", qty=999, confidence=0.9, reason="x")], ["AAPL"], RISK))
    assert o["qty"] == 20  # $2000 / $100
    assert o["stop"] == 99.0 and o["take"] == 102.0


def test_risk_rejects_low_confidence_unlisted_and_naked_short(tmp_path):
    desk, _ = make_desk(tmp_path)
    ds = [Decision(symbol="AAPL", action="buy", qty=1, confidence=0.3, reason="weak"),
          Decision(symbol="ZZZ", action="buy", qty=1, confidence=0.9, reason="unlisted"),
          Decision(symbol="AAPL", action="sell", qty=1, confidence=0.9, reason="short")]
    assert run(desk.risk_check(ds, ["AAPL"], RISK)) == []


def test_sell_closes_existing_long(tmp_path):
    desk, broker = make_desk(tmp_path)
    px = desk.closes["AAPL"][-1]
    run(broker.bracket("AAPL", "buy", 5, px, px * 0.5, px * 2))
    [o] = run(desk.risk_check([Decision(symbol="AAPL", action="sell", qty=1, confidence=0.9, reason="exit")], ["AAPL"], RISK))
    assert o == {"symbol": "AAPL", "action": "close", "qty": 5, "price": px}
    run(desk.execute([o], RISK))
    assert broker.positions() == {}


def test_daily_loss_limit_halts_trading(tmp_path):
    desk, _ = make_desk(tmp_path)
    desk.account["daily_pnl"] = -500
    out = run(desk.risk_check([Decision(symbol="AAPL", action="buy", qty=1, confidence=0.9, reason="x")], ["AAPL"], RISK))
    assert out == [] and desk.halted


def test_auditor_grades_signals_and_rewards_good_agents(tmp_path):
    desk, _ = make_desk(tmp_path)
    desk.closes["AAPL"] = [100.0]
    for _ in range(12):
        desk._record_signals("good", [Signal(symbol="AAPL", score=1, confidence=1, reason="")])
        desk._record_signals("bad", [Signal(symbol="AAPL", score=-1, confidence=1, reason="")])
    desk.bus.db.execute("UPDATE signals SET ts=?", (time.time() - 1000,))
    desk.closes["AAPL"] = [101.0]  # price rose 1%
    board = {r["agent"]: r for r in run(desk.audit())}
    assert board["good"]["hit_rate"] == 1 and board["bad"]["hit_rate"] == 0
    mult = desk.reward_multipliers()
    assert mult["good"] > 1 > mult["bad"]


def test_full_cycles_on_simulator_without_api_key(tmp_path):
    from nexs.config import agents_cfg

    desk, broker = make_desk(tmp_path, agents_cfg.get()["watchlist"] + ["SPY", "QQQ"])
    for _ in range(40):
        broker.tick()

    async def go():
        for _ in range(3):
            await desk.run_cycle()

    run(go())
    kinds = {m["kind"] for m in desk.bus.recent(500)}
    assert {"focus", "signals", "decisions"} <= kinds
    assert all(s["status"] != "error" for s in desk.state.values())


# ---------- Claude wrapper ----------
class FakeMessages:
    def __init__(self, resp=None, exc=None):
        self.resp, self.exc, self.kwargs = resp, exc, None

    async def parse(self, **kw):
        self.kwargs = kw
        if self.exc:
            raise self.exc
        return self.resp


def fake_llm(resp=None, exc=None):
    llm = LLM()
    msgs = FakeMessages(resp, exc)
    llm.client = SimpleNamespace(beta=SimpleNamespace(messages=msgs))
    return llm, msgs


USAGE = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3)


def test_llm_returns_parsed_output_and_tracks_usage():
    parsed = Signal(symbol="AAPL", score=0.5, confidence=0.7, reason="r")
    llm, msgs = fake_llm(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, usage=USAGE))
    assert run(llm.structured("claude-opus-5-5", "low", "sys", "u", Signal)) == parsed
    assert msgs.kwargs["fallbacks"] == "default" and msgs.kwargs["output_config"] == {"effort": "low"}
    assert msgs.kwargs["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert llm.usage == {"input": 10, "output": 5, "cache_read": 3, "calls": 1}


def test_llm_haiku_omits_effort_and_fallbacks():
    llm, msgs = fake_llm(SimpleNamespace(stop_reason="end_turn", parsed_output=None, usage=USAGE))
    run(llm.structured("claude-haiku-4-5", "low", "s", "u", Signal))
    assert "output_config" not in msgs.kwargs and "fallbacks" not in msgs.kwargs


def test_llm_refusal_returns_none():
    llm, _ = fake_llm(SimpleNamespace(stop_reason="refusal", parsed_output=None, usage=USAGE))
    assert run(llm.structured("claude-opus-5-5", "low", "s", "u", Signal)) is None


def test_llm_auth_failure_switches_to_rules():
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.AuthenticationError("no key", response=httpx2.Response(401, request=req), body=None)
    for exc in (err, TypeError("Could not resolve authentication method")):
        llm, _ = fake_llm(exc=exc)
        assert run(llm.structured("claude-opus-5-5", "low", "s", "u", Signal)) is None
        assert llm.available is False


def test_ibkr_refuses_live_account_orders_without_explicit_opt_in(monkeypatch):
    from nexs.broker import IBKRBroker

    b = IBKRBroker("127.0.0.1", 4002, 1)
    b.mode = "live"
    monkeypatch.delenv("NEXS_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="NEXS_ALLOW_LIVE"):
        run(b.bracket("AAPL", "buy", 1, 100, 99, 102))
