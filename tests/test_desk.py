import asyncio
import json
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
    assert {k: llm.usage[k] for k in ("input", "output", "cache_read", "calls")} == {"input": 10, "output": 5, "cache_read": 3, "calls": 1}
    assert llm.usage["cost_usd"] == pytest.approx((10 * 4 + 5 * 20 + 3 * 0.2) / 1e6)  # Opus 5.5 list prices


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


# ---------- ideas adopted from TradingAgents ----------
class StubLLM(NoLLM):
    """Answers by schema, records what each agent was asked."""

    def __init__(self, answers):
        super().__init__()
        self.available, self.answers, self.calls = True, answers, []

    async def structured(self, model, effort, system, user, schema):
        self.calls.append((schema.__name__, system, user))
        ans = self.answers.get(schema.__name__)
        return ans(system, user) if callable(ans) else ans


def test_bull_bear_debate_reaches_ceo_and_is_graded(tmp_path):
    from nexs.config import agents_cfg
    from nexs.desk import Argument, ArgumentList

    desk, _ = make_desk(tmp_path)
    desk.llm = StubLLM({"ArgumentList": lambda system, user: ArgumentList(arguments=[
        Argument(symbol="AAPL", argument="متفائل" if "المحلل المتفائل" in system else "متشائم", strength=0.8)])})
    desk.last_signals["technical"] = {"ts": time.time(), "signals": [Signal(symbol="AAPL", score=0.7, confidence=0.9, reason="up")]}
    run(desk.run_debate(["AAPL"], agents_cfg.get()))
    assert desk.debate["AAPL"]["bull"]["argument"] == "متفائل" and desk.debate["AAPL"]["bear"]["argument"] == "متشائم"
    scores = dict(desk.bus.db.execute("SELECT agent, score FROM signals").fetchall())
    assert scores == {"bull": 0.8, "bear": -0.8}
    assert {m["kind"] for m in desk.bus.recent()} == {"debate"}


def test_debate_skipped_when_no_analyst_has_a_view(tmp_path):
    from nexs.config import agents_cfg

    desk, _ = make_desk(tmp_path)
    desk.llm = StubLLM({})
    run(desk.run_debate(["AAPL"], agents_cfg.get()))
    assert desk.llm.calls == [] and desk.debate == {}


def test_graded_ceo_trades_become_lessons_for_the_ceo(tmp_path):
    desk, _ = make_desk(tmp_path)
    desk.closes["AAPL"] = [100.0]
    desk._record_signals("ceo", [Signal(symbol="AAPL", score=1, confidence=0.8, reason="breakout")])
    desk.bus.db.execute("UPDATE signals SET ts=?", (time.time() - 1000,))
    desk.closes["AAPL"] = [99.0]
    run(desk.audit({"use_llm": False}))
    [lesson] = desk.lessons()
    assert "breakout" in lesson and "-1.00%" in lesson and "wrong call" in lesson
    from nexs.config import agents_cfg
    from nexs.desk import DecisionList

    desk.llm = StubLLM({"DecisionList": DecisionList(decisions=[], summary="")})
    run(desk.ceo(["AAPL"], agents_cfg.get()))
    [(_, _, user)] = desk.llm.calls
    assert json.loads(user)["lessons_from_your_past_trades"] == [lesson]


def test_reflection_uses_claude_when_available(tmp_path):
    from nexs.desk import Lesson, LessonList

    desk, _ = make_desk(tmp_path)
    desk.closes["AAPL"] = [100.0]
    desk._record_signals("ceo", [Signal(symbol="AAPL", score=1, confidence=1, reason="x")])
    desk.bus.db.execute("UPDATE signals SET ts=?", (time.time() - 1000,))
    desk.closes["AAPL"] = [102.0]
    sid = desk.bus.db.execute("SELECT id FROM signals").fetchone()[0]
    desk.llm = StubLLM({"LessonList": LessonList(lessons=[Lesson(id=sid, lesson="قرار صحيح")])})
    run(desk.audit({"use_llm": True, "prompt": "p"}))
    assert desk.lessons() == ["قرار صحيح"]


def test_fundamentals_rules_and_per_agent_horizon(tmp_path, monkeypatch):
    from nexs import fundamentals as fund
    from nexs.config import agents_cfg

    monkeypatch.setattr(fund, "fetch", lambda s: {"recommendationMean": 1.8, "targetMeanPrice": 120, "currentPrice": 100})
    desk, _ = make_desk(tmp_path)
    [sig] = run(desk.fundamentals(["AAPL"], {"use_llm": False}))
    assert sig.score > 0.5 and "target upside +20%" in sig.reason
    desk._record_signals("fundamentals", [sig])
    horizon = desk.bus.db.execute("SELECT horizon FROM signals").fetchone()[0]
    assert horizon == agents_cfg.get()["agents"]["fundamentals"]["horizon"] == 86400


def test_fundamentals_survives_data_outage(tmp_path, monkeypatch):
    from nexs import fundamentals as fund

    def boom(s):
        raise ConnectionError("yahoo down")

    monkeypatch.setattr(fund, "fetch", boom)
    desk, _ = make_desk(tmp_path)
    assert run(desk.fundamentals(["AAPL"], {"use_llm": False})) == []


# ---------- Claude only during US market hours, with a cost counter ----------
def test_llm_window_follows_new_york_session():
    from datetime import datetime
    from nexs.market import ET, llm_window

    h = {"pre_open_minutes": 60, "holidays": ["2026-11-26"]}
    at = lambda *a: datetime(*a, tzinfo=ET)
    assert llm_window(h, at(2026, 10, 8, 9, 0))["phase"] == "pre_open"  # Thursday, 30 min before the open
    assert llm_window(h, at(2026, 10, 8, 12, 0)) == {"active": True, "phase": "open", "next_start": None}
    after = llm_window(h, at(2026, 10, 8, 16, 30))
    assert not after["active"] and after["next_start"] == at(2026, 10, 9, 8, 30).timestamp()
    assert llm_window(h, at(2026, 10, 10, 12, 0))["next_start"] == at(2026, 10, 12, 8, 30).timestamp()  # weekend
    assert llm_window(h, at(2026, 11, 25, 17, 0))["next_start"] == at(2026, 11, 27, 8, 30).timestamp()  # Thanksgiving
    assert llm_window({"market_only": False}, at(2026, 10, 10, 3, 0))["active"]


def test_closed_market_runs_ceo_on_rules_without_calling_claude(tmp_path):
    from nexs.config import agents_cfg

    desk, _ = make_desk(tmp_path)
    desk.llm = StubLLM({})
    desk.llm_gate = {"active": False, "phase": "closed", "next_start": None}
    res = run(desk.ceo(["AAPL"], agents_cfg.get()))
    assert desk.llm.calls == [] and res.decisions and "(rules)" in res.decisions[0].reason


def test_claude_cost_is_recorded_per_agent_and_daily_budget_stops_it(tmp_path):
    parsed = Signal(symbol="AAPL", score=0.5, confidence=0.7, reason="r")
    llm, _ = fake_llm(SimpleNamespace(stop_reason="end_turn", parsed_output=parsed, model="claude-sonnet-5-5",
                                      usage=SimpleNamespace(input_tokens=1_000_000, output_tokens=0,
                                                            cache_read_input_tokens=0, cache_creation_input_tokens=0)))
    desk = Desk(SimBroker(["AAPL"], seed=1), Bus(tmp_path / "c.db"), llm)
    run(desk._step("news", llm.structured("claude-opus-5-5", "low", "s", "u", Signal)))
    cost = desk.llm_costs()
    assert cost["today"] == cost["month"] == 2.0  # billed at the model that served it (Sonnet 5.5, $2/M input)
    assert cost["today_by_agent"] == {"news": 2.0}
    desk._update_llm_gate({"claude": {"market_only": False, "daily_budget_usd": 2}})
    assert desk.llm_gate["phase"] == "budget" and not desk._use_llm({"use_llm": True})
    desk._update_llm_gate({"claude": {"market_only": False, "daily_budget_usd": 0}})
    assert desk._use_llm({"use_llm": True})


# ---------- Claude studies only real trade candidates ----------
def _study_desk(tmp_path, monkeypatch):
    from nexs import fundamentals as fund
    from nexs.desk import DecisionList

    monkeypatch.setattr(fund, "fetch", lambda s: {})
    desk, broker = make_desk(tmp_path)

    async def no_headlines(s):
        return []

    broker.headlines = no_headlines
    desk.llm = StubLLM({"DecisionList": DecisionList(decisions=[Decision(symbol="AAPL", action="buy", qty=5,
                                                                         confidence=0.9, reason="studied")],
                                                     summary="buy AAPL"),
                        "ArgumentList": None})
    desk.last_signals["technical"] = {"ts": time.time(), "signals": [Signal(symbol="AAPL", score=1, confidence=1, reason="up")]}
    return desk


CFG_STUDY = {"market_only": False, "max_studies_per_day": 10, "study_cooldown_minutes": 30}


def test_trade_candidate_is_studied_once_then_cools_down(tmp_path, monkeypatch):
    from nexs.config import agents_cfg

    desk = _study_desk(tmp_path, monkeypatch)
    cfg = {**agents_cfg.get(), "claude": CFG_STUDY}
    res = run(desk.decide(["AAPL", "SPY"], cfg, RISK))
    [buy] = [d for d in res.decisions if d.action != "hold"]
    assert buy.reason == "studied" and desk.studies_today() == 1
    assert [s for s, _, _ in desk.llm.calls].count("DecisionList") == 1
    n = len(desk.llm.calls)
    res = run(desk.decide(["AAPL", "SPY"], cfg, RISK))  # same signal a cycle later: cooldown, no new spend
    assert len(desk.llm.calls) == n and all(d.action == "hold" for d in res.decisions)


def test_no_trade_without_study(tmp_path, monkeypatch):
    from nexs.config import agents_cfg

    desk = _study_desk(tmp_path, monkeypatch)
    cfg = {**agents_cfg.get(), "claude": CFG_STUDY}
    desk.llm_gate = {"active": False, "phase": "closed", "next_start": None}
    res = run(desk.decide(["AAPL"], cfg, RISK))
    assert desk.llm.calls == [] and all(d.action == "hold" for d in res.decisions)
    assert "not studied" in res.decisions[0].reason
    desk.llm_gate = {"active": True, "phase": "open", "next_start": None}
    res = run(desk.decide(["AAPL"], {**cfg, "claude": {**CFG_STUDY, "max_studies_per_day": 0}}, RISK))
    assert desk.llm.calls == [] and "daily study limit" in res.decisions[0].reason
    res = run(desk.decide(["AAPL"], cfg, {**RISK, "trading_enabled": False}))
    assert desk.llm.calls == []  # nothing could be executed, so no study is bought


def test_tradingview_alert_expires(tmp_path):
    from nexs.config import agents_cfg

    desk, _ = make_desk(tmp_path)
    assert desk.tv_alert({"symbol": "AAPL", "action": "buy"})["ok"]
    cfg = agents_cfg.get()
    desk.tv_alerts["AAPL"] = (time.time() - 3600, desk.tv_alerts["AAPL"][1])  # an hour old
    desk._refresh_tv(cfg)
    assert "tradingview" not in desk.last_signals and desk.tv_alerts == {}


# ---------- closed trades: wins and losses ----------
def test_stop_and_target_exits_are_booked_as_losses_and_wins(tmp_path):
    desk, broker = make_desk(tmp_path, ("AAPL", "MSFT", "SPY"))
    for s, px in (("AAPL", 100.0), ("MSFT", 200.0)):
        broker.hist[s].append(px)
        desk.closes[s] = [px]
    orders = [{"symbol": "AAPL", "action": "buy", "qty": 10, "price": 100.0, "stop": 99.0, "take": 102.0},
              {"symbol": "MSFT", "action": "buy", "qty": 5, "price": 200.0, "stop": 198.0, "take": 204.0}]
    run(desk.execute(orders, RISK))
    assert set(desk.book) == {"AAPL", "MSFT"}
    broker.hist["AAPL"].append(102.5)  # through the target
    broker.hist["MSFT"].append(197.0)  # through the stop
    broker.drift = {s: 0.0 for s in broker.drift}
    broker.rng.gauss = lambda mu, sigma: 0.0  # freeze the random walk so the next tick keeps these prices
    broker.tick()
    desk._settle()
    s = desk.trade_stats()
    assert (s["count"], s["wins"], s["losses"]) == (2, 1, 1) and desk.book == {}
    by = {t["symbol"]: t for t in s["recent"]}
    assert by["AAPL"]["reason"] == "take" and by["AAPL"]["pnl"] == pytest.approx(25.0)
    assert by["MSFT"]["reason"] == "stop" and by["MSFT"]["pnl"] == pytest.approx(-15.0)
    assert s["net"] == pytest.approx(10.0) and s["profit_factor"] == pytest.approx(25 / 15, abs=0.01)


# ---------- demo broker: real prices, virtual money ----------
def test_demo_broker_fills_in_session_hits_stop_in_bar_and_persists(tmp_path, monkeypatch):
    from nexs import broker as br

    t0 = 1_800_000_000.0
    bars = [(t0 + 60 * i, 100.0, 100.2, 99.8, 100.0) for i in range(5)]
    monkeypatch.setattr(br.DemoBroker, "_fetch", lambda self, sym: list(bars))
    monkeypatch.setattr(br, "is_open", lambda ts=None, holidays=(): False)
    b = br.DemoBroker(["AAPL"], path=tmp_path / "demo.json")
    r = run(b.bracket("AAPL", "buy", 10, 100.0, 99.0, 102.0))
    assert r["status"].startswith("rejected") and b.pos == {}  # market closed: no fill

    monkeypatch.setattr(br, "is_open", lambda ts=None, holidays=(): True)
    r = run(b.bracket("AAPL", "buy", 10, 100.0, 99.0, 102.0))
    assert r["status"] == "filled" and r["price"] == pytest.approx(100.05)  # 0.05% slippage
    assert b.cash == pytest.approx(100_000 - 1000.5 - 1.0)  # cost + $1 minimum commission

    b.checked["AAPL"] = t0 + 240
    bars.append((t0 + 300, 99.5, 99.6, 98.7, 98.9))  # this minute trades through the 99.00 stop
    b.fetched["AAPL"] = 0
    run(b.bars("AAPL"))
    assert b.pos == {} and b.exits[-1]["reason"] == "stop" and b.exits[-1]["price"] == 99.0

    again = br.DemoBroker(["AAPL"], path=tmp_path / "demo.json")  # a restart keeps the account
    assert again.cash == pytest.approx(b.cash) and again.pos == {}
