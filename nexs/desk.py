"""The trading desk: one CEO agent orchestrating analyst, risk, execution and auditor agents each cycle."""
import asyncio
import json
import logging
import math
import time
from collections import deque
from typing import Literal

from pydantic import BaseModel

from .bus import Bus
from . import fundamentals as fund
from . import market
from .config import agents_cfg, risk_cfg
from .indicators import clamp, pct_change, rsi, sma
from .llm import LLM, current_agent

log = logging.getLogger("nexs.desk")

BUILTIN = {"ceo", "scanner", "technical", "news", "macro", "fundamentals", "bull", "bear", "risk", "execution", "auditor",
           "tradingview"}
SIGNAL_HORIZON_S = 300  # signals are graded on the price move over the next 5 minutes
POSITIVE = ("beat", "upgrade", "launch", "record", "surge", "raises", "approval", "partnership")
NEGATIVE = ("miss", "downgrade", "cut", "probe", "lawsuit", "recall", "plunge", "investigation", "layoff")


class Signal(BaseModel):
    symbol: str
    score: float  # -1 bearish .. 1 bullish
    confidence: float  # 0 .. 1
    reason: str


class SignalList(BaseModel):
    signals: list[Signal]


class MarketView(BaseModel):
    score: float
    confidence: float
    reason: str


class Argument(BaseModel):
    symbol: str
    argument: str
    strength: float  # 0 .. 1, how convincing the case is


class ArgumentList(BaseModel):
    arguments: list[Argument]


class Lesson(BaseModel):
    id: int
    lesson: str


class LessonList(BaseModel):
    lessons: list[Lesson]


class Decision(BaseModel):
    symbol: str
    action: Literal["buy", "sell", "hold"]
    qty: int
    confidence: float
    reason: str


class DecisionList(BaseModel):
    decisions: list[Decision]
    summary: str


class Desk:
    def __init__(self, broker, bus: Bus, llm: LLM):
        self.broker, self.bus, self.llm = broker, bus, llm
        self.cycle = 0
        self.halted, self.halt_reason = False, ""
        self.order_times: deque[float] = deque()
        self.state: dict[str, dict] = {}
        self.last_signals: dict[str, dict] = {}  # agent -> {"ts", "signals": [Signal]}
        self.closes: dict[str, list[float]] = {}
        self.account = {"equity": 0.0, "daily_pnl": 0.0}
        self.wake = asyncio.Event()
        self.debate: dict[str, dict] = {}
        self.studied_at: dict[str, float] = {}  # symbol -> last Claude study, for the cooldown
        self.tv_alerts: dict[str, tuple[float, Signal]] = {}  # symbol -> latest TradingView alert
        self.book: dict[str, dict] = {}  # open trades: symbol -> {side, qty, entry, ts}; settled into closed_trades
        self.llm.on_usage = self._record_usage
        # Claude only runs inside this gate (market hours + daily budget); refreshed at the start of every cycle.
        self.llm_gate = {"active": True, "phase": "always", "next_start": None}

    # ---------- plumbing ----------
    def send(self, src: str, dst: str, kind: str, payload) -> None:
        self.bus.publish(self.cycle, src, dst, kind, payload)

    def _active(self, cfg: dict, agent: str) -> bool:
        a = cfg["agents"].get(agent) or {}
        return bool(a.get("enabled")) and self.cycle % max(1, int(a.get("every", 1))) == 0

    def _use_llm(self, a: dict, default: bool = False) -> bool:
        return bool(a.get("use_llm", default)) and self.llm.available and self.llm_gate["active"]

    def _update_llm_gate(self, cfg: dict) -> None:
        c = cfg.get("claude") or {}
        gate = market.llm_window(c)
        budget = float(c.get("daily_budget_usd") or 0)
        if gate["active"] and budget and self.llm_costs()["today"] >= budget:
            gate = {"active": False, "phase": "budget", "next_start": None}
        if gate["phase"] != self.llm_gate["phase"]:
            log.info("Claude agents: %s", gate["phase"])
        self.llm_gate = gate

    async def _step(self, agent: str, coro):
        self.state[agent] = {**self.state.get(agent, {}), "status": "running", "since": time.time()}
        self.bus.broadcast({"type": "agent", "id": agent, "status": "running"})
        t0 = time.perf_counter()
        token = current_agent.set(agent)
        try:
            result = await coro
            status = "idle"
        except Exception as e:  # one agent failing must never stop the desk
            log.exception("agent %s failed", agent)
            self.send(agent, "ceo", "error", {"error": str(e)})
            result, status = None, "error"
        finally:
            current_agent.reset(token)
        ms = round((time.perf_counter() - t0) * 1000)
        self.state[agent] = {"status": status, "last_ms": ms, "last_run": time.time(), "cycle": self.cycle}
        self.bus.broadcast({"type": "agent", "id": agent, "status": status, "last_ms": ms})
        return result

    async def run_forever(self) -> None:
        while True:
            try:
                await self.run_cycle()
            except Exception:
                log.exception("cycle %s failed", self.cycle)
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=float(agents_cfg.get().get("cycle_seconds", 30)))
            except asyncio.TimeoutError:
                pass

    # ---------- one cycle ----------
    async def run_cycle(self) -> None:
        cfg, risk = agents_cfg.get(), risk_cfg.get()
        self.cycle += 1
        self._update_llm_gate(cfg)
        watch = list(dict.fromkeys(cfg.get("watchlist", [])))
        proxies = cfg.get("market_proxies", ["SPY"])
        symbols = list(dict.fromkeys(watch + proxies))
        await asyncio.wait_for(self.broker.ensure_connected(), 30)
        closes = await asyncio.wait_for(  # a hung data request must not freeze the desk
            asyncio.gather(*(self.broker.bars(s, 60) for s in symbols), return_exceptions=True), 30)
        self.closes = {s: c for s, c in zip(symbols, closes) if isinstance(c, list) and c}
        self.account = await asyncio.wait_for(self.broker.account(), 15)
        self._settle()
        self.bus.broadcast({"type": "cycle", "cycle": self.cycle})

        focus = watch
        if self._active(cfg, "scanner"):
            focus = await self._step("scanner", self.scanner(watch, cfg["agents"]["scanner"])) or watch
            for dst in ("technical", "news"):
                self.send("scanner", dst, "focus", {"symbols": focus})
        self._refresh_tv(cfg)
        focus = list(dict.fromkeys(focus + [s for s in self.tv_alerts if s in watch]))  # alerted symbols always get a look

        # The every-cycle scan is rules only (free). Claude is spent in study(), on trade candidates.
        rules = lambda agent: {**cfg["agents"].get(agent, {}), "use_llm": False}
        analysts = {"technical": self.technical(focus), "news": self.news(focus, rules("news")),
                    "macro": self.macro(proxies, rules("macro")),
                    "fundamentals": self.fundamentals(focus, rules("fundamentals"))}
        for agent, a in cfg["agents"].items():
            if agent not in BUILTIN:
                analysts[agent] = self.custom(focus, rules(agent))  # custom agents have no rules: study only
        runs = {a: c for a, c in analysts.items() if self._active(cfg, a)}
        for a, c in analysts.items():
            if a not in runs:
                c.close()  # never awaited; avoid "coroutine was never awaited" warnings
        results = await asyncio.gather(*(self._step(a, c) for a, c in runs.items()))
        for agent, sigs in zip(runs, results):
            if sigs is None:
                continue
            self.last_signals[agent] = {"ts": time.time(), "signals": sigs}
            self._record_signals(agent, sigs)
            self.send(agent, "ceo", "signals", [s.model_dump() for s in sigs])

        if self._active(cfg, "auditor"):
            board = await self._step("auditor", self.audit(cfg["agents"]["auditor"]))
            if board:
                self.send("auditor", "ceo", "leaderboard", board)

        if not self._active(cfg, "ceo"):
            return
        decisions = await self._step("ceo", self.decide(focus, cfg, risk))
        if not decisions:
            return
        self._record_signals("ceo", [Signal(symbol=d.symbol, score={"buy": 1, "sell": -1}[d.action],
                                            confidence=d.confidence, reason=d.reason)
                                     for d in decisions.decisions if d.action != "hold"])
        self.send("ceo", "risk", "decisions", decisions.model_dump())

        approved = await self._step("risk", self.risk_check(decisions.decisions, watch, risk))
        if approved and self._active(cfg, "execution"):
            fills = await self._step("execution", self.execute(approved, risk))
            if fills:
                self.send("execution", "auditor", "fills", fills)

    # ---------- agents ----------
    async def scanner(self, watch: list[str], a: dict) -> list[str]:
        ranked = sorted((s for s in watch if s in self.closes),
                        key=lambda s: abs(pct_change(self.closes[s], 15)), reverse=True)
        held = [s for s in self.broker.positions() if s in watch]  # always keep watching open positions
        return list(dict.fromkeys(held + ranked[: int(a.get("top_n", 4))]))

    async def technical(self, focus: list[str]) -> list[Signal]:
        out = []
        for s in focus:
            c = self.closes.get(s)
            if not c or len(c) < 31:
                continue
            trend = (sma(c, 10) / sma(c, 30) - 1) * 100
            mom, r = pct_change(c, 5), rsi(c)
            score = 0.6 * math.tanh(trend * 5) + 0.4 * math.tanh(mom * 3)
            if r > 75:
                score -= 0.3
            elif r < 25:
                score += 0.3
            score = clamp(score)
            out.append(Signal(symbol=s, score=round(score, 3), confidence=round(min(1.0, 0.3 + abs(score)), 2),
                              reason=f"trend {trend:+.2f}% mom5 {mom:+.2f}% RSI {r:.0f}"))
        return out

    async def news(self, focus: list[str], a: dict) -> list[Signal]:
        heads = dict(zip(focus, await asyncio.gather(*(self.broker.headlines(s) for s in focus))))
        heads = {s: h for s, h in heads.items() if h}
        if not heads:
            return []  # nothing to read: skip the model call entirely
        if self._use_llm(a):
            res = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                            json.dumps(heads, ensure_ascii=False), SignalList)
            if res:
                return [Signal(symbol=x.symbol, score=clamp(x.score), confidence=clamp(x.confidence, 0, 1),
                               reason=x.reason) for x in res.signals if x.symbol in heads]
        out = []
        for s, h in heads.items():
            text = " ".join(h).lower()
            score = clamp(0.4 * (sum(w in text for w in POSITIVE) - sum(w in text for w in NEGATIVE)))
            out.append(Signal(symbol=s, score=score, confidence=0.5 if score else 0.2, reason="keywords: " + h[-1][:80]))
        return out

    async def macro(self, proxies: list[str], a: dict) -> list[Signal]:
        moves = {p: {f"{n}m": round(pct_change(self.closes[p], n), 3) for n in (5, 15, 30)}
                 for p in proxies if p in self.closes}
        if not moves:
            return []
        view = None
        if self._use_llm(a):
            view = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                             json.dumps(moves), MarketView)
        if view is None:
            avg = sum(m["15m"] for m in moves.values()) / len(moves)
            view = MarketView(score=math.tanh(avg * 2), confidence=0.4, reason=f"avg 15m move {avg:+.2f}%")
        anchor = proxies[0]
        return [Signal(symbol=anchor, score=clamp(view.score), confidence=clamp(view.confidence, 0, 1), reason=view.reason)]

    async def fundamentals(self, focus: list[str], a: dict) -> list[Signal]:
        got = await asyncio.wait_for(asyncio.gather(*(asyncio.to_thread(fund.fetch, s) for s in focus),
                                                    return_exceptions=True), 30)
        data = {s: d for s, d in zip(focus, got) if isinstance(d, dict) and d}
        if not data:
            return []
        if self._use_llm(a) and a.get("prompt"):
            res = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                            json.dumps(data, ensure_ascii=False), SignalList)
            if res:
                return [Signal(symbol=x.symbol, score=clamp(x.score), confidence=clamp(x.confidence, 0, 1),
                               reason=x.reason) for x in res.signals if x.symbol in data]
        out = []
        for s, d in data.items():
            score, why = fund.rule_score(d)
            if score:
                out.append(Signal(symbol=s, score=round(clamp(score), 3), confidence=0.4, reason=why))
        return out

    async def run_debate(self, focus: list[str], cfg: dict) -> None:
        """Bull and bear researchers argue the strongest candidates before the CEO decides (from TradingAgents)."""
        self.debate = {}
        A = cfg["agents"]
        sides = [x for x in ("bull", "bear") if self._active(cfg, x) and A[x].get("use_llm") and A[x].get("prompt")]
        if not sides or not (self.llm.available and self.llm_gate["active"]):
            return
        weights = self._weights(cfg)
        views = {s: [{"agent": ag, "weight": round(weights.get(ag, 1), 2), **x.model_dump(exclude={"symbol"})}
                     for ag, v in self.last_signals.items() if ag not in ("bull", "bear")
                     for x in v["signals"] if x.symbol == s]
                 for s in focus if s in self.closes}
        net = {s: sum(v["weight"] * v["score"] * v["confidence"] for v in vs) for s, vs in views.items()}
        n = int(A["bull" if "bull" in sides else "bear"].get("max_symbols", 2))
        cands = [s for s in sorted(net, key=lambda s: abs(net[s]), reverse=True)[:n] if net[s]]
        if not cands:
            return  # nobody has a view yet; nothing to argue about
        positions = self.broker.positions()
        brief = {s: {"price": round(self.closes[s][-1], 2), "rsi": round(rsi(self.closes[s])),
                     **{f"chg_{m}m": round(pct_change(self.closes[s], m), 3) for m in (5, 30)},
                     "position": positions.get(s), "analyst_signals": views[s]} for s in cands}

        async def argue(side: str):
            a = A[side]
            return await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                             json.dumps(brief, ensure_ascii=False), ArgumentList)

        results = await asyncio.gather(*(self._step(side, argue(side)) for side in sides))
        for side, res in zip(sides, results):
            if not res:
                continue
            sign = 1 if side == "bull" else -1
            args = [x for x in res.arguments if x.symbol in brief]
            sigs = [Signal(symbol=x.symbol, score=sign * clamp(x.strength, 0, 1), confidence=clamp(x.strength, 0, 1),
                           reason=x.argument[:300]) for x in args]
            self.last_signals[side] = {"ts": time.time(), "signals": sigs}
            self._record_signals(side, sigs)  # debaters are graded and rewarded like everyone else
            for x in args:
                self.debate.setdefault(x.symbol, {})[side] = {"argument": x.argument, "strength": x.strength}
            self.send(side, "ceo", "debate", [x.model_dump() for x in args])

    async def custom(self, focus: list[str], a: dict) -> list[Signal]:
        """Agents added from the UI: a prompt plus price stats and headlines in, graded signals out."""
        if not (self._use_llm(a, default=True) and a.get("prompt")):
            return []
        data = {}
        for s in focus:
            c = self.closes.get(s)
            if c and len(c) > 30:
                data[s] = {"price": round(c[-1], 2), "rsi": round(rsi(c)),
                           **{f"chg_{n}m": round(pct_change(c, n), 3) for n in (1, 5, 15, 30)},
                           "headlines": await self.broker.headlines(s)}
        res = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"),
                                        a["prompt"] + "\nأعد لكل سهم score من -1 إلى 1 و confidence من 0 إلى 1.",
                                        json.dumps(data, ensure_ascii=False), SignalList)
        if not res:
            return []
        return [Signal(symbol=x.symbol, score=clamp(x.score), confidence=clamp(x.confidence, 0, 1), reason=x.reason)
                for x in res.signals if x.symbol in data]

    def _weights(self, cfg: dict) -> dict[str, float]:
        mult = self.reward_multipliers() if cfg["agents"].get("auditor", {}).get("auto_reward") else {}
        return {a: float(v.get("weight", 1.0)) * mult.get(a, 1.0) for a, v in cfg["agents"].items()}

    # ---------- trade studies: Claude is spent only on real trade candidates ----------
    async def decide(self, focus: list[str], cfg: dict, risk: dict) -> DecisionList:
        """Rules scan every symbol for free; a buy/sell candidate gets a full Claude study before any order."""
        c = cfg.get("claude") or {}
        rules = self._ceo_rules(self._ceo_report(focus, cfg))
        if c.get("trade_without_study"):
            return rules  # owner opted out of studies: the rule decision goes straight to risk
        picks = self._study_candidates(rules, c, risk)
        if not picks:
            return self._hold(rules, [], "no candidate to study")
        if not (self.llm.available and self.llm_gate["active"]):
            why = "no API key" if not self.llm.available else f"Claude off: {self.llm_gate['phase']}"
            return self._hold(rules, [], f"not studied ({why})")
        if self.studies_today() >= int(c.get("max_studies_per_day", 10)):
            return self._hold(rules, [], "daily study limit reached")
        return self._hold(await self.study(picks, cfg), picks, "")

    def _study_candidates(self, rules: DecisionList, c: dict, risk: dict) -> list[str]:
        if self.halted or not risk.get("trading_enabled", False):
            return []  # nothing could be executed, so a study would be wasted money
        positions, now = self.broker.positions(), time.time()
        cool = 60 * float(c.get("study_cooldown_minutes", 30))
        out = []
        for d in rules.decisions:
            qty = positions.get(d.symbol, {"qty": 0})["qty"]
            if d.action == "hold" or now - self.studied_at.get(d.symbol, 0) < cool:
                continue
            if (d.action == "buy" and qty > 0) or (d.action == "sell" and qty <= 0 and not risk.get("allow_short")):
                continue  # already long, or a short the risk manager would refuse anyway
            out.append(d)
        out.sort(key=lambda d: -d.confidence)
        return [d.symbol for d in out[: int(c.get("max_symbols_per_study", 2))]]

    @staticmethod
    def _hold(res: DecisionList, keep: list[str], note: str) -> DecisionList:
        """Only decisions for `keep` may act; every other buy/sell becomes hold."""
        out = [d if d.symbol in keep or d.action == "hold" else
               d.model_copy(update={"action": "hold", "reason": f"{d.reason}; {note}"}) for d in res.decisions]
        acted = [d for d in out if d.action != "hold"]
        summary = res.summary if keep else (f"{len(res.decisions) - len(acted)} hold" + (f" ({note})" if note else ""))
        return DecisionList(decisions=out, summary=summary)

    def studies_today(self) -> int:
        return self.bus.db.execute("SELECT COUNT(*) FROM messages WHERE kind='study' AND ts>=?",
                                   (market.et_day_start(),)).fetchone()[0]

    async def study(self, picks: list[str], cfg: dict) -> DecisionList:
        """News, fundamentals and custom analysts read the candidates with Claude, bull and bear argue, the CEO decides."""
        A, c = cfg["agents"], cfg.get("claude") or {}
        self.send("ceo", "ceo", "study", {"symbols": picks, "n": self.studies_today() + 1,
                                          "max": int(c.get("max_studies_per_day", 10))})
        for s in picks:
            self.studied_at[s] = time.time()
        jobs = {"news": self.news(picks, A.get("news", {})), "fundamentals": self.fundamentals(picks, A.get("fundamentals", {}))}
        jobs.update({ag: self.custom(picks, a) for ag, a in A.items() if ag not in BUILTIN})
        for ag in [ag for ag in jobs if not (A.get(ag) or {}).get("enabled")]:
            jobs.pop(ag).close()
        results = await asyncio.gather(*(self._step(ag, j) for ag, j in jobs.items()))
        for ag, sigs in zip(jobs, results):
            if not sigs:
                continue
            others = [x for x in self.last_signals.get(ag, {}).get("signals", []) if x.symbol not in picks]
            self.last_signals[ag] = {"ts": time.time(), "signals": others + sigs}
            self._record_signals(ag, sigs)
            self.send(ag, "ceo", "signals", [s.model_dump() for s in sigs])
        await self.run_debate(picks, cfg)
        return await self.ceo(picks, cfg)

    def _ceo_report(self, focus: list[str], cfg: dict) -> dict:
        weights = self._weights(cfg)
        now = time.time()
        positions = self.broker.positions()
        report = {
            "account": self.account,
            "directives_from_owner": cfg.get("directives", []),
            "market": [s.model_dump() for s in self.last_signals.get("macro", {}).get("signals", [])],
            "lessons_from_your_past_trades": self.lessons(),
            "symbols": {},
        }
        for s in focus:
            if s not in self.closes:
                continue
            report["symbols"][s] = {
                "price": round(self.closes[s][-1], 2),
                "position": positions.get(s),
                "signals": [
                    {"agent": ag, "weight": round(weights.get(ag, 1), 2), "age_s": round(now - v["ts"]),
                     **x.model_dump(exclude={"symbol"})}
                    for ag, v in self.last_signals.items() if ag not in ("macro", "bull", "bear")
                    for x in v["signals"] if x.symbol == s
                ],
            }
            if s in self.debate:
                report["symbols"][s]["debate"] = self.debate[s]
        return report

    async def ceo(self, focus: list[str], cfg: dict) -> DecisionList:
        a = cfg["agents"]["ceo"]
        report = self._ceo_report(focus, cfg)
        if self._use_llm(a):
            res = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "medium"), a["prompt"],
                                            json.dumps(report, ensure_ascii=False), DecisionList)
            if res:
                res.decisions = [d for d in res.decisions if d.symbol in report["symbols"]]
                return res
        return self._ceo_rules(report)

    @staticmethod
    def _ceo_rules(report: dict) -> DecisionList:
        tilt = sum(m["score"] * m["confidence"] for m in report["market"]) * 0.3
        out = []
        for s, info in report["symbols"].items():
            sigs = info["signals"]
            wsum = sum(x["weight"] for x in sigs) or 1
            score = sum(x["weight"] * x["score"] * x["confidence"] for x in sigs) / wsum + tilt
            action = "buy" if score > 0.35 else "sell" if score < -0.35 else "hold"
            out.append(Decision(symbol=s, action=action, qty=1_000_000, confidence=round(min(1.0, 0.4 + abs(score)), 2),
                                reason=f"weighted score {score:+.2f} (rules)"))
        acted = [d for d in out if d.action != "hold"]
        return DecisionList(decisions=out, summary=f"{len(acted)} actions from rules" if acted else "hold all")

    async def risk_check(self, decisions: list[Decision], watch: list[str], risk: dict) -> list[dict]:
        if self.account["daily_pnl"] <= -abs(risk.get("daily_loss_limit", 1e12)):
            self.halt(f"daily loss limit hit ({self.account['daily_pnl']:.0f})")
        now = time.time()
        while self.order_times and now - self.order_times[0] > 3600:
            self.order_times.popleft()
        positions = self.broker.positions()
        exposure = sum(abs(p["qty"]) * self.closes.get(s, [p["avg"]])[-1] for s, p in positions.items())
        approved, rejected = [], []
        for d in decisions:
            if d.action == "hold":
                continue
            px = self.closes.get(d.symbol, [0])[-1]
            pos = positions.get(d.symbol, {"qty": 0})["qty"]
            closing = (d.action == "sell" and pos > 0) or (d.action == "buy" and pos < 0)
            reason = None
            if self.halted:
                reason = f"halted: {self.halt_reason}"
            elif not risk.get("trading_enabled", False):
                reason = "trading disabled"
            elif d.symbol not in watch:
                reason = "not in watchlist"
            elif d.confidence < risk.get("min_confidence", 0.6):
                reason = f"confidence {d.confidence:.2f} below minimum"
            elif len(self.order_times) >= risk.get("max_orders_per_hour", 10):
                reason = "hourly order limit"
            elif not px or math.isnan(px):
                reason = "no price"
            elif d.action == "sell" and pos <= 0 and not risk.get("allow_short", False):
                reason = "short selling disabled"
            if reason:
                rejected.append({"symbol": d.symbol, "action": d.action, "reason": reason})
                continue
            if closing:
                approved.append({"symbol": d.symbol, "action": "close", "qty": abs(pos), "price": px})
                continue
            room = min(risk["max_position_value"] - abs(pos) * px, risk["max_total_exposure"] - exposure)
            qty = min(d.qty, math.floor(max(0, room) / px))
            if qty < 1:
                rejected.append({"symbol": d.symbol, "action": d.action, "reason": "position/exposure limit"})
                continue
            sl, tp = risk["stop_loss_pct"] / 100, risk["take_profit_pct"] / 100
            sign = 1 if d.action == "buy" else -1
            approved.append({"symbol": d.symbol, "action": d.action, "qty": qty, "price": px,
                             "stop": round(px * (1 - sign * sl), 2), "take": round(px * (1 + sign * tp), 2)})
            exposure += qty * px
        if rejected:
            self.send("risk", "ceo", "rejected", rejected)
        if approved:
            self.send("risk", "execution", "approved", approved)
        return approved

    async def execute(self, orders: list[dict], risk: dict) -> list[dict]:
        self._settle()  # book any stop/target exits before a new entry on the same symbol
        fills = []
        for o in orders:
            if self.halted:  # kill switch pressed mid-cycle
                break
            try:
                if o["action"] == "close":
                    res = await self.broker.close(o["symbol"])
                else:
                    res = await self.broker.bracket(o["symbol"], o["action"], o["qty"], o["price"], o["stop"], o["take"])
                status = res["status"]
            except Exception as e:
                res, status = {"price": o["price"]}, f"error: {e}"
            self.order_times.append(time.time())
            self.bus.db.execute("INSERT INTO trades(ts,symbol,side,qty,price,status,note) VALUES(?,?,?,?,?,?,?)",
                                (time.time(), o["symbol"], o["action"], o["qty"], res.get("price"), status,
                                 json.dumps(o)))
            self.bus.db.commit()
            fills.append({**o, "status": status, "fill_price": res.get("price")})
            if o["action"] in ("buy", "sell") and status.lower() in ("filled", "submitted", "presubmitted", "pendingsubmit"):
                t = self.book.get(o["symbol"])
                px = res.get("price") or o["price"]
                if t and t["side"] == o["action"]:  # adding to the same side: average the entry
                    t["entry"] = (t["entry"] * t["qty"] + px * o["qty"]) / (t["qty"] + o["qty"])
                    t["qty"] += o["qty"]
                else:
                    # An IBKR order may still be working ("Submitted"): it only counts once a position shows up.
                    self.book[o["symbol"]] = {"side": o["action"], "qty": o["qty"], "entry": px, "ts": time.time(),
                                              "seen": status.lower() == "filled"}
        self._settle()
        return fills

    # ---------- closed trades: wins and losses ----------
    def _settle(self) -> None:
        """Move trades whose position is gone (stop, target, close, flatten) into closed_trades with their P&L."""
        exits = {}
        log_ = getattr(self.broker, "exits", None)
        while log_:
            e = log_.popleft()
            exits[e["symbol"]] = e
        positions = self.broker.positions()
        for sym, t in list(self.book.items()):
            if sym in positions and sym not in exits:
                t["seen"] = True
                continue
            if not t.get("seen", True) and sym not in exits:
                if time.time() - t["ts"] > 86400:  # never filled (cancelled or expired): drop it, nothing to book
                    del self.book[sym]
                continue
            # the broker's own exit fill when it reports one (simulator); otherwise the last price we saw
            e = exits.get(sym) or {"price": self.closes.get(sym, [t["entry"]])[-1], "reason": "closed", "ts": time.time()}
            sign = 1 if t["side"] == "buy" else -1
            pnl = sign * (e["price"] - t["entry"]) * t["qty"]
            self.bus.db.execute(
                "INSERT INTO closed_trades(mode,symbol,side,qty,entry,exit,pnl,pnl_pct,opened,closed,reason) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (self.broker.mode, sym, t["side"], t["qty"], t["entry"], e["price"], pnl,
                 pnl / (t["entry"] * t["qty"]) * 100 if t["entry"] else 0, t["ts"], e["ts"], e["reason"]))
            del self.book[sym]
        self.bus.db.commit()
        for sym, p in positions.items():  # positions opened before this run (e.g. after a restart on IBKR)
            if sym not in self.book and p["qty"]:
                self.book[sym] = {"side": "buy" if p["qty"] > 0 else "sell", "qty": abs(p["qty"]), "entry": p["avg"],
                                  "ts": time.time(), "seen": True}

    def trade_stats(self, limit: int = 50) -> dict:
        """Win/loss summary for the current broker mode, so simulator results never mix with paper ones."""
        rows = self.bus.db.execute(
            "SELECT symbol,side,qty,entry,exit,pnl,pnl_pct,opened,closed,reason FROM closed_trades WHERE mode=? "
            "ORDER BY id DESC", (self.broker.mode,)).fetchall()
        wins = [r[5] for r in rows if r[5] > 0]
        losses = [r[5] for r in rows if r[5] <= 0]
        keys = ("symbol", "side", "qty", "entry", "exit", "pnl", "pnl_pct", "opened", "closed", "reason")
        return {
            "count": len(rows), "wins": len(wins), "losses": len(losses),
            "win_rate": round(len(wins) / len(rows), 3) if rows else None,
            "net": round(sum(wins) + sum(losses), 2),
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "profit_factor": round(sum(wins) / -sum(losses), 2) if losses and sum(losses) < 0 else None,
            "recent": [dict(zip(keys, r)) for r in rows[:limit]],
        }

    # ---------- auditor: scoring & rewards ----------
    def _record_signals(self, agent: str, sigs: list[Signal]) -> None:
        horizon = agents_cfg.get()["agents"].get(agent, {}).get("horizon", SIGNAL_HORIZON_S)
        rows = [(time.time(), agent, s.symbol, s.score, s.confidence, self.closes[s.symbol][-1], horizon, s.reason)
                for s in sigs if s.symbol in self.closes and s.score]
        self.bus.db.executemany(
            "INSERT INTO signals(ts,agent,symbol,score,confidence,price,horizon,reason) VALUES(?,?,?,?,?,?,?,?)", rows)
        self.bus.db.commit()

    async def audit(self, a: dict | None = None) -> list[dict]:
        db, now = self.bus.db, time.time()
        due = db.execute("SELECT id,symbol,score,confidence,price FROM signals WHERE outcome IS NULL AND ts+horizon<=?",
                         (now,)).fetchall()
        for sid, sym, score, conf, px in due:
            cur = self.closes.get(sym)
            if cur and px:
                ret = (cur[-1] / px - 1) * 100
                db.execute("UPDATE signals SET outcome=? WHERE id=?", (math.copysign(1, score) * ret * conf, sid))
        db.commit()
        await self.reflect(a or {})
        return self.leaderboard()

    async def reflect(self, a: dict) -> None:
        """Turn each graded CEO decision into a short lesson the CEO reads next time (TradingAgents' reflection)."""
        db = self.bus.db
        rows = db.execute("""SELECT id,symbol,score,confidence,outcome,reason,horizon FROM signals
                             WHERE agent='ceo' AND outcome IS NOT NULL AND lesson IS NULL ORDER BY id LIMIT 5""").fetchall()
        if not rows:
            return
        trades = [{"id": i, "symbol": sym, "action": "buy" if sc > 0 else "sell", "confidence": c,
                   "move_pct": round(o / (math.copysign(1, sc) * c), 3), "minutes": round(h / 60), "reason": r}
                  for i, sym, sc, c, o, r, h in rows]
        lessons = {}
        if self._use_llm(a) and a.get("prompt"):
            res = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                            json.dumps(trades, ensure_ascii=False), LessonList)
            lessons = {x.id: x.lesson for x in res.lessons} if res else {}
        for t in trades:
            right = (t["move_pct"] > 0) == (t["action"] == "buy")
            text = lessons.get(t["id"]) or (
                f"{t['action']} {t['symbol']} ({t['reason']}) → {t['move_pct']:+.2f}% in {t['minutes']}m: "
                + ("correct call." if right else "wrong call; weigh these signals less next time."))
            db.execute("UPDATE signals SET lesson=? WHERE id=?", (text, t["id"]))
        db.commit()
        self.send("auditor", "ceo", "lessons", [db.execute("SELECT lesson FROM signals WHERE id=?", (t["id"],)).fetchone()[0]
                                               for t in trades])

    def lessons(self, n: int = 6) -> list[str]:
        return [r[0] for r in self.bus.db.execute(
            "SELECT lesson FROM signals WHERE agent='ceo' AND lesson IS NOT NULL ORDER BY id DESC LIMIT ?", (n,))]

    def leaderboard(self) -> list[dict]:
        rows = self.bus.db.execute(
            """SELECT agent, COUNT(*), AVG(outcome>0), AVG(outcome), SUM(outcome) FROM
               (SELECT agent, outcome FROM signals WHERE outcome IS NOT NULL ORDER BY id DESC LIMIT 2000)
               GROUP BY agent ORDER BY SUM(outcome) DESC""").fetchall()
        mult = self.reward_multipliers()
        return [{"agent": a, "graded": n, "hit_rate": round(h or 0, 3), "avg": round(avg or 0, 4),
                 "points": round(tot or 0, 3), "reward_x": round(mult.get(a, 1.0), 2)} for a, n, h, avg, tot in rows]

    def reward_multipliers(self) -> dict[str, float]:
        """Agents that earn points get more say with the CEO (0.25x .. 2x), based on their last 50 graded signals."""
        rows = self.bus.db.execute(
            """SELECT agent, COUNT(*), AVG(outcome) FROM (SELECT agent, outcome,
                 ROW_NUMBER() OVER (PARTITION BY agent ORDER BY id DESC) rn FROM signals WHERE outcome IS NOT NULL)
               WHERE rn<=50 GROUP BY agent""").fetchall()
        return {a: clamp(1 + 4 * avg, 0.25, 2.0) for a, n, avg in rows if n >= 10}

    # ---------- TradingView alerts ----------
    def tv_alert(self, body: dict) -> dict:
        """A TradingView webhook alert becomes a signal; it never places an order itself (CEO + study + risk still decide)."""
        cfg = agents_cfg.get()
        a = cfg["agents"].get("tradingview") or {}
        sym = str(body.get("symbol", "")).split(":")[-1].strip().upper()  # "NASDAQ:AAPL" -> "AAPL"
        action = str(body.get("action", "")).strip().lower()
        reason = None
        if not a.get("enabled"):
            reason = "tradingview agent is disabled"
        elif action not in ("buy", "sell"):
            reason = f"action must be buy or sell, got {action!r}"
        elif sym not in cfg.get("watchlist", []):
            reason = f"{sym or '?'} is not in the watchlist"
        if reason:
            self.send("tradingview", "ceo", "alert_ignored", {"symbol": sym, "action": action, "reason": reason})
            return {"ok": False, "reason": reason}
        try:
            conf = clamp(float(body.get("confidence", a.get("confidence", 0.7))), 0, 1)
        except (TypeError, ValueError):
            conf = float(a.get("confidence", 0.7))
        note = str(body.get("reason") or body.get("message") or "").strip()
        sig = Signal(symbol=sym, score=1.0 if action == "buy" else -1.0, confidence=conf,
                     reason=("TradingView: " + (note or action))[:300])
        self.tv_alerts[sym] = (time.time(), sig)
        self._refresh_tv(cfg)
        self._record_signals("tradingview", [sig])  # graded and rewarded like every other agent
        self.send("tradingview", "ceo", "signals", [sig.model_dump()])
        self.wake.set()  # react now instead of waiting for the next cycle
        return {"ok": True, "symbol": sym, "action": action}

    def _refresh_tv(self, cfg: dict) -> None:
        """Alerts are events: each stays in the CEO's view for ttl_minutes, then expires."""
        ttl = 60 * float((cfg["agents"].get("tradingview") or {}).get("ttl_minutes", 15))
        now = time.time()
        self.tv_alerts = {s: (ts, sig) for s, (ts, sig) in self.tv_alerts.items() if now - ts < ttl}
        if self.tv_alerts:
            self.last_signals["tradingview"] = {"ts": max(ts for ts, _ in self.tv_alerts.values()),
                                                "signals": [sig for _, sig in self.tv_alerts.values()]}
        else:
            self.last_signals.pop("tradingview", None)

    # ---------- Claude cost ----------
    def _record_usage(self, agent: str, model: str, inp: int, out: int, read: int, write: int, cost: float) -> None:
        self.bus.db.execute("INSERT INTO llm_usage(ts,agent,model,input,output,cache_read,cache_write,cost) "
                            "VALUES(?,?,?,?,?,?,?,?)", (time.time(), agent, model, inp, out, read, write, cost))
        self.bus.db.commit()

    def llm_costs(self) -> dict:
        """Approximate Claude spend in USD for today and this month (New York time, like the trading day)."""
        db = self.bus.db
        by_agent = dict(db.execute("SELECT agent, SUM(cost) FROM llm_usage WHERE ts>=? GROUP BY agent",
                                   (market.et_day_start(),)).fetchall())
        month = db.execute("SELECT COALESCE(SUM(cost),0), COUNT(*) FROM llm_usage WHERE ts>=?",
                           (market.et_month_start(),)).fetchone()
        return {"today": round(sum(by_agent.values()), 4), "month": round(month[0], 4), "month_calls": month[1],
                "today_by_agent": {a: round(v, 4) for a, v in sorted(by_agent.items(), key=lambda x: -x[1])}}

    # ---------- owner controls ----------
    def halt(self, reason: str) -> None:
        if not self.halted:
            self.halted, self.halt_reason = True, reason
            self.send("risk", "ceo", "halt", {"reason": reason})

    def resume(self) -> None:
        self.halted, self.halt_reason = False, ""
        self.send("owner", "ceo", "resume", {})
