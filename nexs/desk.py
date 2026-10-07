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
from .config import agents_cfg, risk_cfg
from .indicators import clamp, pct_change, rsi, sma
from .llm import LLM

log = logging.getLogger("nexs.desk")

BUILTIN = {"ceo", "scanner", "technical", "news", "macro", "risk", "execution", "auditor"}
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

    # ---------- plumbing ----------
    def send(self, src: str, dst: str, kind: str, payload) -> None:
        self.bus.publish(self.cycle, src, dst, kind, payload)

    def _active(self, cfg: dict, agent: str) -> bool:
        a = cfg["agents"].get(agent) or {}
        return bool(a.get("enabled")) and self.cycle % max(1, int(a.get("every", 1))) == 0

    async def _step(self, agent: str, coro):
        self.state[agent] = {**self.state.get(agent, {}), "status": "running", "since": time.time()}
        self.bus.broadcast({"type": "agent", "id": agent, "status": "running"})
        t0 = time.perf_counter()
        try:
            result = await coro
            status = "idle"
        except Exception as e:  # one agent failing must never stop the desk
            log.exception("agent %s failed", agent)
            self.send(agent, "ceo", "error", {"error": str(e)})
            result, status = None, "error"
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
        watch = list(dict.fromkeys(cfg.get("watchlist", [])))
        proxies = cfg.get("market_proxies", ["SPY"])
        symbols = list(dict.fromkeys(watch + proxies))
        await asyncio.wait_for(self.broker.ensure_connected(), 30)
        closes = await asyncio.wait_for(  # a hung data request must not freeze the desk
            asyncio.gather(*(self.broker.bars(s, 60) for s in symbols), return_exceptions=True), 30)
        self.closes = {s: c for s, c in zip(symbols, closes) if isinstance(c, list) and c}
        self.account = await asyncio.wait_for(self.broker.account(), 15)
        self.bus.broadcast({"type": "cycle", "cycle": self.cycle})

        focus = watch
        if self._active(cfg, "scanner"):
            focus = await self._step("scanner", self.scanner(watch, cfg["agents"]["scanner"])) or watch
            for dst in ("technical", "news"):
                self.send("scanner", dst, "focus", {"symbols": focus})

        analysts = {"technical": self.technical(focus), "news": self.news(focus, cfg["agents"].get("news", {})),
                    "macro": self.macro(proxies, cfg["agents"].get("macro", {}))}
        for agent, a in cfg["agents"].items():
            if agent not in BUILTIN:
                analysts[agent] = self.custom(focus, a)
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
            board = await self._step("auditor", self.audit())
            if board:
                self.send("auditor", "ceo", "leaderboard", board)

        if not self._active(cfg, "ceo"):
            return
        decisions = await self._step("ceo", self.ceo(focus, cfg))
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
        if a.get("use_llm") and self.llm.available:
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
        if a.get("use_llm") and self.llm.available:
            view = await self.llm.structured(a.get("model", "claude-opus-5-5"), a.get("effort", "low"), a["prompt"],
                                             json.dumps(moves), MarketView)
        if view is None:
            avg = sum(m["15m"] for m in moves.values()) / len(moves)
            view = MarketView(score=math.tanh(avg * 2), confidence=0.4, reason=f"avg 15m move {avg:+.2f}%")
        anchor = proxies[0]
        return [Signal(symbol=anchor, score=clamp(view.score), confidence=clamp(view.confidence, 0, 1), reason=view.reason)]

    async def custom(self, focus: list[str], a: dict) -> list[Signal]:
        """Agents added from the UI: a prompt plus price stats and headlines in, graded signals out."""
        if not (a.get("use_llm", True) and self.llm.available and a.get("prompt")):
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

    async def ceo(self, focus: list[str], cfg: dict) -> DecisionList:
        a = cfg["agents"]["ceo"]
        weights = self._weights(cfg)
        now = time.time()
        positions = self.broker.positions()
        report = {
            "account": self.account,
            "directives_from_owner": cfg.get("directives", []),
            "market": [s.model_dump() for s in self.last_signals.get("macro", {}).get("signals", [])],
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
                    for ag, v in self.last_signals.items() if ag != "macro"
                    for x in v["signals"] if x.symbol == s
                ],
            }
        if a.get("use_llm") and self.llm.available:
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
        return fills

    # ---------- auditor: scoring & rewards ----------
    def _record_signals(self, agent: str, sigs: list[Signal]) -> None:
        rows = [(time.time(), agent, s.symbol, s.score, s.confidence, self.closes[s.symbol][-1], SIGNAL_HORIZON_S)
                for s in sigs if s.symbol in self.closes and s.score]
        self.bus.db.executemany(
            "INSERT INTO signals(ts,agent,symbol,score,confidence,price,horizon) VALUES(?,?,?,?,?,?,?)", rows)
        self.bus.db.commit()

    async def audit(self) -> list[dict]:
        db, now = self.bus.db, time.time()
        due = db.execute("SELECT id,symbol,score,confidence,price FROM signals WHERE outcome IS NULL AND ts+horizon<=?",
                         (now,)).fetchall()
        for sid, sym, score, conf, px in due:
            cur = self.closes.get(sym)
            if cur and px:
                ret = (cur[-1] / px - 1) * 100
                db.execute("UPDATE signals SET outcome=? WHERE id=?", (math.copysign(1, score) * ret * conf, sid))
        db.commit()
        return self.leaderboard()

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

    # ---------- owner controls ----------
    def halt(self, reason: str) -> None:
        if not self.halted:
            self.halted, self.halt_reason = True, reason
            self.send("risk", "ceo", "halt", {"reason": reason})

    def resume(self) -> None:
        self.halted, self.halt_reason = False, ""
        self.send("owner", "ceo", "resume", {})
