"""Brokers: a built-in simulator (works out of the box) and Interactive Brokers via ib_async."""
import asyncio
import json
import math
import os
import random
import time
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

from .config import ROOT, agents_cfg
from .market import ET, is_open


class SimBroker:
    """Random-walk market with regime shifts, synthetic headlines and bracket-order fills."""

    mode = "sim"

    def __init__(self, symbols: list[str], cash: float = 100_000, seed: int | None = None):
        self.rng = random.Random(seed)
        self.hist: dict[str, deque] = {}
        self.drift: dict[str, float] = {}
        self.cash = cash
        self.start_equity = cash
        self.pos: dict[str, dict] = {}  # sym -> {qty, avg, stop, take}
        self.news: dict[str, list[str]] = {}
        self.exits: deque = deque(maxlen=500)  # fills that closed a position; the desk drains them to book wins/losses
        for s in symbols:
            self._ensure(s)
        self._task = None

    def _ensure(self, sym: str) -> None:
        if sym not in self.hist:
            p = self.rng.uniform(50, 500)
            self.hist[sym] = deque([p], maxlen=600)
            self.drift[sym] = 0.0
            self.news[sym] = []

    async def connect(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def ensure_connected(self) -> None:
        pass

    async def _run(self) -> None:
        while True:
            self.tick()
            await asyncio.sleep(1)

    def tick(self) -> None:
        for sym, h in self.hist.items():
            if self.rng.random() < 0.01:  # regime change, sometimes with a headline that explains it
                self.drift[sym] = self.rng.gauss(0, 0.0006)
                if abs(self.drift[sym]) > 0.0004:
                    good = self.drift[sym] > 0
                    self.news[sym].append(
                        f"{sym}: " + (self.rng.choice(["beats earnings estimates", "analyst upgrade", "new product launch"])
                                      if good else self.rng.choice(["guidance cut", "regulatory probe", "analyst downgrade"]))
                    )
                    self.news[sym] = self.news[sym][-5:]
            h.append(h[-1] * math.exp(self.drift[sym] + self.rng.gauss(0, 0.0015)))
        for sym, p in list(self.pos.items()):  # bracket exits
            px = self.hist[sym][-1]
            long = p["qty"] > 0
            hit_take = px >= p["take"] if long else px <= p["take"]
            hit_stop = px <= p["stop"] if long else px >= p["stop"]
            if hit_take or hit_stop:
                self.cash += p["qty"] * px
                del self.pos[sym]
                self.exits.append({"symbol": sym, "price": px, "reason": "take" if hit_take else "stop", "ts": time.time()})

    async def price(self, sym: str) -> float:
        self._ensure(sym)
        return self.hist[sym][-1]

    async def bars(self, sym: str, n: int = 60) -> list[float]:
        self._ensure(sym)
        return list(self.hist[sym])[-n:]

    async def headlines(self, sym: str) -> list[str]:
        return list(self.news.get(sym, []))

    def positions(self) -> dict[str, dict]:
        return {s: {"qty": p["qty"], "avg": p["avg"]} for s, p in self.pos.items()}

    async def account(self) -> dict:
        equity = self.cash + sum(p["qty"] * self.hist[s][-1] for s, p in self.pos.items())
        return {"equity": equity, "daily_pnl": equity - self.start_equity}

    async def bracket(self, sym: str, side: str, qty: float, price: float, stop: float, take: float) -> dict:
        signed = qty if side == "buy" else -qty
        p = self.pos.get(sym)
        self.cash -= signed * price
        if p:
            total = p["qty"] + signed
            p.update(avg=(p["avg"] * p["qty"] + price * signed) / total, qty=total, stop=stop, take=take)
        else:
            self.pos[sym] = {"qty": signed, "avg": price, "stop": stop, "take": take}
        return {"status": "filled", "price": price}

    async def close(self, sym: str) -> dict:
        p = self.pos.pop(sym, None)
        px = self.hist[sym][-1]
        if p:
            self.cash += p["qty"] * px
            self.exits.append({"symbol": sym, "price": px, "reason": "close", "ts": time.time()})
        return {"status": "closed" if p else "no position", "price": px}

    async def flatten(self) -> None:
        for sym in list(self.pos):
            await self.close(sym)


def _yahoo_json(url: str) -> dict:
    # Plain JSON instead of yfinance: Windows Smart App Control blocks pandas' DLLs, which yfinance needs.
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.load(r)


class DemoBroker:
    """Demo account on real prices: Yahoo Finance 1-minute bars, virtual cash, simulated bracket fills.

    Orders and stop/target exits only fill during the regular session. Fills pay a small slippage and an
    IBKR-like commission, and the account is saved to disk so a restart keeps cash and positions.
    """

    mode = "demo"
    SLIPPAGE = 0.0005  # 0.05% against us on every fill
    REFRESH_S = 20

    def __init__(self, symbols: list[str], cash: float = 100_000, path: Path | None = None):
        self.path = path or Path(os.environ.get("NEXS_DEMO_FILE", ROOT / "data" / "demo_account.json"))
        self.symbols = list(symbols)
        self.bars_: dict[str, list[tuple]] = {}  # sym -> [(ts, open, high, low, close)]
        self.fetched: dict[str, float] = {}
        self.checked: dict[str, float] = {}  # sym -> last bar time already checked for stop/target hits
        self.news_cache: dict[str, tuple[float, list[str]]] = {}
        self.exits: deque = deque(maxlen=500)
        s = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.cash = s.get("cash", cash)
        self.pos: dict[str, dict] = s.get("pos", {})  # sym -> {qty, avg, stop, take}
        self.day, self.day_equity = s.get("day", ""), s.get("day_equity", self.cash)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cash": self.cash, "pos": self.pos, "day": self.day,
                                         "day_equity": self.day_equity}), encoding="utf-8")

    @staticmethod
    def _holidays():
        return (agents_cfg.get().get("claude") or {}).get("holidays") or []

    async def connect(self) -> None:
        await asyncio.gather(*(self._refresh(s) for s in self.symbols))

    async def ensure_connected(self) -> None:
        pass

    def _fetch(self, sym: str) -> list[tuple]:
        d = _yahoo_json(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?range=2d&interval=1m&includePrePost=true")
        res = d["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        return [(t, o, h, lo, c) for t, o, h, lo, c in zip(res["timestamp"], q["open"], q["high"], q["low"], q["close"])
                if None not in (o, h, lo, c)]

    async def _refresh(self, sym: str) -> None:
        if time.time() - self.fetched.get(sym, 0) < self.REFRESH_S:
            return
        self.bars_[sym] = await asyncio.to_thread(self._fetch, sym)
        self.fetched[sym] = time.time()
        self._check_exits(sym)

    def _check_exits(self, sym: str) -> None:
        """Walk the new 1-minute bars: a stop or target inside a bar's range fills there (stop first if both)."""
        p, bars = self.pos.get(sym), self.bars_.get(sym) or []
        if not p:
            return
        holidays = self._holidays()
        for t, o, h, lo, c in bars:
            if t <= self.checked.get(sym, p.get("opened", 0)) or not is_open(t, holidays):
                continue
            long = p["qty"] > 0
            stop_hit = lo <= p["stop"] if long else h >= p["stop"]
            take_hit = h >= p["take"] if long else lo <= p["take"]
            if stop_hit or take_hit:
                level = p["stop"] if stop_hit else p["take"]
                # a gap through the level fills at the bar's open, not at the level
                px = min(o, level) if (long and stop_hit) or (not long and take_hit) else max(o, level)
                self._fill_exit(sym, px, "stop" if stop_hit else "take", t)
                break
        self.checked[sym] = bars[-1][0] if bars else time.time()

    def _fill_exit(self, sym: str, px: float, reason: str, ts: float) -> None:
        p = self.pos.pop(sym)
        self.cash += p["qty"] * px - self._commission(p["qty"])
        self.exits.append({"symbol": sym, "price": px, "reason": reason, "ts": ts})
        self._save()

    @staticmethod
    def _commission(qty: float) -> float:
        return max(1.0, 0.005 * abs(qty))

    def _last(self, sym: str) -> float:
        bars = self.bars_.get(sym)
        return bars[-1][4] if bars else float("nan")

    async def price(self, sym: str) -> float:
        await self._refresh(sym)
        return self._last(sym)

    async def bars(self, sym: str, n: int = 60) -> list[float]:
        await self._refresh(sym)
        return [b[4] for b in self.bars_.get(sym, [])][-n:]

    async def headlines(self, sym: str) -> list[str]:
        hit = self.news_cache.get(sym)
        if hit and time.time() - hit[0] < 600:
            return hit[1]
        try:
            d = await asyncio.to_thread(_yahoo_json,
                                        f"https://query1.finance.yahoo.com/v1/finance/search?q={sym}&newsCount=6&quotesCount=0")
            heads = [n["title"] for n in d.get("news", []) if n.get("title")]
        except Exception:
            heads = hit[1] if hit else []
        self.news_cache[sym] = (time.time(), heads)
        return heads

    def positions(self) -> dict[str, dict]:
        return {s: {"qty": p["qty"], "avg": p["avg"]} for s, p in self.pos.items()}

    async def account(self) -> dict:
        equity = self.cash + sum(p["qty"] * (self._last(s) if s in self.bars_ else p["avg"]) for s, p in self.pos.items())
        today = datetime.now(ET).date().isoformat()
        if self.day != today:  # first look of a new trading day: today's P&L starts from here
            self.day, self.day_equity = today, equity
            self._save()
        return {"equity": equity, "daily_pnl": equity - self.day_equity}

    async def bracket(self, sym: str, side: str, qty: float, price: float, stop: float, take: float) -> dict:
        if not is_open(None, self._holidays()):
            return {"status": "rejected: market closed", "price": price}
        await self._refresh(sym)
        px = self._last(sym) * (1 + self.SLIPPAGE if side == "buy" else 1 - self.SLIPPAGE)
        signed = qty if side == "buy" else -qty
        self.cash -= signed * px + self._commission(qty)
        p = self.pos.get(sym)
        if p:
            total = p["qty"] + signed
            p.update(avg=(p["avg"] * p["qty"] + px * signed) / total, qty=total, stop=stop, take=take)
        else:
            self.pos[sym] = {"qty": signed, "avg": px, "stop": stop, "take": take, "opened": time.time()}
        self.checked[sym] = time.time()  # only bars after the entry can hit the stop or target
        self._save()
        return {"status": "filled", "price": px}

    async def close(self, sym: str) -> dict:
        if sym not in self.pos:
            return {"status": "no position", "price": None}
        if not is_open(None, self._holidays()):
            return {"status": "rejected: market closed", "price": None}
        await self._refresh(sym)
        long = self.pos[sym]["qty"] > 0
        px = self._last(sym) * (1 - self.SLIPPAGE if long else 1 + self.SLIPPAGE)
        self._fill_exit(sym, px, "close", time.time())
        return {"status": "closed", "price": px}

    async def flatten(self) -> None:
        for sym in list(self.pos):
            await self.close(sym)


class IBKRBroker:
    """Interactive Brokers through TWS / IB Gateway. Paper accounts (DU...) only unless NEXS_ALLOW_LIVE=1."""

    def __init__(self, host: str, port: int, client_id: int):
        from ib_async import IB

        self.ib = IB()
        self.host, self.port, self.client_id = host, port, client_id
        self.contracts = {}
        self.news_codes = ""
        self.account_id = ""
        self._pnl = None
        self.mode = "paper"

    async def connect(self) -> None:
        await self.ib.connectAsync(self.host, self.port, clientId=self.client_id)
        # 1 = live data (needs a market data subscription), 3 = delayed (free)
        self.ib.reqMarketDataType(int(os.environ.get("IBKR_MARKET_DATA_TYPE", "3")))
        accounts = self.ib.managedAccounts()
        self.account_id = accounts[0] if accounts else ""
        self.mode = "paper" if all(a.startswith("DU") for a in accounts) else "live"
        try:
            providers = await self.ib.reqNewsProvidersAsync()
            self.news_codes = "+".join(p.code for p in providers)
        except Exception:
            self.news_codes = ""
        if self.account_id:
            self._pnl = self.ib.reqPnL(self.account_id)

    async def ensure_connected(self) -> None:
        """IB Gateway restarts daily; reconnect transparently instead of failing every cycle."""
        if not self.ib.isConnected():
            self.contracts.clear()
            await self.connect()

    async def _contract(self, sym: str):
        if sym not in self.contracts:
            from ib_async import Stock

            c = Stock(sym, "SMART", "USD")
            await self.ib.qualifyContractsAsync(c)
            self.contracts[sym] = c
        return self.contracts[sym]

    async def bars(self, sym: str, n: int = 60) -> list[float]:
        bars = await self.ib.reqHistoricalDataAsync(
            await self._contract(sym), endDateTime="", durationStr=f"{max(n, 2) * 60} S",
            barSizeSetting="1 min", whatToShow="TRADES", useRTH=False,
        )
        return [b.close for b in bars][-n:]

    async def price(self, sym: str) -> float:
        [t] = await self.ib.reqTickersAsync(await self._contract(sym))
        px = t.marketPrice()
        if px is None or math.isnan(px):
            closes = await self.bars(sym, 2)
            px = closes[-1] if closes else float("nan")
        return px

    async def headlines(self, sym: str) -> list[str]:
        if not self.news_codes:
            return []
        c = await self._contract(sym)
        news = await self.ib.reqHistoricalNewsAsync(c.conId, self.news_codes, "", "", 10)
        return [n.headline for n in news or []]

    def positions(self) -> dict[str, dict]:
        return {p.contract.symbol: {"qty": p.position, "avg": p.avgCost} for p in self.ib.positions() if p.position}

    async def account(self) -> dict:
        vals = await self.ib.accountSummaryAsync(self.account_id)
        equity = next((float(v.value) for v in vals if v.tag == "NetLiquidation"), 0.0)
        daily = self._pnl.dailyPnL if self._pnl and not math.isnan(self._pnl.dailyPnL or float("nan")) else 0.0
        return {"equity": equity, "daily_pnl": daily}

    async def bracket(self, sym: str, side: str, qty: float, price: float, stop: float, take: float) -> dict:
        if self.mode == "live" and os.environ.get("NEXS_ALLOW_LIVE") != "1":
            raise RuntimeError("Live account detected; set NEXS_ALLOW_LIVE=1 to allow real-money orders")
        c = await self._contract(sym)
        limit = round(price * (1.001 if side == "buy" else 0.999), 2)  # marketable limit, bounded slippage
        orders = self.ib.bracketOrder(side.upper(), qty, limit, round(take, 2), round(stop, 2))
        trades = [self.ib.placeOrder(c, o) for o in orders]
        await asyncio.sleep(1)
        return {"status": trades[0].orderStatus.status, "price": limit, "order_id": trades[0].order.orderId}

    async def close(self, sym: str) -> dict:
        """Cancel the symbol's resting bracket legs, then exit the position at market."""
        from ib_async import MarketOrder

        for t in self.ib.openTrades():
            if t.contract.symbol == sym:
                self.ib.cancelOrder(t.order)
        p = self.positions().get(sym)
        if not p:
            return {"status": "no position", "price": None}
        trade = self.ib.placeOrder(await self._contract(sym), MarketOrder("SELL" if p["qty"] > 0 else "BUY", abs(p["qty"])))
        await asyncio.sleep(1)
        return {"status": trade.orderStatus.status, "price": None, "order_id": trade.order.orderId}

    async def flatten(self) -> None:
        self.ib.reqGlobalCancel()
        for sym in list(self.positions()):
            await self.close(sym)


def make_broker(symbols: list[str]):
    kind = os.environ.get("NEXS_BROKER", "sim")
    if kind == "ibkr":
        return IBKRBroker(
            os.environ.get("IBKR_HOST", "127.0.0.1"),
            int(os.environ.get("IBKR_PORT", "4002")),  # 4002 = IB Gateway paper, 7497 = TWS paper
            int(os.environ.get("IBKR_CLIENT_ID", "17")),
        )
    if kind == "demo":
        return DemoBroker(symbols)
    return SimBroker(symbols, seed=int(time.time()))
