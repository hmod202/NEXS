"""Brokers: a built-in simulator (works out of the box) and Interactive Brokers via ib_async."""
import asyncio
import math
import os
import random
import time
from collections import deque


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
            if (long and (px <= p["stop"] or px >= p["take"])) or (not long and (px >= p["stop"] or px <= p["take"])):
                self.cash += p["qty"] * px
                del self.pos[sym]

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
        return {"status": "closed" if p else "no position", "price": px}

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
    return SimBroker(symbols, seed=int(time.time()))
