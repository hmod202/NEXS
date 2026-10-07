"""Web server: dashboard, live WebSocket feed, and live-editing API for agents and risk limits."""
import asyncio
import logging
import os
import re
from contextlib import asynccontextmanager

from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse

from .broker import SimBroker, make_broker
from .bus import Bus
from .config import ROOT, agents_cfg, risk_cfg
from .desk import BUILTIN, Desk
from .llm import LLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("nexs")

EDITABLE = {"name", "role", "enabled", "use_llm", "model", "effort", "every", "weight", "prompt", "top_n", "auto_reward"}
RISK_KEYS = {"trading_enabled", "max_position_value", "max_total_exposure", "max_orders_per_hour", "daily_loss_limit",
             "min_confidence", "stop_loss_pct", "take_profit_pct", "allow_short"}

desk: Desk | None = None
broker_error = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    global desk, broker_error
    bus = Bus()
    cfg = agents_cfg.get()
    symbols = cfg.get("watchlist", []) + cfg.get("market_proxies", [])
    broker = make_broker(symbols)
    try:
        await broker.connect()
    except Exception as e:  # keep the desk alive on the simulator and show the problem in the UI
        broker_error = f"IBKR connection failed: {e}"
        log.error(broker_error)
        broker = SimBroker(symbols)
        await broker.connect()
    desk = Desk(broker, bus, LLM())
    tasks = [asyncio.create_task(desk.run_forever()), asyncio.create_task(_state_pump())]
    yield
    for t in tasks:
        t.cancel()


app = FastAPI(lifespan=lifespan)
TOKEN = os.environ.get("NEXS_TOKEN", "")


@app.middleware("http")
async def require_token(request, call_next):
    # The API can place orders: when exposed beyond localhost, set NEXS_TOKEN and open /?token=...
    if TOKEN and request.url.path.startswith("/api") and request.query_params.get("token") != TOKEN:
        return JSONResponse({"detail": "bad token"}, status_code=401)
    return await call_next(request)


async def _state_pump():
    while True:
        await asyncio.sleep(2)
        try:
            desk.bus.broadcast({"type": "state", **await state()})
        except Exception:
            log.exception("state pump")


@app.get("/")
async def index():
    return FileResponse(ROOT / "web" / "index.html")


@app.get("/api/state")
async def state():
    cfg = agents_cfg.get()
    mult = desk.reward_multipliers()
    agents = {
        aid: {**a, "id": aid, "builtin": aid in BUILTIN, "reward_x": round(mult.get(aid, 1.0), 2),
              **desk.state.get(aid, {"status": "disabled" if not a.get("enabled") else "idle"})}
        for aid, a in cfg["agents"].items()
    }
    prices = {s: round(c[-1], 2) for s, c in desk.closes.items()}
    # the simulator is cheap to query live; IBKR is refreshed once per cycle by the desk
    account = await desk.broker.account() if desk.broker.mode == "sim" else desk.account
    return {
        "cycle": desk.cycle, "mode": desk.broker.mode, "broker_error": broker_error,
        "halted": desk.halted, "halt_reason": desk.halt_reason,
        "account": account, "positions": desk.broker.positions(), "prices": prices,
        "agents": agents, "leaderboard": desk.leaderboard(),
        "llm": {"status": desk.llm.status, "available": desk.llm.available, "usage": desk.llm.usage},
        "settings": {k: cfg.get(k) for k in ("cycle_seconds", "watchlist", "market_proxies", "directives")},
        "risk": risk_cfg.get(),
        "trades": [dict(zip(("ts", "symbol", "side", "qty", "price", "status"), r)) for r in desk.bus.db.execute(
            "SELECT ts,symbol,side,qty,price,status FROM trades ORDER BY id DESC LIMIT 30").fetchall()],
    }


@app.get("/api/messages")
async def messages(limit: int = 200):
    return desk.bus.recent(min(limit, 1000))


@app.put("/api/agents/{aid}")
async def update_agent(aid: str, patch: dict = Body(...)):
    cfg = agents_cfg.get()
    if aid not in cfg["agents"]:
        raise HTTPException(404, "unknown agent")
    cfg["agents"][aid].update({k: v for k, v in patch.items() if k in EDITABLE})
    agents_cfg.save(cfg)
    desk.send("owner", aid, "config", {k: v for k, v in patch.items() if k in EDITABLE and k != "prompt"})
    return cfg["agents"][aid]


@app.post("/api/agents")
async def add_agent(body: dict = Body(...)):
    aid = re.sub(r"[^a-z0-9_]", "", str(body.get("id", "")).lower())
    cfg = agents_cfg.get()
    if not aid or aid in cfg["agents"]:
        raise HTTPException(400, "id must be new and use a-z, 0-9, _")
    if not body.get("prompt"):
        raise HTTPException(400, "prompt is required")
    cfg["agents"][aid] = {"name": body.get("name", aid), "role": body.get("role", ""), "enabled": True,
                          "use_llm": True, "model": body.get("model", "claude-opus-5-5"), "effort": "low",
                          "every": int(body.get("every", 2)), "weight": 1.0, "prompt": body["prompt"]}
    agents_cfg.save(cfg)
    desk.send("owner", "ceo", "hire", {"agent": aid, "name": cfg["agents"][aid]["name"]})
    return cfg["agents"][aid]


@app.delete("/api/agents/{aid}")
async def remove_agent(aid: str):
    if aid in BUILTIN:
        raise HTTPException(400, "built-in agents can be disabled, not removed")
    cfg = agents_cfg.get()
    cfg["agents"].pop(aid, None)
    agents_cfg.save(cfg)
    desk.last_signals.pop(aid, None)
    return {"ok": True}


@app.put("/api/risk")
async def update_risk(patch: dict = Body(...)):
    r = risk_cfg.get()
    r.update({k: v for k, v in patch.items() if k in RISK_KEYS})
    risk_cfg.save(r)
    desk.send("owner", "risk", "limits", {k: v for k, v in patch.items() if k in RISK_KEYS})
    return r


@app.put("/api/settings")
async def update_settings(patch: dict = Body(...)):
    cfg = agents_cfg.get()
    if "watchlist" in patch:
        cfg["watchlist"] = [s.strip().upper() for s in patch["watchlist"] if s.strip()]
    if "cycle_seconds" in patch:
        cfg["cycle_seconds"] = max(5, int(patch["cycle_seconds"]))
    agents_cfg.save(cfg)
    return {k: cfg[k] for k in ("watchlist", "cycle_seconds")}


@app.post("/api/directive")
async def add_directive(body: dict = Body(...)):
    text = str(body.get("text", "")).strip()
    if not text:
        raise HTTPException(400, "empty directive")
    cfg = agents_cfg.get()
    cfg.setdefault("directives", []).append(text)
    agents_cfg.save(cfg)
    desk.send("owner", "ceo", "directive", {"text": text})
    return cfg["directives"]


@app.delete("/api/directive/{index}")
async def remove_directive(index: int):
    cfg = agents_cfg.get()
    if 0 <= index < len(cfg.get("directives", [])):
        cfg["directives"].pop(index)
        agents_cfg.save(cfg)
    return cfg.get("directives", [])


@app.post("/api/halt")
async def halt(body: dict = Body(default={})):
    desk.halt("stopped by owner")
    if body.get("flatten"):
        await desk.broker.flatten()
        desk.send("owner", "execution", "flatten", {})
    return {"halted": True}


@app.post("/api/resume")
async def resume():
    desk.resume()
    return {"halted": False}


@app.post("/api/cycle")
async def run_now():
    desk.wake.set()
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    if TOKEN and sock.query_params.get("token") != TOKEN:
        await sock.close(code=4401)
        return
    await sock.accept()
    q: asyncio.Queue = asyncio.Queue()
    desk.bus.subscribers.add(q)
    try:
        await sock.send_json({"type": "hello", "messages": desk.bus.recent(150), "state": await state()})
        while True:
            await sock.send_json(await q.get())
    except WebSocketDisconnect:
        pass
    finally:
        desk.bus.subscribers.discard(q)


def main():
    import uvicorn

    uvicorn.run(app, host=os.environ.get("NEXS_HOST", "127.0.0.1"), port=int(os.environ.get("NEXS_PORT", "8000")))


if __name__ == "__main__":
    main()
