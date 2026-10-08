from fastapi.testclient import TestClient

from nexs.app import app


def test_live_editing_endpoints():
    with TestClient(app) as c:
        s = c.get("/api/state").json()
        assert s["mode"] == "sim" and "ceo" in s["agents"]

        c.put("/api/agents/news", json={"every": 5, "prompt": "new prompt", "bogus": 1})
        news = c.get("/api/state").json()["agents"]["news"]
        assert news["every"] == 5 and news["prompt"] == "new prompt" and "bogus" not in news

        assert c.post("/api/agents", json={"id": "earnings", "name": "محلل الأرباح", "prompt": "p"}).status_code == 200
        assert "earnings" in c.get("/api/state").json()["agents"]
        assert c.delete("/api/agents/ceo").status_code == 400
        assert c.delete("/api/agents/earnings").status_code == 200

        assert c.post("/api/directive", json={"text": "avoid TSLA"}).json() == ["avoid TSLA"]
        c.put("/api/risk", json={"max_position_value": 500, "evil": True})
        r = c.get("/api/state").json()["risk"]
        assert r["max_position_value"] == 500 and "evil" not in r

        c.put("/api/settings", json={"claude": {"pre_open_minutes": 30, "daily_budget_usd": 5, "evil": 1}})
        s = c.get("/api/state").json()
        assert s["settings"]["claude"]["pre_open_minutes"] == 30 and "evil" not in s["settings"]["claude"]
        assert {"gate", "cost"} <= s["llm"].keys() and s["llm"]["cost"]["today"] == 0

        c.post("/api/halt", json={})
        assert c.get("/api/state").json()["halted"] is True
        c.post("/api/resume")
        assert c.get("/api/state").json()["halted"] is False

        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"


def test_tradingview_webhook_turns_alerts_into_signals(monkeypatch):
    from nexs import app as appmod

    with TestClient(app) as c:
        monkeypatch.delenv("NEXS_TV_SECRET", raising=False)
        assert c.post("/webhook/tradingview", json={"symbol": "AAPL", "action": "buy"}).status_code == 503
        monkeypatch.setenv("NEXS_TV_SECRET", "s3cret")
        assert c.post("/webhook/tradingview", json={"secret": "nope", "symbol": "AAPL", "action": "buy"}).status_code == 401
        assert c.post("/webhook/tradingview", content="not json").status_code == 400
        r = c.post("/webhook/tradingview", json={"secret": "s3cret", "symbol": "XYZ", "action": "buy"}).json()
        assert r["ok"] is False and "watchlist" in r["reason"]
        r = c.post("/webhook/tradingview", content='{"secret":"s3cret","symbol":"NASDAQ:AAPL","action":"sell",'
                                                  '"confidence":0.9,"reason":"RSI cross"}').json()
        assert r == {"ok": True, "symbol": "AAPL", "action": "sell"}
        [sig] = appmod.desk.last_signals["tradingview"]["signals"]
        assert (sig.symbol, sig.score, sig.confidence) == ("AAPL", -1.0, 0.9) and "RSI cross" in sig.reason
