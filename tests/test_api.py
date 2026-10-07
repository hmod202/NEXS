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

        c.post("/api/halt", json={})
        assert c.get("/api/state").json()["halted"] is True
        c.post("/api/resume")
        assert c.get("/api/state").json()["halted"] is False

        with c.websocket_connect("/ws") as ws:
            assert ws.receive_json()["type"] == "hello"
