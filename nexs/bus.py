"""Message bus: every agent-to-agent message is stored in SQLite and pushed live to the UI."""
import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path

from .config import ROOT

DB_PATH = Path(os.environ.get("NEXS_DB", ROOT / "data" / "nexs.db"))


class Bus:
    def __init__(self, db_path: Path = DB_PATH):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY, ts REAL, cycle INT,
                src TEXT, dst TEXT, kind TEXT, payload TEXT);
            CREATE TABLE IF NOT EXISTS signals(id INTEGER PRIMARY KEY, ts REAL, agent TEXT, symbol TEXT,
                score REAL, confidence REAL, price REAL, horizon REAL, outcome REAL, reason TEXT, lesson TEXT);
            CREATE TABLE IF NOT EXISTS trades(id INTEGER PRIMARY KEY, ts REAL, symbol TEXT, side TEXT,
                qty REAL, price REAL, status TEXT, note TEXT);
            CREATE TABLE IF NOT EXISTS llm_usage(id INTEGER PRIMARY KEY, ts REAL, agent TEXT, model TEXT,
                input INT, output INT, cache_read INT, cache_write INT, cost REAL);
            CREATE INDEX IF NOT EXISTS llm_usage_ts ON llm_usage(ts);
            CREATE TABLE IF NOT EXISTS closed_trades(id INTEGER PRIMARY KEY, mode TEXT, symbol TEXT, side TEXT,
                qty REAL, entry REAL, exit REAL, pnl REAL, pnl_pct REAL, opened REAL, closed REAL, reason TEXT);
            """
        )
        for col in ("reason", "lesson"):  # databases created before these columns existed
            try:
                self.db.execute(f"ALTER TABLE signals ADD COLUMN {col} TEXT")
            except sqlite3.OperationalError:
                pass
        self.subscribers: set[asyncio.Queue] = set()

    def publish(self, cycle: int, src: str, dst: str, kind: str, payload) -> dict:
        msg = {"ts": time.time(), "cycle": cycle, "src": src, "dst": dst, "kind": kind, "payload": payload}
        cur = self.db.execute(
            "INSERT INTO messages(ts,cycle,src,dst,kind,payload) VALUES(?,?,?,?,?,?)",
            (msg["ts"], cycle, src, dst, kind, json.dumps(payload, ensure_ascii=False, default=str)),
        )
        self.db.commit()
        msg["id"] = cur.lastrowid
        self.broadcast({"type": "message", **msg})
        return msg

    def broadcast(self, event: dict) -> None:
        for q in list(self.subscribers):
            if q.qsize() < 500:  # drop events for a stalled browser tab instead of growing memory
                q.put_nowait(event)

    def recent(self, limit: int = 200) -> list[dict]:
        rows = self.db.execute(
            "SELECT id,ts,cycle,src,dst,kind,payload FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [
            {"id": r[0], "ts": r[1], "cycle": r[2], "src": r[3], "dst": r[4], "kind": r[5], "payload": json.loads(r[6])}
            for r in reversed(rows)
        ]
