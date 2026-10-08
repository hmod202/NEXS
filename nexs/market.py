"""US equity market hours: Claude agents run only around the session; outside it they fall back to rules."""
from datetime import date, datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
OPEN, CLOSE = dtime(9, 30), dtime(16, 0)


def _trading_day(d: date, holidays: set[str]) -> bool:
    return d.weekday() < 5 and d.isoformat() not in holidays


def llm_window(hours: dict | None, now: datetime | None = None) -> dict:
    """Whether Claude agents should run now.

    hours: the `llm_hours` block of agents.yaml. Returns {"active", "phase", "next_start"}, where phase is
    "always" | "pre_open" | "open" | "closed" and next_start is a Unix time (None while active).
    """
    h = hours or {}
    if not h.get("market_only", True):
        return {"active": True, "phase": "always", "next_start": None}
    now = (now or datetime.now(ET)).astimezone(ET)
    holidays = {str(d) for d in h.get("holidays") or []}
    pre = timedelta(minutes=float(h.get("pre_open_minutes", 60)))
    post = timedelta(minutes=float(h.get("post_close_minutes", 0)))
    for i in range(15):  # long enough to cross any holiday weekend
        d = now.date() + timedelta(days=i)
        if not _trading_day(d, holidays):
            continue
        opens = datetime.combine(d, OPEN, ET)
        start, end = opens - pre, datetime.combine(d, CLOSE, ET) + post
        if start <= now < end:
            return {"active": True, "phase": "pre_open" if now < opens else "open", "next_start": None}
        if now < start:
            return {"active": False, "phase": "closed", "next_start": start.timestamp()}
    return {"active": False, "phase": "closed", "next_start": None}


def is_open(ts: float | None = None, holidays=()) -> bool:
    """Regular session (9:30-16:00 New York, trading days); demo orders and stop/target fills only happen inside it."""
    now = datetime.fromtimestamp(ts, ET) if ts else datetime.now(ET)
    return _trading_day(now.date(), {str(h) for h in holidays}) and OPEN <= now.time() < CLOSE


def et_day_start(now: datetime | None = None) -> float:
    """Unix time of midnight New York time today; the cost counter's day and budget follow the trading day."""
    now = (now or datetime.now(ET)).astimezone(ET)
    return datetime.combine(now.date(), dtime(0), ET).timestamp()


def et_month_start(now: datetime | None = None) -> float:
    now = (now or datetime.now(ET)).astimezone(ET)
    return datetime.combine(now.date().replace(day=1), dtime(0), ET).timestamp()
