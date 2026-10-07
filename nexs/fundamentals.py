"""Company fundamentals from Yahoo Finance (yfinance), cached for a day. Idea from TradingAgents (Apache-2.0)."""
import time

KEYS = ("shortName", "sector", "marketCap", "trailingPE", "forwardPE", "pegRatio", "priceToBook", "revenueGrowth",
        "earningsGrowth", "profitMargins", "debtToEquity", "freeCashflow", "recommendationMean",
        "numberOfAnalystOpinions", "targetMeanPrice", "currentPrice", "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
        "shortPercentOfFloat")
_cache: dict[str, tuple[float, dict]] = {}


def fetch(sym: str) -> dict:
    hit = _cache.get(sym)
    if hit and time.time() - hit[0] < 86400:
        return hit[1]
    import yfinance as yf

    info = yf.Ticker(sym).info or {}
    data = {k: info[k] for k in KEYS if info.get(k) is not None}
    _cache[sym] = (time.time(), data)
    return data


def rule_score(d: dict) -> tuple[float, str]:
    """Analyst consensus (1 = strong buy .. 5 = sell) plus upside to the mean price target."""
    parts, score = [], 0.0
    if "recommendationMean" in d:
        score += (3 - d["recommendationMean"]) / 2
        parts.append(f"consensus {d['recommendationMean']:.1f}")
    if d.get("targetMeanPrice") and d.get("currentPrice"):
        upside = d["targetMeanPrice"] / d["currentPrice"] - 1
        score += max(-0.5, min(0.5, upside * 2))
        parts.append(f"target upside {upside:+.0%}")
    return score, ", ".join(parts) or "no analyst data"
