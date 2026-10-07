def sma(xs: list[float], n: int) -> float:
    return sum(xs[-n:]) / min(n, len(xs))


def rsi(xs: list[float], n: int = 14) -> float:
    deltas = [b - a for a, b in zip(xs[-n - 1:], xs[-n:])]
    gain = sum(d for d in deltas if d > 0)
    loss = -sum(d for d in deltas if d < 0)
    if loss == 0:
        return 100.0 if gain else 50.0
    return 100 - 100 / (1 + gain / loss)


def pct_change(xs: list[float], n: int) -> float:
    if len(xs) <= n or xs[-n - 1] == 0:
        return 0.0
    return (xs[-1] / xs[-n - 1] - 1) * 100


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))
