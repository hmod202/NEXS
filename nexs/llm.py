"""Claude calls with validated structured output. Returns None on any failure so agents fall back to rules."""
import logging
from contextvars import ContextVar

import anthropic
from pydantic import BaseModel

log = logging.getLogger("nexs.llm")

# Models that accept effort + server-side refusal fallbacks ("default" form).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")

# USD per million tokens: (input, output, cache read). Cache writes cost 1.25x input. Thinking bills as output.
# Approximate list prices; the Console's billing page is the source of truth.
PRICES = {
    "claude-fable-5-1": (10.0, 50.0, 0.25), "claude-fable-5": (10.0, 50.0, 1.0),
    "claude-opus-5-5": (4.0, 20.0, 0.20), "claude-opus-5": (5.0, 25.0, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 0.50), "claude-opus-4-7": (5.0, 25.0, 0.50), "claude-opus-4-6": (5.0, 25.0, 0.50),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20), "claude-sonnet-5": (2.0, 10.0, 0.20), "claude-sonnet-4-6": (3.0, 15.0, 0.30),
    "claude-haiku-5-5": (0.10, 0.50, 0.01), "claude-haiku-4-5": (1.0, 5.0, 0.10),
}

# Which agent is calling; set by Desk._step so usage can be attributed without threading it through every call.
current_agent: ContextVar[str] = ContextVar("nexs_agent", default="")


def cost_usd(model: str, inp: int, out: int, cache_read: int, cache_write: int) -> float:
    pin, pout, pread = PRICES.get(model, PRICES["claude-opus-5-5"])
    return (inp * pin + out * pout + cache_read * pread + cache_write * pin * 1.25) / 1e6


class LLM:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic()
        self.available = True
        self.status = "ready"
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "calls": 0, "cost_usd": 0.0}
        self.on_usage = None  # callback(agent, model, input, output, cache_read, cache_write, cost); the desk persists it

    async def structured(self, model: str, effort: str, system: str, user: str, schema: type[BaseModel]):
        if not self.available:
            return None
        kwargs = {}
        if not model.startswith("claude-haiku"):
            kwargs["output_config"] = {"effort": effort}
        if model in _FALLBACK_MODELS:
            kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        try:
            resp = await self.client.beta.messages.parse(
                model=model,
                max_tokens=16000,
                # Stable per-agent prompt first so it is cached; the changing market data goes in `user`.
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_format=schema,
                **kwargs,
            )
        except (anthropic.AuthenticationError, TypeError):  # TypeError: SDK found no credentials at all
            self.available, self.status = False, "no API key — running on rules"
            log.warning("Anthropic authentication failed; LLM agents switch to rule-based mode")
            return None
        except anthropic.RateLimitError:
            self.status = "rate limited"
            return None
        except anthropic.APIStatusError as e:
            self.status = f"API error {e.status_code}"
            log.warning("Claude API error %s: %s", e.status_code, e.message)
            return None
        except anthropic.APIConnectionError:
            self.status = "connection error"
            return None
        u = resp.usage
        read, write = u.cache_read_input_tokens or 0, getattr(u, "cache_creation_input_tokens", 0) or 0
        billed = getattr(resp, "model", None) or model  # a refusal fallback may have served it on another model
        cost = cost_usd(billed, u.input_tokens, u.output_tokens, read, write)
        self.usage["calls"] += 1
        self.usage["input"] += u.input_tokens
        self.usage["output"] += u.output_tokens
        self.usage["cache_read"] += read
        self.usage["cost_usd"] += cost
        if self.on_usage:
            self.on_usage(current_agent.get(), billed, u.input_tokens, u.output_tokens, read, write, cost)
        if resp.stop_reason == "refusal":
            self.status = "refused"
            return None
        self.status = "ok"
        return resp.parsed_output
