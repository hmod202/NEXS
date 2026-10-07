"""Claude calls with validated structured output. Returns None on any failure so agents fall back to rules."""
import logging

import anthropic
from pydantic import BaseModel

log = logging.getLogger("nexs.llm")

# Models that accept effort + server-side refusal fallbacks ("default" form).
_FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5")


class LLM:
    def __init__(self):
        self.client = anthropic.AsyncAnthropic()
        self.available = True
        self.status = "ready"
        self.usage = {"input": 0, "output": 0, "cache_read": 0, "calls": 0}

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
        self.usage["calls"] += 1
        self.usage["input"] += u.input_tokens
        self.usage["output"] += u.output_tokens
        self.usage["cache_read"] += u.cache_read_input_tokens or 0
        if resp.stop_reason == "refusal":
            self.status = "refused"
            return None
        self.status = "ok"
        return resp.parsed_output
