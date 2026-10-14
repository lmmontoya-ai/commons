"""An offline stand-in for ``OpenRouterClient``.

``FakeClient`` returns canned replies in the OpenAI chat-completion response
shape and never touches the network. It exercises the full model path
(prompt rendering, request, parsing, retries, logging) at zero cost, and it
records every request it receives so tests can inspect the exact messages.
"""
from __future__ import annotations

import itertools
import json
from collections.abc import Callable, Iterable

from .config import ModelSpec
from .openrouter import CallResult

ReplyFn = Callable[[ModelSpec, list[dict]], str]


def action_reply(action: str) -> str:
    """The reply text of a well-formed answer."""
    return json.dumps({"action": action})


class FakeClient:
    def __init__(self, replies: Iterable[str] | ReplyFn, *,
                 provider: str = "fake", cost_usd: float = 0.0) -> None:
        if callable(replies):
            self._reply = replies
        else:
            cycle = itertools.cycle(list(replies))
            self._reply = lambda spec, messages: next(cycle)
        self.provider = provider
        self.cost_usd = cost_usd
        self.requests: list[tuple[ModelSpec, list[dict]]] = []

    def chat(self, spec: ModelSpec, messages: list[dict]) -> CallResult:
        self.requests.append((spec, [dict(m) for m in messages]))
        text = self._reply(spec, messages)
        raw = {
            "id": f"fake-{len(self.requests)}",
            "model": spec.slug,
            "provider": self.provider,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                      "cost": self.cost_usd},
        }
        return CallResult(raw=raw, provider=self.provider, model=spec.slug,
                          usage=raw["usage"], cost_usd=self.cost_usd,
                          latency_s=0.0, attempts=1)


def constant_action_client(action: str) -> FakeClient:
    """A fake model that always answers ``{"action": <action>}``."""
    return FakeClient([action_reply(action)])
