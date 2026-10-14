"""Agents: a model behind an OpenAI-compatible client, or a scripted policy,
behind one interface: ``act(state, agent, ...) -> Decision``.

Every agent receives the serialized state at the START of the turn, before
any seat's move is applied, so the three choices are simultaneous.

``ModelAgent`` follows the elicitation the studied campaigns used: the actor
prompt is rendered once per decision (``commons.render.actor_messages``) and
sent unchanged on every attempt, with no corrective re-prompt; up to three
attempts; each reply is parsed with the chosen parser. After the attempts
run out, ``on_invalid="wait"`` records WAIT with ``invalid=True`` (phase 2)
and ``on_invalid="abandon"`` abandons the episode (phase 3). Provider errors
never reach this table; the client retries or abandons below this layer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.env import action_space, legal_action_ids
from ..render import actor_messages, messages_digest
from .config import ModelSpec
from .parse import PARSERS, parse_action

ATTEMPTS = 3
ON_INVALID = ("wait", "abandon")


class InvalidActionAbandon(RuntimeError):
    """No legal action after the attempt budget, with on_invalid="abandon"."""


@dataclass
class Decision:
    action: str
    invalid: bool = False
    attempts: int = 1
    digest: str | None = None


class ScriptedAgent:
    """A deterministic policy from ``commons.policies``, run through the same
    loop as models so mixed episodes are possible."""

    def __init__(self, name: str, policy) -> None:
        self.name = name
        self.policy = policy

    def describe(self) -> dict:
        return {"kind": "scripted", "name": self.name}

    def act(self, state: dict[str, Any], agent: str, *,
            sham_display: dict[str, str] | None = None,
            logger=None, meter=None) -> Decision:
        action = self.policy(state, agent)
        if action not in legal_action_ids(agent):
            return Decision("WAIT", invalid=True)
        return Decision(action)


class ModelAgent:
    def __init__(self, spec: ModelSpec, client, *, attempts: int = ATTEMPTS,
                 parser: str = "lenient", on_invalid: str = "wait") -> None:
        if parser not in PARSERS:
            raise ValueError(f"unknown parser {parser!r}; expected {PARSERS}")
        if on_invalid not in ON_INVALID:
            raise ValueError(
                f"unknown on_invalid {on_invalid!r}; expected {ON_INVALID}")
        self.spec = spec
        self.client = client
        self.attempts = attempts
        self.parser = parser
        self.on_invalid = on_invalid

    def describe(self) -> dict:
        return {"kind": "model", "attempts": self.attempts,
                "parser": self.parser, "on_invalid": self.on_invalid,
                **self.spec.describe()}

    def act(self, state: dict[str, Any], agent: str, *,
            sham_display: dict[str, str] | None = None,
            logger=None, meter=None) -> Decision:
        messages = actor_messages(state, agent, sham_display=sham_display)
        digest = messages_digest(messages)
        space = action_space(agent)
        turn = int(state["turn"])
        last_error = "no attempt"
        for attempt in range(1, self.attempts + 1):
            result = (meter.call(self.client, self.spec, messages)
                      if meter is not None
                      else self.client.chat(self.spec, messages))
            action, error = parse_action(result.content(), space, self.parser)
            if logger is not None:
                logger.call(turn=turn, agent=agent, digest=digest,
                            model=self.describe(), attempt=attempt,
                            purpose="act", result=result, action=action,
                            error=error)
            if action is not None:
                return Decision(action, invalid=False, attempts=attempt,
                                digest=digest)
            last_error = error or "invalid reply"
        if self.on_invalid == "abandon":
            raise InvalidActionAbandon(
                f"{self.spec.slug} produced no legal action for {agent} at "
                f"turn {turn} after {self.attempts} attempts ({last_error})")
        return Decision("WAIT", invalid=True, attempts=self.attempts,
                        digest=digest)
