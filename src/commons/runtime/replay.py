"""Freeze a decision and re-sample the actor's policy there.

The world side replays from the final state's event log alone
(``replay_states``). The model side re-sends the byte-identical actor prompt
``n`` times, which is how the studied campaigns estimated a target's
next-action distribution at a preserved decision (two batches of 16 draws).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..core.env import action_space, replay_states
from ..render import actor_messages, messages_digest
from .config import ModelSpec
from .log import read_records
from .parse import parse_action


def load_final_state(path: str | Path) -> dict[str, Any]:
    """The final state recorded in an episode log written by ``run_episode``."""
    for record in read_records(path):
        if record.get("kind") == "result":
            return record["final_state"]
    raise ValueError(f"{path}: no result record (abandoned or incomplete)")


def load_sham_display(path: str | Path) -> dict[str, str] | None:
    for record in read_records(path):
        if record.get("kind") == "meta":
            return record.get("sham_display")
    return None


def states_from_log(path: str | Path) -> dict[int, dict[str, Any]]:
    """Every state of a logged episode, turn 0 through the final turn."""
    return replay_states(load_final_state(path))


@dataclass
class PolicyBatch:
    turn: int
    agent: str
    digest: str
    actions: list[str | None] = field(default_factory=list)
    invalid_draws: int = 0
    providers: list[str | None] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return dict(Counter(a for a in self.actions if a is not None))


def sample_policy(client, spec: ModelSpec, state: dict[str, Any], agent: str,
                  *, n: int, attempts: int = 3, parser: str = "lenient",
                  sham_display: dict[str, str] | None = None,
                  meter=None) -> PolicyBatch:
    """``n`` independent draws of ``agent``'s action at the frozen ``state``.

    Each draw sends the identical actor prompt up to ``attempts`` times until
    a legal action parses. A draw that never parses is recorded as None and
    counted in ``invalid_draws``.
    """
    messages = actor_messages(state, agent, sham_display=sham_display)
    batch = PolicyBatch(turn=int(state["turn"]), agent=agent,
                        digest=messages_digest(messages))
    space = action_space(agent)
    for _ in range(n):
        action = None
        for _attempt in range(attempts):
            result = (meter.call(client, spec, messages) if meter is not None
                      else client.chat(spec, messages))
            batch.providers.append(result.provider)
            action, _error = parse_action(result.content(), space, parser)
            if action is not None:
                break
        if action is None:
            batch.invalid_draws += 1
        batch.actions.append(action)
    return batch
