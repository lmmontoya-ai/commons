"""Scripted seat policies for zero-cost episodes, baselines, and tests.

A policy is a function ``(state, agent) -> action id``. It sees the full
serialized state, which includes every agent's hidden objective, so a
scripted policy is an audit instrument or a baseline, never a stand-in for a
model's information. The lookahead policies read only the mover's OWN
objective (through ``objective_utility``).

Every policy is deterministic given the state: randomized policies seed their
generator from (seed, scenario, turn, agent).
"""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from ..core.env import legal_action_ids
from .lookahead import BETA, action_values, softmax

Policy = Callable[[dict[str, Any], str], str]


def _rng(tag: str, state: dict[str, Any], agent: str) -> random.Random:
    return random.Random(
        f"commons_scripted_{tag}:{state.get('seed')}:{state.get('scenario_id')}"
        f":{state.get('turn')}:{agent}")


def wait_policy(state: dict[str, Any], agent: str) -> str:
    """Always WAIT."""
    return "WAIT"


def uniform_policy(state: dict[str, Any], agent: str) -> str:
    """Uniform over the eleven legal cells."""
    return _rng("uniform", state, agent).choice(legal_action_ids(agent))


def greedy_policy(state: dict[str, Any], agent: str) -> str:
    """Argmax of the exact two-turn lookahead for the mover's own objective.
    Ties go to the first tied action in menu order."""
    values = action_values(state, agent)
    top = max(values.values())
    for action in legal_action_ids(agent):
        if values[action] >= top - 1e-9:
            return action
    raise AssertionError("unreachable: some action attains the maximum")


def softmax_policy(state: dict[str, Any], agent: str,
                   beta: float = BETA) -> str:
    """Sample from softmax(beta * lookahead value), beta = 2 by default."""
    probabilities = softmax(action_values(state, agent), beta)
    actions = sorted(probabilities)
    return _rng("softmax", state, agent).choices(
        actions, weights=[probabilities[a] for a in actions])[0]


POLICIES: dict[str, Policy] = {
    "wait": wait_policy,
    "uniform": uniform_policy,
    "greedy": greedy_policy,
    "softmax": softmax_policy,
}


def fixed_sequence(moves: list[str], default: str = "WAIT") -> Policy:
    """A policy that plays ``moves[turn]`` and ``default`` after the list
    runs out. Useful for scripted test scenarios."""
    def policy(state: dict[str, Any], agent: str) -> str:
        turn = int(state["turn"])
        return moves[turn] if turn < len(moves) else default
    return policy
