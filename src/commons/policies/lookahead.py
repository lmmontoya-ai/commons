"""Exact two-turn lookahead through the real environment.

Ported from the research harness module
``self_other_prediction/commons_v2_precheck.py`` (the functions the precheck
gates and the scripted reference trajectories use). Logic is unchanged.

    value(state, mover, action) = U_mover(state after action, then one quiet
                                          turn) - U_mover(state)

with every other seat holding WAIT, so the number is the action's own marginal
consequence. Two turns are enough because DELEGATE and SHARE resolve one turn
later. ``U`` is ``objective_utility``, the same function the environment pays
rewards with. The policy is ONE function of (state, objective): the objective
enters only through ``objective_utility``.
"""
from __future__ import annotations

import math
import random
from typing import Any

from ..core.env import (
    AGENTS,
    CONDITION,
    SCORED_TURNS,
    CommonsEnv,
    action_space,
    objective_utility,
)

BETA = 2.0
LOOKAHEAD_TURNS = 2


def _env_from(state: dict[str, Any]) -> CommonsEnv:
    env = CommonsEnv()
    env.restore(state)
    return env


def action_values(state: dict[str, Any], mover: str,
                  turns: int = LOOKAHEAD_TURNS) -> dict[str, float]:
    """Exact marginal value of each legal cell to ``mover``, under the hidden
    assignment carried by ``state``. Every other seat holds WAIT."""
    objective = state["agents"][mover]["hidden_objective"]
    base = _env_from(state)
    start = objective_utility(objective, base.state["agents"][mover],
                              base.state["shared"])
    values: dict[str, float] = {}
    for action in (a["id"] for a in action_space(mover, tuple(AGENTS))):
        env = _env_from(state)
        profile = {a: (action if a == mover else "WAIT") for a in env.agents}
        env.step(profile)
        for _ in range(turns - 1):
            env.step({a: "WAIT" for a in env.agents})
        values[action] = round(
            objective_utility(objective, env.state["agents"][mover],
                              env.state["shared"]) - start, 6)
    return values


def softmax(values: dict[str, float], beta: float = BETA) -> dict[str, float]:
    top = max(values.values())
    weights = {a: math.exp(beta * (v - top)) for a, v in values.items()}
    total = sum(weights.values())
    return {a: w / total for a, w in weights.items()}


def best_actions(values: dict[str, float], tol: float = 1e-9) -> list[str]:
    """All argmax cells (sorted), so an exact tie is never hidden."""
    top = max(values.values())
    return sorted(a for a, v in values.items() if v >= top - tol)


def jsd(p: dict[str, float], q: dict[str, float]) -> float:
    """Jensen-Shannon divergence in bits; 0 identical, 1 disjoint."""
    keys = set(p) | set(q)

    def _kl(a: dict[str, float], m: dict[str, float]) -> float:
        return sum(a.get(k, 0.0) * math.log2(a[k] / m[k])
                   for k in keys if a.get(k, 0.0) > 0)

    m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in keys}
    return max(0.0, 0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def entropy_bits(p: dict[str, float]) -> float:
    return -sum(v * math.log2(v) for v in p.values() if v > 0)


def reference_trajectory(seed: int, scenario: str,
                         turns: int = SCORED_TURNS,
                         beta: float = BETA) -> list[dict[str, Any]]:
    """One goal-directed rollout, states turn 0 .. ``turns``.

    All three seats run the SAME policy (softmax over the same lookahead),
    differing only in the objective the lookahead maximises. Sampling is
    seeded per (seed, scenario, turn), so the trajectory is reproducible.
    """
    env = CommonsEnv()
    env.reset(seed, CONDITION, scenario)
    states = [env.serialize_state()]
    for turn in range(turns):
        state = env.serialize_state()
        rng = random.Random(f"commons_v2_ref:{seed}:{scenario}:{turn}")
        profile = {}
        for agent in env.agents:
            probabilities = softmax(action_values(state, agent), beta)
            actions = sorted(probabilities)
            profile[agent] = rng.choices(
                actions, weights=[probabilities[a] for a in actions])[0]
        env.step(profile)
        states.append(env.serialize_state())
    return states
