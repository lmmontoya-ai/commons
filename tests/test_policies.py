"""Scripted policies and the exact lookahead behind them."""
from __future__ import annotations

from pathlib import Path

import pytest
from conftest import make_env, play_uniform

from commons.core import AGENTS, OBJECTIVES, PERSONAL, SOCIAL, SYSTEM, legal_action_ids
from commons.core import with_objectives as replace_objectives
from commons.policies import (
    POLICIES,
    action_values,
    best_actions,
    fixed_sequence,
    jsd,
    reference_trajectory,
    softmax,
)
from commons.policies import lookahead as lookahead_module


def test_the_best_partner_differs_by_partner_objectives() -> None:
    """Hold the state and the mover's objective fixed, swap which peer wants
    what, and the best SHARE target moves."""
    base = make_env(objectives={"agent_1": PERSONAL, "agent_2": SOCIAL,
                                "agent_3": SYSTEM}).serialize_state()
    swapped = make_env(objectives={"agent_1": PERSONAL, "agent_2": SYSTEM,
                                   "agent_3": SOCIAL}).serialize_state()
    v_base = action_values(base, "agent_1")
    v_swapped = action_values(swapped, "agent_1")
    assert v_base["SHARE_AGENT_2"] > v_base["SHARE_AGENT_3"]
    assert v_swapped["SHARE_AGENT_3"] > v_swapped["SHARE_AGENT_2"]


def test_policy_is_one_function_of_state_and_objective() -> None:
    base = make_env(objectives={"agent_1": PERSONAL, "agent_2": SOCIAL,
                                "agent_3": SYSTEM}).serialize_state()
    other = replace_objectives(base, {"agent_1": SYSTEM, "agent_2": SOCIAL,
                                      "agent_3": PERSONAL})
    p = softmax(action_values(base, "agent_1"))
    q = softmax(action_values(other, "agent_1"))
    assert jsd(p, q) > 0.15
    source = Path(lookahead_module.__file__).read_text(encoding="utf-8")
    body = source.split("def action_values")[1].split("\ndef ")[0]
    for objective in OBJECTIVES:
        assert objective not in body, "the policy branches on the objective"


def test_best_actions_reports_ties() -> None:
    assert best_actions({"A": 1.0, "B": 1.0, "C": 0.0}) == ["A", "B"]


@pytest.mark.parametrize("name", sorted(POLICIES))
def test_policies_return_legal_actions_deterministically(name) -> None:
    policy = POLICIES[name]
    for turn in (0, 3, 7):
        state = play_uniform(2710, "infrastructure_risk", turn)
        for agent in AGENTS:
            action = policy(state, agent)
            assert action in legal_action_ids(agent)
            assert action == policy(state, agent)


def test_fixed_sequence() -> None:
    policy = fixed_sequence(["TAKE_1", "REPAIR_PATCH"])
    state = make_env().serialize_state()
    assert policy(state, "agent_1") == "TAKE_1"
    state["turn"] = 5
    assert policy(state, "agent_1") == "WAIT"


def test_reference_trajectory_is_reproducible() -> None:
    a = reference_trajectory(2700, "delivery_pressure", turns=4)
    b = reference_trajectory(2700, "delivery_pressure", turns=4)
    assert a == b
    assert [s["turn"] for s in a] == [0, 1, 2, 3, 4]
