"""The sham-objective arm: visible labels, zero payoff role."""
from __future__ import annotations

import json
import random

import pytest
from conftest import SCEN

from commons.core import (
    AGENTS,
    BRIEFING,
    CONDITION,
    EPISODE_TURNS,
    OBJECTIVES,
    SHAM_BRIEFING,
    SHAM_GOAL_TEXT,
    SHAM_LABELS,
    SHAM_MENU_TEXT,
    SYSTEM,
    CommonsEnv,
    ShamCommonsEnv,
    ShamObjectiveError,
    assert_inert,
    legal_action_ids,
    replay_states,
    restore_sham,
)
from commons.render import actor_messages

DISPLAY = dict(zip(AGENTS, SHAM_LABELS))


def test_sham_requires_sham_labels() -> None:
    env = ShamCommonsEnv()
    with pytest.raises(ShamObjectiveError):
        env.reset(1, CONDITION, SCEN)
    with pytest.raises(ShamObjectiveError):
        env.reset(1, CONDITION, SCEN, objectives=dict(zip(AGENTS, OBJECTIVES)))
    with pytest.raises(ShamObjectiveError):
        env.reset(1, CONDITION, SCEN, objectives={"agent_1": SHAM_LABELS[0]})


def test_every_seat_effectively_holds_system_health() -> None:
    env = ShamCommonsEnv()
    state = env.reset(5, CONDITION, SCEN, objectives=DISPLAY)
    assert {s["hidden_objective"] for s in state["agents"].values()} == {SYSTEM}


def test_label_is_visible_to_its_holder_only() -> None:
    env = ShamCommonsEnv()
    env.reset(5, CONDITION, SCEN, objectives=DISPLAY)
    for agent in AGENTS:
        own = env.observe(agent)
        assert own["self"]["hidden_objective"] == DISPLAY[agent]
        assert own["self"]["private_goal"] == SHAM_GOAL_TEXT[DISPLAY[agent]]
        assert own["environment_briefing"] == SHAM_BRIEFING
        public = json.dumps(env.public_observation(agent))
        for label in SHAM_LABELS:
            assert f'"{label}"' not in public
    assert SHAM_MENU_TEXT in SHAM_BRIEFING and SHAM_BRIEFING != BRIEFING


def test_assert_inert_on_a_full_episode() -> None:
    rng = random.Random(2)
    moves = [{a: rng.choice(legal_action_ids(a)) for a in AGENTS}
             for _ in range(EPISODE_TURNS)]
    report = assert_inert(7, CONDITION, SCEN, AGENTS, moves)
    assert report["states_identical"]
    assert report["label_visible_and_differs"]
    assert report["goal_text_visible_and_differs"]
    assert report["effective_is_real"]
    assert report["menu_substituted_in_briefing"]


def test_sham_state_equals_active_state_with_all_system_health() -> None:
    sham = ShamCommonsEnv()
    sham.reset(9, CONDITION, SCEN, objectives=DISPLAY)
    active = CommonsEnv()
    active.reset(9, CONDITION, SCEN, objectives={a: SYSTEM for a in AGENTS})
    rng = random.Random(9)
    for _ in range(EPISODE_TURNS):
        moves = {a: rng.choice(legal_action_ids(a)) for a in AGENTS}
        sham.step(dict(moves))
        active.step(dict(moves))
    assert sham.serialize_state() == active.serialize_state()
    # so a sham state replays with the ordinary replay
    final = sham.serialize_state()
    assert replay_states(final)[EPISODE_TURNS] == final


def test_sham_actor_prompt_shows_label_and_never_the_effective_objective() -> None:
    env = ShamCommonsEnv()
    state = env.reset(3, CONDITION, SCEN, objectives=DISPLAY)
    for agent in AGENTS:
        text = "\n".join(m["content"] for m in
                         actor_messages(state, agent, sham_display=DISPLAY))
        assert DISPLAY[agent] in text
        assert SYSTEM not in text
        assert SHAM_MENU_TEXT in text
    restored = restore_sham(state, DISPLAY)
    assert restored.display_map() == DISPLAY
    with pytest.raises(ShamObjectiveError):
        restored.set_display({"agent_1": SYSTEM})
