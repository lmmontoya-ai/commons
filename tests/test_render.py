"""Actor and reader prompts: structure, information content, determinism,
and byte equality with prompt texts exported from the original harness."""
from __future__ import annotations

import random
from collections import Counter

import pytest
from conftest import GOLDEN, canonical, make_env, play_uniform

from commons.core import (
    AGENTS,
    BRIEFING,
    CONDITION,
    GOAL_TEXT,
    OBJECTIVES,
    CommonsEnv,
    legal_action_ids,
)
from commons.render import (
    COUNT_CONTRACT,
    EM_DASH,
    HISTORY_CONTRACT,
    INSTRUCTION_VARIANTS,
    actor_messages,
    goal_naming_messages,
    instruction_variant_for,
    instruction_variant_map,
    messages_digest,
    objective_menu,
    reader_messages,
    reader_observation,
)
from commons.render.views import _all_keys


def _state_at(scenario: str, seed: int, turn: int) -> dict:
    """Uniform play from random.Random(f"golden:{scenario}:{seed}"), the
    trajectory the golden files were exported from."""
    env = CommonsEnv()
    env.reset(seed, CONDITION, scenario)
    rng = random.Random(f"golden:{scenario}:{seed}")
    for _ in range(turn):
        env.step({a: rng.choice(legal_action_ids(a)) for a in AGENTS})
    return env.serialize_state()


def _text(messages) -> str:
    return "".join(f"===== {m['role']} =====\n{m['content']}\n\n" for m in messages)


GOLDEN_CASES = [
    ("actor_delivery_pressure_s2700_t0_agent_1.txt", "delivery_pressure", 2700, 0,
     "agent_1", "actor"),
    ("reader_ordered_delivery_pressure_s2700_t3_agent_2.txt", "delivery_pressure",
     2700, 3, "agent_2", "ordered"),
    ("reader_tally_delivery_pressure_s2700_t3_agent_2.txt", "delivery_pressure",
     2700, 3, "agent_2", "tally"),
    ("goal_ordered_delivery_pressure_s2700_t3_agent_2.txt", "delivery_pressure",
     2700, 3, "agent_2", "goal-ordered"),
]


@pytest.mark.parametrize("name,scenario,seed,turn,agent,view", GOLDEN_CASES)
def test_prompts_match_harness_export_byte_for_byte(name, scenario, seed, turn,
                                                   agent, view) -> None:
    state = _state_at(scenario, seed, turn)
    if view == "actor":
        messages = actor_messages(state, agent)
    elif view.startswith("goal-"):
        messages = goal_naming_messages(state, agent, view[5:])
    else:
        # the golden export keyed the instruction variant on luna (seed % 3 == 0)
        variant = instruction_variant_for("openrouter_gpt56_luna_high", agent,
                                          scenario)
        messages = reader_messages(
            state, agent, view, instruction_variant=variant,
            shuffle_seed_context={"campaign": "commons_v2_campaign",
                                  "snapshot_id": f"golden-{scenario}-{seed}-{turn}",
                                  "forecast_suite_version": "m4.3-fs1"})
    expected = (GOLDEN / name).read_bytes().decode("utf-8")
    assert _text(messages) == expected


def test_actor_prompt_structure() -> None:
    state = make_env().serialize_state()
    for agent in AGENTS:
        system, user = (m["content"] for m in actor_messages(state, agent))
        assert system.startswith(BRIEFING + "\n\n")
        assert ", ".join(legal_action_ids(agent)) in system
        assert BRIEFING not in user
        assert user.startswith("## Task\nChoose your next action.")
        own = state["agents"][agent]["hidden_objective"]
        assert own in user and GOAL_TEXT[own] in user
        for other in set(OBJECTIVES) - {own}:
            assert other not in user
        assert user.rstrip().endswith("```")


def test_prompts_are_deterministic() -> None:
    state = play_uniform(2701, "coordination_debt", 5)
    for agent in AGENTS:
        assert actor_messages(state, agent) == actor_messages(state, agent)
        for view in ("board", "ordered", "shuffled", "tally"):
            a = reader_messages(state, agent, view)
            assert a == reader_messages(state, agent, view)
            assert messages_digest(a) == messages_digest(
                reader_messages(state, agent, view))


def test_reader_views_carry_no_hidden_objective() -> None:
    state = play_uniform(2702, "infrastructure_risk", 6)
    for agent in AGENTS:
        for view in ("board", "ordered", "shuffled", "tally"):
            obs = reader_observation(state, agent, view)
            body = canonical({k: v for k, v in obs.items()
                              if k != "environment_briefing"})
            for objective in OBJECTIVES:
                assert objective not in body
            for text in GOAL_TEXT.values():
                assert text not in body


def test_reader_views_differ_only_in_history() -> None:
    state = play_uniform(2703, "delivery_pressure", 5)
    target = "agent_3"
    board = reader_observation(state, target, "board")
    ordered = reader_observation(state, target, "ordered")
    shuffled = reader_observation(state, target, "shuffled")
    tally = reader_observation(state, target, "tally")
    assert board["target_observation"]["recent_events"] == []
    assert board["public_state"]["recent_events"] == []
    events = ordered["public_state"]["recent_events"]
    assert len(events) > 3
    assert ordered["target_observation"]["recent_events"] == events
    assert shuffled["public_state"]["recent_events"] != events
    assert sorted(map(canonical, shuffled["public_state"]["recent_events"])) == \
        sorted(map(canonical, events))
    assert "recent_events" not in _all_keys(tally)
    counts = {(c["actor"], c["action"]): c["count"]
              for c in tally["public_state"]["recent_event_counts"]}
    assert counts == dict(Counter((e["actor"], e["action"]) for e in events))
    assert HISTORY_CONTRACT in ordered["environment_briefing"]
    assert COUNT_CONTRACT in tally["environment_briefing"]
    for obs in (board, ordered, shuffled, tally):
        assert obs["perspective"] == "forecasting_target"
        assert obs["target_id"] == target
        assert obs["instruction"] == INSTRUCTION_VARIANTS["V1"]
    strip = {"recent_events", "recent_event_counts", "environment_briefing"}

    def without_history(value):
        if isinstance(value, dict):
            return {k: without_history(v) for k, v in value.items() if k not in strip}
        return value

    assert without_history(board) == without_history(ordered) == \
        without_history(shuffled) == without_history(tally)


def test_bilateral_outcomes_are_masked_for_readers() -> None:
    env = make_env(objectives={"agent_1": "PERSONAL_THROUGHPUT",
                               "agent_2": "SOCIAL_CREDIT",
                               "agent_3": "SYSTEM_HEALTH"})
    env.step({"agent_1": "SHARE_AGENT_2", "agent_2": "WAIT", "agent_3": "WAIT"})
    env.step({a: "WAIT" for a in AGENTS})
    state = env.serialize_state()
    text = reader_messages(state, "agent_1", "ordered")[1]["content"]
    assert "bilateral_exchange" in text
    assert "exchange_reciprocated" not in text
    own = actor_messages(state, "agent_1")[1]["content"]
    assert "exchange_reciprocated" in own


def test_instruction_variant_map_is_complete_for_the_paper_roster() -> None:
    mapping = instruction_variant_map()
    assert len(mapping) == 108
    assert set(mapping.values()) == {"V1", "V2", "V3"}
    for model in ("openrouter_gpt56_luna_high", "openrouter_sonnet5_adaptive_r2",
                  "openrouter_grok45", "openrouter_gemini37_flash",
                  "openrouter_glm53"):
        for agent in AGENTS:
            for scenario in ("delivery_pressure", "infrastructure_risk",
                             "coordination_debt"):
                assert instruction_variant_for(model, agent, scenario) in \
                    INSTRUCTION_VARIANTS
    with pytest.raises(KeyError):
        instruction_variant_for("unknown_model", "agent_1", "delivery_pressure")
    with pytest.raises(ValueError):
        reader_observation(make_env().serialize_state(), "agent_1", "ordered",
                           instruction_variant="V9")
    with pytest.raises(ValueError):
        reader_observation(make_env().serialize_state(), "agent_1", "telepathy")


def test_goal_naming_prompt() -> None:
    state = play_uniform(2704, "coordination_debt", 4)
    menu = objective_menu()
    assert [m["id"] for m in menu] == list(OBJECTIVES)
    system, user = (m["content"] for m in goal_naming_messages(state, "agent_2"))
    assert system.startswith(BRIEFING + "\n\n")
    assert "Identify the target agent's hidden objective." in system
    assert user.startswith("## Task\nIdentify target hidden objective.")
    for objective in OBJECTIVES:
        assert f"`{objective}`" in user


def test_rendered_separator_is_the_studied_em_dash() -> None:
    user = actor_messages(make_env().serialize_state(), "agent_1")[1]["content"]
    assert f"`TAKE_1` {EM_DASH} Take one supply unit" in user
