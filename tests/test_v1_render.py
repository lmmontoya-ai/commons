"""Commons v1 prompts: byte equality with prompt texts exported from the
original harness, structure, and what each reader view may contain."""
from __future__ import annotations

import json
from collections import Counter

import pytest
from conftest import GOLDEN

from commons.render import INSTRUCTION_VARIANTS
from commons.v1 import (
    AGENTS,
    BRIEFING,
    PRIVATE_GOALS,
    actor_messages,
    legal_action_ids,
    position,
    position_actor_messages,
    position_reader_messages,
    position_state,
    reader_messages,
    reader_observation,
    rollout_states,
)

GOLDEN_POSITION = "concordia_predictive_commons-seed1600-turn8-agent_1-3356000a8932@t8"


def _all_keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for child in value.values() for k in _all_keys(child)}
    if isinstance(value, list):
        return {k for child in value for k in _all_keys(child)}
    return set()


def _text(messages) -> str:
    return "".join(f"===== {m['role']} =====\n{m['content']}\n\n" for m in messages)


def _late_state() -> dict:
    moves = [["TAKE_1", "REPAIR_PATCH", "DELEGATE_AGENT_1"],
             ["TRADE_AGENT_3", "INSPECT_QUICK", "WAIT"],
             ["REPAIR_FULL", "INSPECT_THOROUGH", "TRADE_AGENT_2"],
             ["TAKE_3", "DELEGATE_AGENT_3", "INSPECT_QUICK"]] * 2
    return rollout_states(1612, "infrastructure_risk", moves)[-1]


@pytest.mark.parametrize("name, build", [
    ("v1_actor_seed1600_agent_1_t8.txt",
     lambda pos: position_actor_messages(pos)),
    ("v1_reader_board_seed1600_agent_1_t8.txt",
     lambda pos: position_reader_messages(pos, "board")),
    ("v1_reader_ordered_seed1600_agent_1_t8.txt",
     lambda pos: position_reader_messages(pos, "ordered")),
])
def test_prompts_equal_the_harness_export_byte_for_byte(name, build):
    expected = (GOLDEN / name).read_bytes()
    assert _text(build(position(GOLDEN_POSITION))).encode("utf-8") == expected


def test_json_message_layout():
    state = _late_state()
    for messages, task in ((actor_messages(state, "agent_2"), "choose_action"),
                           (reader_messages(state, "agent_2"),
                            "forecast_target_action")):
        system, user = messages
        assert system["role"] == "system" and user["role"] == "user"
        assert system["content"].startswith(BRIEFING + "\n\n")
        assert system["content"].endswith(", ".join(legal_action_ids("agent_2")) + ".")
        payload = json.loads(user["content"])
        assert sorted(payload) == ["action_space", "observation", "return_schema", "task"]
        assert payload["task"] == task
        assert payload["return_schema"] == {
            "action": "one valid action id from action_space"}
        assert [a["id"] for a in payload["action_space"]] == legal_action_ids("agent_2")
        assert "environment_briefing" not in json.dumps(payload["observation"])
        # Exactly json.dumps with sorted keys and default separators.
        assert user["content"] == json.dumps(payload, sort_keys=True)


def test_actor_sees_its_private_goal_and_acknowledgements():
    observation = json.loads(actor_messages(_late_state(), "agent_3")[1]["content"])[
        "observation"]
    assert observation["self"]["private_goal"] == PRIVATE_GOALS["agent_3"]
    own = [e for e in observation["recent_events"] if e["action"] == "action_feedback"]
    assert own and all(e["actor"] == "agent_3" for e in own)
    assert len(observation["recent_events"]) == 24


@pytest.mark.parametrize("target", AGENTS)
def test_reader_views_hold_public_information_only(target):
    state = _late_state()
    for view in ("board", "ordered", "shuffled"):
        text = json.dumps(reader_messages(state, target, view))
        assert "action_feedback" not in text
        assert "private_goal" not in text
        for goal in PRIVATE_GOALS.values():
            assert goal not in text
        observation = json.loads(json.loads(text)[1]["content"])["observation"]
        keys = _all_keys(observation)
        for private in ("task_progress", "energy", "reputation", "coordination_load"):
            assert private not in keys


def test_board_view_has_no_event_list_but_keeps_the_board():
    state = _late_state()
    board = reader_observation(state, "agent_1", "board")
    assert board["target_observation"]["recent_events"] == []
    assert board["public_state"]["recent_events"] == []
    assert board["public_state"]["shared"]["open_trades"]
    ordered = reader_observation(state, "agent_1", "ordered")
    assert len(ordered["public_state"]["recent_events"]) == 24
    strip = lambda o: {k: v for k, v in o.items() if k != "recent_events"}  # noqa: E731
    assert strip(board["public_state"]) == strip(ordered["public_state"])


def test_shuffled_view_reorders_events_only():
    state = _late_state()
    ordered = reader_observation(state, "agent_1", "ordered")
    shuffled = reader_observation(state, "agent_1", "shuffled",
                                  shuffle_seed_context={"k": 1})
    for key in ("target_observation", "public_state"):
        a, b = ordered[key]["recent_events"], shuffled[key]["recent_events"]
        assert a != b
        assert Counter(json.dumps(e, sort_keys=True) for e in a) == Counter(
            json.dumps(e, sort_keys=True) for e in b)
    again = reader_observation(state, "agent_1", "shuffled",
                               shuffle_seed_context={"k": 1})
    assert again == shuffled


def test_instruction_variants_change_only_the_instruction():
    state = _late_state()
    observations = {v: reader_observation(state, "agent_2", "ordered",
                                          instruction_variant=v)
                    for v in INSTRUCTION_VARIANTS}
    for variant, observation in observations.items():
        assert observation["instruction"] == INSTRUCTION_VARIANTS[variant]
        rest = {k: x for k, x in observation.items() if k != "instruction"}
        assert rest == {k: x for k, x in observations["V1"].items()
                        if k != "instruction"}


def test_aliases_and_errors():
    state = _late_state()
    assert reader_messages(state, "agent_1", "situation") == reader_messages(
        state, "agent_1", "board_only")
    assert reader_messages(state, "agent_1", "full_record") == reader_messages(
        state, "agent_1", "full_history")
    with pytest.raises(ValueError):
        reader_messages(state, "agent_1", "tally")
    with pytest.raises(ValueError):
        reader_messages(state, "agent_1", "ordered", instruction_variant="V4")
    with pytest.raises(ValueError):
        actor_messages(state, "agent_1", prompt_format="yaml")


def test_markdown_rendering_carries_the_same_frame():
    pos = position(GOLDEN_POSITION)
    markdown = position_reader_messages(pos, "ordered", prompt_format="markdown")
    json_messages = position_reader_messages(pos, "ordered")
    assert markdown[0] == json_messages[0]
    assert markdown[1]["content"].startswith("## Task\n")
    assert "## Response" in markdown[1]["content"]
    assert position_state(pos)["turn"] == 8


def test_show_v1_prompt_script_reports_a_match():
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "show_v1_prompt.py"
    for view in ("board", "ordered", "actor"):
        out = subprocess.run([sys.executable, str(script), "--position", "0",
                              "--view", view], capture_output=True, text=True,
                             encoding="utf-8", check=True).stdout
        assert out.startswith("===== system =====\n")
        assert out.rstrip().endswith("(match)")
    listing = subprocess.run([sys.executable, str(script), "--list"],
                             capture_output=True, text=True, check=True).stdout
    assert len(listing.splitlines()) == 48
