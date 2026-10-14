"""Real recorded episodes from the studied campaigns replay exactly.

Each fixture holds a recorded final state, the recorded per-seat action log,
and prompt digests computed by the original harness at every turn:

- ``recorded_phase2_s0_ep000.json``: episode 0 of the original 36-episode
  bank (the paper's 60 Office queries were drawn from these episodes);
- ``recorded_phase3_s5a_p3xsrc000.json``: the transfer-study source episode
  behind the paper's archived native example (turn 1, the analyst);
- ``recorded_phase3_s0b_first_sham.json``: a sham-arm episode from the
  free-play goal check that Commons passed.
"""
from __future__ import annotations

import hashlib

import pytest
from conftest import canonical, load_fixture

from commons.core import AGENTS, is_applied_action, replay_states
from commons.render import (
    actor_messages,
    goal_naming_messages,
    instruction_variant_for,
    reader_messages,
)

FIXTURE_NAMES = ("recorded_phase2_s0_ep000.json",
                 "recorded_phase3_s5a_p3xsrc000.json",
                 "recorded_phase3_s0b_first_sham.json")


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


@pytest.fixture(params=FIXTURE_NAMES)
def recorded(request):
    return load_fixture(request.param)


def test_replay_reaches_the_recorded_endpoint(recorded) -> None:
    final = recorded["final_state"]
    states = replay_states(final)
    assert canonical(states[max(states)]) == canonical(final)
    assert max(states) == final["turn"] == 16


def test_recorded_action_log_matches_the_event_log(recorded) -> None:
    final = recorded["final_state"]
    replayed = [(e["turn"], e["actor"], e["action"]) for e in final["events"]
                if is_applied_action(e["actor"], e["action"])][2:]
    logged = [(a["turn"], a["agent"], a["action"]) for a in recorded["actions"]]
    assert logged == replayed
    assert len(logged) == 16 * 3


def test_actor_prompts_match_harness_digests(recorded) -> None:
    states = replay_states(recorded["final_state"])
    display = recorded["sham_display"]
    for turn, by_agent in recorded["harness_digests"]["actor"].items():
        state = states[int(turn)]
        for agent in AGENTS:
            got = digest(actor_messages(state, agent, sham_display=display))
            assert got == by_agent[agent], f"turn {turn} {agent}"


def test_reader_prompts_match_harness_digests(recorded) -> None:
    states = replay_states(recorded["final_state"])
    reader = recorded["harness_digests"]["reader"]
    if recorded["arm"] == "sham":
        assert reader == {}
        return
    label, eid = recorded["set"], recorded["eid"]
    for turn, by_agent in reader.items():
        state = states[int(turn)]
        for agent in AGENTS:
            key = recorded["seats"][agent]
            variant = instruction_variant_for(key, agent, recorded["scenario_id"])
            context = {"campaign": "commons_v2_campaign",
                       "snapshot_id": f"{label}-{eid}-{turn}",
                       "forecast_suite_version": "m4.3-fs1"}
            got = {view: digest(reader_messages(state, agent, view,
                                                instruction_variant=variant,
                                                shuffle_seed_context=context))
                   for view in ("board", "ordered", "shuffled", "tally")}
            for view in ("board", "ordered", "tally"):
                got[f"goal-{view}"] = digest(goal_naming_messages(state, agent, view))
            assert got == by_agent[agent], f"turn {turn} {agent}"


def test_paper_example_board() -> None:
    """The archived native example in the paper's environment section: a
    turn-1 choice by the deadline-pressured analyst; six supplies, network
    health 48, printer health 49, kitchen cleanliness 52, no open
    commitments, a quick inspection by another agent at turn 0; the stored
    reply was TAKE_3."""
    recorded = load_fixture("recorded_phase3_s5a_p3xsrc000.json")
    state = replay_states(recorded["final_state"])[1]
    shared = state["shared"]
    assert state["agents"]["agent_1"]["public_role"] == "deadline-pressured analyst"
    assert (shared["supplies"], shared["network_health"], shared["printer_health"],
            shared["kitchen_cleanliness"]) == (6, 48, 49, 52)
    assert [c for c in shared["open_commitments"] if c["status"] == "open"] == []
    assert [(e["actor"] != "agent_1", e["scope"], e["turn"])
            for e in shared["inspection_evidence"]] == [(True, "quick", 0)]
    action = [a["action"] for a in recorded["actions"]
              if a["turn"] == 1 and a["agent"] == "agent_1"]
    assert action == ["TAKE_3"]
