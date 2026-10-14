"""Commons v1 world: menu, reset, transitions, visibility, determinism,
replay. Expected values follow the research harness module
``self_other_prediction/envs/social_sim.py`` (protocol-v8 path)."""
from __future__ import annotations

import json
import random
import subprocess
import sys

import pytest

from commons.v1 import (
    ACTION_FEEDBACK,
    ACTION_SUPPORT_SIZE,
    AGENTS,
    CONDITION,
    PRIVATE_GOALS,
    PUBLIC_ROLES,
    SCENARIO_IDS,
    CommonsV1Env,
    action_family,
    action_space,
    applied_actions,
    legal_action_ids,
    replay_states,
    rollout_states,
    state_id,
    state_sha256,
)


def make(seed: int = 1600, scenario: str = "delivery_pressure",
         feedback: bool = True) -> CommonsV1Env:
    env = CommonsV1Env(action_feedback=feedback)
    env.reset(seed, CONDITION, scenario)
    return env


def uniform_moves(rng: random.Random) -> list[str]:
    return [rng.choice(legal_action_ids(a)) for a in AGENTS]


# ------------------------------------------------------------------ menu

def test_menu_has_eleven_cells_per_seat_and_never_names_the_mover():
    assert ACTION_SUPPORT_SIZE == 11
    for agent in AGENTS:
        ids = legal_action_ids(agent)
        assert len(ids) == len(set(ids)) == 11
        assert ids[:7] == ["TAKE_1", "TAKE_3", "REPAIR_PATCH", "REPAIR_FULL",
                           "INSPECT_QUICK", "INSPECT_THOROUGH", "WAIT"]
        assert not any(agent.upper() in i for i in ids)
        peers = [p for p in AGENTS if p != agent]
        assert ids[7:] == [x for p in peers
                           for x in (f"DELEGATE_{p.upper()}", f"TRADE_{p.upper()}")]


def test_menu_does_not_depend_on_state():
    env = make()
    rng = random.Random(1)
    first = {a: env.action_space(a) for a in AGENTS}
    for _ in range(5):
        env.step(dict(zip(AGENTS, uniform_moves(rng))))
        assert {a: env.action_space(a) for a in AGENTS} == first
    assert first["agent_2"] == action_space("agent_2")


def test_action_families():
    assert action_family("TAKE_3") == "TAKE_RESOURCE"
    assert action_family("REPAIR_FULL") == "REPAIR"
    assert action_family("DELEGATE_AGENT_2") == "DELEGATE"
    assert action_family("TRADE_AGENT_1") == "OFFER_TRADE"
    assert action_family("INSPECT_QUICK") == "INSPECT"
    assert action_family("WAIT") == "WAIT"
    assert {action_family(a) for a in legal_action_ids("agent_1")} == set(ACTION_FEEDBACK)


@pytest.mark.parametrize("moves", [
    {"agent_1": "DELEGATE_AGENT_1"},
    {"agent_2": "TRADE_AGENT_2"},
    {"agent_1": "SHARE_AGENT_2"},
    {"agent_4": "WAIT"},
])
def test_off_menu_actions_are_rejected_before_anything_changes(moves):
    env = make()
    before = env.serialize_state()
    with pytest.raises(ValueError):
        env.step(moves)
    assert env.serialize_state() == before


# ----------------------------------------------------------------- reset

def test_reset_requires_the_studied_condition_and_a_scenario():
    env = CommonsV1Env()
    with pytest.raises(ValueError):
        env.reset(1600, "tradeoff_v7", "delivery_pressure")
    with pytest.raises(ValueError):
        env.reset(1600, "tradeoff_v8_target_knows_predicted", "delivery_pressure")
    with pytest.raises(ValueError):
        env.reset(1600, CONDITION, None)
    with pytest.raises(ValueError):
        env.reset(1600, CONDITION, "unknown")


def test_reset_is_a_function_of_seed_and_scenario():
    for scenario in SCENARIO_IDS:
        assert make(1600, scenario).serialize_state() == make(1600, scenario).serialize_state()
    states = {json.dumps(make(seed).state["agents"], sort_keys=True)
              + json.dumps(make(seed).state["shared"], sort_keys=True)
              for seed in range(1600, 1630)}
    assert len(states) > 1


def test_reset_state_shape_and_opening_events():
    env = make(1600, "delivery_pressure")
    state = env.serialize_state()
    assert list(state) == ["seed", "condition", "scenario_id", "turn", "agents",
                           "shared", "events"]
    for agent in AGENTS:
        record = state["agents"][agent]
        assert list(record) == ["task_progress", "energy", "reputation",
                                "public_role", "private_goal", "coordination_load"]
        assert record["public_role"] == PUBLIC_ROLES[agent]
        assert record["private_goal"] == PRIVATE_GOALS[agent]
        assert 2 <= record["task_progress"] <= 4
    shared = state["shared"]
    assert shared["supplies"] in (8, 9, 10)
    assert shared["network_health"] in (50, 52, 54)
    assert shared["kitchen_cleanliness"] == 54
    # Opening events are applied, then recorded at turn 0.
    assert [(e["actor"], e["action"]) for e in state["events"]] == [
        ("agent_2", "INSPECT_QUICK"), ("agent_3", "TRADE_AGENT_1")]
    assert shared["inspection_evidence"] == [
        {"actor": "agent_2", "scope": "quick", "turn": 0,
         "finding": "one additional symptom recorded"}]
    assert shared["open_trades"][0]["offerer"] == "agent_3"
    assert state["agents"]["agent_3"]["reputation"] == 51
    assert state["agents"]["agent_2"]["reputation"] == 54
    assert all("content" in e for e in state["events"])


def test_infrastructure_opening_delegation_loads_the_maintainer():
    state = make(1601, "infrastructure_risk").serialize_state()
    assert state["agents"]["agent_2"]["coordination_load"] == 1
    assert state["shared"]["open_commitments"] == [
        {"owner": "agent_2", "requester": "agent_3", "task": "shared_follow_up",
         "created_turn": 0, "status": "open"}]


# ----------------------------------------------------------- transitions

def test_take_under_contention_follows_seat_order():
    env = make()
    env.state["shared"]["supplies"] = 2
    before = {a: dict(env.state["agents"][a]) for a in AGENTS}
    result = env.step({"agent_1": "TAKE_3", "agent_2": "TAKE_3", "agent_3": "TAKE_1"})
    after = env.state["agents"]
    assert after["agent_1"]["task_progress"] == before["agent_1"]["task_progress"] + 2
    assert after["agent_2"]["task_progress"] == before["agent_2"]["task_progress"]
    assert after["agent_3"]["task_progress"] == before["agent_3"]["task_progress"]
    assert after["agent_2"]["energy"] == before["agent_2"]["energy"] - 1
    assert after["agent_2"]["reputation"] == before["agent_2"]["reputation"] - 2
    assert result.rewards == {"agent_1": pytest.approx(1.4), "agent_2": 0.6,
                              "agent_3": 0.6}
    assert env.state["shared"]["supplies"] == 0


def test_repair_inspect_delegate_trade_and_wait():
    env = make()
    s0 = env.serialize_state()
    result = env.step({"agent_1": "REPAIR_FULL", "agent_2": "INSPECT_THOROUGH",
                       "agent_3": "DELEGATE_AGENT_1"})
    s1 = env.serialize_state()
    a0, a1 = s0["agents"], s1["agents"]
    # Repairs add 10 to both meters before the 1-point decay of the turn.
    assert s1["shared"]["network_health"] == s0["shared"]["network_health"] + 10 - 1
    assert s1["shared"]["printer_health"] == s0["shared"]["printer_health"] + 10 - 1
    assert s1["shared"]["kitchen_cleanliness"] == s0["shared"]["kitchen_cleanliness"] - 2
    assert a1["agent_1"]["energy"] == a0["agent_1"]["energy"] - 2
    assert a1["agent_1"]["reputation"] == a0["agent_1"]["reputation"] + 2
    assert a1["agent_2"]["energy"] == a0["agent_2"]["energy"] - 1
    assert s1["shared"]["inspection_evidence"][-1] == {
        "actor": "agent_2", "scope": "thorough", "turn": 0,
        "finding": "fault scope narrowed"}
    assert a1["agent_1"]["coordination_load"] == a0["agent_1"]["coordination_load"] + 1
    assert s1["shared"]["open_commitments"][-1]["requester"] == "agent_3"
    assert result.rewards == {"agent_1": 1.2, "agent_2": 0.7, "agent_3": 0.8}
    result = env.step({"agent_1": "TRADE_AGENT_2", "agent_2": "WAIT",
                       "agent_3": "INSPECT_QUICK"})
    assert env.state["shared"]["open_trades"][-1] == {
        "offerer": "agent_1", "recipient": "agent_2", "give": "future_help",
        "want": "task_assistance", "created_turn": 1, "status": "open"}
    assert result.rewards == {"agent_1": 0.7, "agent_2": 0.0, "agent_3": 0.4}
    assert env.state["turn"] == 2


def test_meters_are_clamped_and_supplies_never_negative():
    env = make()
    env.state["shared"].update(network_health=0, printer_health=0,
                               kitchen_cleanliness=1, supplies=0)
    env.step({a: "WAIT" for a in AGENTS})
    shared = env.state["shared"]
    assert (shared["network_health"], shared["printer_health"],
            shared["kitchen_cleanliness"], shared["supplies"]) == (0, 0, 0, 0)


def test_step_touches_no_randomness(monkeypatch):
    env = make()

    def boom(*args, **kwargs):
        raise AssertionError("step used random")

    for name in ("random", "randint", "choice", "shuffle", "Random"):
        monkeypatch.setattr(random, name, boom)
    for _ in range(3):
        env.step({"agent_1": "TAKE_3", "agent_2": "REPAIR_PATCH",
                  "agent_3": "TRADE_AGENT_2"})


# ------------------------------------------------------------ visibility

def test_acknowledgements_are_private_to_the_actor():
    env = make()
    env.step({"agent_1": "TAKE_1", "agent_2": "WAIT", "agent_3": "INSPECT_QUICK"})
    feedback = [e for e in env.state["events"] if e["action"] == "action_feedback"]
    assert [(e["actor"], e["scope"]) for e in feedback] == [
        ("agent_1", "private"), ("agent_2", "private"), ("agent_3", "private")]
    assert feedback[0]["content"] == ACTION_FEEDBACK["TAKE_RESOURCE"]
    public = env.public_state()["recent_events"]
    assert all(e["action"] != "action_feedback" for e in public)
    own = env.observe("agent_1")["recent_events"]
    assert [e["actor"] for e in own if e["action"] == "action_feedback"] == ["agent_1"]
    assert all(e["action"] != "action_feedback"
               for e in env.public_observation("agent_1")["recent_events"])
    # Rendered events carry no turn and no visibility bookkeeping.
    assert all(set(e) <= {"actor", "action", "kind", "content"} for e in own)


def test_feedback_can_be_switched_off():
    env = make(feedback=False)
    env.step({a: "WAIT" for a in AGENTS})
    assert all(e["action"] != "action_feedback" for e in env.state["events"])


def test_public_views_hide_private_records():
    env = make()
    public = env.public_observation("agent_1")
    assert public["self"] == {"public_role": PUBLIC_ROLES["agent_1"]}
    assert all(set(v) == {"public_role"} for v in env.public_state()["agents"].values())
    text = json.dumps(env.public_state()) + json.dumps(public)
    for goal in PRIVATE_GOALS.values():
        assert goal not in text
    assert env.observe("agent_1")["self"]["private_goal"] == PRIVATE_GOALS["agent_1"]


def test_history_horizon_of_24_binds_at_turn_8():
    env = make()
    rng = random.Random(7)
    for _ in range(8):
        env.step(dict(zip(AGENTS, uniform_moves(rng))))
    assert len(env.state["events"]) == 2 + 8 * 6
    assert len(env.public_state()["recent_events"]) == 24
    actor_view = env.observe("agent_1")["recent_events"]
    assert len(actor_view) == 24
    # The actor sees 26 public events plus its 8 acknowledgements; the last
    # 24 of those 34 cover turns 2 to 7, four events per turn.
    assert sum(e["action"] == "action_feedback" for e in actor_view) == 6


# ------------------------------------------------- determinism and replay

def test_rollout_is_deterministic_and_matches_step_by_step_play():
    rng = random.Random(11)
    moves = [uniform_moves(rng) for _ in range(8)]
    a = rollout_states(1605, "coordination_debt", moves)
    b = rollout_states(1605, "coordination_debt",
                       [dict(zip(AGENTS, m)) for m in moves])
    assert a == b and len(a) == 9
    env = make(1605, "coordination_debt")
    for turn, m in enumerate(moves):
        assert env.serialize_state() == a[turn]
        env.step(dict(zip(AGENTS, m)))
    assert env.serialize_state() == a[-1]
    assert state_id(a[-1]) == env.state_id()
    assert len(state_sha256(a[-1])) == 64


def test_replay_from_the_final_state_recovers_every_state():
    rng = random.Random(5)
    for scenario in SCENARIO_IDS:
        for feedback in (True, False):
            moves = [uniform_moves(rng) for _ in range(8)]
            states = rollout_states(1610, scenario, moves, action_feedback=feedback)
            assert replay_states(states[-1]) == states
            assert [[x for _, x in turn] for turn in applied_actions(states[-1])] == moves


def test_replay_refuses_a_tampered_state():
    states = rollout_states(1610, "delivery_pressure", [["WAIT"] * 3] * 3)
    final = json.loads(json.dumps(states[-1]))
    final["shared"]["supplies"] += 1
    with pytest.raises(ValueError):
        replay_states(final)
    final = json.loads(json.dumps(states[-1]))
    final["events"][0]["action"] = "INSPECT_THOROUGH"
    with pytest.raises(ValueError):
        replay_states(final)


def test_importing_commons_does_not_load_v1():
    code = "import sys, commons; print('commons.v1' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "False"
