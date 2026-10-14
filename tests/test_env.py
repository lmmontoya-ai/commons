"""The world: menu, hidden objectives, belief-coupled outcomes, determinism,
replay, and invariants. Adapted from the research harness's environment
tests (only the environment-level ones)."""
from __future__ import annotations

import json
import random
import sys

import pytest
from conftest import SCEN, make_env, play_uniform

from commons.core import (
    ACTION_SUPPORT_SIZE,
    AGENTS,
    CONDITION,
    DELEGATION_OUTCOME,
    EPISODE_TURNS,
    GOAL_TEXT,
    OBJECTIVE_PERMUTATIONS,
    OBJECTIVES,
    PERSONAL,
    RESOLUTION_ACTIONS,
    SCENARIO_IDS,
    SCORED_TURNS,
    SHARE_OUTCOME,
    SOCIAL,
    SYSTEM,
    CommonsEnv,
    action_space,
    assign_objectives,
    is_applied_action,
    legal_action_ids,
    objective_utility,
    replay_states,
    with_objectives,
)

# ------------------------------------------------------------------- menu

def test_menu_is_eleven_cells_and_constant_across_every_state() -> None:
    assert ACTION_SUPPORT_SIZE == 11
    env = make_env()
    reference = {a: env.action_space(a) for a in env.agents}
    for agent in env.agents:
        assert len(reference[agent]) == 11
        assert len({x["id"] for x in reference[agent]}) == 11
        # no cell names the mover itself
        assert agent.upper() not in " ".join(x["id"] for x in reference[agent])
    rng = random.Random(7)
    for _ in range(EPISODE_TURNS):
        env.step({a: rng.choice(legal_action_ids(a)) for a in env.agents})
        for agent in env.agents:
            assert env.action_space(agent) == reference[agent]


def test_menu_layout() -> None:
    ids = legal_action_ids("agent_1")
    assert ids == ["TAKE_1", "TAKE_3", "REPAIR_PATCH", "REPAIR_FULL",
                   "INSPECT_QUICK", "INSPECT_THOROUGH", "WAIT",
                   "DELEGATE_AGENT_2", "SHARE_AGENT_2",
                   "DELEGATE_AGENT_3", "SHARE_AGENT_3"]
    assert legal_action_ids("agent_2")[7:] == [
        "DELEGATE_AGENT_1", "SHARE_AGENT_1", "DELEGATE_AGENT_3", "SHARE_AGENT_3"]


def test_resolution_events_are_never_legal_choices() -> None:
    for agent in AGENTS:
        for action in RESOLUTION_ACTIONS:
            assert not is_applied_action(agent, action)
        for action in legal_action_ids(agent):
            assert is_applied_action(agent, action)


# ------------------------------------------------------ objective assignment

def test_assignment_is_a_deterministic_permutation() -> None:
    for seed in range(1600, 1640):
        first = assign_objectives(seed)
        assert first == assign_objectives(seed)
        assert sorted(first.values()) == sorted(OBJECTIVES)
    drawn = {tuple(assign_objectives(s)[a] for a in AGENTS)
             for s in range(1600, 1700)}
    assert drawn == set(OBJECTIVE_PERMUTATIONS), "assignment is not surjective"


def test_independent_mode_is_available_and_deterministic() -> None:
    a = assign_objectives(9, AGENTS, "independent")
    assert a == assign_objectives(9, AGENTS, "independent")
    assert set(a.values()) <= set(OBJECTIVES)
    with pytest.raises(ValueError):
        assign_objectives(9, AGENTS, "no_such_mode")


def test_explicit_objectives_override_the_draw_and_are_validated() -> None:
    forced = {"agent_1": SYSTEM, "agent_2": SYSTEM, "agent_3": PERSONAL}
    env = make_env(objectives=forced)
    assert env.objective_assignment() == forced
    assert env.state["objective_assignment_mode"] == "explicit"
    with pytest.raises(ValueError):
        make_env(objectives={"agent_1": SYSTEM})
    with pytest.raises(ValueError):
        make_env(objectives={a: "MAXIMISE_CHAOS" for a in AGENTS})


def test_reset_fails_closed_on_a_missing_scenario_in_the_studied_condition() -> None:
    env = CommonsEnv()
    with pytest.raises(ValueError):
        env.reset(1600, CONDITION, None)
    with pytest.raises(ValueError):
        env.reset(1600, CONDITION, "no_such_scenario")


def test_prediction_awareness_conditions_are_refused() -> None:
    with pytest.raises(ValueError):
        CommonsEnv().reset(1600, "hidden_objectives_v2_target_knows_predicted", SCEN)


def test_other_condition_labels_resolve_the_scenario_from_the_seed() -> None:
    env = CommonsEnv()
    state = env.reset(4, "smoke")
    assert state["scenario_id"] == SCENARIO_IDS[4 % 3]


# --------------------------------------------------------------- hiddenness

def test_objectives_are_visible_to_the_holder_and_to_nobody_else() -> None:
    env = make_env()
    truth = env.objective_assignment()
    for agent in env.agents:
        own = env.observe(agent)
        assert own["self"]["hidden_objective"] == truth[agent]
        assert own["self"]["private_goal"] == GOAL_TEXT[truth[agent]]
        outside = json.dumps({k: v for k, v in own.items() if k != "self"})
        for objective in OBJECTIVES:
            assert objective not in outside.replace(env.briefing(), "")
        public = env.public_observation(agent)
        assert set(public["self"]) == {"public_role"}
    board = json.dumps(env.public_state())
    for objective in OBJECTIVES:
        assert objective not in board
    for text in GOAL_TEXT.values():
        assert text not in board


def test_public_role_cannot_proxy_for_the_objective() -> None:
    seen: dict[str, set[str]] = {a: set() for a in AGENTS}
    for seed in range(1600, 1640):
        for agent, objective in assign_objectives(seed).items():
            seen[agent].add(objective)
    assert all(v == set(OBJECTIVES) for v in seen.values())


def test_share_outcomes_are_bilateral_and_delegation_outcomes_are_public() -> None:
    env = make_env(objectives={"agent_1": PERSONAL, "agent_2": SOCIAL,
                               "agent_3": SYSTEM})
    env.step({"agent_1": "SHARE_AGENT_2", "agent_2": "WAIT",
              "agent_3": "DELEGATE_AGENT_2"})
    env.step({a: "WAIT" for a in env.agents})
    sharer = [e["action"] for e in env.visible_history("agent_1")]
    recipient = [e["action"] for e in env.visible_history("agent_2")]
    outsider = [e["action"] for e in env.visible_history("agent_3")]
    public = [e["action"] for e in env.visible_history(None)]
    assert "exchange_reciprocated" in sharer
    assert "exchange_reciprocated" in recipient
    assert "exchange_reciprocated" not in outsider
    assert "bilateral_exchange" in outsider
    assert "bilateral_exchange" in public
    assert "follow_up_completed" in public


def test_the_event_log_never_writes_an_objective_label() -> None:
    env = make_env()
    rng = random.Random(3)
    for _ in range(EPISODE_TURNS):
        env.step({a: rng.choice(legal_action_ids(a)) for a in env.agents})
    log = json.dumps(env.state["events"])
    for objective in OBJECTIVES:
        assert objective not in log
    for text in GOAL_TEXT.values():
        assert text not in log


def test_history_horizon_is_96_and_board_view_drops_all_events() -> None:
    env = make_env()
    assert env.history_horizon == 96
    assert env.visible_history(None, limit=0) == []
    assert len(env.visible_history(None)) == 2  # the two opening events


# ------------------------------------------- payoff coupling to the partner

def _delegation_health_delta(owner_objective: str) -> float:
    env = make_env(objectives={"agent_1": SOCIAL, "agent_2": owner_objective,
                               "agent_3": PERSONAL})
    env.step({"agent_1": "DELEGATE_AGENT_2", "agent_2": "WAIT",
              "agent_3": "WAIT"})
    mid = env.state["shared"]["network_health"]
    env.step({a: "WAIT" for a in env.agents})
    return env.state["shared"]["network_health"] - mid


def test_delegation_payoff_depends_on_the_recipients_objective() -> None:
    deltas = {o: _delegation_health_delta(o) for o in OBJECTIVES}
    assert deltas[SYSTEM] > deltas[SOCIAL] - 1 > deltas[PERSONAL]
    assert deltas[SYSTEM] - deltas[PERSONAL] == (
        DELEGATION_OUTCOME[SYSTEM]["health"]
        - DELEGATION_OUTCOME[PERSONAL]["health"])


def test_delegation_credits_and_debits_the_delegator_by_partner_type() -> None:
    # infrastructure_risk is the one scenario whose opening events leave
    # agent_1 alone, so the ledger below is the delegation's own effect.
    for owner_objective in OBJECTIVES:
        env = make_env(scenario=SCENARIO_IDS[1],
                       objectives={"agent_1": SOCIAL, "agent_2": owner_objective,
                                   "agent_3": PERSONAL})
        env.step({"agent_1": "DELEGATE_AGENT_2", "agent_2": "WAIT",
                  "agent_3": "WAIT"})
        env.step({a: "WAIT" for a in env.agents})
        me = env.state["agents"]["agent_1"]
        assert me["credit"] == DELEGATION_OUTCOME[owner_objective]["delegator_credit"]
        assert me["coordination_debt"] == DELEGATION_OUTCOME[
            owner_objective]["delegator_debt"]


def test_delegation_status_is_public_on_the_board() -> None:
    env = make_env(scenario=SCENARIO_IDS[1],
                   objectives={"agent_1": SOCIAL, "agent_2": PERSONAL,
                               "agent_3": SYSTEM})
    env.step({"agent_1": "DELEGATE_AGENT_2", "agent_2": "WAIT", "agent_3": "WAIT"})
    env.step({a: "WAIT" for a in env.agents})
    statuses = [c["status"] for c in env.public_state()["shared"]["open_commitments"]
                if c["requester"] == "agent_1"]
    assert statuses == ["lapsed"]


def test_share_payoff_sign_flips_with_the_recipients_objective() -> None:
    net: dict[str, int] = {}
    for recipient_objective in OBJECTIVES:
        env = make_env(scenario=SCENARIO_IDS[1],
                       objectives={"agent_1": PERSONAL,
                                   "agent_2": recipient_objective,
                                   "agent_3": SYSTEM})
        before = env.state["agents"]["agent_1"]["task_progress"]
        env.step({"agent_1": "SHARE_AGENT_2", "agent_2": "WAIT",
                  "agent_3": "WAIT"})
        env.step({a: "WAIT" for a in env.agents})
        net[recipient_objective] = (
            env.state["agents"]["agent_1"]["task_progress"] - before)
    assert net[SOCIAL] > 0 > net[PERSONAL]
    assert net[PERSONAL] == net[SYSTEM] == -1
    assert net[SOCIAL] == SHARE_OUTCOME[SOCIAL]["sharer_progress"] - 1


def test_reward_is_the_agents_own_objective_utility_delta() -> None:
    env = make_env(objectives={"agent_1": PERSONAL, "agent_2": SYSTEM,
                               "agent_3": SOCIAL})
    before = {a: objective_utility(s["hidden_objective"], s, env.state["shared"])
              for a, s in env.state["agents"].items()}
    result = env.step({"agent_1": "TAKE_1", "agent_2": "REPAIR_FULL",
                       "agent_3": "DELEGATE_AGENT_2"})
    for agent, state in env.state["agents"].items():
        after = objective_utility(state["hidden_objective"], state,
                                  env.state["shared"])
        assert result.rewards[agent] == pytest.approx(after - before[agent])


def test_seat_order_resolves_supply_contention() -> None:
    """Choices are simultaneous but applied in dict order: with three units
    left, the first seat to TAKE_3 gets all of them."""
    for order in (("agent_1", "agent_2"), ("agent_2", "agent_1")):
        env = make_env()
        env.state["shared"]["supplies"] = 3
        progress = {a: s["task_progress"] for a, s in env.state["agents"].items()}
        env.step({order[0]: "TAKE_3", order[1]: "TAKE_3", "agent_3": "WAIT"})
        gained = {a: env.state["agents"][a]["task_progress"] - progress[a]
                  for a in order}
        assert gained == {order[0]: 3, order[1]: 0}


# -------------------------------------------------------------- determinism

@pytest.mark.parametrize("scenario", SCENARIO_IDS)
def test_same_seed_twice_is_the_same_episode(scenario: str) -> None:
    for seed in (1600, 1607, 1631):
        a, b = play_uniform(seed, scenario, EPISODE_TURNS), play_uniform(
            seed, scenario, EPISODE_TURNS)
        assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_different_seeds_give_different_episodes() -> None:
    states = {json.dumps(play_uniform(s, SCEN, EPISODE_TURNS), sort_keys=True)
              for s in range(1600, 1606)}
    assert len(states) > 1


@pytest.mark.parametrize("scenario", SCENARIO_IDS)
def test_replay_from_the_event_log_is_byte_identical(scenario: str) -> None:
    for seed in (1600, 1613):
        final = play_uniform(seed, scenario, EPISODE_TURNS)
        states = replay_states(final)
        assert len(states) == EPISODE_TURNS + 1
        assert json.dumps(states[EPISODE_TURNS], sort_keys=True) == \
            json.dumps(final, sort_keys=True)
        assert states[0]["turn"] == 0
        assert states[SCORED_TURNS]["turn"] == SCORED_TURNS


def test_replay_detects_a_tampered_log() -> None:
    final = play_uniform(1600, SCEN, EPISODE_TURNS)
    final["events"] = [e for e in final["events"]
                       if not (e["turn"] == 3 and e["actor"] == "agent_2")]
    with pytest.raises(ValueError):
        replay_states(final)


def test_step_touches_no_randomness(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env()
    module = sys.modules["commons.core.env"]

    class Forbidden:
        def __getattr__(self, name):  # pragma: no cover - must never run
            raise AssertionError(f"step() reached random.{name}")

    monkeypatch.setattr(module, "random", Forbidden())
    rng = random.Random(11)
    for _ in range(EPISODE_TURNS):
        env.step({a: rng.choice(legal_action_ids(a)) for a in env.agents})


def test_snapshot_restore_round_trips() -> None:
    env = make_env()
    env.step({a: "WAIT" for a in env.agents})
    before = env.snapshot()
    env.step({"agent_1": "TAKE_3", "agent_2": "REPAIR_FULL",
              "agent_3": "SHARE_AGENT_1"})
    assert env.serialize_state() != before
    env.restore(before)
    assert env.serialize_state() == before


def test_with_objectives_changes_only_the_assignment() -> None:
    base = make_env(objectives={"agent_1": PERSONAL, "agent_2": SOCIAL,
                                "agent_3": SYSTEM}).serialize_state()
    other = with_objectives(base, {"agent_1": SYSTEM, "agent_3": PERSONAL})
    assert other["shared"] == base["shared"]
    assert other["events"] == base["events"]
    assert other["agents"]["agent_1"]["hidden_objective"] == SYSTEM
    assert other["agents"]["agent_1"]["private_goal"] == GOAL_TEXT[SYSTEM]
    assert other["objective_assignment_mode"] == "counterfactual"


# ------------------------------------------------------------- invariants

def test_stocks_stay_in_range_over_full_episodes() -> None:
    for seed in range(1600, 1612):
        for scenario in SCENARIO_IDS:
            env = make_env(seed, scenario)
            rng = random.Random(seed)
            for _ in range(EPISODE_TURNS):
                env.step({a: rng.choice(legal_action_ids(a)) for a in env.agents})
                shared = env.state["shared"]
                assert 0 <= shared["network_health"] <= 100
                assert 0 <= shared["printer_health"] <= 100
                assert 0 <= shared["kitchen_cleanliness"] <= 100
                assert shared["supplies"] >= 0
                for state in env.state["agents"].values():
                    assert state["energy"] >= 0
                    assert state["private_supplies"] >= 0
                    assert state["task_progress"] >= state["private_supplies"]


def test_unfunded_actions_are_legal_no_ops_not_errors() -> None:
    env = make_env()
    for state in env.state["agents"].values():
        state["energy"] = 0
        state["private_supplies"] = 0
    before = env.serialize_state()
    env.step({"agent_1": "REPAIR_FULL", "agent_2": "TAKE_3",
              "agent_3": "SHARE_AGENT_1"})
    after = env.state["agents"]
    assert after["agent_1"]["reputation"] == before["agents"]["agent_1"]["reputation"]
    assert after["agent_2"]["task_progress"] == \
        before["agents"]["agent_2"]["task_progress"]
    env.step({a: "WAIT" for a in env.agents})
    assert env.state["pending_shares"][-1]["status"] == "empty"


def test_illegal_actions_raise() -> None:
    env = make_env()
    with pytest.raises(ValueError):
        env.step({"agent_1": "TRADE_AGENT_2"})
    with pytest.raises(ValueError):
        env.step({"agent_1": "DELEGATE_AGENT_1"})


def test_briefing_is_ascii_states_the_menu_and_names_no_agent_ids() -> None:
    env = make_env()
    briefing = env.briefing()
    assert briefing.isascii()
    assert "agent_1" not in briefing
    for objective in OBJECTIVES:
        assert objective in briefing
    assert "hold three different objectives" in briefing


def test_action_space_function_matches_method() -> None:
    env = make_env()
    for agent in AGENTS:
        assert env.action_space(agent) == action_space(agent)
