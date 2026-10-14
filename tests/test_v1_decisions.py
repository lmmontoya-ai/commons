"""The frozen Commons v1 decisions of the report, rebuilt from seeds and
action labels and checked against the digests the study recorded.

The expected digests in ``src/commons/v1/data`` were exported from the
research harness records (``scripts/check_equivalence_v1.py --write-data``):
the M4.3 snapshot manifest, the M4.3 forecast-suite prompt fixtures, the
M4.4 run store, and the nine-forecaster panel manifest."""
from __future__ import annotations

import json
from collections import Counter
from importlib import resources

import pytest

from commons.v1 import (
    AGENTS,
    SCENARIO_IDS,
    bank_metadata,
    bank_observations,
    legal_action_ids,
    legal_actions,
    messages_sha256,
    observation_sha256,
    position_actor_messages,
    position_reader_messages,
    position_state,
    positions,
    positions_metadata,
    state_id,
    state_sha256,
    suite_observations,
    trajectories,
    trajectory_states,
    turn8_state,
)

TRAJECTORY_KEYS = {
    "snapshot_id", "campaign_id", "seed", "scenario_id", "target_id", "turn",
    "instruction_variant", "actions", "state_sha256", "state_id",
    "snapshot_sha256", "observation_sha256_by_arm",
    "suite_observation_sha256_by_arm"}
POSITION_KEYS = {"position_id", "trajectory_id", "turn", "target_id",
                 "state_sha256", "prompt_sha256"}


def test_frozen_files_hold_only_environment_inputs_and_digests():
    for name, key, allowed in (("m43_turn8_bank.json", "trajectories", TRAJECTORY_KEYS),
                               ("m44_positions.json", "positions", POSITION_KEYS)):
        text = (resources.files("commons.v1") / "data" / name).read_text(encoding="utf-8")
        payload = json.loads(text)
        assert {k for row in payload[key] for k in row} == allowed
        lowered = text.lower()
        for marker in ("openrouter", "reasoning", "response_id", "cost", "usd",
                       "gpt", "sonnet", "grok", "claude", "gemini", "d:\\", "c:\\",
                       "/users/"):
            assert marker not in lowered, marker


def test_bank_design():
    rows = trajectories()
    assert len(rows) == 90
    assert bank_metadata()["condition"] == "tradeoff_v8"
    assert Counter((r["target_id"], r["scenario_id"]) for r in rows) == {
        (a, s): 10 for a in AGENTS for s in SCENARIO_IDS}
    assert Counter(r["instruction_variant"] for r in rows) == {"V1": 30, "V2": 30, "V3": 30}
    for row in rows:
        assert row["turn"] == len(row["actions"]) == 8
        for moves in row["actions"]:
            assert [m in legal_action_ids(a) for a, m in zip(AGENTS, moves)] == [True] * 3


def test_every_turn8_state_rebuilds_to_the_recorded_digests():
    for row in trajectories():
        state = turn8_state(row)
        assert state["turn"] == 8
        assert state_sha256(state) == row["state_sha256"], row["snapshot_id"]
        assert state_id(state) == row["state_id"], row["snapshot_id"]


def test_bank_reader_observations_match_the_snapshot_manifest():
    for row in trajectories():
        observed = {arm: observation_sha256(o) for arm, o in bank_observations(row).items()}
        assert observed == row["observation_sha256_by_arm"], row["snapshot_id"]


def test_suite_reader_observations_match_the_prompt_fixtures():
    for row in trajectories():
        observed = {arm: observation_sha256(o)
                    for arm, o in suite_observations(row).items()}
        assert observed == row["suite_observation_sha256_by_arm"], row["snapshot_id"]


def test_position_design():
    rows = positions()
    assert len(rows) == 48
    assert positions_metadata()["schema"] == "commons.v1.m44_positions/1"
    assert sum(1 for p in rows if p["turn"] == 8) == 12
    assert len({p["trajectory_id"] for p in rows}) == 23
    bank = {r["snapshot_id"]: r for r in trajectories()}
    for p in rows:
        assert p["position_id"] == f"{p['trajectory_id']}@t{p['turn']}"
        assert p["target_id"] == bank[p["trajectory_id"]]["target_id"]
        assert legal_actions(p) == legal_action_ids(p["target_id"])


def test_every_position_state_rebuilds_to_its_digest():
    bank = {r["snapshot_id"]: r for r in trajectories()}
    for p in positions():
        state = position_state(p)
        assert state["turn"] == p["turn"]
        assert state_sha256(state) == p["state_sha256"], p["position_id"]
        assert state == trajectory_states(p["trajectory_id"])[p["turn"]]
        if p["turn"] == 8:
            assert p["state_sha256"] == bank[p["trajectory_id"]]["state_sha256"]


@pytest.mark.parametrize("prompt_format, views", [
    ("json", ("board", "ordered", "shuffled")),
    ("markdown", ("board", "ordered", "shuffled")),
])
def test_position_reader_prompts_match_the_recorded_digests(prompt_format, views):
    for p in positions():
        for view in views:
            digest = messages_sha256(
                position_reader_messages(p, view, prompt_format=prompt_format))
            assert digest == p["prompt_sha256"][prompt_format][view], (
                p["position_id"], view)


def test_position_actor_prompts_match_the_recorded_digests():
    for p in positions():
        assert messages_sha256(position_actor_messages(p)) == p["prompt_sha256"]["json"][
            "actor"], p["position_id"]
