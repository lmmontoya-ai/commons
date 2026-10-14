"""The frozen Commons v1 decisions behind the report's Office ordinary-play
results, rebuilt from seeds and action labels.

Two frozen inputs ship with the package (``commons/v1/data``):

``m43_turn8_bank.json``
    The 90 trajectories of the M4.3 turn-8 replay bank (10 generator models
    x 3 target seats x 3 scenarios, seeds 1600 to 1622). Each holds its seed,
    scenario, target seat, the 8 turns of seat-ordered action labels that
    produced it, and the digests recorded by the study (state, observation
    arms). The report's Office distinctness figure was scored at the 90
    turn-8 states.

``m44_positions.json``
    The 48 decision positions of the M4.4 battery (16 per target model,
    turns 0 to 8, 12 of them at turn 8) taken from 23 of those trajectories.
    The report's nine-forecaster panel and the ordinary arm of its
    perspective study used these positions, with the situation-only
    (``board``) and full-record (``ordered``) reader views.

Only environment inputs are stored: ids, seeds, scenario ids, seats, turns,
action labels, the reader instruction variant, and digests. No model text,
model reply, provider record, or cost is included. Which model played a
trajectory's target seat is not stored either.

States are rebuilt by deterministic replay (``rollout_states``) and checked
against the stored digests by the test suite. The shuffle seed contexts below
are the ones the studied prompts used, so shuffled prompts are reproduced
byte for byte too.
"""
from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

from .env import legal_action_ids, rollout_states
from .render import (
    Message,
    actor_messages,
    canonical_view,
    reader_messages,
    reader_observation,
)

BANK_RESOURCE = "m43_turn8_bank.json"
POSITIONS_RESOURCE = "m44_positions.json"

# Campaign label mixed into the shuffle seed of the bank's own frozen
# observation arms (``snapshot_manifest.json``).
BANK_MANIFEST_CAMPAIGN = "predictive_commons_m4_turn8_cohort10"
# Version label mixed into the shuffle seed of the forecast-suite prompts
# (M4.3 suite fixtures and every M4.4 position prompt).
FORECAST_SUITE_VERSION = "m4.3-fs1"

# Harness arm names of the three reader views.
ARMS = {"board": "state_only", "ordered": "full_history",
        "shuffled": "shuffled_history"}


def _load(name: str) -> dict[str, Any]:
    resource = resources.files("commons.v1") / "data" / name
    return json.loads(resource.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _bank() -> dict[str, Any]:
    return _load(BANK_RESOURCE)


@lru_cache(maxsize=1)
def _positions() -> dict[str, Any]:
    return _load(POSITIONS_RESOURCE)


def bank_metadata() -> dict[str, Any]:
    """Top-level fields of the bank file (everything but the rows)."""
    return {k: v for k, v in _bank().items() if k != "trajectories"}


def positions_metadata() -> dict[str, Any]:
    """Top-level fields of the positions file (everything but the rows)."""
    return {k: v for k, v in _positions().items() if k != "positions"}


# ============================================================ trajectories

def trajectories() -> list[dict[str, Any]]:
    """The 90 frozen trajectories, sorted by ``snapshot_id``."""
    return json.loads(json.dumps(_bank()["trajectories"]))


def trajectory(snapshot_id: str) -> dict[str, Any]:
    for row in _bank()["trajectories"]:
        if row["snapshot_id"] == snapshot_id:
            return json.loads(json.dumps(row))
    raise KeyError(snapshot_id)


@lru_cache(maxsize=128)
def _states(snapshot_id: str) -> str:
    row = trajectory(snapshot_id)
    return json.dumps(rollout_states(row["seed"], row["scenario_id"],
                                     row["actions"], action_feedback=True))


def trajectory_states(row: dict[str, Any] | str) -> list[dict[str, Any]]:
    """Every state of a frozen trajectory, turn 0 through turn 8."""
    snapshot_id = row if isinstance(row, str) else row["snapshot_id"]
    return json.loads(_states(snapshot_id))


def turn8_state(row: dict[str, Any] | str) -> dict[str, Any]:
    """The frozen turn-8 decision state of a trajectory."""
    return trajectory_states(row)[-1]


def bank_seed_context(row: dict[str, Any]) -> dict[str, Any]:
    """Shuffle seed of the bank's own frozen observation arms."""
    return {"campaign": BANK_MANIFEST_CAMPAIGN,
            "snapshot_id": row["snapshot_id"]}


def suite_seed_context(row: dict[str, Any],
                       snapshot_id: str | None = None) -> dict[str, Any]:
    """Shuffle seed of the forecast-suite prompts: the trajectory's own
    campaign id, the decision id, and the suite version."""
    return {"campaign": row["campaign_id"],
            "snapshot_id": snapshot_id or row["snapshot_id"],
            "forecast_suite_version": FORECAST_SUITE_VERSION}


def bank_observations(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The bank's three frozen reader observations at turn 8 (neutral V1
    instruction), keyed by harness arm name. Their digests are the
    ``observation_sha256_by_arm`` of ``snapshot_manifest.json``."""
    state = turn8_state(row)
    return {ARMS[view]: reader_observation(
                state, row["target_id"], view,
                shuffle_seed_context=bank_seed_context(row))
            for view in ARMS}


def suite_observations(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """The forecast suite's three reader observations at turn 8 (with the
    trajectory's instruction variant), keyed by harness arm name."""
    state = turn8_state(row)
    return {ARMS[view]: reader_observation(
                state, row["target_id"], view,
                instruction_variant=row["instruction_variant"],
                shuffle_seed_context=suite_seed_context(row))
            for view in ARMS}


# =============================================================== positions

def positions() -> list[dict[str, Any]]:
    """The 48 frozen M4.4 positions, sorted by ``position_id``."""
    return json.loads(json.dumps(_positions()["positions"]))


def position(position_id: str) -> dict[str, Any]:
    for row in _positions()["positions"]:
        if row["position_id"] == position_id:
            return json.loads(json.dumps(row))
    raise KeyError(position_id)


def position_state(pos: dict[str, Any]) -> dict[str, Any]:
    """The decision state of a position, rebuilt by replay."""
    return trajectory_states(pos["trajectory_id"])[int(pos["turn"])]


def legal_actions(pos: dict[str, Any]) -> list[str]:
    """The target's eleven legal action ids at a position."""
    return legal_action_ids(pos["target_id"])


def position_reader_messages(pos: dict[str, Any], view: str, *,
                             prompt_format: str = "json") -> list[Message]:
    """The exact reader messages of the M4.4 battery at a position.

    ``view="board"`` is the report's "situation only" view and
    ``view="ordered"`` its "full record" view (the nine-forecaster panel and
    the standard arm of the perspective study).
    """
    row = trajectory(pos["trajectory_id"])
    return reader_messages(
        position_state(pos), pos["target_id"], canonical_view(view),
        instruction_variant=row["instruction_variant"],
        shuffle_seed_context=suite_seed_context(row, pos["position_id"]),
        prompt_format=prompt_format)


def position_actor_messages(pos: dict[str, Any], *,
                            prompt_format: str = "json") -> list[Message]:
    """The exact acting messages of the target at a position (the prompt of
    every replayed target draw)."""
    return actor_messages(position_state(pos), pos["target_id"],
                          prompt_format=prompt_format)
