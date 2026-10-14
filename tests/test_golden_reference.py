"""Digests computed by the original research harness, recomputed here.

``tests/golden/harness_reference.jsonl`` was written by
``scripts/check_equivalence.py --write-golden`` from the HARNESS side: for
twelve seeded episodes (three scenarios, four seeds, uniform random play) it
holds a digest prefix of every intermediate state, of every actor prompt,
and of every reader and goal-naming prompt in turns 0 to 8. This test needs
no harness: it replays the same episodes with this package and compares.
"""
from __future__ import annotations

import hashlib
import json
import random

import pytest
from conftest import GOLDEN, canonical

from commons.core import AGENTS, CONDITION, CommonsEnv, legal_action_ids
from commons.render import (
    actor_messages,
    goal_naming_messages,
    instruction_variant_for,
    reader_messages,
)
from commons.render.messages import Message

_LINES = (GOLDEN / "harness_reference.jsonl").read_text(encoding="utf-8").splitlines()
HEADER = json.loads(_LINES[0])
EPISODES = [json.loads(line) for line in _LINES[1:] if line.strip()]


def digest(value) -> str:
    full = hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()
    return full[:HEADER["digest_chars"]]


def reader_digests(state, agent, model_key, position_id) -> dict[str, str]:
    variant = instruction_variant_for(model_key, agent, state["scenario_id"])
    seed_context = {"campaign": "commons_v2_campaign", "snapshot_id": position_id,
                    "forecast_suite_version": "m4.3-fs1"}
    out: dict[str, list[Message]] = {}
    for view in ("board", "ordered", "shuffled", "tally"):
        out[view] = reader_messages(state, agent, view, instruction_variant=variant,
                                    shuffle_seed_context=seed_context)
    for view in ("board", "ordered", "tally"):
        out[f"goal-{view}"] = goal_naming_messages(state, agent, view)
    return {k: digest(v) for k, v in out.items()}


@pytest.mark.parametrize(
    "episode", EPISODES, ids=[f"{e['scenario']}-{e['seed']}" for e in EPISODES])
def test_episode_matches_harness_digests(episode) -> None:
    scenario, seed = episode["scenario"], episode["seed"]
    env = CommonsEnv()
    env.reset(seed, CONDITION, scenario)
    rng = random.Random(f"golden:{scenario}:{seed}")
    for row in episode["turns"]:
        state = env.serialize_state()
        assert digest(state) == row["state"], f"state at turn {row['turn']}"
        for agent in AGENTS:
            assert digest(actor_messages(state, agent)) == row["actor"][agent]
            if agent in row["reader"]:
                expected = row["reader"][agent]
                got = reader_digests(state, agent, episode["model_key"],
                                     f"golden-{scenario}-{seed}-{row['turn']}")
                assert got == expected, f"reader views at turn {row['turn']} {agent}"
        moves = {a: rng.choice(legal_action_ids(a)) for a in AGENTS}
        assert moves == row["moves"]
        env.step(moves)
    assert digest(env.serialize_state()) == episode["final_state"]


def test_reference_covers_every_scenario() -> None:
    scenarios = {e["scenario"] for e in EPISODES}
    assert scenarios == {"delivery_pressure", "infrastructure_risk",
                         "coordination_debt"}
    assert len(EPISODES) == 12
    assert HEADER["digest_chars"] == 16
