from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from commons.core import CONDITION, SCENARIO_IDS, CommonsEnv, legal_action_ids

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
GOLDEN = Path(__file__).resolve().parent / "golden"

SCEN = SCENARIO_IDS[0]


def make_env(seed: int = 1600, scenario: str = SCEN,
             objectives: dict[str, str] | None = None) -> CommonsEnv:
    env = CommonsEnv()
    env.reset(seed, CONDITION, scenario, objectives=objectives)
    return env


def play_uniform(seed: int, scenario: str, turns: int, tag: str = "test") -> dict:
    env = make_env(seed, scenario)
    rng = random.Random(f"{tag}:{scenario}:{seed}")
    for _ in range(turns):
        env.step({a: rng.choice(legal_action_ids(a)) for a in env.agents})
    return env.serialize_state()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture
def env() -> CommonsEnv:
    return make_env()
