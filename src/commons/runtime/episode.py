"""Run one Commons episode end to end and log it.

The loop matches the studied campaigns: at each turn every seat chooses from
the same start-of-turn state, then the three choices are applied with
``CommonsEnv.step`` in the fixed seat order agent_1, agent_2, agent_3. After
the last turn the final state is replay-verified with ``replay_states``.

An abandoned episode (provider down past caps, the cost cap tripping, or a
seat with no legal action under ``on_invalid="abandon"``) writes an
``abandoned`` record and no result: it is quarantined, never truncated.
"""
from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

from .. import __version__
from ..core.env import (
    AGENTS,
    CONDITION,
    ENV_VERSION,
    EPISODE_TURNS,
    SCORED_TURNS,
    CommonsEnv,
    objective_utility,
    replay_states,
)
from ..core.sham import ShamCommonsEnv
from .agents import InvalidActionAbandon
from .log import EpisodeLogger
from .openrouter import BudgetExceeded, EpisodeAbandoned


def _code_version() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).parent).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run_episode(*, seed: int, scenario_id: str, agents: dict[str, Any],
                out_dir: str | Path,
                objectives: dict[str, str] | None = None,
                sham_display: dict[str, str] | None = None,
                turns: int = EPISODE_TURNS, condition: str = CONDITION,
                meter=None, episode_id: str | None = None) -> dict | None:
    """Play one episode. Returns the result record, or None if abandoned.

    ``agents`` maps each seat (agent_1, agent_2, agent_3) to an agent.
    ``objectives`` fixes the hidden assignment; when omitted it is drawn from
    the seed. ``sham_display`` runs the sham arm instead: every seat's
    effective objective is SYSTEM_HEALTH and the given sham labels are shown.
    """
    if set(agents) != set(AGENTS):
        raise ValueError(f"agents must cover exactly {AGENTS}; got {sorted(agents)}")
    if sham_display is not None and objectives is not None:
        raise ValueError("pass objectives (active arm) or sham_display "
                         "(sham arm), not both")
    episode_id = episode_id or uuid.uuid4().hex[:12]
    if sham_display is not None:
        env: CommonsEnv = ShamCommonsEnv()
        env.reset(seed, condition, scenario_id, objectives=sham_display)
        arm = "sham"
    else:
        env = CommonsEnv()
        env.reset(seed, condition, scenario_id, objectives=objectives)
        arm = "active"
    logger = EpisodeLogger(Path(out_dir) / f"{episode_id}.jsonl", {
        "episode_id": episode_id,
        "seed": seed, "scenario_id": scenario_id, "condition": condition,
        "arm": arm,
        "objectives": env.objective_assignment(),
        "sham_display": sham_display,
        "turns": turns, "scored_turns": SCORED_TURNS,
        "seats": {a: agents[a].describe() for a in AGENTS},
        "env_version": ENV_VERSION, "package_version": __version__,
        "code_version": _code_version(),
    })
    totals = {a: 0.0 for a in AGENTS}
    invalid = 0
    try:
        for turn in range(turns):
            state = env.serialize_state()
            moves: dict[str, str] = {}
            for agent in AGENTS:
                decision = agents[agent].act(state, agent,
                                             sham_display=sham_display,
                                             logger=logger, meter=meter)
                moves[agent] = decision.action
                invalid += int(decision.invalid)
                logger.write({"kind": "decision", "turn": turn,
                              "agent": agent, "action": decision.action,
                              "invalid": decision.invalid,
                              "attempts": decision.attempts,
                              "digest": decision.digest})
            step = env.step({a: moves[a] for a in AGENTS})
            for agent, reward in step.rewards.items():
                totals[agent] = round(totals[agent] + reward, 6)
            logger.write({"kind": "turn", "turn": turn, "actions": moves,
                          "rewards": step.rewards})
        final_state = env.serialize_state()
        states = replay_states(final_state)
        if len(states) != turns + 1:
            raise RuntimeError(
                f"replay produced {len(states)} states, expected {turns + 1}")
        result = {
            "episode_id": episode_id,
            "arm": arm,
            "turns": turns,
            "invalid_decisions": invalid,
            "objective_utility": {
                a: objective_utility(s["hidden_objective"], s,
                                     final_state["shared"])
                for a, s in final_state["agents"].items()},
            "total_reward": totals,
            "replay_verified": True,
            "final_state": final_state,
        }
        logger.result(result)
        return result
    except (EpisodeAbandoned, BudgetExceeded, InvalidActionAbandon) as e:
        logger.abandon(str(e))
        return None
    finally:
        if meter is not None:
            logger.write({"kind": "spend", "calls": meter.calls,
                          "spent_usd": round(meter.spent_usd, 4),
                          "unpriced_calls": meter.unpriced_calls})
        logger.close()
