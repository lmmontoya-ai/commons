"""Print the exact prompt an actor or reader receives at one decision.

Zero cost: renders locally, calls no model.

    # actor prompt for agent_1 at turn 0 of a seeded episode:
    python scripts/show_prompts.py --seed 2700 --scenario delivery_pressure \
        --turn 0 --agent agent_1 --view actor

    # reader prompt (ordered view) at turn 3, reached by scripted play:
    python scripts/show_prompts.py --seed 2700 --scenario delivery_pressure \
        --turn 3 --agent agent_2 --view ordered --variant V2

    # from a logged episode:
    python scripts/show_prompts.py --log data/runs/smoke/<id>.jsonl \
        --turn 5 --agent agent_3 --view tally

Views: actor, board, ordered, shuffled, tally, goal-board, goal-ordered,
goal-tally. Without --log, the state at --turn is reached by letting every
seat play --policy (default greedy) from the reset state.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commons.core import AGENTS, CONDITION, OBJECTIVES, SCENARIO_IDS, CommonsEnv  # noqa: E402
from commons.policies import POLICIES  # noqa: E402
from commons.render import (  # noqa: E402
    actor_messages,
    goal_naming_messages,
    messages_digest,
    reader_messages,
)
from commons.runtime import load_sham_display, states_from_log  # noqa: E402

VIEWS = ("actor", "board", "ordered", "shuffled", "tally",
         "goal-board", "goal-ordered", "goal-tally")


def scripted_state(seed: int, scenario: str, turn: int, policy: str,
                   objectives: dict[str, str] | None) -> dict:
    env = CommonsEnv()
    env.reset(seed, CONDITION, scenario, objectives=objectives)
    for _ in range(turn):
        state = env.serialize_state()
        env.step({a: POLICIES[policy](state, a) for a in AGENTS})
    return env.serialize_state()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--log", default=None, help="episode JSONL from run_episode.py")
    ap.add_argument("--seed", type=int, default=2700)
    ap.add_argument("--scenario", default=SCENARIO_IDS[0], choices=SCENARIO_IDS)
    ap.add_argument("--objectives", default=None,
                    help="three objectives in seat order; default: drawn from the seed")
    ap.add_argument("--policy", default="greedy", choices=sorted(POLICIES))
    ap.add_argument("--turn", type=int, default=0)
    ap.add_argument("--agent", default="agent_1", choices=AGENTS)
    ap.add_argument("--view", default="actor", choices=VIEWS)
    ap.add_argument("--variant", default="V1", choices=("V1", "V2", "V3"),
                    help="reader instruction variant")
    args = ap.parse_args()

    sham = None
    if args.log:
        states = states_from_log(args.log)
        if args.turn not in states:
            raise SystemExit(f"turn {args.turn} not in the log (0..{max(states)})")
        state = states[args.turn]
        sham = load_sham_display(args.log)
    else:
        objectives = None
        if args.objectives:
            values = [v.strip() for v in args.objectives.split(",")]
            if len(values) != 3 or any(v not in OBJECTIVES for v in values):
                raise SystemExit(f"--objectives needs three of {list(OBJECTIVES)}")
            objectives = dict(zip(AGENTS, values))
        state = scripted_state(args.seed, args.scenario, args.turn, args.policy,
                               objectives)

    if args.view == "actor":
        messages = actor_messages(state, args.agent, sham_display=sham)
    elif args.view.startswith("goal-"):
        messages = goal_naming_messages(state, args.agent, args.view[5:])
    else:
        messages = reader_messages(state, args.agent, args.view,
                                   instruction_variant=args.variant)
    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    for message in messages:
        out.write(f"===== {message['role']} =====\n{message['content']}\n\n")
    out.write(f"===== digest =====\n{messages_digest(messages)}\n")


if __name__ == "__main__":
    main()
