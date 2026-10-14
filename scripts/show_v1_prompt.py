"""Print the exact prompt at one frozen Commons v1 decision of the report.

Zero cost: rebuilds the decision from the frozen inputs in
``src/commons/v1/data`` and renders it locally; calls no model.

    # list the 48 positions of the nine-forecaster panel:
    python scripts/show_v1_prompt.py --list

    # situation-only reader prompt at the first position:
    python scripts/show_v1_prompt.py --position 0 --view board

    # full-record reader prompt, or the target's own acting prompt:
    python scripts/show_v1_prompt.py --position 0 --view ordered
    python scripts/show_v1_prompt.py --position 0 --view actor

    # a turn of one of the 90 bank trajectories (any seat):
    python scripts/show_v1_prompt.py --trajectory 0 --turn 8 --agent agent_2 --view actor

Views: actor, board (situation only), ordered (full record), shuffled.
``--position`` takes an index or a position id; ``--trajectory`` an index or
a snapshot id. For a position the output ends with the recorded digest and
whether the rendered prompt matches it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commons.v1 import (  # noqa: E402
    AGENTS,
    actor_messages,
    messages_sha256,
    position_actor_messages,
    position_reader_messages,
    positions,
    reader_messages,
    suite_seed_context,
    trajectories,
    trajectory_states,
)

VIEWS = ("actor", "board", "ordered", "shuffled")


def pick(rows: list[dict], key: str, value: str) -> dict:
    if value.isdigit():
        return rows[int(value)]
    for row in rows:
        if row[key] == value:
            return row
    raise SystemExit(f"no {key} {value!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--list", action="store_true", help="list the 48 positions")
    ap.add_argument("--position", default=None)
    ap.add_argument("--trajectory", default=None)
    ap.add_argument("--turn", type=int, default=8)
    ap.add_argument("--agent", default=None, choices=AGENTS,
                    help="seat (default: the trajectory's target seat)")
    ap.add_argument("--view", default="board", choices=VIEWS)
    ap.add_argument("--format", default="json", choices=("json", "markdown"))
    args = ap.parse_args()

    out = sys.stdout
    if hasattr(out, "reconfigure"):
        out.reconfigure(encoding="utf-8")
    if args.list:
        for i, p in enumerate(positions()):
            out.write(f"{i:2d}  {p['position_id']}  target={p['target_id']}\n")
        return
    recorded = None
    if args.position is not None:
        pos = pick(positions(), "position_id", args.position)
        if args.view == "actor":
            messages = position_actor_messages(pos, prompt_format=args.format)
            recorded = pos["prompt_sha256"].get(args.format, {}).get("actor")
        else:
            messages = position_reader_messages(pos, args.view,
                                                prompt_format=args.format)
            recorded = pos["prompt_sha256"][args.format][args.view]
    elif args.trajectory is not None:
        row = pick(trajectories(), "snapshot_id", args.trajectory)
        if not 0 <= args.turn <= 8:
            raise SystemExit("--turn must be between 0 and 8")
        state = trajectory_states(row)[args.turn]
        agent = args.agent or row["target_id"]
        if args.view == "actor":
            messages = actor_messages(state, agent, prompt_format=args.format)
        else:
            messages = reader_messages(
                state, agent, args.view,
                instruction_variant=row["instruction_variant"],
                shuffle_seed_context=suite_seed_context(
                    row, f"{row['snapshot_id']}@t{args.turn}"),
                prompt_format=args.format)
    else:
        raise SystemExit("pass --list, --position, or --trajectory")

    for message in messages:
        out.write(f"===== {message['role']} =====\n{message['content']}\n\n")
    digest = messages_sha256(messages)
    out.write(f"===== digest =====\n{digest}\n")
    if recorded is not None:
        out.write(f"recorded: {recorded} ({'match' if recorded == digest else 'MISMATCH'})\n")


if __name__ == "__main__":
    main()
