"""Run one Commons episode with scripted seats, offline fake models, or LLMs.

    # zero cost, no key needed (scripted seats):
    python scripts/run_episode.py --seed 2700 --scenario delivery_pressure \
        --agent-1 scripted:greedy --agent-2 scripted:softmax \
        --agent-3 scripted:uniform --out data/runs/smoke

    # zero cost through the full model path (offline fake model):
    python scripts/run_episode.py --seed 2700 --scenario delivery_pressure \
        --agent-1 fake:TAKE_1 --agent-2 fake:REPAIR_PATCH \
        --agent-3 fake:WAIT --out data/runs/fake

    # LLM seats (needs OPENROUTER_API_KEY and a models file; spends money):
    python scripts/run_episode.py --seed 2700 --scenario delivery_pressure \
        --objectives PERSONAL_THROUGHPUT,SYSTEM_HEALTH,SOCIAL_CREDIT \
        --models models.json --agent-1 model:luna --agent-2 model:grok45 \
        --agent-3 model:sonnet5 --cap-usd 2 --out data/runs/pilot

Seat specs: ``scripted:<wait|uniform|greedy|softmax>``, ``fake:<ACTION_ID>``,
or ``model:<name in the models file>``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from commons.core import (  # noqa: E402
    AGENTS,
    CONDITION,
    EPISODE_TURNS,
    OBJECTIVES,
    SCENARIO_IDS,
    SHAM_LABELS,
    legal_action_ids,
)
from commons.policies import POLICIES  # noqa: E402
from commons.runtime import (  # noqa: E402
    CostMeter,
    ModelAgent,
    ModelSpec,
    OpenRouterClient,
    ScriptedAgent,
    constant_action_client,
    load_models,
    run_episode,
)


def make_agent(spec_str: str, seat: str, models, client, args):
    kind, _, name = spec_str.partition(":")
    if kind == "scripted":
        if name not in POLICIES:
            raise SystemExit(f"unknown policy {name!r}; have {sorted(POLICIES)}")
        return ScriptedAgent(name, POLICIES[name])
    if kind == "fake":
        if name not in legal_action_ids(seat):
            raise SystemExit(f"{name!r} is not a legal action for {seat}; "
                             f"have {legal_action_ids(seat)}")
        spec = ModelSpec(slug=f"fake/{name.lower()}", label=f"fake:{name}",
                         price_in=0.0, price_out=0.0)
        return ModelAgent(spec, constant_action_client(name),
                          parser=args.parser, on_invalid=args.on_invalid)
    if kind == "model":
        if models is None or name not in models:
            raise SystemExit(f"unknown model {name!r}; pass --models with it defined")
        return ModelAgent(models[name], client, parser=args.parser,
                          on_invalid=args.on_invalid)
    raise SystemExit("seat must be scripted:<name>, fake:<ACTION_ID> or "
                     f"model:<name>, got {spec_str!r}")


def parse_triple(text: str | None, allowed, what: str) -> dict[str, str] | None:
    if text is None:
        return None
    values = [v.strip() for v in text.split(",")]
    if len(values) != len(AGENTS) or any(v not in allowed for v in values):
        raise SystemExit(f"--{what} needs three comma-separated values from "
                         f"{list(allowed)}, in seat order {list(AGENTS)}")
    return dict(zip(AGENTS, values))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=2700)
    ap.add_argument("--scenario", default=SCENARIO_IDS[0], choices=SCENARIO_IDS)
    ap.add_argument("--objectives", default=None,
                    help="three objectives in seat order; default: drawn from the seed")
    ap.add_argument("--sham", default=None,
                    help="run the sham arm with these three display labels "
                         f"(from {list(SHAM_LABELS)})")
    ap.add_argument("--agent-1", default="scripted:greedy")
    ap.add_argument("--agent-2", default="scripted:greedy")
    ap.add_argument("--agent-3", default="scripted:greedy")
    ap.add_argument("--turns", type=int, default=EPISODE_TURNS)
    ap.add_argument("--models", default=None, help="model registry JSON")
    ap.add_argument("--parser", default="lenient", choices=("lenient", "strict"))
    ap.add_argument("--on-invalid", default="wait", choices=("wait", "abandon"))
    ap.add_argument("--cap-usd", type=float, default=1.0)
    ap.add_argument("--out", default="data/runs/adhoc")
    ap.add_argument("--episode-id", default=None)
    args = ap.parse_args()

    objectives = parse_triple(args.objectives, OBJECTIVES, "objectives")
    sham = parse_triple(args.sham, SHAM_LABELS, "sham")
    seat_specs = {"agent_1": args.agent_1, "agent_2": args.agent_2,
                  "agent_3": args.agent_3}
    models = load_models(args.models) if args.models else None
    needs_client = any(s.startswith("model:") for s in seat_specs.values())
    client = OpenRouterClient() if needs_client else None
    meter = CostMeter(cap_usd=args.cap_usd)
    agents = {seat: make_agent(spec, seat, models, client, args)
              for seat, spec in seat_specs.items()}

    result = run_episode(seed=args.seed, scenario_id=args.scenario,
                         agents=agents, out_dir=args.out,
                         objectives=objectives, sham_display=sham,
                         turns=args.turns, condition=CONDITION, meter=meter,
                         episode_id=args.episode_id)
    if result is None:
        print("episode ABANDONED (see log); spent "
              f"${meter.spent_usd:.4f} over {meter.calls} calls")
        raise SystemExit(2)
    log = Path(args.out) / f"{result['episode_id']}.jsonl"
    print(f"episode {result['episode_id']} ({result['arm']} arm), "
          f"{result['turns']} turns, replay verified: {result['replay_verified']}")
    for agent in AGENTS:
        objective = result["final_state"]["agents"][agent]["hidden_objective"]
        print(f"  {agent}: objective {objective}, final utility "
              f"{result['objective_utility'][agent]:g}, total reward "
              f"{result['total_reward'][agent]:g}")
    print(f"invalid decisions: {result['invalid_decisions']} | "
          f"${meter.spent_usd:.4f} over {meter.calls} calls | log: {log}")


if __name__ == "__main__":
    main()
