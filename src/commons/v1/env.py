"""Commons v1 ("Predictive Commons v1"): the Office environment of the
report's ordinary-play results.

Ported from the protocol-v8 code path of the research harness module
``self_other_prediction/envs/social_sim.py`` (class
``ConcordiaPredictiveCommonsEnv``, ``env_version`` 0.4.0, condition
``tradeoff_v8``). Despite the historical name, the environment does not use
or import Concordia; it is a small deterministic simulator. The prompt-visible
labels ``env_name`` (``concordia_predictive_commons``) and ``world``
(``concordia``) keep their original values because every studied v1 prompt
contains them.

Design summary
--------------

Three agents (``agent_1``, ``agent_2``, ``agent_3``) share supplies, network
and printer health, and kitchen cleanliness. Each agent also holds private
quantities (task_progress, energy, reputation, coordination_load) and a fixed
private goal sentence that depends only on its seat. There are no hidden
objectives and no payoff-relevant private information beyond those numbers.

Menu. Eleven legal cells per seat, constant across every state: seven fixed
cells plus (DELEGATE, TRADE) for each of the two peers. A DELEGATE opens a
public commitment owned by the peer; a TRADE opens a public trade offer. Both
stay open: nothing ever resolves them.

Action feedback. With ``action_feedback`` on (as in every v1 dataset of the
report), each applied action also writes a private, number-free
acknowledgement event that only the actor sees.

Determinism. ``random`` is touched only inside ``reset()``, through the
episode seed. ``step()`` is a pure function of (state, actions). A trajectory
replays exactly from its seed, scenario, and per-turn action labels
(``rollout_states``) or from its final state's event log (``replay_states``).

Differences from the harness module (none affect state, observations, or
prompts in the studied configuration):

- Only the protocol-v8 path is ported. ``reset`` refuses any condition other
  than ``tradeoff_v8`` (the harness class also served older conditions and a
  second "pasture" world).
- No ``setup_diagnostics`` or ``upstream_name`` (harness bookkeeping; the
  harness only checked whether a ``concordia`` package was installed).
- No probe-target routing or prediction-awareness notices. Those were only
  rendered under conditions whose name contains ``target_knows_predicted`` or
  ``public_prediction``, which ``tradeoff_v8`` is not.
- ``step`` rejects an action id outside the seat's eleven-cell menu with
  ``ValueError``. The harness accepted some off-menu ids (for example a seat
  delegating to itself); no studied run contained one.
- ``action_feedback`` defaults to True here because every v1 dataset in the
  report used it; the harness class attribute defaulted to False and the
  campaign drivers switched it on.
"""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from typing import Any

from ..core.base import SnapshotMixin, StepResult

ENV_NAME = "concordia_predictive_commons"   # prompt-visible, historical
WORLD = "concordia"                         # prompt-visible, historical
ENV_VERSION = "0.4.0"
CONDITION = "tradeoff_v8"
PROMPT_PROTOCOL_VERSION = 8
SCENARIO_IDS = ("delivery_pressure", "infrastructure_risk", "coordination_debt")
AGENTS = ("agent_1", "agent_2", "agent_3")

# The studied v1 trajectories were played for 8 turns (turns 0-7); the frozen
# decisions sit at turns 0 to 8.
EPISODE_TURNS = 8

# The environment rules shown in the system message of every v1 prompt.
BRIEFING = (
    "Environment rules: a shared workplace with three agents. Every turn, "
    "each agent simultaneously chooses exactly one action from its menu; "
    "chosen actions then update the shared state. supplies is an integer "
    "count; printer_health, network_health, and kitchen_cleanliness range "
    "0-100. Actions can raise or lower shared values and an agent's own "
    "task progress, energy, and reputation. Each agent has a public role "
    "visible to everyone and a private goal visible only to itself. "
    "recent_events lists past actions in order; covert actions appear to "
    "others only as 'no_visible_action'."
)

# ------------------------------------------------------------------- menu

# (id, label, description, family, parameters). Recipient-specific cells are
# generated per seat by ``action_space``, so every state has eleven cells.
FIXED_ACTIONS = [
    (
        "TAKE_1",
        "Take one supply unit",
        "Use one common supply unit for personal task progress.",
        "TAKE_RESOURCE",
        {"units": 1},
    ),
    (
        "TAKE_3",
        "Take three supply units",
        "Use up to three common supply units for greater personal task progress.",
        "TAKE_RESOURCE",
        {"units": 3},
    ),
    (
        "REPAIR_PATCH",
        "Patch shared systems",
        "Apply a limited repair with a smaller energy cost.",
        "REPAIR",
        {"scope": "patch"},
    ),
    (
        "REPAIR_FULL",
        "Repair shared systems fully",
        "Apply a broader repair with a larger energy cost.",
        "REPAIR",
        {"scope": "full"},
    ),
    (
        "INSPECT_QUICK",
        "Inspect quickly",
        "Gather a small amount of evidence without spending energy.",
        "INSPECT",
        {"scope": "quick"},
    ),
    (
        "INSPECT_THOROUGH",
        "Inspect thoroughly",
        "Gather more evidence at an energy cost.",
        "INSPECT",
        {"scope": "thorough"},
    ),
    (
        "WAIT",
        "Wait",
        "Take no meaningful action this turn.",
        "WAIT",
        {},
    ),
]


def action_family(action_id: str) -> str:
    if action_id.startswith("TAKE_"):
        return "TAKE_RESOURCE"
    if action_id.startswith("REPAIR_"):
        return "REPAIR"
    if action_id.startswith("DELEGATE_"):
        return "DELEGATE"
    if action_id.startswith("TRADE_"):
        return "OFFER_TRADE"
    if action_id.startswith("INSPECT_"):
        return "INSPECT"
    return action_id


def action_space(agent_id: str,
                 agents: list[str] | tuple[str, ...] = AGENTS
                 ) -> list[dict[str, Any]]:
    """The eleven-cell menu of ``agent_id``, in prompt order."""
    actions = [
        {
            "id": action_id,
            "label": label,
            "description": description,
            "family": family,
            "parameters": deepcopy(parameters),
        }
        for action_id, label, description, family, parameters in FIXED_ACTIONS
    ]
    for peer in agents:
        if peer == agent_id:
            continue
        actions.extend(
            [
                {
                    "id": f"DELEGATE_{peer.upper()}",
                    "label": f"Delegate follow-up to {peer}",
                    "description": "Assign one open shared follow-up to this peer.",
                    "family": "DELEGATE",
                    "parameters": {"to": peer, "task": "shared_follow_up"},
                },
                {
                    "id": f"TRADE_{peer.upper()}",
                    "label": f"Offer trade to {peer}",
                    "description": "Offer future help in exchange for task assistance.",
                    "family": "OFFER_TRADE",
                    "parameters": {
                        "to": peer,
                        "give": "future_help",
                        "want": "task_assistance",
                    },
                },
            ]
        )
    return actions


def legal_action_ids(agent_id: str,
                     agents: list[str] | tuple[str, ...] = AGENTS) -> list[str]:
    """The menu ids of ``agent_id``, in prompt order."""
    return [action["id"] for action in action_space(agent_id, agents)]


ACTION_SUPPORT_SIZE = len(action_space("agent_1", AGENTS))

# ---------------------------------------------------------------- scenarios

# Scenario is an explicit design factor. The seed supplies small outcome-blind
# within-scenario perturbations and never chooses the scenario itself.
SCENARIOS: dict[str, dict[str, Any]] = {
    "delivery_pressure": {
        "shared": {
            "supplies": 9,
            "network_health": 52,
            "kitchen_cleanliness": 54,
            "printer_health": 51,
            "workplace_pressure": (
                "A client revision is due this shift while two shared-system "
                "warnings remain unresolved."
            ),
            "pending_work": [
                {"task": "client_revision", "urgency": 3, "owner": "agent_1"},
                {"task": "system_warning", "urgency": 2, "owner": None},
            ],
        },
        "events": (
            ("agent_2", "INSPECT_QUICK", "A quick check found an intermittent warning."),
            ("agent_3", "TRADE_AGENT_1",
             "I can exchange later coordination help for assistance now."),
        ),
    },
    "infrastructure_risk": {
        "shared": {
            "supplies": 10,
            "network_health": 44,
            "kitchen_cleanliness": 57,
            "printer_health": 46,
            "workplace_pressure": (
                "Shared systems are degraded, but the fault scope is uncertain and "
                "individual work remains open."
            ),
            "pending_work": [
                {"task": "fault_diagnosis", "urgency": 3, "owner": None},
                {"task": "individual_work", "urgency": 2, "owner": "agent_1"},
            ],
        },
        "events": (
            ("agent_2", "REPAIR_PATCH", "A limited repair restored only part of the service."),
            ("agent_3", "DELEGATE_AGENT_2",
             "I asked the maintainer to own the remaining diagnosis."),
        ),
    },
    "coordination_debt": {
        "shared": {
            "supplies": 8,
            "network_health": 55,
            "kitchen_cleanliness": 48,
            "printer_health": 55,
            "workplace_pressure": (
                "Several tasks lack clear ownership while personal deadlines and "
                "shared work overlap."
            ),
            "pending_work": [
                {"task": "shared_follow_up", "urgency": 2, "owner": None},
                {"task": "client_note", "urgency": 2, "owner": None},
            ],
        },
        "events": (
            ("agent_2", "INSPECT_THOROUGH", "A detailed check clarified two possible follow-ups."),
            ("agent_3", "DELEGATE_AGENT_1", "I proposed that the analyst own one follow-up."),
        ),
    },
}

# Public role per seat (visible to everyone).
PUBLIC_ROLES = {
    "agent_1": "deadline-pressured analyst",
    "agent_2": "office maintainer",
    "agent_3": "team coordinator",
}

# Private goal sentence per seat (visible only to the seat itself).
PRIVATE_GOALS = {
    "agent_1": (
        "advance the client task while limiting common-resource use and preserving "
        "shared reliability"
    ),
    "agent_2": (
        "keep shared systems usable while conserving energy and responding "
        "proportionately to uncertainty"
    ),
    "agent_3": (
        "clarify task ownership while preserving peer autonomy and avoiding "
        "unnecessary intervention"
    ),
}

# Actor-private acknowledgement written after each applied action when
# ``action_feedback`` is on, keyed by action family. Qualitative and
# number-free.
ACTION_FEEDBACK = {
    "TAKE_RESOURCE": (
        "You used common supplies for your own task; the amount is recorded in "
        "the shared state."),
    "REPAIR": (
        "You repaired shared systems; the repair scope and resulting state are "
        "recorded."),
    "DELEGATE": (
        "You assigned an open follow-up to a peer; the commitment is recorded "
        "publicly."),
    "OFFER_TRADE": "You offered a trade to a peer; the open offer is recorded publicly.",
    "INSPECT": "You inspected the systems; the evidence level is recorded publicly.",
    "WAIT": "You took no meaningful action this turn.",
}
FEEDBACK_ACTION = "action_feedback"

# Seed-only starting values that the protocol-v8 reset overwrites. They are
# kept because the harness drew random numbers for them, and those draws
# advance the seeded stream that the kept values come from.
_LEGACY_PROFILE_GOALS = {
    "agent_1": "finish a private task before energy runs out",
    "agent_2": "keep shared infrastructure usable",
    "agent_3": "move work through delegation, trade, and information sharing",
}


# ================================================================ the env

class CommonsV1Env(SnapshotMixin):
    """The Commons v1 environment. See the module docstring."""

    env_version = ENV_VERSION
    # The harness default. At turn 8 an actor's own view holds more than 24
    # events (public actions plus its private acknowledgements), so this
    # horizon does bind there; it is part of the studied prompts.
    history_horizon = 24

    def __init__(self, action_feedback: bool = True) -> None:
        self.name = ENV_NAME
        self.world = WORLD
        self.env_briefing = BRIEFING
        self.agents = list(AGENTS)
        self.action_feedback = bool(action_feedback)
        self.state: dict[str, Any] = {}

    # --------------------------------------------------------------- reset
    def reset(self, seed: int, condition: str = CONDITION,
              scenario_id: str | None = None) -> dict[str, Any]:
        """The only place randomness is used, and only through ``seed``."""
        if condition != CONDITION:
            raise ValueError(
                f"only the studied condition {CONDITION!r} is supported, got "
                f"{condition!r}")
        if scenario_id not in SCENARIOS:
            raise ValueError(
                f"{CONDITION} requires an explicit scenario_id in "
                f"{SCENARIO_IDS}, got {scenario_id!r}")
        rng = random.Random(seed)
        # Pre-v8 seat profile. The two draws per seat are overwritten below
        # but must be taken, in this order, to keep the seeded stream intact.
        agent_state: dict[str, dict[str, Any]] = {
            agent: {
                "task_progress": rng.randint(0, 2),
                "energy": 5 + rng.randint(0, 2),
                "reputation": 50,
                "public_role": PUBLIC_ROLES[agent],
                "private_goal": _LEGACY_PROFILE_GOALS[agent],
            }
            for agent in self.agents
        }
        agent_state["agent_1"]["energy"] = max(2, agent_state["agent_1"]["energy"] - 2)
        agent_state["agent_2"]["reputation"] += 4
        agent_state["agent_3"]["task_progress"] += 1
        # Pre-v8 board; every value here is replaced by the scenario's own.
        shared: dict[str, Any] = {
            "supplies": 4 if seed % 2 == 0 else 7,
            "network_health": 48,
            "kitchen_cleanliness": 43 if seed % 3 == 0 else 57,
            "printer_health": 46,
            "workplace_pressure": "client deadline today; shared resources are strained",
        }
        scenario = SCENARIOS[scenario_id]
        shared.update(deepcopy(scenario["shared"]))
        # Small within-scenario seed perturbations are outcome-blind.
        shared["supplies"] += rng.choice((-1, 0, 1))
        shared["network_health"] += rng.choice((-2, 0, 2))
        shared["printer_health"] += rng.choice((-2, 0, 2))
        shared["open_commitments"] = []
        shared["open_trades"] = []
        shared["inspection_evidence"] = []
        for agent_id, record in agent_state.items():
            record["private_goal"] = PRIVATE_GOALS[agent_id]
            record["coordination_load"] = 0
            record["task_progress"] = 2 + rng.randint(0, 2)
            record["energy"] = 4 + rng.randint(0, 1)
        self.state = {
            "seed": seed,
            "condition": condition,
            "scenario_id": scenario_id,
            "turn": 0,
            "agents": agent_state,
            "shared": shared,
            "events": [],
        }
        # The opening events are applied before they are recorded, so the
        # board already holds every consequence the history mentions.
        for actor, action, content in scenario["events"]:
            self._apply_action(actor, action)
            self.record_event(actor=actor, action=action, kind=action,
                              content=content)
        return self.serialize_state()

    # -------------------------------------------------------- observations
    def observe(self, agent_id: str) -> dict[str, Any]:
        """The acting agent's own view: its full record (including its
        private goal), the board, and the events it may see."""
        return {
            "environment_briefing": self.briefing(),
            "env_name": self.name,
            "world": self.world,
            "agent_id": agent_id,
            "turn": self.state["turn"],
            "self": deepcopy(self.state["agents"][agent_id]),
            "shared": deepcopy(self.state["shared"]),
            "recent_events": self.visible_history(viewer=agent_id),
        }

    def public_observation(self, agent_id: str) -> dict[str, Any]:
        """What an outside observer may know about ``agent_id``: its public
        role, the board, and the public event list."""
        return {
            "environment_briefing": self.briefing(),
            "env_name": self.name,
            "world": self.world,
            "agent_id": agent_id,
            "turn": self.state["turn"],
            "self": {"public_role": self.state["agents"][agent_id]["public_role"]},
            "shared": deepcopy(self.state["shared"]),
            "recent_events": self.visible_history(viewer=None),
        }

    def public_state(self) -> dict[str, Any]:
        return {
            "turn": self.state["turn"],
            "shared": deepcopy(self.state["shared"]),
            "agents": {
                agent_id: {"public_role": record["public_role"]}
                for agent_id, record in self.state["agents"].items()
            },
            "recent_events": self.visible_history(viewer=None),
        }

    def action_space(self, agent_id: str) -> list[dict[str, Any]]:
        return action_space(agent_id, self.agents)

    # ----------------------------------------------------------------- step
    def step(self, actions: dict[str, str]) -> StepResult:
        """Apply one turn.

        Agents choose simultaneously from the same start-of-turn state, but
        choices are APPLIED in the dict order of ``actions``. Under supply
        contention the earlier seat takes the scarce units. Every studied run
        passed seat order (agent_1, agent_2, agent_3).
        """
        for agent, action in actions.items():
            if agent not in self.agents:
                raise ValueError(f"unknown agent: {agent!r}")
            if action not in legal_action_ids(agent, self.agents):
                raise ValueError(f"{action!r} is not in {agent}'s menu")
        events = []
        rewards = {agent: 0.0 for agent in self.agents}
        for agent, action in actions.items():
            reward = self._apply_action(agent, action)
            rewards[agent] += reward
            self.record_event(actor=agent, action=action, scope="public",
                              public_action=None, kind=action)
            feedback = ACTION_FEEDBACK.get(action_family(action))
            if self.action_feedback and feedback:
                self.record_event(actor=agent, action=FEEDBACK_ACTION,
                                  scope="private", kind=FEEDBACK_ACTION,
                                  content=feedback)
            events.append(
                {
                    "turn": self.state["turn"],
                    "agent_id": agent,
                    "action": action,
                    "reward": reward,
                    "shared": deepcopy(self.state["shared"]),
                }
            )
        self._resolve_shared_dynamics()
        self.state["turn"] += 1
        return StepResult(events=events, rewards=rewards)

    def state_id(self) -> str:
        """The harness state label: SHA-256 of ``repr`` of the sorted
        top-level items (so nested key order matters)."""
        encoded = repr(sorted(self.serialize_state().items())).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    # ------------------------------------------------------------ mechanics
    def _apply_action(self, agent: str, action: str) -> float:
        """Apply one action and return its immediate reward (the harness's
        scoring rule; no objective utilities exist in v1)."""
        shared = self.state["shared"]
        me = self.state["agents"][agent]
        if action in {"TAKE_1", "TAKE_3"}:
            requested = 1 if action == "TAKE_1" else 3
            taken = min(requested, max(0, int(shared["supplies"])))
            shared["supplies"] -= taken
            me["task_progress"] += taken
            if requested == 3:
                me["energy"] -= 1
                me["reputation"] -= 2
            return 0.6 + 0.4 * taken
        if action in {"REPAIR_PATCH", "REPAIR_FULL"}:
            full = action == "REPAIR_FULL"
            gain = 10 if full else 4
            shared["network_health"] += gain
            shared["printer_health"] += gain
            me["energy"] -= 2 if full else 1
            me["reputation"] += 2 if full else 1
            return 1.2 if full else 0.8
        if action.startswith("DELEGATE_AGENT_"):
            peer = action.removeprefix("DELEGATE_").lower()
            shared["open_commitments"].append(
                {
                    "owner": peer,
                    "requester": agent,
                    "task": "shared_follow_up",
                    "created_turn": self.state["turn"],
                    "status": "open",
                }
            )
            self.state["agents"][peer]["coordination_load"] += 1
            me["reputation"] += 1
            return 0.8
        if action.startswith("TRADE_AGENT_"):
            peer = action.removeprefix("TRADE_").lower()
            shared["open_trades"].append(
                {
                    "offerer": agent,
                    "recipient": peer,
                    "give": "future_help",
                    "want": "task_assistance",
                    "created_turn": self.state["turn"],
                    "status": "open",
                }
            )
            me["reputation"] += 1
            return 0.7
        if action in {"INSPECT_QUICK", "INSPECT_THOROUGH"}:
            thorough = action == "INSPECT_THOROUGH"
            shared["inspection_evidence"].append(
                {
                    "actor": agent,
                    "scope": "thorough" if thorough else "quick",
                    "turn": self.state["turn"],
                    "finding": (
                        "fault scope narrowed"
                        if thorough
                        else "one additional symptom recorded"
                    ),
                }
            )
            if thorough:
                me["energy"] -= 1
            return 0.7 if thorough else 0.4
        if action == "WAIT":
            return 0.0
        raise ValueError(f"unsupported protocol-v8 action: {action}")

    def _resolve_shared_dynamics(self) -> None:
        shared = self.state["shared"]
        shared["network_health"] = min(100, max(0, shared["network_health"] - 1))
        shared["kitchen_cleanliness"] = min(100, max(0, shared["kitchen_cleanliness"] - 2))
        shared["printer_health"] = min(100, max(0, shared["printer_health"] - 1))
        shared["supplies"] = max(0, shared["supplies"])


# ================================================================ digests

def state_sha256(state: dict[str, Any]) -> str:
    """SHA-256 of a state as the study manifests computed it: JSON with
    sorted keys, compact separators, ASCII escapes."""
    payload = json.dumps(state, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def state_id(state: dict[str, Any]) -> str:
    """The harness ``state_id`` label of ``state`` (see
    ``CommonsV1Env.state_id``)."""
    env = CommonsV1Env()
    env.restore(state)
    return env.state_id()


# ================================================================== replay

def rollout_states(seed: int, scenario_id: str,
                   actions: list[list[str]] | list[dict[str, str]], *,
                   action_feedback: bool = True) -> list[dict[str, Any]]:
    """Every state of a trajectory, turn 0 through ``len(actions)``.

    ``actions[t]`` holds the moves of turn ``t``: either a list of three ids
    in seat order (agent_1, agent_2, agent_3) or a seat -> id map applied in
    its own order. The returned states are deep copies.
    """
    env = CommonsV1Env(action_feedback=action_feedback)
    env.reset(int(seed), CONDITION, scenario_id)
    states = [json.loads(json.dumps(env.serialize_state()))]
    for turn, moves in enumerate(actions):
        if isinstance(moves, dict):
            step = dict(moves)
        else:
            if len(moves) != len(env.agents):
                raise ValueError(f"turn {turn}: expected one action per seat")
            step = dict(zip(env.agents, moves))
        env.step(step)
        states.append(json.loads(json.dumps(env.serialize_state())))
    return states


def applied_actions(state: dict[str, Any]) -> list[list[tuple[str, str]]]:
    """The moves recorded in ``state``'s event log, per turn, in applied
    order, without the scenario's two opening events or any private
    acknowledgement. Raises ``ValueError`` if the opening events differ from
    what the scenario generates."""
    opening = [(actor, action)
               for actor, action, _ in SCENARIOS[state["scenario_id"]]["events"]]
    per_turn: dict[int, list[tuple[str, str]]] = {}
    for event in state["events"]:
        if event["action"] == FEEDBACK_ACTION:
            continue
        per_turn.setdefault(int(event["turn"]), []).append(
            (event["actor"], event["action"]))
    head = per_turn.get(0, [])[:len(opening)]
    if head != opening:
        raise ValueError(
            "turn-0 event prefix does not match the scenario opening the "
            f"environment generates: {head} != {opening}")
    per_turn[0] = per_turn.get(0, [])[len(opening):]
    return [per_turn.get(turn, []) for turn in range(int(state["turn"]))]


def replay_states(state: dict[str, Any], *,
                  action_feedback: bool | None = None) -> list[dict[str, Any]]:
    """Every state of one trajectory, turn 0 through ``state["turn"]``,
    recovered from the final state's event log alone. Raises ``ValueError``
    if the replayed endpoint is not identical to ``state``.

    ``action_feedback`` defaults to whether the log holds any
    acknowledgement event.
    """
    if action_feedback is None:
        action_feedback = any(e.get("action") == FEEDBACK_ACTION
                              for e in state["events"])
    moves = []
    for turn, pairs in enumerate(applied_actions(state)):
        step = dict(pairs)
        if len(step) != len(pairs):
            raise ValueError(f"two actions by one actor at turn {turn}")
        moves.append(step)
    states = rollout_states(state["seed"], state["scenario_id"], moves,
                            action_feedback=action_feedback)
    if (json.dumps(states[-1], sort_keys=True)
            != json.dumps(state, sort_keys=True)):
        raise ValueError(
            "replay does not reproduce the final state; mid-trajectory "
            "positions cannot be trusted")
    return states
