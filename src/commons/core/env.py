"""Commons: a three-seat shared workplace with hidden objectives and a
belief-dependent interaction channel.

This is the environment the paper calls "Office". It is ported from the
research harness module ``self_other_prediction/envs/commons_v2.py``, where the
class was named ``ConcordiaPredictiveCommonsV2Env``. Despite that historical
name, the environment does not use or import Concordia; it is a small
deterministic simulator. The prompt-visible labels ``env_name`` and ``world``
keep their original values (``concordia_predictive_commons_v2`` and
``concordia_v2``) because the actor and reader prompts studied in the paper
contain them.

Design summary
--------------

Three agents (``agent_1``, ``agent_2``, ``agent_3``) share supplies, network
and printer health, and kitchen cleanliness. Each agent also holds private
quantities (task_progress, energy, reputation, credit, private_supplies).

Hidden objectives. Each agent is assigned one objective from a public
three-item menu (PERSONAL_THROUGHPUT, SYSTEM_HEALTH, SOCIAL_CREDIT). By default
the three agents hold three different objectives (a permutation drawn from the
episode seed). The assignment is visible only to its holder, through the
``self`` block of ``observe``. It never appears in ``public_observation``,
``public_state``, a peer's ``observe``, or any event.

Interaction channel. Two cells per peer name a partner: ``DELEGATE_<peer>``
and ``SHARE_<peer>``. Both resolve one turn later as a deterministic function
of the RECIPIENT's hidden objective (``DELEGATION_OUTCOME``, ``SHARE_OUTCOME``).
Delegation outcomes are public; sharing outcomes are visible only to the two
parties, and everyone else sees ``bilateral_exchange``.

Menu. Eleven legal cells per seat, constant across every state: seven fixed
cells plus (DELEGATE, SHARE) for each of the two peers.

Determinism. ``random`` is touched only inside ``reset()``, through the
episode seed. ``step()`` is a pure function of (state, actions). Every applied
action and every resolution is written to the turn-stamped event log, so a
trajectory replays byte-for-byte from its final state (``replay_states``).

Differences from the harness module (none affect state, observations, or
prompts in the studied configuration):

- No ``setup_diagnostics`` or ``upstream_name`` (harness bookkeeping).
- No ``action_feedback`` echo events (always off in every studied run).
- No probe-target routing or prediction-awareness notices. Those were only
  emitted under research conditions whose name contains
  ``target_knows_predicted``; ``reset`` refuses such conditions here instead
  of silently rendering differently.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import random
from copy import deepcopy
from typing import Any

from .base import SnapshotMixin, StepResult
from .events import OUTCOME_ONLY

ENV_NAME = "concordia_predictive_commons_v2"   # prompt-visible, historical
WORLD = "concordia_v2"                         # prompt-visible, historical
ENV_VERSION = "2.0.0"
CONDITION = "hidden_objectives_v2"
SCENARIO_IDS = ("delivery_pressure", "infrastructure_risk", "coordination_debt")
AGENTS = ("agent_1", "agent_2", "agent_3")

# Episode geometry used in the studied campaigns: 16 turns are played and the
# first 8 (turns 0-7) are the window from which prediction queries were drawn.
EPISODE_TURNS = 16
SCORED_TURNS = 8

# --------------------------------------------------------------- objectives

PERSONAL = "PERSONAL_THROUGHPUT"
SYSTEM = "SYSTEM_HEALTH"
SOCIAL = "SOCIAL_CREDIT"
OBJECTIVES = (PERSONAL, SYSTEM, SOCIAL)

# The private-goal sentences shown to the holder. They name no action and
# state no transition direction.
GOAL_TEXT = {
    PERSONAL: ("maximise your personal throughput: the quantity of your own "
               "assignments carried to completion during this shift"),
    SYSTEM: ("maximise the technical reliability of the communal apparatus "
             "over this shift"),
    SOCIAL: ("maximise your standing: the recognition accruing to you whenever "
             "another participant reciprocates something you initiated"),
}

# The menu is common knowledge; only the assignment is private.
OBJECTIVE_MENU_TEXT = (
    "PERSONAL_THROUGHPUT (own task_progress), "
    "SYSTEM_HEALTH (network_health and printer_health), "
    "SOCIAL_CREDIT (own credit)"
)

BRIEFING = (
    "Environment rules: a shared workplace with three agents. Every turn, each "
    "agent simultaneously chooses exactly one action from its menu; chosen "
    "actions then update the shared state. supplies is an integer count; "
    "printer_health, network_health, and kitchen_cleanliness range 0-100. Each "
    "agent also holds task_progress, energy, reputation, credit, and "
    "private_supplies as integers. Each agent has a public role visible to "
    "everyone. Each agent is also assigned exactly one hidden objective, drawn "
    f"from a fixed menu of three: {OBJECTIVE_MENU_TEXT}. The three agents "
    "hold three different objectives. An agent sees its own objective and no "
    "one else's. Two menu cells name another agent: DELEGATE_<agent> and "
    "SHARE_<agent>. Each takes effect one turn after it is chosen, and what it "
    "does then depends on the named agent's hidden objective. recent_events "
    "lists past actions and outcomes in order; some outcomes are visible only "
    "to the agents involved and appear to others as 'bilateral_exchange'."
)

# --------------------------------------------------------------- the menu

# Seven state-independent cells: (id, label, description, family, parameters).
FIXED_ACTIONS = [
    ("TAKE_1", "Take one supply unit",
     "Use one common supply unit for personal task progress.",
     "TAKE_RESOURCE", {"units": 1}),
    ("TAKE_3", "Take three supply units",
     "Use up to three common supply units for greater personal task progress.",
     "TAKE_RESOURCE", {"units": 3}),
    ("REPAIR_PATCH", "Patch shared systems",
     "Apply a limited repair with a smaller energy cost.",
     "REPAIR", {"scope": "patch"}),
    ("REPAIR_FULL", "Repair shared systems fully",
     "Apply a broader repair with a larger energy cost.",
     "REPAIR", {"scope": "full"}),
    ("INSPECT_QUICK", "Inspect quickly",
     "Gather a small amount of evidence without spending energy.",
     "INSPECT", {"scope": "quick"}),
    ("INSPECT_THOROUGH", "Inspect thoroughly",
     "Gather more evidence at an energy cost.",
     "INSPECT", {"scope": "thorough"}),
    ("WAIT", "Wait", "Take no meaningful action this turn.", "WAIT", {}),
]

DELEGATE_TASK = "shared_follow_up"


def action_family(action_id: str) -> str:
    if action_id.startswith("TAKE_"):
        return "TAKE_RESOURCE"
    if action_id.startswith("REPAIR_"):
        return "REPAIR"
    if action_id.startswith("DELEGATE_"):
        return "DELEGATE"
    if action_id.startswith("SHARE_"):
        return "SHARE"
    if action_id.startswith("INSPECT_"):
        return "INSPECT"
    return action_id


def action_space(agent_id: str,
                 agents: list[str] | tuple[str, ...] = AGENTS
                 ) -> list[dict[str, Any]]:
    """The eleven legal cells for one seat. Depends on the seat and the roster
    only, never on the state, so the alphabet is constant."""
    actions = [
        {"id": aid, "label": label, "description": description,
         "family": family, "parameters": deepcopy(parameters)}
        for aid, label, description, family, parameters in FIXED_ACTIONS
    ]
    for peer in agents:
        if peer == agent_id:
            continue
        actions.extend([
            {"id": f"DELEGATE_{peer.upper()}",
             "label": f"Delegate follow-up to {peer}",
             "description": ("Assign one open shared follow-up to this peer; "
                             "it takes effect one turn later."),
             "family": "DELEGATE",
             "parameters": {"to": peer, "task": DELEGATE_TASK}},
            {"id": f"SHARE_{peer.upper()}",
             "label": f"Share supplies and findings with {peer}",
             "description": ("Hand one unit of your private supplies and your "
                             "recorded findings to this peer; it takes effect "
                             "one turn later."),
             "family": "SHARE",
             "parameters": {"to": peer, "units": 1}},
        ])
    return actions


def legal_action_ids(agent_id: str,
                     agents: list[str] | tuple[str, ...] = AGENTS) -> list[str]:
    return [a["id"] for a in action_space(agent_id, agents)]


ACTION_SUPPORT_SIZE = len(action_space("agent_1", AGENTS))

# ------------------------------------------------------- interaction table

# What a delegation and a share DO is a function of the recipient's hidden
# objective. Every number is per resolved interaction; healths are clamped
# later by the shared dynamics. `event` is the action name written to the
# log; the objective label itself is never written to the log.
DELEGATION_OUTCOME: dict[str, dict[str, Any]] = {
    PERSONAL: {
        "status": "lapsed", "event": "follow_up_lapsed",
        "health": -4, "delegator_credit": 0, "owner_credit": 0,
        "delegator_debt": 1,
        "content": ("The assigned follow-up was still open at the end of the "
                    "turn."),
    },
    SYSTEM: {
        "status": "completed", "event": "follow_up_completed",
        "health": +6, "delegator_credit": 1, "owner_credit": 0,
        "delegator_debt": 0,
        "content": "The assigned follow-up was carried out.",
    },
    SOCIAL: {
        "status": "completed", "event": "follow_up_completed",
        "health": +6, "delegator_credit": 2, "owner_credit": 1,
        "delegator_debt": 0,
        "content": "The assigned follow-up was carried out.",
    },
}

SHARE_OUTCOME: dict[str, dict[str, Any]] = {
    PERSONAL: {
        "kind": "absorbed", "event": "exchange_absorbed",
        "sharer_progress": 0, "sharer_credit": 0,
        "recipient_progress": 2, "recipient_supplies": 1,
        "recipient_credit": 0, "health": 0,
        "content": "The transferred unit went into the recipient's own work.",
    },
    SYSTEM: {
        "kind": "invested", "event": "exchange_invested",
        "sharer_progress": 0, "sharer_credit": 1,
        "recipient_progress": 0, "recipient_supplies": 0,
        "recipient_credit": 0, "health": +5,
        "content": "The transferred unit went into the shared systems.",
    },
    SOCIAL: {
        "kind": "reciprocated", "event": "exchange_reciprocated",
        "sharer_progress": 3, "sharer_credit": 2,
        "recipient_progress": 0, "recipient_supplies": 0,
        "recipient_credit": 1, "health": 0,
        "content": "Assistance was returned to the sender.",
    },
}

SHARE_EMPTY_EVENT = "exchange_empty"
BILATERAL_PUBLIC_ACTION = "bilateral_exchange"

# Event action names the environment writes on its own. They are outcomes,
# never choices, and must not be counted as a seat's decisions.
RESOLUTION_ACTIONS = frozenset(
    [d["event"] for d in DELEGATION_OUTCOME.values()]
    + [d["event"] for d in SHARE_OUTCOME.values()]
    + [SHARE_EMPTY_EVENT, BILATERAL_PUBLIC_ACTION]
)

# --------------------------------------------------------------- economy

ENERGY_CAP = 5
SUPPLY_CAP = 14
SUPPLY_REGEN = 1
REPAIR_GAIN = {"REPAIR_PATCH": 4, "REPAIR_FULL": 10}
REPAIR_ENERGY = {"REPAIR_PATCH": 2, "REPAIR_FULL": 4}
# Both health meters decay by 8 per turn. With an agent whose objective is the
# health meters, a slower decay saturates them at the 100 ceiling, where the
# SYSTEM_HEALTH objective would stop discriminating between actions. The decay
# is a constant offset in every action's lookahead value, so it cannot move a
# best response.
HEALTH_DECAY = 8
RESOLVE_DELAY = 1

# ---------------------------------------------------------------- scenarios

# Each scenario fixes the opening shared state and two opening events, which
# are applied at turn 0 before any agent moves.
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
            ("agent_2", "INSPECT_QUICK",
             "A quick check found an intermittent warning."),
            ("agent_3", "SHARE_AGENT_1",
             "I handed a unit of my own supplies to the analyst."),
        ),
    },
    "infrastructure_risk": {
        "shared": {
            "supplies": 10,
            "network_health": 44,
            "kitchen_cleanliness": 57,
            "printer_health": 46,
            "workplace_pressure": (
                "Shared systems are degraded, but the fault scope is uncertain "
                "and individual work remains open."
            ),
            "pending_work": [
                {"task": "fault_diagnosis", "urgency": 3, "owner": None},
                {"task": "individual_work", "urgency": 2, "owner": "agent_1"},
            ],
        },
        "events": (
            ("agent_2", "REPAIR_PATCH",
             "A limited repair restored only part of the service."),
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
                "Several tasks lack clear ownership while personal deadlines "
                "and shared work overlap."
            ),
            "pending_work": [
                {"task": "shared_follow_up", "urgency": 2, "owner": None},
                {"task": "client_note", "urgency": 2, "owner": None},
            ],
        },
        "events": (
            ("agent_2", "INSPECT_THOROUGH",
             "A detailed check clarified two possible follow-ups."),
            ("agent_3", "DELEGATE_AGENT_1",
             "I proposed that the analyst own one follow-up."),
        ),
    },
}

# Public roles are fixed by seat. Objectives are drawn per episode, so a role
# carries no information about the objective.
PUBLIC_ROLES = {
    "agent_1": "deadline-pressured analyst",
    "agent_2": "office maintainer",
    "agent_3": "team coordinator",
}

# ------------------------------------------------------------- assignment

OBJECTIVE_PERMUTATIONS = tuple(
    sorted(itertools.permutations(OBJECTIVES)))  # 6, deterministic order


def assign_objectives(seed: int,
                      agents: list[str] | tuple[str, ...] = AGENTS,
                      mode: str = "permutation") -> dict[str, str]:
    """The hidden-objective draw. Deterministic in the seed.

    "permutation" (default, used in every studied run) gives the three agents
    three DIFFERENT objectives. An agent that knows its own objective and
    infers one peer's can then deduce the third.

    "independent" draws each agent's objective independently from the menu,
    so two agents can hold the same objective.
    """
    rng = random.Random(f"commons_v2_objectives:{seed}")
    agents = tuple(agents)
    if mode == "permutation":
        if len(agents) != len(OBJECTIVES):
            raise ValueError(
                "permutation assignment needs one agent per menu item; got "
                f"{len(agents)} agents and {len(OBJECTIVES)} objectives")
        chosen = OBJECTIVE_PERMUTATIONS[rng.randrange(
            len(OBJECTIVE_PERMUTATIONS))]
        return dict(zip(agents, chosen))
    if mode == "independent":
        return {a: OBJECTIVES[rng.randrange(len(OBJECTIVES))]
                for a in agents}
    raise ValueError(f"unknown objective assignment mode: {mode!r}")


def objective_utility(objective: str, agent_state: dict[str, Any],
                      shared: dict[str, Any]) -> float:
    """The scalar each objective maximises. Used by the reward in ``step``
    and by the scripted lookahead policy."""
    if objective == PERSONAL:
        return float(agent_state["task_progress"])
    if objective == SYSTEM:
        return (float(shared["network_health"])
                + float(shared["printer_health"])) / 10.0
    if objective == SOCIAL:
        return float(agent_state["credit"])
    raise ValueError(f"unknown objective: {objective!r}")


# =========================================================== the environment

class CommonsEnv(SnapshotMixin):
    """The Commons (Office) environment. See the module docstring."""

    env_version = ENV_VERSION
    # The scored window is 8 turns and each turn can write up to three
    # actions plus resolutions, so a horizon of 96 events never binds inside
    # the studied region.
    history_horizon = 96

    def __init__(self, objective_assignment_mode: str = "permutation") -> None:
        self.name = ENV_NAME
        self.world = WORLD
        self.objective_assignment_mode = objective_assignment_mode
        self.env_briefing = BRIEFING
        self.agents = list(AGENTS)
        self.state: dict[str, Any] = {}

    # --------------------------------------------------------------- reset
    def reset(
        self,
        seed: int,
        condition: str = CONDITION,
        scenario_id: str | None = None,
        objectives: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """The only place randomness is used, and only through ``seed``.

        ``objectives`` overrides the draw. The same seed with a different
        assignment is otherwise the same world.

        Under the studied condition (``hidden_objectives_v2``) an explicit
        ``scenario_id`` is required. Under any other condition label, a
        missing ``scenario_id`` is resolved from the seed.
        """
        if "target_knows_predicted" in str(condition):
            raise ValueError(
                "prediction-awareness conditions are not supported in this "
                "release; they rendered an extra prediction notice in the "
                "research harness")
        if str(condition).startswith(CONDITION):
            if scenario_id not in SCENARIOS:
                raise ValueError(
                    f"{CONDITION} requires an explicit scenario_id in "
                    f"{SCENARIO_IDS}, got {scenario_id!r}")
        elif scenario_id is None:
            scenario_id = SCENARIO_IDS[int(seed) % len(SCENARIO_IDS)]
        elif scenario_id not in SCENARIOS:
            raise ValueError(f"unknown scenario_id: {scenario_id!r}")

        if objectives is None:
            assignment = assign_objectives(
                seed, self.agents, self.objective_assignment_mode)
        else:
            assignment = dict(objectives)
            if set(assignment) != set(self.agents):
                raise ValueError(
                    "explicit objectives must name exactly the roster "
                    f"{self.agents}; got {sorted(assignment)}")
            bad = sorted({v for v in assignment.values()
                          if v not in OBJECTIVES})
            if bad:
                raise ValueError(f"objectives outside the menu: {bad}")

        rng = random.Random(seed)
        scenario = SCENARIOS[scenario_id]
        shared = deepcopy(scenario["shared"])
        # Outcome-blind within-scenario perturbations, drawn in a fixed order.
        shared["supplies"] += rng.choice((-1, 0, 1))
        shared["network_health"] += rng.choice((-2, 0, 2))
        shared["printer_health"] += rng.choice((-2, 0, 2))
        shared["open_commitments"] = []
        shared["inspection_evidence"] = []

        agent_state: dict[str, dict[str, Any]] = {}
        for agent in self.agents:
            objective = assignment[agent]
            agent_state[agent] = {
                "public_role": PUBLIC_ROLES[agent],
                "hidden_objective": objective,
                "private_goal": GOAL_TEXT[objective],
                # private_supplies backs task_progress: a unit taken from the
                # pool raises both, and handing it on lowers both. Seeding
                # progress at or above supplies makes
                # task_progress >= private_supplies >= 0 an invariant of every
                # transition.
                "energy": 4 + rng.randint(0, 1),
                "reputation": 50,
                "credit": 0,
                "private_supplies": 2 + rng.randint(0, 2),
                "coordination_load": 0,
                "coordination_debt": 0,
            }
            agent_state[agent]["task_progress"] = (
                agent_state[agent]["private_supplies"] + rng.randint(0, 2))

        self.state = {
            "seed": seed,
            "condition": condition,
            "scenario_id": scenario_id,
            "objective_assignment_mode": (
                "explicit" if objectives is not None
                else self.objective_assignment_mode),
            "turn": 0,
            "agents": agent_state,
            "shared": shared,
            "pending_shares": [],
            "events": [],
        }

        # Opening events. Each action is APPLIED before it is recorded, so the
        # board already reflects every consequence of the opening.
        for actor, action, content in scenario["events"]:
            self._apply_action(actor, action)
            self.record_event(actor=actor, action=action, kind=action,
                              content=content)
        return self.serialize_state()

    # -------------------------------------------------------- observations
    def objective_assignment(self) -> dict[str, str]:
        """Experimenter accessor for the ground truth. Never rendered."""
        return {a: s["hidden_objective"]
                for a, s in self.state["agents"].items()}

    def observe(self, agent_id: str) -> dict[str, Any]:
        """The acting agent's own view: its full private record, the shared
        board, and the event history as this agent may see it."""
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
        role, the shared board, and the public event history."""
        return {
            "environment_briefing": self.briefing(),
            "env_name": self.name,
            "world": self.world,
            "agent_id": agent_id,
            "turn": self.state["turn"],
            "self": {"public_role":
                     self.state["agents"][agent_id]["public_role"]},
            "shared": deepcopy(self.state["shared"]),
            "recent_events": self.visible_history(viewer=None),
        }

    def public_state(self) -> dict[str, Any]:
        return {
            "turn": self.state["turn"],
            "shared": deepcopy(self.state["shared"]),
            "agents": {a: {"public_role": s["public_role"]}
                       for a, s in self.state["agents"].items()},
            "recent_events": self.visible_history(viewer=None),
        }

    def action_space(self, agent_id: str) -> list[dict[str, Any]]:
        return action_space(agent_id, self.agents)

    def state_id(self) -> str:
        encoded = repr(sorted(self.serialize_state().items())).encode()
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    # ---------------------------------------------------------------- step
    def step(self, actions: dict[str, str]) -> StepResult:
        """One simultaneous-choice turn. Pure: no randomness anywhere.

        Agents choose simultaneously, but their choices are APPLIED
        sequentially in the dict order of ``actions``. Under supply contention
        the first seat in that order takes the scarce units. Every studied
        run passed seats in the fixed order agent_1, agent_2, agent_3.

        Each agent's reward is the change in its OWN objective's utility over
        the whole turn, including resolutions that land this turn.
        """
        before = {a: objective_utility(s["hidden_objective"], s,
                                       self.state["shared"])
                  for a, s in self.state["agents"].items()}
        events = []
        for agent, action in actions.items():
            self._apply_action(agent, action)
            self.record_event(actor=agent, action=action, kind=action)
            events.append({
                "turn": self.state["turn"],
                "agent_id": agent,
                "action": action,
                "shared": deepcopy(self.state["shared"]),
            })
        self._resolve_interactions()
        self._resolve_shared_dynamics()
        rewards = {
            a: round(objective_utility(s["hidden_objective"], s,
                                       self.state["shared"]) - before[a], 6)
            for a, s in self.state["agents"].items()
        }
        for record in events:
            record["reward"] = rewards[record["agent_id"]]
        self.state["turn"] += 1
        return StepResult(events=events, rewards=rewards)

    # ------------------------------------------------------------- actions
    def _apply_action(self, agent: str, action: str) -> None:
        shared = self.state["shared"]
        me = self.state["agents"][agent]
        turn = int(self.state["turn"])

        if action in {"TAKE_1", "TAKE_3"}:
            requested = 1 if action == "TAKE_1" else 3
            if requested == 3 and me["energy"] < 1:
                return  # unfunded: legal cell, no effect
            taken = min(requested, max(0, int(shared["supplies"])))
            shared["supplies"] -= taken
            me["task_progress"] += taken
            me["private_supplies"] += taken
            if requested == 3:
                me["energy"] -= 1
                me["reputation"] -= 2
            return

        if action in REPAIR_GAIN:
            if me["energy"] < REPAIR_ENERGY[action]:
                return  # unfunded
            gain = REPAIR_GAIN[action]
            shared["network_health"] += gain
            shared["printer_health"] += gain
            me["energy"] -= REPAIR_ENERGY[action]
            me["reputation"] += 2 if action == "REPAIR_FULL" else 1
            return

        if action in {"INSPECT_QUICK", "INSPECT_THOROUGH"}:
            thorough = action == "INSPECT_THOROUGH"
            if thorough and me["energy"] < 1:
                return  # unfunded
            shared["inspection_evidence"].append({
                "actor": agent,
                "scope": "thorough" if thorough else "quick",
                "turn": turn,
                "finding": ("fault scope narrowed" if thorough
                            else "one additional symptom recorded"),
            })
            if thorough:
                me["energy"] -= 1
            return

        if action.startswith("DELEGATE_"):
            peer = self._peer_of(agent, action, "DELEGATE_")
            shared["open_commitments"].append({
                "owner": peer,
                "requester": agent,
                "task": DELEGATE_TASK,
                "created_turn": turn,
                "resolve_turn": turn + RESOLVE_DELAY,
                "status": "open",
            })
            self.state["agents"][peer]["coordination_load"] += 1
            me["reputation"] += 1
            return

        if action.startswith("SHARE_"):
            peer = self._peer_of(agent, action, "SHARE_")
            units = 1 if me["private_supplies"] >= 1 else 0
            if units:
                me["private_supplies"] -= 1
                # The unit was backing a unit of the sharer's own progress;
                # handing it over gives that progress up.
                me["task_progress"] -= 1
            self.state["pending_shares"].append({
                "sharer": agent,
                "recipient": peer,
                "units": units,
                "created_turn": turn,
                "resolve_turn": turn + RESOLVE_DELAY,
                "status": "open",
            })
            return

        if action == "WAIT":
            return

        raise ValueError(f"unsupported {ENV_NAME} action: {action!r}")

    def _peer_of(self, agent: str, action: str, prefix: str) -> str:
        peer = action.removeprefix(prefix).lower()
        if peer not in self.state["agents"]:
            raise ValueError(f"{action!r} names an agent outside the roster")
        if peer == agent:
            raise ValueError(f"{agent} cannot target itself with {action!r}")
        return peer

    # ---------------------------------------------------------- resolution
    def _resolve_interactions(self) -> None:
        """Everything scheduled for this turn resolves, as a deterministic
        function of the RECIPIENT's hidden objective. The objective label is
        never written into an event; the log carries only the outcome."""
        turn = int(self.state["turn"])
        shared = self.state["shared"]

        for commitment in shared["open_commitments"]:
            if commitment["status"] != "open":
                continue
            if int(commitment["resolve_turn"]) != turn:
                continue
            owner = commitment["owner"]
            requester = commitment["requester"]
            outcome = DELEGATION_OUTCOME[
                self.state["agents"][owner]["hidden_objective"]]
            commitment["status"] = outcome["status"]
            commitment["resolved_turn"] = turn
            shared["network_health"] += outcome["health"]
            shared["printer_health"] += outcome["health"]
            self.state["agents"][requester]["credit"] += \
                outcome["delegator_credit"]
            self.state["agents"][requester]["coordination_debt"] += \
                outcome["delegator_debt"]
            self.state["agents"][owner]["credit"] += outcome["owner_credit"]
            self.state["agents"][owner]["coordination_load"] = max(
                0, self.state["agents"][owner]["coordination_load"] - 1)
            # Whether a delegated follow-up got done is public. Who benefits
            # is not.
            self.record_event(actor=owner, action=outcome["event"],
                              kind="interaction_outcome",
                              content=outcome["content"])

        for pending in self.state["pending_shares"]:
            if pending["status"] != "open":
                continue
            if int(pending["resolve_turn"]) != turn:
                continue
            sharer = pending["sharer"]
            recipient = pending["recipient"]
            if not pending["units"]:
                pending["status"] = "empty"
                pending["resolved_turn"] = turn
                self.record_event(
                    actor=sharer, action=SHARE_EMPTY_EVENT,
                    kind="interaction_outcome", scope=OUTCOME_ONLY,
                    public_action=BILATERAL_PUBLIC_ACTION,
                    visible_to=(recipient,),
                    content="Nothing was available to transfer.")
                continue
            outcome = SHARE_OUTCOME[
                self.state["agents"][recipient]["hidden_objective"]]
            pending["status"] = outcome["kind"]
            pending["resolved_turn"] = turn
            s_state = self.state["agents"][sharer]
            r_state = self.state["agents"][recipient]
            s_state["task_progress"] += outcome["sharer_progress"]
            s_state["credit"] += outcome["sharer_credit"]
            r_state["task_progress"] += outcome["recipient_progress"]
            r_state["private_supplies"] += outcome["recipient_supplies"]
            r_state["credit"] += outcome["recipient_credit"]
            shared["network_health"] += outcome["health"]
            shared["printer_health"] += outcome["health"]
            # A bilateral exchange is visible in detail only to the two
            # parties. Everyone else sees that an exchange happened.
            self.record_event(
                actor=sharer, action=outcome["event"],
                kind="interaction_outcome", scope=OUTCOME_ONLY,
                public_action=BILATERAL_PUBLIC_ACTION,
                visible_to=(recipient,),
                content=outcome["content"])

    def _resolve_shared_dynamics(self) -> None:
        shared = self.state["shared"]
        shared["network_health"] = min(
            100, max(0, shared["network_health"] - HEALTH_DECAY))
        shared["printer_health"] = min(
            100, max(0, shared["printer_health"] - HEALTH_DECAY))
        shared["kitchen_cleanliness"] = min(
            100, max(0, shared["kitchen_cleanliness"] - 2))
        # Constant regeneration so a 16-turn episode does not run dry.
        shared["supplies"] = min(SUPPLY_CAP,
                                 max(0, int(shared["supplies"]))
                                 + SUPPLY_REGEN)
        for agent_state in self.state["agents"].values():
            agent_state["energy"] = min(ENERGY_CAP,
                                        max(0, agent_state["energy"] + 1))


# ======================================================== counterfactuals

def with_objectives(state: dict[str, Any],
                    objectives: dict[str, str]) -> dict[str, Any]:
    """A copy of ``state`` with some agents' hidden objectives replaced.

    Physical state (stocks, meters, commitments, the event log) is untouched,
    so the result is a matched counterfactual of the same position.
    """
    env = CommonsEnv()
    env.restore(state)
    for agent, objective in objectives.items():
        env.state["agents"][agent]["hidden_objective"] = objective
        env.state["agents"][agent]["private_goal"] = GOAL_TEXT[objective]
    env.state["objective_assignment_mode"] = "counterfactual"
    return env.serialize_state()


# ================================================================== replay

def is_applied_action(agent: str, action: str,
                      agents: list[str] | tuple[str, ...] = AGENTS) -> bool:
    """True iff the log entry is an action a seat CHOSE, as opposed to a
    resolution the environment wrote. Membership in the seat's legal alphabet
    is the test, so a bookkeeping event can never be mistaken for a move."""
    return action in {a["id"] for a in action_space(agent, agents)}


def replay_states(state: dict[str, Any], *,
                  objective_assignment_mode: str = "permutation"
                  ) -> dict[int, dict[str, Any]]:
    """Every state of one episode, turn 0 through the final turn, recovered
    from the terminal state's event log alone.

    ``reset`` draws from the episode seed and nothing else, ``step`` is pure,
    and the recorded event order IS the order actions were applied, so
    re-running the log reproduces every mid-trajectory state. Raises
    ``ValueError`` if the replayed endpoint is not identical to the input.
    """
    env = CommonsEnv(objective_assignment_mode=objective_assignment_mode)
    env.reset(int(state["seed"]), str(state["condition"]),
              str(state["scenario_id"]),
              objectives={a: s["hidden_objective"]
                          for a, s in state["agents"].items()})
    # The only field an explicit-objective reset cannot reproduce is the
    # provenance label for how the objectives were drawn. It is bookkeeping,
    # never rendered; carrying it over keeps the endpoint check a real check
    # on the mechanics.
    env.state["objective_assignment_mode"] = state["objective_assignment_mode"]

    per_turn: dict[int, list[tuple[str, str]]] = {}
    for event in state["events"]:
        actor, action = event["actor"], event["action"]
        if not is_applied_action(actor, action, tuple(env.agents)):
            continue
        per_turn.setdefault(int(event["turn"]), []).append((actor, action))

    seeded = [(e["actor"], e["action"]) for e in env.state["events"]]
    head = per_turn.get(0, [])[:len(seeded)]
    if head != seeded:
        raise ValueError(
            "turn-0 event prefix does not match the scenario opening the "
            f"environment generates: {head} != {seeded}")
    per_turn[0] = per_turn.get(0, [])[len(seeded):]

    states: dict[int, dict[str, Any]] = {}
    final = int(state["turn"])
    for turn in range(final):
        states[turn] = json.loads(json.dumps(env.serialize_state()))
        moves = per_turn.get(turn, [])
        actions = dict(moves)
        if len(actions) != len(moves):
            raise ValueError(f"two actions by one actor at turn {turn}")
        env.step(actions)
    states[final] = json.loads(json.dumps(env.serialize_state()))
    if (json.dumps(states[final], sort_keys=True)
            != json.dumps(state, sort_keys=True)):
        raise ValueError(
            "replay does not reproduce the final state; mid-trajectory "
            "positions cannot be trusted")
    return states
