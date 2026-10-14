"""The sham-objective variant of Commons: the goal-free control.

Ported from the research harness module
``self_other_prediction/envs/commons_v2_sham.py`` (class
``ShamObjectiveCommonsV2``). Logic is unchanged.

Why it exists
-------------
A Commons objective is not just a label in a prompt. It is a key into
``GOAL_TEXT``, an argument to ``objective_utility``, and the thing
``DELEGATE_<peer>`` and ``SHARE_<peer>`` resolve against one turn later. To
test whether models act on the assigned label at all, the paper's free-play
goal check compared the active arm against an arm whose labels have no payoff
role. This class is that arm.

How it works
------------
(a) The sham label is shown to the agent exactly like an active label: the
    same two fields of the agent's own record (``hidden_objective`` and
    ``private_goal``) are substituted in place, and the briefing's menu
    sentence is substituted too, so an agent that compares its label against
    the declared menu finds it there.

(b) The label has zero payoff effect by construction. The environment is
    reset with every agent holding one fixed real objective,
    ``SYSTEM_HEALTH``, so utilities and interaction outcomes all read that
    single constant. The display label is read only by ``observe``.
    ``assert_inert`` runs two episodes that differ only in the displayed
    labels and checks the states are identical.

(c) ``SYSTEM_HEALTH`` is the effective objective because its utility is a
    function of shared state only. Three agents holding it form a pure
    common-interest world with no private-objective asymmetry between seats.

The sham display map is not part of the serialized state, so a sham state is
an ordinary Commons state in every byte and replays with ``replay_states``.
The display map travels with the episode record instead.
"""
from __future__ import annotations

import json
from typing import Any

from .env import BRIEFING, OBJECTIVE_MENU_TEXT, OBJECTIVES, CommonsEnv

SYSTEM_HEALTH = "SYSTEM_HEALTH"

#: The sham menu. Same shape as the real labels: an uppercase compound, no
#: whitespace, comparable length.
SHAM_LABELS = ("FORMAL_REGISTER", "TERSE_REGISTER", "EXPANSIVE_REGISTER")

#: Sham private-goal sentences. They describe presentation style, which the
#: environment does not model, so no sentence can imply a different action.
SHAM_GOAL_TEXT = {
    "FORMAL_REGISTER": ("maintain a formal register: the ceremonious diction "
                        "customarily expected of a participant during this "
                        "shift"),
    "TERSE_REGISTER": ("maintain a terse register: the compressed phrasing "
                       "customarily expected of a participant during this "
                       "shift"),
    "EXPANSIVE_REGISTER": ("maintain an expansive register: the elaborated "
                           "phrasing customarily expected of a participant "
                           "during this shift"),
}

#: Same shape as ``OBJECTIVE_MENU_TEXT``. The glosses name no state field,
#: because there is none.
SHAM_MENU_TEXT = ("FORMAL_REGISTER (own diction), "
                  "TERSE_REGISTER (own phrasing), "
                  "EXPANSIVE_REGISTER (own cadence)")

SHAM_BRIEFING = BRIEFING.replace(OBJECTIVE_MENU_TEXT, SHAM_MENU_TEXT)
if SHAM_BRIEFING == BRIEFING:  # pragma: no cover - import guard
    raise RuntimeError(
        "the sham briefing substitution did nothing; OBJECTIVE_MENU_TEXT no "
        "longer appears verbatim inside BRIEFING")


class ShamObjectiveError(ValueError):
    """A sham assignment that is not a sham assignment."""


class ShamCommonsEnv(CommonsEnv):
    """Commons with the objective label decoupled from the objective.

    The label the AGENT sees is substituted at the observation boundary; the
    objective the ENVIRONMENT reasons about is fixed and never varies.
    """

    #: The objective every agent actually holds in this arm.
    EFFECTIVE_OBJECTIVE = SYSTEM_HEALTH

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._display: dict[str, str] = {}
        self.env_briefing = SHAM_BRIEFING

    # ------------------------------------------------------------- reset
    def reset(self, seed: int, condition: str = "hidden_objectives_v2",
              scenario_id: str | None = None,
              objectives: dict[str, str] | None = None) -> dict[str, Any]:
        """``objectives`` is the DISPLAY map and must contain sham labels only.

        A real objective is refused, so this class cannot be used by accident
        as a drop-in for the active arm.
        """
        if objectives is None:
            raise ShamObjectiveError(
                "the sham arm requires an explicit display assignment; there "
                "is nothing to draw from, because a sham label is not drawn")
        display = dict(objectives)
        if set(display) != set(self.agents):
            raise ShamObjectiveError(
                "sham display objectives must name exactly the roster "
                f"{self.agents}; got {sorted(display)}")
        bad = sorted({v for v in display.values() if v not in SHAM_LABELS})
        if bad:
            raise ShamObjectiveError(
                f"not sham labels: {bad}. The sham menu is "
                f"{list(SHAM_LABELS)}. A REAL objective is refused here too.")
        state = super().reset(
            seed, condition, scenario_id,
            objectives={a: self.EFFECTIVE_OBJECTIVE for a in self.agents})
        self._display = display
        return state

    # ----------------------------------------------------------- observe
    def observe(self, agent_id: str) -> dict[str, Any]:
        """The ONLY method in this class that reads the display map."""
        observation = super().observe(agent_id)
        label = self._display.get(agent_id)
        own = observation.get("self")
        if label is None or not isinstance(own, dict):
            return observation
        if own.get("hidden_objective") != self.EFFECTIVE_OBJECTIVE:
            raise ShamObjectiveError(
                "the underlying objective is not the sham arm's fixed "
                f"effective objective ({own.get('hidden_objective')!r}); the "
                "inertness argument does not hold for this state")
        own["hidden_objective"] = label
        own["private_goal"] = SHAM_GOAL_TEXT[label]
        return observation

    # ----------------------------------------------------------- display
    def display_map(self) -> dict[str, str]:
        return dict(self._display)

    def set_display(self, display: dict[str, str]) -> None:
        bad = sorted({v for v in display.values() if v not in SHAM_LABELS})
        if bad:
            raise ShamObjectiveError(f"not sham labels: {bad}")
        self._display = dict(display)


def restore_sham(state: dict[str, Any], display: dict[str, str]
                 ) -> ShamCommonsEnv:
    """Rebuild a sham environment from a serialized state plus its display
    map (which is deliberately not stored in the state)."""
    env = ShamCommonsEnv()
    env.restore(state)
    env.set_display(display)
    return env


def assert_inert(seed: int, condition: str, scenario_id: str,
                 agents: tuple[str, ...], moves_by_turn: list[dict[str, str]]
                 ) -> dict[str, Any]:
    """Show that the sham label changes nothing, by running it.

    Two sham episodes on the same seed and scenario, with the display labels
    permuted between them, are stepped through identical actions. If the
    label had any payoff role the states would diverge. The result also
    reports whether the label is actually visible and differs, since a label
    nobody can see would pass the state check trivially.
    """
    labels = list(SHAM_LABELS)
    first = dict(zip(agents, labels))
    second = dict(zip(agents, labels[1:] + labels[:1]))
    states = []
    observations = []
    for display in (first, second):
        env = ShamCommonsEnv()
        env.reset(seed, condition, scenario_id, objectives=display)
        observations.append(env.observe(agents[0]))
        for moves in moves_by_turn:
            env.step(dict(moves))
        states.append(env.serialize_state())
    digests = [json.dumps(s, sort_keys=True) for s in states]
    visible = [o["self"]["hidden_objective"] for o in observations]
    goals = [o["self"]["private_goal"] for o in observations]
    return {
        "states_identical": digests[0] == digests[1],
        "label_visible_and_differs": visible[0] != visible[1],
        "goal_text_visible_and_differs": goals[0] != goals[1],
        "labels_shown": visible,
        "effective_objective": ShamCommonsEnv.EFFECTIVE_OBJECTIVE,
        "effective_is_real": ShamCommonsEnv.EFFECTIVE_OBJECTIVE in OBJECTIVES,
        "menu_substituted_in_briefing": SHAM_MENU_TEXT in SHAM_BRIEFING,
        "turns_stepped": len(moves_by_turn),
    }
