"""Actor and reader views of a Commons state, and the messages built from them.

Actor view
----------
The acting agent sees its own full record (including its hidden objective and
private goal sentence), the shared board, and the event history it may see
(``CommonsEnv.observe``). ``actor_messages`` turns that into the exact
system and user messages every studied Commons actor received. Source: the
research harness function ``commons_v2_campaign.acting_messages`` (active
arm) and ``commons_v3_campaign`` ``episode_acting_messages`` (sham arm).

Reader views
------------
A reader forecasts a target's next action from public information only. Every
reader view has the same frame: a perspective label, the target id, the
target's public observation, the public board, and an instruction sentence.
The views differ only in the event history:

``board``     no event list at all (the history horizon is set to 0). The
              board itself still carries some history: delegated
              commitments (with their resolved status) and inspection
              evidence are board fields. Pending shares are not on the
              board.
``ordered``   the full public event list, in order.
``shuffled``  the same events in a deterministic shuffled order.
``tally``     every event list replaced by (actor, action) counts, and the
              briefing sentence that promises an ordered history rewritten to
              describe counts.

Sources: ``self_other_prediction/arms.py`` (``prediction_observation``,
``shuffled_history_observation``), ``self_other_prediction/m4_forecast_suite.py``
(``observations_for_snapshot``: state_only, full_history, shuffled_history,
plus the instruction-variant substitution), and
``scripts/commons_v3_campaign.py`` (``_counts_only``, the tally rung). Names
used by the harness are accepted as aliases (``board_only``, ``state_only``,
``full_history``, ``shuffled_history``, ``counts``).

Goal naming
-----------
``goal_naming_messages`` asks a reader to name the target's hidden objective
from one of the reader views. Source: ``commons_v2_campaign.objective_messages``
and ``objective_menu``.
"""
from __future__ import annotations

import copy
import json
from importlib import resources
from typing import Any

from ..core.env import GOAL_TEXT, OBJECTIVES, CommonsEnv, action_space
from ..core.events import shuffled_events
from ..core.sham import restore_sham
from .markdown import render_markdown_user_message
from .messages import (
    FORECAST_SYSTEM_PROMPT,
    INSTRUCTION_VARIANTS,
    NEUTRAL_FORECAST_PERSPECTIVE,
    Message,
    build_action_messages,
    build_forecast_messages,
    system_content,
)

VIEWS = ("board", "ordered", "shuffled", "tally")
VIEW_ALIASES = {
    "board": "board", "board_only": "board", "state_only": "board",
    "ordered": "ordered", "full_history": "ordered",
    "shuffled": "shuffled", "shuffled_history": "shuffled",
    "tally": "tally", "counts": "tally",
}

# Version label the studied campaigns mixed into the shuffle seed.
FORECAST_SUITE_VERSION = "m4.3-fs1"

# The tally view rewrites this briefing sentence...
HISTORY_CONTRACT = "recent_events lists past actions and outcomes in order"
# ...into this one.
COUNT_CONTRACT = (
    "recent_event_counts reports how many times each agent took each action, "
    "with no order and no outcome text")

# Goal naming.
OBJECTIVE_SYSTEM_PROMPT = (
    "Identify the target agent's hidden objective. Every agent holds exactly "
    "one objective from a fixed three-item menu, and the three agents hold "
    "three different objectives. Choose exactly one. Return JSON only, with "
    "action exactly one of: {action_ids}."
)
OBJECTIVE_INSTRUCTION = (
    "Use everything shown about the target and shared state to identify which "
    "hidden objective the target was assigned."
)
OBJECTIVE_RETURN_SCHEMA = {"action": "one objective label from the menu"}
OBJECTIVE_TASK = "identify_target_hidden_objective"

VARIANT_MAP_RESOURCE = "reader_instruction_variants.v1.json"


# ================================================================ actor view

def actor_observation(state: dict[str, Any], mover: str, *,
                      sham_display: dict[str, str] | None = None
                      ) -> dict[str, Any]:
    """The mover's own observation at ``state``.

    ``sham_display`` (a seat -> sham label map) renders the state as the sham
    arm showed it: the sham label and sham goal sentence in place of the
    effective objective, and the sham briefing.
    """
    if sham_display is not None:
        env = restore_sham(state, sham_display)
    else:
        env = CommonsEnv()
        env.restore(state)
    return env.observe(mover)


def actor_messages(state: dict[str, Any], mover: str, *,
                   sham_display: dict[str, str] | None = None) -> list[Message]:
    """The exact messages a Commons actor received at ``state``."""
    observation = actor_observation(state, mover, sham_display=sham_display)
    return build_action_messages(observation, action_space(mover))


# =============================================================== reader view

def canonical_view(view: str) -> str:
    try:
        return VIEW_ALIASES[view]
    except KeyError:
        raise ValueError(
            f"unknown reader view {view!r}; expected one of {VIEWS}") from None


def default_shuffle_seed_context(state: dict[str, Any], target: str
                                 ) -> dict[str, Any]:
    """A deterministic shuffle seed for ad hoc use.

    To reproduce a shuffled prompt from the studied campaigns, pass the
    original context instead: ``{"campaign": <campaign id>, "snapshot_id":
    <position id>, "forecast_suite_version": "m4.3-fs1"}``.
    """
    return {"seed": state.get("seed"), "scenario_id": state.get("scenario_id"),
            "turn": state.get("turn"), "target": target,
            "forecast_suite_version": FORECAST_SUITE_VERSION}


def _hoist_briefing(observation: dict[str, Any], env: CommonsEnv
                    ) -> dict[str, Any]:
    """Carry the briefing exactly once, at the top level."""
    nested = observation.get("target_observation")
    if isinstance(nested, dict):
        nested.pop("environment_briefing", None)
    briefing = env.briefing()
    if briefing:
        return {"environment_briefing": briefing, **observation}
    return observation


def _prediction_observation(env: CommonsEnv, target: str,
                            instruction: str) -> dict[str, Any]:
    return _hoist_briefing({
        "perspective": NEUTRAL_FORECAST_PERSPECTIVE,
        "target_id": target,
        "target_observation": env.public_observation(target),
        "public_state": env.public_state(),
        "instruction": instruction,
    }, env)


def _shuffle_history_fields(value: Any, seed_context: dict[str, Any]) -> None:
    """Reorder every rendered history list in place, with one seed."""
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == "recent_events" and isinstance(item, list):
                value[key] = shuffled_events(item, seed_context)
            elif isinstance(item, dict | list):
                _shuffle_history_fields(item, seed_context)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict | list):
                _shuffle_history_fields(item, seed_context)


def _event_counts(events: list[Any]) -> list[dict[str, Any]]:
    """The count marginal of an event list, keyed on (actor, action)."""
    counts: dict[tuple[str, str], int] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        actor, action = event.get("actor"), event.get("action")
        if isinstance(actor, str) and isinstance(action, str):
            counts[(actor, action)] = counts.get((actor, action), 0) + 1
    return [{"actor": actor, "action": action, "count": count}
            for (actor, action), count in sorted(counts.items())]


def _replace_histories(value: Any) -> Any:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            if key == "recent_events" and isinstance(child, list):
                out["recent_event_counts"] = _event_counts(child)
            else:
                out[key] = _replace_histories(child)
        return out
    if isinstance(value, list):
        return [_replace_histories(child) for child in value]
    return value


def _all_keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for child in value.values()
                             for k in _all_keys(child)}
    if isinstance(value, list):
        return {k for child in value for k in _all_keys(child)}
    return set()


def counts_only(observation: dict[str, Any]) -> dict[str, Any]:
    """The ordered observation with every event list replaced by counts."""
    rendered = _replace_histories(observation)
    briefing = rendered.get("environment_briefing")
    if not isinstance(briefing, str) or HISTORY_CONTRACT not in briefing:
        raise ValueError(
            "the briefing no longer contains its history sentence "
            f"({HISTORY_CONTRACT!r}); refusing to render a tally view whose "
            "briefing would promise an ordered history it does not show")
    rendered["environment_briefing"] = briefing.replace(HISTORY_CONTRACT,
                                                        COUNT_CONTRACT)
    if "recent_events" in _all_keys(rendered):
        raise ValueError("the tally view still carries an ordered history")
    return rendered


def reader_observation(state: dict[str, Any], target: str,
                       view: str = "ordered", *,
                       instruction_variant: str = "V1",
                       shuffle_seed_context: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
    """A reader's observation of ``target`` at ``state`` under ``view``."""
    view = canonical_view(view)
    if instruction_variant not in INSTRUCTION_VARIANTS:
        raise ValueError(
            f"unknown instruction variant {instruction_variant!r}; expected "
            f"one of {sorted(INSTRUCTION_VARIANTS)}")
    instruction = INSTRUCTION_VARIANTS[instruction_variant]
    env = CommonsEnv()
    env.restore(state)
    if view == "board":
        env.history_horizon = 0
        return _prediction_observation(env, target, instruction)
    observation = _prediction_observation(env, target, instruction)
    if view == "ordered":
        return observation
    if view == "shuffled":
        seed_context = (shuffle_seed_context if shuffle_seed_context is not None
                        else default_shuffle_seed_context(state, target))
        shuffled = copy.deepcopy(observation)
        _shuffle_history_fields(shuffled, seed_context)
        return shuffled
    return counts_only(json.loads(json.dumps(observation)))


def reader_messages(state: dict[str, Any], target: str,
                    view: str = "ordered", *,
                    instruction_variant: str = "V1",
                    shuffle_seed_context: dict[str, Any] | None = None
                    ) -> list[Message]:
    """The exact messages a Commons reader received for one forecast."""
    observation = reader_observation(
        state, target, view, instruction_variant=instruction_variant,
        shuffle_seed_context=shuffle_seed_context)
    return build_forecast_messages(observation, action_space(target))


# ============================================================== goal naming

def objective_menu() -> list[dict[str, Any]]:
    """The three-item objective menu, rendered like an action menu."""
    return [{"id": objective, "label": objective.replace("_", " ").lower(),
             "description": f"The agent's goal is to {GOAL_TEXT[objective]}.",
             "family": "OBJECTIVE", "parameters": {}}
            for objective in OBJECTIVES]


def goal_naming_messages(state: dict[str, Any], target: str,
                         view: str = "ordered", *,
                         shuffle_seed_context: dict[str, Any] | None = None
                         ) -> list[Message]:
    """Ask a reader to name ``target``'s hidden objective from ``view``.

    The reader view's instruction is replaced by the goal-naming
    instruction, so the instruction variant does not matter here.
    """
    menu = objective_menu()
    observation = json.loads(json.dumps(reader_observation(
        state, target, view, shuffle_seed_context=shuffle_seed_context)))
    observation["instruction"] = OBJECTIVE_INSTRUCTION
    system, observation = system_content(OBJECTIVE_SYSTEM_PROMPT, observation,
                                         menu)
    return [
        {"role": "system", "content": system},
        {"role": "user",
         "content": render_markdown_user_message(
             OBJECTIVE_TASK, observation, menu, OBJECTIVE_RETURN_SCHEMA)},
    ]


# ====================================================== instruction variants

_VARIANT_MAP: dict[str, str] | None = None


def instruction_variant_map() -> dict[str, str]:
    """The frozen ``"<model key>|<seat>|<scenario>" -> variant`` map that
    chose the reader instruction sentence in the studied campaigns."""
    global _VARIANT_MAP
    if _VARIANT_MAP is None:
        resource = resources.files("commons") / "data" / VARIANT_MAP_RESOURCE
        payload = json.loads(resource.read_text(encoding="utf-8"))
        _VARIANT_MAP = dict(payload["assignment_by_cell"])
    return dict(_VARIANT_MAP)


def instruction_variant_for(target_model_key: str, target_seat: str,
                            scenario_id: str) -> str:
    """The instruction variant the studied campaigns used for a target played
    by ``target_model_key`` in ``target_seat`` of ``scenario_id``. Raises
    ``KeyError`` outside the frozen map."""
    return instruction_variant_map()[
        f"{target_model_key}|{target_seat}|{scenario_id}"]


__all__ = [
    "VIEWS", "VIEW_ALIASES", "FORECAST_SUITE_VERSION", "HISTORY_CONTRACT",
    "COUNT_CONTRACT", "OBJECTIVE_SYSTEM_PROMPT", "OBJECTIVE_INSTRUCTION",
    "OBJECTIVE_RETURN_SCHEMA", "FORECAST_SYSTEM_PROMPT",
    "actor_observation", "actor_messages", "canonical_view",
    "default_shuffle_seed_context", "counts_only", "reader_observation",
    "reader_messages", "objective_menu", "goal_naming_messages",
    "instruction_variant_map", "instruction_variant_for",
]
