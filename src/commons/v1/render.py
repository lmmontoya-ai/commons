"""Actor and reader prompts of Commons v1 (prompt protocol v8).

Every v1 prompt in the report was a JSON prompt: the system message is the
environment briefing, a blank line, and a task sentence with the menu ids
filled in; the user message is ``json.dumps`` (sorted keys, default
separators, ASCII escapes) of ``{"task", "observation", "return_schema",
"action_space"}``. The action-only return schema was used throughout.

Sources in the research harness: ``model_clients.py``
(``build_action_messages``, ``build_forecast_messages``, ``_user_content``
with ``prompt_format="json"``), ``arms.py`` (``prediction_observation``,
``shuffled_history_observation``), ``m4_forecast_suite.py``
(``observations_for_snapshot``) and ``scripts/commons_m44.py``
(``forecast_messages``, ``acting_messages``).

Actor view
----------
The acting agent sees its own record (including its private goal), the
board, and the events it may see, which include its own private action
acknowledgements. ``actor_messages`` gives the exact messages of every v1
acting call and every replayed target draw.

Reader views
------------
A reader forecasts a target's next action from public information only.
Every view has the same frame (perspective label, target id, the target's
public observation, the public state, an instruction sentence) and differs
only in the event history:

``board``     no event list (history horizon 0). The report calls this
              view "situation only". The board itself still carries some
              history: open commitments, open trade offers, and inspection
              evidence, each with the turn it was created.
``ordered``   the public event list in order, at most the last 24 events.
              The report calls this view "full record".
``shuffled``  the same events in a deterministic shuffled order.

Harness names are accepted as aliases (``board_only``, ``state_only``,
``situation``, ``full_history``, ``full_record``, ``shuffled_history``).

``prompt_format="markdown"`` renders the same observation with the markdown
user message of :mod:`commons.render.markdown`, as in the rendering factor of
the harness study M4.4. No report-v4 result used it.
"""
from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

from ..core.events import shuffled_events
from ..render.markdown import render_markdown_user_message
from ..render.messages import (
    ACTION_SYSTEM_PROMPT,
    FORECAST_SYSTEM_PROMPT,
    INSTRUCTION_VARIANTS,
    NEUTRAL_FORECAST_PERSPECTIVE,
    RETURN_SCHEMA_ACTION_ONLY,
    Message,
    system_content,
)
from .env import CommonsV1Env, action_space

PROMPT_FORMATS = ("json", "markdown")
VIEWS = ("board", "ordered", "shuffled")
VIEW_ALIASES = {
    "board": "board", "board_only": "board", "state_only": "board",
    "situation": "board",
    "ordered": "ordered", "full_history": "ordered", "full_record": "ordered",
    "shuffled": "shuffled", "shuffled_history": "shuffled",
    "shuffled_record": "shuffled",
}

ACT_TASK = "choose_action"
FORECAST_TASK = "forecast_target_action"


def canonical_view(view: str) -> str:
    try:
        return VIEW_ALIASES[view]
    except KeyError:
        raise ValueError(
            f"unknown reader view {view!r}; expected one of {VIEWS}") from None


def _restore(state: dict[str, Any]) -> CommonsV1Env:
    env = CommonsV1Env()
    env.restore(state)
    return env


def user_content(task: str, observation: dict[str, Any],
                 menu: list[dict[str, Any]], return_schema: Any,
                 prompt_format: str = "json") -> str:
    """The user message: JSON (the v1 rendering) or markdown."""
    if prompt_format == "markdown":
        return render_markdown_user_message(task, observation, menu,
                                            return_schema)
    if prompt_format != "json":
        raise ValueError(
            f"unknown prompt_format {prompt_format!r}; expected one of "
            f"{PROMPT_FORMATS}")
    payload: dict[str, Any] = {
        "task": task,
        "observation": observation,
        "return_schema": return_schema,
        "action_space": menu,
    }
    return json.dumps(payload, sort_keys=True)


def build_messages(template: str, task: str, observation: dict[str, Any],
                   menu: list[dict[str, Any]],
                   prompt_format: str = "json") -> list[Message]:
    system, observation = system_content(template, observation, menu)
    return [
        {"role": "system", "content": system},
        {"role": "user",
         "content": user_content(task, observation, menu,
                                 RETURN_SCHEMA_ACTION_ONLY, prompt_format)},
    ]


# ================================================================ actor view

def actor_observation(state: dict[str, Any], mover: str) -> dict[str, Any]:
    """The mover's own observation at ``state``."""
    return _restore(state).observe(mover)


def actor_messages(state: dict[str, Any], mover: str, *,
                   prompt_format: str = "json") -> list[Message]:
    """The exact messages a v1 actor (or a replayed target draw) received."""
    return build_messages(ACTION_SYSTEM_PROMPT, ACT_TASK,
                          actor_observation(state, mover), action_space(mover),
                          prompt_format)


# =============================================================== reader view

def default_shuffle_seed_context(state: dict[str, Any], target: str
                                 ) -> dict[str, Any]:
    """A deterministic shuffle seed for ad hoc use.

    The studied prompts used other contexts; ``commons.v1.decisions`` builds
    them for every frozen decision.
    """
    return {"seed": state.get("seed"), "scenario_id": state.get("scenario_id"),
            "turn": state.get("turn"), "target": target}


def _prediction_observation(env: CommonsV1Env, target: str,
                            instruction: str) -> dict[str, Any]:
    observation = {
        "perspective": NEUTRAL_FORECAST_PERSPECTIVE,
        "target_id": target,
        "target_observation": env.public_observation(target),
        "public_state": env.public_state(),
        "instruction": instruction,
    }
    # The briefing is carried exactly once, at the top level.
    observation["target_observation"].pop("environment_briefing", None)
    return {"environment_briefing": env.briefing(), **observation}


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


def reader_observation(state: dict[str, Any], target: str,
                       view: str = "ordered", *,
                       instruction_variant: str = "V1",
                       shuffle_seed_context: dict[str, Any] | None = None
                       ) -> dict[str, Any]:
    """A reader's observation of ``target`` at ``state`` under ``view``.

    ``instruction_variant`` picks one of the three instruction paraphrases
    (``commons.render.INSTRUCTION_VARIANTS``); V1 is the neutral sentence.
    """
    view = canonical_view(view)
    if instruction_variant not in INSTRUCTION_VARIANTS:
        raise ValueError(
            f"unknown instruction variant {instruction_variant!r}; expected "
            f"one of {sorted(INSTRUCTION_VARIANTS)}")
    instruction = INSTRUCTION_VARIANTS[instruction_variant]
    env = _restore(state)
    if view == "board":
        env.history_horizon = 0
        return _prediction_observation(env, target, instruction)
    observation = _prediction_observation(env, target, instruction)
    if view == "ordered":
        return observation
    seed_context = (shuffle_seed_context if shuffle_seed_context is not None
                    else default_shuffle_seed_context(state, target))
    shuffled = copy.deepcopy(observation)
    _shuffle_history_fields(shuffled, seed_context)
    return shuffled


def reader_messages(state: dict[str, Any], target: str,
                    view: str = "ordered", *,
                    instruction_variant: str = "V1",
                    shuffle_seed_context: dict[str, Any] | None = None,
                    prompt_format: str = "json") -> list[Message]:
    """The exact messages a v1 reader received for one forecast."""
    observation = reader_observation(
        state, target, view, instruction_variant=instruction_variant,
        shuffle_seed_context=shuffle_seed_context)
    return build_messages(FORECAST_SYSTEM_PROMPT, FORECAST_TASK, observation,
                          action_space(target), prompt_format)


# =================================================================== digests

def observation_sha256(observation: dict[str, Any]) -> str:
    """SHA-256 of an observation as the M4.3 manifests computed it: JSON with
    sorted keys, compact separators, ASCII escapes."""
    payload = json.dumps(observation, sort_keys=True, separators=(",", ":"),
                         default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def messages_sha256(messages: list[Message]) -> str:
    """SHA-256 of a message list as the forecast panels computed it
    (``prompt_sha256``): JSON with sorted keys, compact separators, UTF-8
    without escapes."""
    payload = json.dumps(messages, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
