"""Message builders: observation -> the exact chat messages sent to a model.

Ported from the research harness modules
``self_other_prediction/model_clients.py`` (``_system_content``,
``build_action_messages``, ``build_forecast_messages``, the action-only
return schema) and ``self_other_prediction/prompt_templates.py`` (the system
task sentences and the reader instruction texts). Only the markdown user
message format is supported, because every studied Commons prompt used it.

Composition (identical for actors and readers):

* system message = environment briefing, a blank line, then the task
  sentence with the menu's action ids filled in;
* user message = the markdown rendering of the observation (with the
  briefing removed), the action menu, and the return schema.
"""
from __future__ import annotations

from typing import Any

from .markdown import render_markdown_user_message

Message = dict[str, str]

# System-layer task framing. ``{action_ids}`` is filled with the menu ids.
ACTION_SYSTEM_PROMPT = (
    "Choose one valid action id for the current agent. Weigh the agent's "
    "private goal, public role, recent actions, and shared-state pressures. "
    "Return JSON only, with action exactly one of: "
    "{action_ids}."
)

FORECAST_SYSTEM_PROMPT = (
    "Forecast the target agent's next action. Choose exactly one valid "
    "action id. Return JSON only, with action exactly one of: {action_ids}."
)

# The action-only return schema used by every studied Commons actor and
# reader call.
RETURN_SCHEMA_ACTION_ONLY = {"action": "one valid action id from action_space"}

# Reader framing. The perspective label is shared by every reader view.
NEUTRAL_FORECAST_PERSPECTIVE = "forecasting_target"

# The three paraphrases of the reader instruction. In the studied campaigns
# one of them was chosen per (target model, target seat, scenario) by a
# frozen Latin-square map; see src/commons/data/reader_instruction_variants.v1.json and
# ``commons.render.views.instruction_variant_for``.
INSTRUCTION_VARIANTS = {
    "V1": (
        "Use everything shown about the target and shared state to forecast "
        "the target's next action."
    ),
    "V2": (
        "Based on the information provided about the target and the shared "
        "situation, forecast the target's next action."
    ),
    "V3": (
        "Forecast which action the target will take next, given what is shown "
        "about the target and the shared state."
    ),
}


def system_content(template: str, observation: dict[str, Any],
                   action_space: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    """Compose the system message: environment rules first, then the task.

    The briefing is popped from a copy of the observation so it renders
    exactly once, in the system layer.
    """
    observation = dict(observation)
    briefing = observation.pop("environment_briefing", "")
    task = template.format(action_ids=", ".join(action["id"] for action in action_space))
    content = f"{briefing}\n\n{task}" if briefing else task
    return content, observation


def build_messages(template: str, task: str, observation: dict[str, Any],
                   action_space: list[dict[str, Any]],
                   return_schema: Any = None) -> list[Message]:
    """Generic builder: system template + markdown user message."""
    schema = RETURN_SCHEMA_ACTION_ONLY if return_schema is None else return_schema
    system, observation = system_content(template, observation, action_space)
    return [
        {"role": "system", "content": system},
        {"role": "user",
         "content": render_markdown_user_message(task, observation,
                                                 action_space, schema)},
    ]


def build_action_messages(observation: dict[str, Any],
                          action_space: list[dict[str, Any]]) -> list[Message]:
    """The exact messages sent for every actor call (and every replayed
    target draw at a frozen decision)."""
    return build_messages(ACTION_SYSTEM_PROMPT, "choose_action",
                          observation, action_space)


def build_forecast_messages(observation: dict[str, Any],
                            action_space: list[dict[str, Any]]) -> list[Message]:
    """The exact messages sent for every reader (forecast) call."""
    return build_messages(FORECAST_SYSTEM_PROMPT, "forecast_target_action",
                          observation, action_space)
