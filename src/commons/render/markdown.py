"""Markdown rendering of the user message ("prompt format v2").

Ported from the research harness module
``self_other_prediction/prompt_render_md.py``. The rendering logic is
unchanged, so the bytes of every rendered user message are unchanged. The
only edits are documentation and the removal of the unused debrief task line.

Invariants:

* The environment briefing never appears in the user message. The message
  builders in :mod:`commons.render.messages` pop it into the system message
  before anything here runs.
* The rendered message ends with a ``## Response`` section that restates the
  return schema as JSON inside a fenced block.
* Every field of the observation is shown, including keys with no bespoke
  section (rendered by the generic recursive renderer below).
* Determinism: dict keys are sorted, list order is preserved (events and the
  action menu carry meaning in order), and nothing time- or random-dependent
  is rendered. Two renders of the same input are byte-equal.

Prompt-visible separator: event lines and action lines use an em dash
(U+2014) between the head and the details. It is written below as
``chr(0x2014)`` so the source stays ASCII; the rendered bytes contain the
character itself, exactly as in the studied prompts.
"""
from __future__ import annotations

import json
from typing import Any

EM_DASH = chr(0x2014)  # U+2014, the separator used in the studied prompts

# One plain sentence per task.
TASK_LINES: dict[str, str] = {
    "choose_action": "Choose your next action.",
    "forecast_target_action": "Predict the target agent's next action.",
}

RESPONSE_SENTENCE = "Reply with a single JSON object matching this schema."

# Observation keys with a bespoke section. Everything else is either a scalar
# (collected into "## Setup") or handed to the generic recursive renderer.
_BESPOKE_KEYS = frozenset(
    {
        "environment_briefing",
        "instruction",
        "env_name",
        "world",
        "turn",
        "agent_id",
        "self",
        "shared",
        "recent_events",
        "prediction_context",
        "target_observation",
        "predictor_observation",
        "public_state",
        "serialized_state",
    }
)

_WORLD_KEYS = ("env_name", "world", "turn")
_EVENT_HEAD_KEYS = ("turn", "actor", "action", "content")
_ACTION_HEAD_KEYS = ("id", "label", "description")


# --- scalars ---------------------------------------------------------------


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, str | int | float | bool)


def _scalar_text(value: Any) -> str:
    if value is None:
        return "none"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return value
    return str(value)


def _label(key: Any) -> str:
    """Field label: underscores become spaces, EXCEPT for identifier-like keys.

    Some dicts are keyed by in-world identifiers (``agent_1``, ``TAKE_1``,
    ``DELEGATE_AGENT_2``). Those render exactly as they appear everywhere
    else in the prompt. A key containing a digit or any uppercase letter is
    treated as an identifier and left verbatim.
    """
    text = str(key)
    if any(character.isdigit() for character in text) or text != text.lower():
        return text
    return text.replace("_", " ")


def _title(key: Any) -> str:
    text = str(key)
    label = _label(text)
    if label != text or (text.islower() and text.isalpha()):
        return label[:1].upper() + label[1:] if label else label
    return label


def _inline(value: Any) -> str:
    """One-line rendering used inside list items and ``key: value`` lines."""
    if _is_scalar(value):
        return _scalar_text(value).replace("\n", " ")
    if isinstance(value, list):
        if not value:
            return "(none)"
        return ", ".join(_inline(item) for item in value)
    if isinstance(value, dict):
        if not value:
            return "(none)"
        return "(" + _pairs(value) + ")"
    return _scalar_text(value)


def _pairs(mapping: dict[Any, Any]) -> str:
    return ", ".join(
        f"{_label(key)}={_inline(mapping[key])}" for key in sorted(mapping, key=str)
    )


def _kv_lines(key: Any, value: Any) -> list[str]:
    label = _label(key)
    if isinstance(value, str) and "\n" in value:
        return [f"{label}:", *[f"  {line}" for line in value.split("\n")]]
    return [f"{label}: {_inline(value)}"]


# --- generic recursive renderer --------------------------------------------


def _is_complex(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(value)
    if isinstance(value, list):
        return any(isinstance(item, dict | list) for item in value)
    return False


def _emit_heading(lines: list[str], level: int, title: str) -> None:
    if lines:
        lines.append("")
    lines.append(f"{'#' * level} {title}")


def _emit_section(lines: list[str], title: str, value: Any, level: int) -> None:
    _emit_heading(lines, level, title)
    _emit_value(lines, value, level + 1)


def _emit_value(lines: list[str], value: Any, level: int) -> None:
    if isinstance(value, dict):
        _emit_mapping(lines, value, level)
    elif isinstance(value, list):
        _emit_list(lines, value)
    elif isinstance(value, str) and "\n" in value:
        lines.extend(value.split("\n"))
    else:
        lines.append(_scalar_text(value))


def _emit_mapping(lines: list[str], mapping: dict[Any, Any], level: int) -> None:
    if not mapping:
        lines.append("(none)")
        return
    simple = [key for key in sorted(mapping, key=str) if not _is_complex(mapping[key])]
    nested = [key for key in sorted(mapping, key=str) if _is_complex(mapping[key])]
    for key in simple:
        lines.extend(_kv_lines(key, mapping[key]))
    for key in nested:
        # Event feeds render identically wherever they appear (own view,
        # target view, or nested inside a state dict), so a shuffled history
        # differs from the ordered one in event ORDER only.
        if key == "recent_events" and isinstance(mapping[key], list):
            _emit_heading(lines, level, "Recent events")
            _emit_events(lines, mapping[key])
            continue
        _emit_section(lines, _title(key), mapping[key], level)


def _emit_list(lines: list[str], value: list[Any]) -> None:
    if not value:
        lines.append("(none)")
        return
    if all(_is_scalar(item) for item in value):
        lines.append(", ".join(_scalar_text(item).replace("\n", " ") for item in value))
        return
    for item in value:
        if isinstance(item, dict):
            lines.append(f"- {_pairs(item)}" if item else "- (none)")
        else:
            lines.append(f"- {_inline(item)}")


# --- bespoke sections -------------------------------------------------------


def _event_line(event: Any) -> str:
    if not isinstance(event, dict):
        return f"- {_inline(event)}"
    head_parts: list[str] = []
    if event.get("turn") is not None:
        head_parts.append(f"turn {_scalar_text(event['turn'])}:")
    actor = event.get("actor")
    if actor is not None:
        head_parts.append(_scalar_text(actor))
    # ``kind`` duplicates ``action`` in most events; it is rendered explicitly
    # below only when the two actually differ.
    action = event.get("action")
    if action is None:
        action = event.get("kind")
    if action is not None:
        head_parts.append(_scalar_text(action))
    rest: list[str] = []
    content = event.get("content")
    if content is not None:
        rest.append(_scalar_text(content).replace("\n", " "))
    for key in sorted(event, key=str):
        if key in _EVENT_HEAD_KEYS:
            continue
        if key == "kind" and event.get("kind") == event.get("action"):
            continue
        rest.append(f"{_label(key)}={_inline(event[key])}")
    if not head_parts:
        return f"- {_pairs(event)}"
    line = "- " + " ".join(head_parts)
    if rest:
        line += f" {EM_DASH} " + ", ".join(rest)
    return line


def _emit_events(lines: list[str], events: Any) -> None:
    if isinstance(events, list) and events:
        lines.extend(_event_line(event) for event in events)
    elif isinstance(events, list):
        lines.append("(none)")
    else:
        _emit_value(lines, events, 0)


def _action_line(action: Any) -> str:
    if not isinstance(action, dict):
        return f"- {_inline(action)}"
    line = f"- `{_scalar_text(action.get('id'))}`"
    label = action.get("label")
    if label:
        line += f" {EM_DASH} {_scalar_text(label)}"
    description = action.get("description")
    if description:
        line += f": {_scalar_text(description)}"
    for key in sorted(action, key=str):
        if key in _ACTION_HEAD_KEYS:
            continue
        value = action[key]
        if key == "parameters":
            if isinstance(value, dict) and value:
                params = ", ".join(
                    f"{key_}={_inline(value[key_])}" for key_ in sorted(value, key=str)
                )
                line += f", parameters: {params}"
            continue
        if value in (None, "", {}, []):
            continue
        line += f", {_label(key)}: {_inline(value)}"
    return line


def _emit_observation(
    lines: list[str],
    observation: dict[str, Any],
    level: int,
    self_title: str = "You",
) -> None:
    """Render one observation (or a nested observation one level down).

    Sections appear in a fixed order; only sections whose keys are present
    are emitted. Keys without a bespoke section still render: scalars go to
    "Setup" and structures go through the generic recursive renderer.
    """
    remaining = {
        key: value
        for key, value in observation.items()
        if key != "environment_briefing"
    }

    setup = {
        key: value
        for key, value in remaining.items()
        if key not in _BESPOKE_KEYS and _is_scalar(value)
    }
    if setup:
        _emit_heading(lines, level, "Setup")
        for key in sorted(setup, key=str):
            lines.extend(_kv_lines(key, setup[key]))

    if remaining.get("instruction") is not None:
        _emit_section(lines, "Instruction", remaining["instruction"], level)

    world = {key: remaining[key] for key in _WORLD_KEYS if key in remaining}
    if world:
        _emit_heading(lines, level, "World")
        for key in _WORLD_KEYS:
            if key in world:
                lines.extend(_kv_lines(key, world[key]))

    if "agent_id" in remaining or "self" in remaining:
        # ``self`` is always the observing agent of THIS observation: "You" at
        # the top level, "Target" inside a reader's nested target view, so a
        # reader is never told the target's state is its own.
        _emit_heading(lines, level, self_title)
        if "agent_id" in remaining:
            lines.extend(_kv_lines("agent_id", remaining["agent_id"]))
        if "self" in remaining:
            _emit_value(lines, remaining["self"], level + 1)

    if "shared" in remaining:
        _emit_section(lines, "Shared", remaining["shared"], level)

    if "recent_events" in remaining:
        _emit_heading(lines, level, "Recent events")
        _emit_events(lines, remaining["recent_events"])

    if "prediction_context" in remaining:
        _emit_section(lines, "Prediction context", remaining["prediction_context"], level)

    for key, title, nested_self_title in (
        ("target_observation", "Target observation", "Target"),
        ("predictor_observation", "Your own observation", "You"),
    ):
        value = remaining.get(key)
        if value is None:
            continue
        if isinstance(value, dict):
            _emit_heading(lines, level, title)
            _emit_observation(lines, value, level + 1, self_title=nested_self_title)
        else:
            _emit_section(lines, title, value, level)

    for key in ("public_state", "serialized_state"):
        if key in remaining:
            _emit_section(lines, "State", remaining[key], level)

    for key in sorted(remaining, key=str):
        if key in _BESPOKE_KEYS or key in setup:
            continue
        _emit_section(lines, _title(key), remaining[key], level)


def _emit_response(lines: list[str], return_schema: Any, level: int) -> None:
    _emit_heading(lines, level, "Response")
    lines.append(RESPONSE_SENTENCE)
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(return_schema, indent=2, sort_keys=True))
    lines.append("```")


def render_markdown_user_message(
    task: str,
    observation: dict[str, Any],
    action_space: list[dict[str, Any]] | None,
    return_schema: Any,
) -> str:
    """Render the markdown user message. Pure function of its arguments."""
    lines: list[str] = ["## Task", TASK_LINES.get(task, f"{_title(task)}.")]
    _emit_observation(lines, observation, 2)
    if action_space is not None:
        _emit_heading(lines, 2, "Actions")
        if action_space:
            lines.extend(_action_line(action) for action in action_space)
        else:
            lines.append("(none)")
    _emit_response(lines, return_schema, 2)
    return "\n".join(lines)
