"""Reply parsing: model text -> one legal action id, or an error.

``parse_json_object``, ``resolve_action_choice`` and ``effective_action_id``
are ported unchanged from the research harness module
``self_other_prediction/model_clients.py``.

Two parsers were used by the studied Commons campaigns:

``lenient`` (phase 2: the original 36-episode bank)
    Extract the first JSON object; accept an ``action`` / ``action_id`` /
    ``selected_action`` / ``choice`` field that equals a legal id or matches
    an id or label after case and punctuation folding; otherwise accept a
    legal id in ``answer`` when ``action`` is null (the answer-field rescue).

``strict`` (phase 3: the later episodes, including the transfer study)
    Extract the first JSON object; its ``action`` field must be exactly a
    legal id.
"""
from __future__ import annotations

import json
from typing import Any

PARSERS = ("lenient", "strict")

ANSWER_RESCUE_ACTION_FIELDS = ("action", "action_id", "choice", "selected_action")


def parse_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Extract the first complete JSON object from a completion.

    Uses ``raw_decode`` so a valid object followed by trailing fragments still
    parses. A leading Markdown code fence is stripped.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None, None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    index = cleaned.find("{")
    while index >= 0:
        try:
            parsed, _ = decoder.raw_decode(cleaned, index)
            return parsed if isinstance(parsed, dict) else None, None
        except json.JSONDecodeError:
            index = cleaned.find("{", index + 1)
    return None, "No JSON object found"


def effective_action_id(
    parsed: dict[str, Any] | None,
    action_ids: list[str] | set[str] | tuple[str, ...],
) -> tuple[str | None, bool]:
    """``(action_id, rescued_via_answer_field)``.

    A valid ``action`` wins. Only when ``action`` is null or absent, ``answer``
    equals exactly one valid id, and no other recognized field holds a
    DIFFERENT valid id, the ``answer`` value is accepted.
    """
    if not isinstance(parsed, dict):
        return None, False
    valid_ids = list(action_ids)
    action = parsed.get("action")
    if isinstance(action, str) and action in valid_ids:
        return action, False
    if action is not None:
        return None, False
    answer = parsed.get("answer")
    if not isinstance(answer, str) or answer not in valid_ids:
        return None, False
    for field_name in ANSWER_RESCUE_ACTION_FIELDS:
        candidate = parsed.get(field_name)
        if isinstance(candidate, str) and candidate in valid_ids and candidate != answer:
            return None, False
    return answer, True


def _action_key(value: str) -> str:
    return "".join(char for char in value.lower() if char.isalnum())


def resolve_action_choice(
    parsed: dict[str, Any] | None, action_space: list[dict[str, Any]]
) -> str | None:
    if not parsed:
        return None
    action_ids = [action["id"] for action in action_space]
    lookup: dict[str, str] = {}
    for action in action_space:
        action_id = action["id"]
        for alias in (action_id, action.get("label", "")):
            key = _action_key(alias)
            if key:
                lookup[key] = action_id

    for action_field in ("action", "action_id", "selected_action", "choice"):
        candidate = parsed.get(action_field)
        if not isinstance(candidate, str):
            continue
        if candidate in action_ids:
            return candidate
        resolved = lookup.get(_action_key(candidate))
        if resolved:
            return resolved
    return None


def parse_action(text: str, action_space: list[dict[str, Any]],
                 parser: str = "lenient") -> tuple[str | None, str | None]:
    """``(action, error)``; exactly one is set."""
    if parser not in PARSERS:
        raise ValueError(f"unknown parser {parser!r}; expected one of {PARSERS}")
    legal = [a["id"] for a in action_space]
    parsed, error = parse_json_object(text)
    if parser == "strict":
        if error:
            return None, f"json parse: {error}"
        action = parsed.get("action") if isinstance(parsed, dict) else None
        if action is None:
            return None, "no action field in reply"
        if str(action) not in legal:
            return None, f"action {str(action)!r} outside legal alphabet"
        return str(action), None
    action = resolve_action_choice(parsed, action_space)
    if action is None:
        action, _rescued = effective_action_id(parsed, legal)
    if action in legal:
        return action, None
    return None, error or "no legal action in reply"
