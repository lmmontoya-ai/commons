"""Typed event log with per-viewer visibility scopes.

Every applied action and every interaction outcome is recorded as one event
dict in ``state["events"]``. Plain dicts keep snapshot and restore a deepcopy.
All observation history is derived from this log through ``view_events``, so
what is public is decided once, by the environment, when the event is
recorded.

Scopes:

- ``public``: every viewer sees the full event.
- ``outcome_only``: viewers outside the event see a masked event whose
  ``action`` is replaced by ``public_action``. In Commons this is how a
  sharing outcome such as ``exchange_reciprocated`` appears to an outsider
  as ``bilateral_exchange``. The actor, and any agent listed in
  ``visible_to``, sees the true event.
- ``private``: only the actor (and ``visible_to``) sees the event at all.
  Commons itself never records a private event; the scope is kept because it
  is part of the visibility contract.

Ported from the research harness module ``self_other_prediction/events.py``.
The functions kept here are byte-for-byte the same logic. Helpers used only by
research probes (identity anonymization, turn stripping) were not ported.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from typing import Any

PUBLIC = "public"
OUTCOME_ONLY = "outcome_only"
PRIVATE = "private"

_SCOPES = {PUBLIC, OUTCOME_ONLY, PRIVATE}
_MASKED_ACTION = "unobserved_activity"

# Fields internal to the visibility mechanism. They are stripped from rendered
# views, so prompts never mention scopes or the true action of a masked event.
_INTERNAL_FIELDS = ("scope", "public_action", "visible_to")


def make_event(
    turn: int,
    actor: str,
    action: str,
    scope: str = PUBLIC,
    public_action: str | None = None,
    visible_to: tuple[str, ...] | list[str] = (),
    **payload: Any,
) -> dict[str, Any]:
    if scope not in _SCOPES:
        raise ValueError(f"unknown event scope: {scope!r}")
    if scope == OUTCOME_ONLY and not public_action:
        public_action = _MASKED_ACTION
    return {
        "turn": turn,
        "actor": actor,
        "action": action,
        "scope": scope,
        "public_action": public_action,
        "visible_to": list(visible_to),
        **payload,
    }


def _can_see_true_event(event: dict[str, Any], viewer: str | None) -> bool:
    if event.get("scope", PUBLIC) == PUBLIC:
        return True
    if viewer is None:
        return False
    return viewer == event.get("actor") or viewer in (event.get("visible_to") or [])


def view_events(
    events: list[dict[str, Any]],
    viewer: str | None = None,
    limit: int | None = None,
    include_turn: bool = False,
) -> list[dict[str, Any]]:
    """Render the event log as seen by ``viewer`` (None = external observer).

    ``include_turn`` defaults to False, so order is carried only by list
    position. A shuffled history then differs from the ordered one only in
    order, with no recoverable timestamps.
    """
    rendered: list[dict[str, Any]] = []
    for event in events:
        scope = event.get("scope", PUBLIC)
        if _can_see_true_event(event, viewer):
            visible = {k: v for k, v in event.items() if k not in _INTERNAL_FIELDS}
        elif scope == OUTCOME_ONLY:
            visible = {
                "turn": event.get("turn"),
                "actor": event.get("actor"),
                "action": event.get("public_action") or _MASKED_ACTION,
            }
        else:  # private and the viewer is not privileged
            continue
        if not include_turn:
            visible = {k: v for k, v in visible.items() if k != "turn"}
        rendered.append(visible)
    if limit is not None:
        # limit=0 must mean "no events" (the board-only reader view);
        # rendered[-0:] would silently return the WHOLE list.
        rendered = rendered[-limit:] if limit > 0 else []
    return rendered


def shuffled_events(
    events: list[dict[str, Any]],
    seed_context: dict[str, Any],
) -> list[dict[str, Any]]:
    """Deterministically reorder rendered events; content is untouched.

    ``random.shuffle`` may return the identity permutation, which would make a
    shuffled view byte-identical to the ordered one. If that happens, rotate
    by a deterministic non-zero offset. A one-event history is necessarily
    unchanged.
    """
    shuffled = copy.deepcopy(events)
    seed = json.dumps(seed_context, sort_keys=True, default=str)
    random.Random(f"{seed}:shuffled_history").shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == events:
        offset = int.from_bytes(
            hashlib.sha256(f"{seed}:shuffle_fallback".encode()).digest()[:8],
            "big",
        ) % (len(shuffled) - 1) + 1
        shuffled = [*shuffled[offset:], *shuffled[:offset]]
    return shuffled
