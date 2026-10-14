"""Snapshot and event-log plumbing shared by the Commons environments.

Ported from the research harness module ``self_other_prediction/envs/base.py``.
Kept: ``StepResult`` and the parts of ``SnapshotMixin`` the environment uses
(snapshot, restore, serialize, event recording, visible history, briefing).
Not ported: the multi-environment protocol class, the probe-target routing,
the omniscient ``prompt_state`` projection, and ``fallback_action``. None of
them change state, observations, or rendered prompts for Commons.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .events import make_event, view_events


@dataclass
class StepResult:
    events: list[dict[str, Any]]
    rewards: dict[str, float] = field(default_factory=dict)


class SnapshotMixin:
    """State snapshotting plus the event-log and visibility plumbing.

    All history lives as typed events in ``state["events"]`` (see
    :mod:`commons.core.events`). Observation histories are derived views over
    that single log, so public versus private is decided at record time.
    """

    state: dict[str, Any]
    # Default horizon; the Commons environment overrides it with 96.
    history_horizon: int = 24
    # Fixed rules briefing, included identically in every prompt. It names no
    # agent id and gives no strategy advice.
    env_briefing: str = ""

    def briefing(self) -> str:
        return self.env_briefing

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.state)

    def restore(self, snapshot: dict[str, Any]) -> None:
        self.state = deepcopy(snapshot)

    def serialize_state(self) -> dict[str, Any]:
        return deepcopy(self.state)

    # --- event log -------------------------------------------------------
    def events(self) -> list[dict[str, Any]]:
        return self.state.setdefault("events", [])

    def record_event(
        self,
        actor: str,
        action: str,
        scope: str = "public",
        public_action: str | None = None,
        visible_to: tuple[str, ...] | list[str] = (),
        **payload: Any,
    ) -> dict[str, Any]:
        event = make_event(
            turn=self.state.get("turn", 0),
            actor=actor,
            action=action,
            scope=scope,
            public_action=public_action,
            visible_to=visible_to,
            **payload,
        )
        self.events().append(event)
        return event

    def visible_history(
        self,
        viewer: str | None,
        limit: int | None = None,
        include_turn: bool = False,
    ) -> list[dict[str, Any]]:
        horizon = self.history_horizon if limit is None else limit
        return view_events(self.events(), viewer, limit=horizon, include_turn=include_turn)
