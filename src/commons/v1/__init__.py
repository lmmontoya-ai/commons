"""commons.v1: Commons v1 ("Predictive Commons v1"), the Office environment
of the report's ordinary-play results.

It is an earlier, separate environment from :mod:`commons.core` (Commons
v2): fixed seat goals instead of hidden objectives, ``TRADE_<peer>`` offers
instead of ``SHARE_<peer>``, 8-turn trajectories, private action
acknowledgements, and JSON prompts (prompt protocol v8). See ``docs/V1.md``.

- :mod:`commons.v1.env`: state, transitions, legal actions, rewards, replay.
- :mod:`commons.v1.render`: actor and reader prompts.
- :mod:`commons.v1.decisions`: the frozen decisions of the report (90 M4.3
  turn-8 states and 48 M4.4 positions), rebuilt from seeds and action labels.

Nothing here imports the v2 environment, and importing this package does not
change :mod:`commons`.
"""
from .decisions import (
    ARMS,
    BANK_MANIFEST_CAMPAIGN,
    FORECAST_SUITE_VERSION,
    bank_metadata,
    bank_observations,
    bank_seed_context,
    legal_actions,
    position,
    position_actor_messages,
    position_reader_messages,
    position_state,
    positions,
    positions_metadata,
    suite_observations,
    suite_seed_context,
    trajectories,
    trajectory,
    trajectory_states,
    turn8_state,
)
from .env import (
    ACTION_FEEDBACK,
    ACTION_SUPPORT_SIZE,
    AGENTS,
    BRIEFING,
    CONDITION,
    ENV_NAME,
    ENV_VERSION,
    EPISODE_TURNS,
    FIXED_ACTIONS,
    PRIVATE_GOALS,
    PROMPT_PROTOCOL_VERSION,
    PUBLIC_ROLES,
    SCENARIO_IDS,
    SCENARIOS,
    WORLD,
    CommonsV1Env,
    action_family,
    action_space,
    applied_actions,
    legal_action_ids,
    replay_states,
    rollout_states,
    state_id,
    state_sha256,
)
from .render import (
    PROMPT_FORMATS,
    VIEW_ALIASES,
    VIEWS,
    actor_messages,
    actor_observation,
    canonical_view,
    default_shuffle_seed_context,
    messages_sha256,
    observation_sha256,
    reader_messages,
    reader_observation,
)

__all__ = [
    "ACTION_FEEDBACK", "ACTION_SUPPORT_SIZE", "AGENTS", "BRIEFING",
    "CONDITION", "ENV_NAME", "ENV_VERSION", "EPISODE_TURNS", "FIXED_ACTIONS",
    "PRIVATE_GOALS", "PROMPT_PROTOCOL_VERSION", "PUBLIC_ROLES",
    "SCENARIO_IDS", "SCENARIOS", "WORLD",
    "CommonsV1Env", "action_family", "action_space", "applied_actions",
    "legal_action_ids", "replay_states", "rollout_states", "state_id",
    "state_sha256",
    "PROMPT_FORMATS", "VIEW_ALIASES", "VIEWS", "actor_messages",
    "actor_observation", "canonical_view", "default_shuffle_seed_context",
    "messages_sha256", "observation_sha256", "reader_messages",
    "reader_observation",
    "ARMS", "BANK_MANIFEST_CAMPAIGN", "FORECAST_SUITE_VERSION",
    "bank_metadata", "bank_observations", "bank_seed_context",
    "legal_actions", "position", "position_actor_messages",
    "position_reader_messages", "position_state", "positions",
    "positions_metadata", "suite_observations", "suite_seed_context",
    "trajectories", "trajectory", "trajectory_states", "turn8_state",
]
