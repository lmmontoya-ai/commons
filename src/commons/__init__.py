"""Commons: a three-agent shared-workplace environment for language-model
agents, with hidden objectives and a belief-dependent interaction channel.

Called "Office" in the accompanying paper.

Subpackages:

- ``commons.core``: the pure world (state, actions, resolution, visibility,
  the sham variant, replay).
- ``commons.render``: pure rendering of actor and reader prompts.
- ``commons.policies``: scripted seats and the exact lookahead they use.
- ``commons.runtime``: the provider client, agents, episode runner, logs.
"""
__version__ = "0.1.0"

from .core import (  # noqa: E402
    AGENTS,
    CONDITION,
    EPISODE_TURNS,
    OBJECTIVES,
    SCENARIO_IDS,
    SCORED_TURNS,
    CommonsEnv,
    ShamCommonsEnv,
    action_space,
    replay_states,
)

__all__ = [
    "__version__",
    "AGENTS", "CONDITION", "EPISODE_TURNS", "OBJECTIVES", "SCENARIO_IDS",
    "SCORED_TURNS", "CommonsEnv", "ShamCommonsEnv", "action_space",
    "replay_states",
]
