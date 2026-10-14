"""commons.render: pure rendering, state -> observation -> chat messages.

Every function here is deterministic. Two renders of the same input are
byte-equal, and the bytes equal the prompts used in the studied campaigns.
"""
import hashlib
import json

from .markdown import EM_DASH, render_markdown_user_message
from .messages import (
    ACTION_SYSTEM_PROMPT,
    FORECAST_SYSTEM_PROMPT,
    INSTRUCTION_VARIANTS,
    NEUTRAL_FORECAST_PERSPECTIVE,
    RETURN_SCHEMA_ACTION_ONLY,
    Message,
    build_action_messages,
    build_forecast_messages,
    build_messages,
    system_content,
)
from .views import (
    COUNT_CONTRACT,
    HISTORY_CONTRACT,
    OBJECTIVE_INSTRUCTION,
    OBJECTIVE_RETURN_SCHEMA,
    OBJECTIVE_SYSTEM_PROMPT,
    VIEWS,
    actor_messages,
    actor_observation,
    canonical_view,
    counts_only,
    default_shuffle_seed_context,
    goal_naming_messages,
    instruction_variant_for,
    instruction_variant_map,
    objective_menu,
    reader_messages,
    reader_observation,
)


def messages_digest(messages: list[Message]) -> str:
    """SHA-256 over the canonical JSON of a message list (role and content,
    in order). Used to label calls in episode logs."""
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


__all__ = [
    "EM_DASH", "render_markdown_user_message",
    "ACTION_SYSTEM_PROMPT", "FORECAST_SYSTEM_PROMPT", "INSTRUCTION_VARIANTS",
    "NEUTRAL_FORECAST_PERSPECTIVE", "RETURN_SCHEMA_ACTION_ONLY", "Message",
    "build_action_messages", "build_forecast_messages", "build_messages",
    "system_content",
    "COUNT_CONTRACT", "HISTORY_CONTRACT", "OBJECTIVE_INSTRUCTION",
    "OBJECTIVE_RETURN_SCHEMA", "OBJECTIVE_SYSTEM_PROMPT", "VIEWS",
    "actor_messages", "actor_observation", "canonical_view", "counts_only",
    "default_shuffle_seed_context", "goal_naming_messages",
    "instruction_variant_for", "instruction_variant_map", "objective_menu",
    "reader_messages", "reader_observation",
    "messages_digest",
]
