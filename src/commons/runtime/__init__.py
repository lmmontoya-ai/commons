"""commons.runtime: the impure layer (providers, episodes, logs, replay).

Everything outside this package is pure. Everything that touches a network or
a clock lives here.
"""
from .agents import ATTEMPTS, Decision, InvalidActionAbandon, ModelAgent, ScriptedAgent
from .config import ModelSpec, load_models, model_spec_from_dict
from .episode import run_episode
from .fake import FakeClient, action_reply, constant_action_client
from .log import EpisodeLogger, read_records
from .openrouter import (
    DEFAULT_API_KEY_ENV,
    DEFAULT_BASE_URL,
    BudgetExceeded,
    CallResult,
    CostMeter,
    EpisodeAbandoned,
    OpenRouterClient,
    ProviderError,
    build_body,
)
from .parse import (
    PARSERS,
    effective_action_id,
    parse_action,
    parse_json_object,
    resolve_action_choice,
)
from .replay import (
    PolicyBatch,
    load_final_state,
    load_sham_display,
    sample_policy,
    states_from_log,
)

__all__ = [
    "ATTEMPTS", "Decision", "InvalidActionAbandon", "ModelAgent",
    "ScriptedAgent",
    "ModelSpec", "load_models", "model_spec_from_dict",
    "run_episode",
    "FakeClient", "action_reply", "constant_action_client",
    "EpisodeLogger", "read_records",
    "DEFAULT_API_KEY_ENV", "DEFAULT_BASE_URL", "BudgetExceeded", "CallResult",
    "CostMeter", "EpisodeAbandoned", "OpenRouterClient", "ProviderError",
    "build_body",
    "PARSERS", "effective_action_id", "parse_action", "parse_json_object",
    "resolve_action_choice",
    "PolicyBatch", "load_final_state", "load_sham_display", "sample_policy",
    "states_from_log",
]
