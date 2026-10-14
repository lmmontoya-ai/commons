"""Model registry: what a 'model' is for a Commons run.

A model is an OpenAI-compatible chat slug plus the generation settings sent
with every request. The fields mirror the wire settings the studied campaigns
sent through OpenRouter: an explicit provider order (fallbacks off by
default), optional ``require_parameters``, a reasoning block, optional prompt
caching, and either a temperature or no temperature at all (some models
reject the parameter).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    slug: str                              # provider model slug
    label: str = ""                        # short name used in logs
    provider_order: tuple[str, ...] = ()   # pinned providers; empty = unpinned
    allow_fallbacks: bool = False          # only sent when provider_order is set
    require_parameters: bool = False       # OpenRouter provider.require_parameters
    temperature: float | None = 1.0        # None = omit the parameter
    max_tokens: int = 8192
    reasoning: dict | None = None          # OpenRouter reasoning block, or None
    cache_control: dict | None = None      # e.g. {"type": "ephemeral"}
    price_in: float | None = None          # USD per 1M input tokens (cost cap)
    price_out: float | None = None         # USD per 1M output tokens

    def describe(self) -> dict:
        return {
            "slug": self.slug, "label": self.label or self.slug,
            "provider_order": list(self.provider_order),
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
            "temperature": self.temperature, "max_tokens": self.max_tokens,
            "reasoning": self.reasoning, "cache_control": self.cache_control,
        }


def model_spec_from_dict(name: str, m: dict) -> ModelSpec:
    return ModelSpec(
        slug=m["slug"], label=m.get("label", name),
        provider_order=tuple(m.get("provider_order", [])),
        allow_fallbacks=bool(m.get("allow_fallbacks", False)),
        require_parameters=bool(m.get("require_parameters", False)),
        temperature=m.get("temperature", 1.0),
        max_tokens=int(m.get("max_tokens", 8192)),
        reasoning=m.get("reasoning"),
        cache_control=m.get("cache_control"),
        price_in=m.get("price_in"), price_out=m.get("price_out"),
    )


def load_models(path: str | Path) -> dict[str, ModelSpec]:
    """Load a registry file: ``{"models": {name: {slug, ...}}}``. Keys that
    start with an underscore are comments and are ignored."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {name: model_spec_from_dict(name, m)
            for name, m in payload["models"].items()
            if not name.startswith("_")}
