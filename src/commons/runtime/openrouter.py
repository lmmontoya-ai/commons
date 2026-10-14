"""The only code that talks to a provider.

``OpenRouterClient`` posts chat-completion requests to an OpenAI-compatible
endpoint, OpenRouter by default. The API key is read from the environment
variable named by ``api_key_env`` (``OPENROUTER_API_KEY`` by default) unless
passed explicitly. It is never written to logs.

Provider errors, timeouts, and malformed bodies are infrastructure, never
behavior: they are retried with backoff, and past the caps the episode is
abandoned (``EpisodeAbandoned``). Nothing here converts an infrastructure
failure into a WAIT.

The request body (``build_body``) reproduces the wire settings the studied
campaigns sent: ``response_format: json_object`` (the prompts ask for one
JSON object), the pinned provider block, the reasoning block, optional prompt
caching, and the temperature unless the model rejects it. ``usage.include``
asks OpenRouter to report cost; it is not prompt-visible.

The transport is injectable, so tests exercise retries and bodies without a
network. The default transport uses only the standard library.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

from .config import ModelSpec

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"
RETRYABLE = {408, 429, 500, 502, 503, 504}

# transport(url, headers, body_bytes, timeout_s) -> (status_code, response_text)
Transport = Callable[[str, dict, bytes, float], tuple[int, str]]


class ProviderError(RuntimeError):
    """A non-retryable provider response."""


class EpisodeAbandoned(RuntimeError):
    """The provider stayed down past the caps: abandon and quarantine."""


class BudgetExceeded(RuntimeError):
    """The hard cost cap tripped; the episode is abandoned, not truncated."""


@dataclass
class CallResult:
    raw: dict                    # the full response body, verbatim
    provider: str | None         # who actually served it, per the response
    model: str                   # the served model id, per the response
    usage: dict
    cost_usd: float | None
    latency_s: float
    attempts: int                # transport attempts (1 = clean)

    def message(self) -> dict:
        return (self.raw.get("choices") or [{}])[0].get("message") or {}

    def content(self) -> str:
        return self.message().get("content") or ""

    def finish_reason(self) -> str | None:
        return (self.raw.get("choices") or [{}])[0].get("finish_reason")


def build_body(spec: ModelSpec, messages: list[dict], *,
               include_usage: bool = True) -> dict:
    """The exact JSON request body for one chat call."""
    body: dict = {
        "model": spec.slug,
        "messages": messages,
        "max_tokens": spec.max_tokens,
        "response_format": {"type": "json_object"},
    }
    if spec.temperature is not None:
        body["temperature"] = spec.temperature
    if spec.provider_order or spec.require_parameters:
        provider: dict = {}
        if spec.provider_order:
            provider["order"] = list(spec.provider_order)
            provider["allow_fallbacks"] = bool(spec.allow_fallbacks)
        if spec.require_parameters:
            provider["require_parameters"] = True
        body["provider"] = provider
    if spec.reasoning is not None:
        body["reasoning"] = dict(spec.reasoning)
    if spec.cache_control is not None:
        body["cache_control"] = dict(spec.cache_control)
    if include_usage:
        body["usage"] = {"include": True}
    return body


def urllib_transport(url: str, headers: dict, body: bytes,
                     timeout_s: float) -> tuple[int, str]:
    """Default transport: a single HTTPS POST with the standard library."""
    request = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.status, response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        return error.code, error.read().decode("utf-8", errors="replace")


class CostMeter:
    """Cumulative spend with a HARD cap, checked before a call is placed
    (worst-case estimate) as well as after. A call whose price cannot be
    determined is an error, never $0, unless ``strict_pricing`` is False."""

    # Covers the longest Commons prompt with ample headroom.
    EST_PROMPT_TOKENS = 16_000

    def __init__(self, cap_usd: float | None = None, *,
                 strict_pricing: bool = True) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self.calls = 0
        self.unpriced_calls = 0
        self.strict_pricing = strict_pricing
        self.reserved_usd = 0.0
        self._lock = threading.Lock()

    def precheck(self, spec: ModelSpec) -> float:
        """Reserve one call's worst-case cost before launch."""
        if self.cap_usd is None:
            return 0.0
        if spec.price_in is None or spec.price_out is None:
            if self.strict_pricing:
                raise BudgetExceeded(
                    f"{spec.slug} has no price entry; refusing to spend "
                    "unmetered money (set price_in and price_out)")
            return 0.0
        worst = (self.EST_PROMPT_TOKENS * spec.price_in
                 + spec.max_tokens * spec.price_out) / 1e6
        with self._lock:
            committed = self.spent_usd + self.reserved_usd
            if committed + worst > self.cap_usd:
                raise BudgetExceeded(
                    f"next call could cost ${worst:.2f}; "
                    f"${self.spent_usd:.2f} spent + "
                    f"${self.reserved_usd:.2f} reserved of "
                    f"${self.cap_usd:.2f}")
            self.reserved_usd += worst
        return worst

    def release(self, reservation: float) -> None:
        if not reservation:
            return
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - reservation)

    def add(self, result: CallResult, spec: ModelSpec, *,
            reservation: float = 0.0) -> None:
        cost = result.cost_usd
        if cost is None and spec.price_in is not None and spec.price_out is not None:
            u = result.usage or {}
            cost = (u.get("prompt_tokens", 0) * spec.price_in
                    + u.get("completion_tokens", 0) * spec.price_out) / 1e6
        with self._lock:
            if reservation:
                self.reserved_usd = max(0.0, self.reserved_usd - reservation)
            self.calls += 1
            if cost is None:
                self.unpriced_calls += 1
                if self.strict_pricing:
                    raise BudgetExceeded(
                        f"call to {spec.slug} returned no usage cost and the "
                        "registry has no prices; unmetered spend refused")
                cost = 0.0
            self.spent_usd += cost
            if self.cap_usd is not None and self.spent_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"spent ${self.spent_usd:.2f} > cap ${self.cap_usd:.2f} "
                    f"after {self.calls} calls")

    def call(self, client, spec: ModelSpec, messages: list[dict]) -> CallResult:
        """Place and account for one call under an atomic reservation."""
        reservation = self.precheck(spec)
        try:
            result = client.chat(spec, messages)
        except BaseException:
            self.release(reservation)
            raise
        self.add(result, spec, reservation=reservation)
        return result


class OpenRouterClient:
    def __init__(self, api_key: str | None = None, *,
                 base_url: str = DEFAULT_BASE_URL,
                 api_key_env: str = DEFAULT_API_KEY_ENV,
                 transport: Transport | None = None,
                 timeout_s: float = 180.0, max_attempts: int = 6,
                 wall_clock_cap_s: float = 900.0, app_name: str = "commons",
                 include_usage: bool = True,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.api_key = api_key or os.environ.get(api_key_env)
        if not self.api_key:
            raise ProviderError(
                f"no API key: pass api_key or set {api_key_env}")
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.transport = transport or urllib_transport
        self.timeout_s = timeout_s
        self.max_attempts = max_attempts
        self.wall_clock_cap_s = wall_clock_cap_s
        self.app_name = app_name
        self.include_usage = include_usage
        self._sleep = sleep

    def chat(self, spec: ModelSpec, messages: list[dict]) -> CallResult:
        body = build_body(spec, messages, include_usage=self.include_usage)
        payload = json.dumps(body).encode("utf-8")
        headers = {"Authorization": f"Bearer {self.api_key}",
                   "Content-Type": "application/json",
                   "X-Title": self.app_name}
        started = time.monotonic()
        last_err = "no attempt made"
        for attempt in range(1, self.max_attempts + 1):
            if time.monotonic() - started > self.wall_clock_cap_s:
                break
            t0 = time.monotonic()
            try:
                status, text = self.transport(self.url, headers, payload,
                                              self.timeout_s)
            except OSError as e:  # connection errors and timeouts
                last_err = f"transport: {e}"
                self._sleep(min(2 ** attempt, 30))
                continue
            latency = time.monotonic() - t0
            if status in RETRYABLE:
                last_err = f"HTTP {status}"
                self._sleep(min(2 ** attempt, 30))
                continue
            if status != 200:
                raise ProviderError(f"HTTP {status}: {text[:500]}")
            try:
                raw = json.loads(text)
            except json.JSONDecodeError:
                last_err = "unparseable 200 body"
                self._sleep(min(2 ** attempt, 30))
                continue
            err = raw.get("error")
            if err:
                code = err.get("code") if isinstance(err, dict) else None
                if isinstance(code, int) and code in RETRYABLE:
                    last_err = f"provider error {code}: {err.get('message')}"
                    self._sleep(min(2 ** attempt, 30))
                    continue
                raise ProviderError(f"provider error: {err}")
            usage = raw.get("usage") or {}
            return CallResult(
                raw=raw,
                provider=raw.get("provider"),
                model=raw.get("model", spec.slug),
                usage=usage,
                cost_usd=usage.get("cost"),
                latency_s=round(latency, 3),
                attempts=attempt,
            )
        raise EpisodeAbandoned(
            f"provider unavailable past caps ({last_err}); episode abandoned")
