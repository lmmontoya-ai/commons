"""The runtime, fully offline: request bodies, the client's retry policy with
an injected transport, reply parsing, agents, the episode runner, policy
replay, and the command-line scripts. No test makes a network call."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest
from conftest import ROOT, make_env

from commons.core import AGENTS, SHAM_LABELS, legal_action_ids
from commons.policies import POLICIES
from commons.render import actor_messages
from commons.runtime import (
    BudgetExceeded,
    CostMeter,
    EpisodeAbandoned,
    FakeClient,
    InvalidActionAbandon,
    ModelAgent,
    ModelSpec,
    OpenRouterClient,
    ProviderError,
    ScriptedAgent,
    action_reply,
    build_body,
    constant_action_client,
    load_models,
    parse_action,
    read_records,
    run_episode,
    sample_policy,
    states_from_log,
)

SPACE = [{"id": a, "label": a.lower()} for a in legal_action_ids("agent_1")]
FREE = ModelSpec(slug="fake/model", price_in=0.0, price_out=0.0)


# ----------------------------------------------------------------- body

def test_build_body_matches_the_studied_wire_settings() -> None:
    sonnet = ModelSpec(slug="anthropic/claude-sonnet-5",
                       provider_order=("Anthropic",), require_parameters=True,
                       temperature=None, max_tokens=8192,
                       reasoning={"exclude": False},
                       cache_control={"type": "ephemeral"})
    body = build_body(sonnet, [{"role": "user", "content": "x"}])
    assert body["response_format"] == {"type": "json_object"}
    assert "temperature" not in body
    assert body["provider"] == {"order": ["Anthropic"], "allow_fallbacks": False,
                                "require_parameters": True}
    assert body["reasoning"] == {"exclude": False}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["usage"] == {"include": True}
    plain = build_body(ModelSpec(slug="x/y"), [], include_usage=False)
    assert plain["temperature"] == 1.0
    assert "provider" not in plain and "reasoning" not in plain and "usage" not in plain


def test_example_registry_loads() -> None:
    models = load_models(ROOT / "models.example.json")
    assert set(models) == {"luna", "sonnet5", "grok45", "gemini37_flash", "glm53"}
    assert models["luna"].temperature is None
    assert models["grok45"].temperature == 1.0
    assert all(m.price_in is not None for m in models.values())


# --------------------------------------------------------------- client

def _ok(content: str = '{"action": "WAIT"}') -> str:
    return json.dumps({"model": "x/y", "provider": "P",
                       "choices": [{"finish_reason": "stop",
                                    "message": {"content": content}}],
                       "usage": {"cost": 0.001, "prompt_tokens": 10,
                                 "completion_tokens": 2}})


def _client(responses, **kw):
    sent = []
    queue = list(responses)

    def transport(url, headers, body, timeout):
        sent.append((url, headers, json.loads(body)))
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = OpenRouterClient(api_key="test-key", transport=transport,
                              sleep=lambda s: None, **kw)
    return client, sent


def test_client_requires_a_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ProviderError):
        OpenRouterClient()
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-env")
    assert OpenRouterClient().api_key == "from-env"


def test_client_retries_then_succeeds() -> None:
    client, sent = _client([(429, "slow down"), OSError("reset"), (200, _ok())])
    result = client.chat(ModelSpec(slug="x/y"), [{"role": "user", "content": "hi"}])
    assert result.content() == '{"action": "WAIT"}'
    assert result.attempts == 3 and result.provider == "P"
    assert len(sent) == 3
    url, headers, body = sent[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert headers["Authorization"] == "Bearer test-key"
    assert body["messages"] == [{"role": "user", "content": "hi"}]


def test_client_raises_on_non_retryable_and_abandons_past_caps() -> None:
    client, _ = _client([(400, "bad request")])
    with pytest.raises(ProviderError):
        client.chat(ModelSpec(slug="x/y"), [])
    client, _ = _client([(503, "")] * 3, max_attempts=3)
    with pytest.raises(EpisodeAbandoned):
        client.chat(ModelSpec(slug="x/y"), [])


def test_cost_meter_enforces_the_cap() -> None:
    meter = CostMeter(cap_usd=0.02)
    spec = ModelSpec(slug="x/y", price_in=1.0, price_out=1.0, max_tokens=100)
    client, _ = _client([(200, _ok())])
    meter.call(client, spec, [])
    assert meter.calls == 1 and meter.spent_usd == pytest.approx(0.001)
    with pytest.raises(BudgetExceeded):
        CostMeter(cap_usd=1.0).precheck(ModelSpec(slug="unpriced/model"))


# ---------------------------------------------------------------- parse

@pytest.mark.parametrize("text,lenient,strict", [
    ('{"action": "TAKE_1"}', "TAKE_1", "TAKE_1"),
    ('```json\n{"action": "WAIT"}\n```', "WAIT", "WAIT"),
    ('{"action": "take_1"}', "TAKE_1", None),
    ('{"action": "Take one supply unit"}', None, None),
    ('{"choice": "REPAIR_FULL"}', "REPAIR_FULL", None),
    ('{"action": null, "answer": "WAIT"}', "WAIT", None),
    ('I pick {"action": "TAKE_3"} because', "TAKE_3", "TAKE_3"),
    ('{"action": "FLY"}', None, None),
    ("no json here", None, None),
])
def test_parsers(text, lenient, strict) -> None:
    assert parse_action(text, SPACE, "lenient")[0] == lenient
    assert parse_action(text, SPACE, "strict")[0] == strict


def test_lenient_parser_resolves_labels() -> None:
    space = [{"id": "TAKE_1", "label": "Take one supply unit"}]
    assert parse_action('{"action": "take one supply unit"}', space)[0] == "TAKE_1"


# --------------------------------------------------------------- agents

def test_model_agent_sends_exactly_the_actor_prompt() -> None:
    state = make_env().serialize_state()
    client = constant_action_client("REPAIR_PATCH")
    decision = ModelAgent(FREE, client).act(state, "agent_2")
    assert decision.action == "REPAIR_PATCH" and not decision.invalid
    assert client.requests[0][1] == actor_messages(state, "agent_2")


def test_model_agent_retries_identically_then_waits_or_abandons() -> None:
    state = make_env().serialize_state()
    client = FakeClient(["not json", '{"action": "FLY"}', '{"action": "nope"}'])
    decision = ModelAgent(FREE, client).act(state, "agent_1")
    assert decision.action == "WAIT" and decision.invalid and decision.attempts == 3
    assert len(client.requests) == 3
    assert client.requests[0][1] == client.requests[2][1]
    with pytest.raises(InvalidActionAbandon):
        ModelAgent(FREE, FakeClient(["x"]), on_invalid="abandon").act(state, "agent_1")
    flaky = FakeClient(["garbage", action_reply("TAKE_1")])
    assert ModelAgent(FREE, flaky).act(state, "agent_1").attempts == 2


def test_scripted_agent_falls_back_to_wait_on_an_illegal_policy_output() -> None:
    state = make_env().serialize_state()
    decision = ScriptedAgent("bad", lambda s, a: "DELEGATE_AGENT_1").act(state, "agent_1")
    assert decision.action == "WAIT" and decision.invalid


# -------------------------------------------------------------- episodes

def test_scripted_episode_runs_logs_and_replays(tmp_path) -> None:
    agents = {a: ScriptedAgent(n, POLICIES[n])
              for a, n in zip(AGENTS, ("greedy", "softmax", "uniform"))}
    result = run_episode(seed=2700, scenario_id="delivery_pressure", agents=agents,
                         out_dir=tmp_path, episode_id="ep")
    assert result is not None and result["replay_verified"]
    assert result["final_state"]["turn"] == 16
    records = read_records(tmp_path / "ep.jsonl")
    kinds = [r["kind"] for r in records]
    assert kinds[0] == "meta" and kinds[-1] == "result"
    assert kinds.count("decision") == 48 and kinds.count("turn") == 16
    states = states_from_log(tmp_path / "ep.jsonl")
    assert states[16] == result["final_state"]
    again = run_episode(seed=2700, scenario_id="delivery_pressure", agents=agents,
                        out_dir=tmp_path, episode_id="ep2")
    assert again["final_state"] == result["final_state"]


def test_model_path_episode_with_fake_clients(tmp_path) -> None:
    agents = {
        "agent_1": ModelAgent(FREE, constant_action_client("TAKE_1")),
        "agent_2": ModelAgent(FREE, FakeClient(["oops"])),
        "agent_3": ScriptedAgent("wait", POLICIES["wait"]),
    }
    meter = CostMeter(cap_usd=1.0)
    result = run_episode(seed=1, scenario_id="coordination_debt", agents=agents,
                         out_dir=tmp_path, episode_id="m", turns=4, meter=meter,
                         objectives={"agent_1": "SOCIAL_CREDIT",
                                     "agent_2": "PERSONAL_THROUGHPUT",
                                     "agent_3": "SYSTEM_HEALTH"})
    assert result["invalid_decisions"] == 4
    records = read_records(tmp_path / "m.jsonl")
    calls = [r for r in records if r["kind"] == "call"]
    assert len(calls) == 4 * (1 + 3)
    assert all("raw" in c and "digest" in c for c in calls)
    assert meter.calls == 16


def test_abandoned_episode_writes_no_result(tmp_path) -> None:
    agents = {a: ModelAgent(FREE, FakeClient(["x"]), on_invalid="abandon")
              for a in AGENTS}
    assert run_episode(seed=1, scenario_id="coordination_debt", agents=agents,
                       out_dir=tmp_path, episode_id="ab", turns=2) is None
    kinds = [r["kind"] for r in read_records(tmp_path / "ab.jsonl")]
    assert "abandoned" in kinds and "result" not in kinds


def test_sham_episode(tmp_path) -> None:
    display = dict(zip(AGENTS, SHAM_LABELS))
    client = constant_action_client("REPAIR_PATCH")
    agents = {a: ModelAgent(FREE, client) for a in AGENTS}
    result = run_episode(seed=3, scenario_id="infrastructure_risk", agents=agents,
                         out_dir=tmp_path, episode_id="s", turns=2,
                         sham_display=display)
    assert result["arm"] == "sham"
    sent = client.requests[0][1]
    assert SHAM_LABELS[0] in sent[1]["content"]
    assert "SYSTEM_HEALTH" not in sent[0]["content"] + sent[1]["content"]
    with pytest.raises(ValueError):
        run_episode(seed=3, scenario_id="infrastructure_risk", agents=agents,
                    out_dir=tmp_path, sham_display=display,
                    objectives={a: "SYSTEM_HEALTH" for a in AGENTS})


def test_sample_policy_at_a_frozen_decision() -> None:
    state = make_env().serialize_state()
    client = FakeClient([action_reply("TAKE_1"), action_reply("WAIT"), "bad", "bad",
                         "bad"])
    batch = sample_policy(client, FREE, state, "agent_1", n=3)
    assert batch.actions == ["TAKE_1", "WAIT", None]
    assert batch.invalid_draws == 1 and batch.counts() == {"TAKE_1": 1, "WAIT": 1}
    assert all(r[1] == actor_messages(state, "agent_1") for r in client.requests)


# --------------------------------------------------------------- scripts

def _run(*args, cwd):
    return subprocess.run([sys.executable, *args], cwd=cwd, capture_output=True,
                          text=True, encoding="utf-8", timeout=600)


def test_run_episode_script_zero_cost(tmp_path) -> None:
    out = tmp_path / "runs"
    proc = _run(str(ROOT / "scripts" / "run_episode.py"), "--seed", "2700",
                "--scenario", "delivery_pressure", "--agent-1", "scripted:greedy",
                "--agent-2", "fake:REPAIR_PATCH", "--agent-3", "scripted:uniform",
                "--out", str(out), "--episode-id", "cli", cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert "replay verified: True" in proc.stdout
    assert (out / "cli.jsonl").exists()
    show = _run(str(ROOT / "scripts" / "show_prompts.py"), "--log",
                str(out / "cli.jsonl"), "--turn", "3", "--agent", "agent_2",
                "--view", "tally", cwd=tmp_path)
    assert show.returncode == 0, show.stderr
    assert "recent event counts" in show.stdout.lower()
