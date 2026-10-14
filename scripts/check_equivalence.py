"""Check that this package reproduces the original research harness exactly.

This is a provenance tool for maintainers, not part of the test suite. It
needs a checkout of the private research harness the environment was
extracted from (repository ``self-other-prediction-smoke-tests``) and must be
run with an interpreter that has the harness's own dependencies installed,
for example the harness's virtual environment:

    <harness>/.venv/Scripts/python.exe -B scripts/check_equivalence.py \
        --harness <harness checkout> --report docs/provenance/equivalence_report.json

The harness is only read. Bytecode writing is disabled, and recorded run
stores are copied to a temporary directory before they are opened.

What is compared, harness implementation versus this package:

1. Constants: briefing, goal texts, menus, outcome tables, scenarios, roles,
   sham texts, the reader instruction variant map.
2. Objective draws for many seeds, in both assignment modes.
3. Seeded synthetic trajectories (uniform random and lookahead play, drawn
   and explicit objectives, all three scenarios): reset states, every step's
   events and rewards, every intermediate state, every observation, and the
   rendered actor, reader (board, ordered, shuffled, tally), and goal-naming
   messages, byte for byte. Replay of every final state.
4. Sham arm: states, observations, actor messages, and ``assert_inert``.
5. Lookahead values and the scripted reference trajectories.
6. Recorded episodes from the studied campaigns: each recorded final state
   is replayed by both implementations, the endpoint must equal the record,
   the recorded action log must equal the replayed moves, and actor and
   reader prompts are compared at every turn.

``--write-golden`` also writes the digest fixture used by
``tests/test_golden_reference.py`` (computed from the HARNESS side).
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import contextlib  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

# Harness files the extracted modules come from (relative to the harness root).
SOURCE_FILES = (
    "src/self_other_prediction/envs/commons_v2.py",
    "src/self_other_prediction/envs/commons_v2_sham.py",
    "src/self_other_prediction/envs/base.py",
    "src/self_other_prediction/events.py",
    "src/self_other_prediction/prompt_render_md.py",
    "src/self_other_prediction/prompt_templates.py",
    "src/self_other_prediction/model_clients.py",
    "src/self_other_prediction/arms.py",
    "src/self_other_prediction/m4_forecast_suite.py",
    "src/self_other_prediction/commons_v2_precheck.py",
    "scripts/commons_v2_campaign.py",
    "scripts/commons_v3_campaign.py",
    "scripts/commons_m44.py",
    "config/predictive_commons_m4_3_forecast_suite_m4_3_1.json",
    "config/predictive_commons_v2.json",
)

# Recorded episode sets: (label, relative path, kind).
PHASE2_S0 = "data/outputs/predictive_commons_v2/campaign/cells/s0"
STORES = (
    ("phase3_s0", "data/outputs/phase3/commons/s0/_store"),
    ("phase3_s0b", "data/outputs/phase3/commons/s0b/_store"),
    ("phase3_s5a", "data/outputs/phase3/commons/s5a/_store"),
    ("s5_replication_s5a",
     "data/outputs/paper_harmonization/s5_replication/run/commons/s5a/_store"),
)

TRIO = ("openrouter_gpt56_luna_high", "openrouter_sonnet5_adaptive_r2",
        "openrouter_grok45")
SUITE_VERSION = "m4.3-fs1"
READER_VIEWS = ("board", "ordered", "shuffled", "tally")
HARNESS_VIEW = {"board": "board_only", "ordered": "ordered",
                "shuffled": "shuffled"}


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class Tally:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failures: list[str] = []

    def check(self, key: str, ok: bool, detail: str = "") -> None:
        self.counts[key] = self.counts.get(key, 0) + 1
        if not ok:
            self.failures.append(f"{key}: {detail}"[:500])

    def equal(self, key: str, a, b, detail: str = "") -> None:
        self.check(key, canonical(a) == canonical(b), detail)


def load(harness: Path):
    sys.path.insert(0, str(harness / "scripts"))
    sys.path.insert(0, str(harness / "src"))
    sys.path.insert(0, str(REPO / "src"))
    with contextlib.redirect_stdout(io.StringIO()):
        import commons_v2_campaign as HV2
        import commons_v3_campaign as HV3
        from self_other_prediction import commons_v2_precheck as HP
        from self_other_prediction import m4_forecast_suite as HFS
        from self_other_prediction import model_clients as HMC
        from self_other_prediction.envs import commons_v2 as H
        from self_other_prediction.envs import commons_v2_sham as HS
    import commons.core as C
    import commons.policies as P
    import commons.render as R
    return dict(HV2=HV2, HV3=HV3, HP=HP, HFS=HFS, HMC=HMC, H=H, HS=HS,
                C=C, P=P, R=R)


# ------------------------------------------------------------ harness side

def h_position(state, mover, model_key, position_id):
    return {"eid": "eq", "seed": state["seed"], "scenario": state["scenario_id"],
            "turn": state["turn"], "mover": mover,
            "mover_instrument": model_key, "position_id": position_id}


def h_reader_views(m, state, mover, model_key, position_id):
    """Harness reader messages for every view, plus goal naming."""
    HV2, HV3, HMC = m["HV2"], m["HV3"], m["HMC"]
    position = h_position(state, mover, model_key, position_id)
    obs = HV2.forecast_observations(position, state)
    out = {}
    for view in ("board", "ordered", "shuffled"):
        out[view] = HV2.forecast_messages(position, HARNESS_VIEW[view], obs)
    counts = HV3._counts_only(json.loads(json.dumps(obs[HV2.VIEWS["ordered"]])))
    out["tally"] = HMC.build_forecast_messages(
        counts, HV2.action_space_of(mover), action_only=True,
        prompt_format="markdown")
    out["goal-board"] = HV2.objective_messages(position, "board_only", obs)
    out["goal-ordered"] = HV2.objective_messages(position, "ordered", obs)
    out["goal-tally"] = HV2.objective_messages(
        position, "ordered", {HV2.VIEWS["ordered"]: counts})
    return out


def p_reader_views(m, state, mover, model_key, position_id):
    R = m["R"]
    variant = R.instruction_variant_for(model_key, mover, state["scenario_id"])
    seed_context = {"campaign": "commons_v2_campaign",
                    "snapshot_id": position_id,
                    "forecast_suite_version": SUITE_VERSION}
    out = {}
    for view in READER_VIEWS:
        out[view] = R.reader_messages(state, mover, view,
                                      instruction_variant=variant,
                                      shuffle_seed_context=seed_context)
    for view in ("board", "ordered", "tally"):
        out[f"goal-{view}"] = R.goal_naming_messages(state, mover, view)
    return out


def h_sham_actor(m, state, mover, display):
    HS, HMC = m["HS"], m["HMC"]
    env = HS.restore_sham(state, display)
    env.set_probe_target(mover)
    return HMC.build_action_messages(env.observe(mover), env.action_space(mover),
                                     action_only=True, prompt_format="markdown")


# ---------------------------------------------------------- comparisons

def compare_state_views(m, t: Tally, state, *, model_key, position_id,
                        readers: bool = True, sham_display=None):
    """Observations and every prompt at one state, for every seat.

    ``model_key`` is one model key, or a seat -> model key map; it selects
    the frozen reader instruction variant for each target seat."""
    H, C, R, HV2 = m["H"], m["C"], m["R"], m["HV2"]
    h_env = H.ConcordiaPredictiveCommonsV2Env()
    h_env.restore(state)
    p_env = C.CommonsEnv()
    p_env.restore(state)
    t.equal("public_state", h_env.public_state(), p_env.public_state())
    for agent in C.AGENTS:
        where = f"seed={state['seed']} turn={state['turn']} {agent}"
        if sham_display is None:
            t.equal("observe", h_env.observe(agent), p_env.observe(agent), where)
            t.equal("actor_messages", HV2.acting_messages(state, agent),
                    R.actor_messages(state, agent), where)
        else:
            t.equal("sham_actor_messages",
                    h_sham_actor(m, state, agent, sham_display),
                    R.actor_messages(state, agent, sham_display=sham_display),
                    where)
        t.equal("public_observation", h_env.public_observation(agent),
                p_env.public_observation(agent), where)
        if readers:
            key = (model_key.get(agent, TRIO[0]) if isinstance(model_key, dict)
                   else model_key)
            h = h_reader_views(m, state, agent, key, position_id)
            p = p_reader_views(m, state, agent, key, position_id)
            for view in h:
                t.equal(f"reader_{view}", h[view], p[view], f"{where} {view}")


def synthetic_episodes(m, t: Tally, args):
    H, C, P = m["H"], m["C"], m["P"]
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    goldens = []
    for scenario in C.SCENARIO_IDS:
        for seed in seeds:
            assignments = [None]
            if seed % 4 == 0:
                assignments += [dict(zip(C.AGENTS, perm))
                                for perm in C.OBJECTIVE_PERMUTATIONS]
            for objectives in assignments:
                for play in ("uniform", "softmax"):
                    if play == "softmax" and objectives is not None:
                        continue
                    h_env = H.ConcordiaPredictiveCommonsV2Env()
                    p_env = C.CommonsEnv()
                    h0 = h_env.reset(seed, H.V2_CONDITION, scenario,
                                     objectives=objectives)
                    p0 = p_env.reset(seed, C.CONDITION, scenario,
                                     objectives=objectives)
                    t.equal("reset_state", h0, p0, f"{scenario} {seed}")
                    rng = random.Random(f"equivalence:{scenario}:{seed}:{play}")
                    model_key = TRIO[seed % 3]
                    readers = objectives is None
                    for turn in range(C.EPISODE_TURNS):
                        state = p_env.serialize_state()
                        if turn <= C.SCORED_TURNS:
                            compare_state_views(
                                m, t, state, model_key=model_key,
                                position_id=f"eq-{scenario}-{seed}-{turn}",
                                readers=readers)
                        if play == "uniform":
                            moves = {a: rng.choice(C.legal_action_ids(a))
                                     for a in C.AGENTS}
                        else:
                            moves = {a: P.softmax_policy(state, a)
                                     for a in C.AGENTS}
                        hr = h_env.step(dict(moves))
                        pr = p_env.step(dict(moves))
                        t.equal("step_events", hr.events, pr.events)
                        t.equal("step_rewards", hr.rewards, pr.rewards)
                        t.equal("step_state", h_env.serialize_state(),
                                p_env.serialize_state(),
                                f"{scenario} {seed} turn {turn}")
                    final = p_env.serialize_state()
                    t.equal("replay_states", H.replay_states(final),
                            C.replay_states(final), f"{scenario} {seed}")
                    t.equal("state_id", h_env.state_id(), p_env.state_id())
    # the in-repo golden grid: uniform play, drawn objectives
    for scenario in C.SCENARIO_IDS:
        for seed in (2700, 2711, 2723, 2734):
            goldens.append(golden_episode(m, scenario, seed))
    return goldens


# Hex prefix length kept in tests/golden. This script itself compares full
# objects; the prefix only keeps the committed fixture small.
GOLDEN_DIGEST_CHARS = 16


def short(value) -> str:
    return digest(value)[:GOLDEN_DIGEST_CHARS]


def golden_episode(m, scenario, seed):
    """Digest prefixes computed from the HARNESS side, for tests/golden."""
    H, HV2 = m["H"], m["HV2"]
    env = H.ConcordiaPredictiveCommonsV2Env()
    env.reset(seed, H.V2_CONDITION, scenario)
    rng = random.Random(f"golden:{scenario}:{seed}")
    model_key = TRIO[seed % 3]
    record = {"scenario": scenario, "seed": seed, "model_key": model_key,
              "turns": []}
    for turn in range(H.V2_EPISODE_TURNS):
        state = env.serialize_state()
        row = {"turn": turn, "state": short(state), "actor": {}, "reader": {}}
        for agent in H.AGENTS:
            row["actor"][agent] = short(HV2.acting_messages(state, agent))
            if turn <= H.V2_SCORED_TURNS:
                views = h_reader_views(m, state, agent, model_key,
                                       f"golden-{scenario}-{seed}-{turn}")
                row["reader"][agent] = {v: short(msgs)
                                        for v, msgs in views.items()}
        moves = {a: rng.choice([x["id"] for x in H.v2_action_space(a)])
                 for a in H.AGENTS}
        row["moves"] = moves
        env.step(moves)
        record["turns"].append(row)
    record["final_state"] = short(env.serialize_state())
    return record


def write_golden_reference(goldens, commit: str) -> None:
    header = {
        "_note": ("Digest prefixes (first 16 hex characters of SHA-256 over "
                  "canonical JSON) computed from the original research "
                  "harness by scripts/check_equivalence.py --write-golden. "
                  "tests/test_golden_reference.py recomputes them with this "
                  "package. One header line, then one episode per line."),
        "harness_commit": commit,
        "digest_chars": GOLDEN_DIGEST_CHARS,
    }
    lines = [json.dumps(header)] + [json.dumps(e, separators=(",", ":"))
                                    for e in goldens]
    path = REPO / "tests" / "golden" / "harness_reference.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def sham_checks(m, t: Tally):
    HS, C = m["HS"], m["C"]
    labels = list(C.SHAM_LABELS)
    displays = [dict(zip(C.AGENTS, labels[i:] + labels[:i])) for i in range(3)]
    for scenario in C.SCENARIO_IDS:
        for seed in range(3000, 3006):
            display = displays[seed % 3]
            h_env = HS.ShamObjectiveCommonsV2()
            p_env = C.ShamCommonsEnv()
            t.equal("sham_reset", h_env.reset(seed, "hidden_objectives_v2",
                                              scenario, objectives=display),
                    p_env.reset(seed, C.CONDITION, scenario,
                                objectives=display))
            rng = random.Random(f"sham:{scenario}:{seed}")
            for turn in range(C.EPISODE_TURNS):
                state = p_env.serialize_state()
                for agent in C.AGENTS:
                    t.equal("sham_observe", h_env.observe(agent),
                            p_env.observe(agent))
                    t.equal("sham_actor_messages",
                            h_sham_actor(m, state, agent, display),
                            m["R"].actor_messages(state, agent,
                                                  sham_display=display))
                moves = {a: rng.choice(C.legal_action_ids(a)) for a in C.AGENTS}
                h_env.step(dict(moves))
                p_env.step(dict(moves))
                t.equal("sham_step_state", h_env.serialize_state(),
                        p_env.serialize_state())
            moves_by_turn = [{a: random.Random(f"{seed}:{i}:{a}").choice(
                C.legal_action_ids(a)) for a in C.AGENTS} for i in range(8)]
            t.equal("sham_assert_inert",
                    HS.assert_inert(seed, "hidden_objectives_v2", scenario,
                                    tuple(C.AGENTS), moves_by_turn),
                    C.assert_inert(seed, C.CONDITION, scenario,
                                   tuple(C.AGENTS), moves_by_turn))


def constant_checks(m, t: Tally):
    H, HS, C, R, HMC, HV2, HFS = (m[k] for k in
                                  ("H", "HS", "C", "R", "HMC", "HV2", "HFS"))
    pairs = {
        "briefing": (H.V2_BRIEFING, C.BRIEFING),
        "goal_text": (H.V2_GOAL_TEXT, C.GOAL_TEXT),
        "objective_menu_text": (H.V2_OBJECTIVE_MENU_TEXT, C.OBJECTIVE_MENU_TEXT),
        "objectives": (list(H.V2_OBJECTIVES), list(C.OBJECTIVES)),
        "permutations": ([list(p) for p in H.OBJECTIVE_PERMUTATIONS],
                         [list(p) for p in C.OBJECTIVE_PERMUTATIONS]),
        "scenarios": (H.V2_SCENARIOS, C.SCENARIOS),
        "public_roles": (H.V2_PUBLIC_ROLES, C.PUBLIC_ROLES),
        "delegation_outcome": (H.DELEGATION_OUTCOME, C.DELEGATION_OUTCOME),
        "share_outcome": (H.SHARE_OUTCOME, C.SHARE_OUTCOME),
        "resolution_actions": (sorted(H.RESOLUTION_ACTIONS),
                               sorted(C.RESOLUTION_ACTIONS)),
        "env_name": (H.ENV_NAME, C.ENV_NAME),
        "history_horizon": (H.ConcordiaPredictiveCommonsV2Env.history_horizon,
                            C.CommonsEnv.history_horizon),
        "geometry": ([H.V2_EPISODE_TURNS, H.V2_SCORED_TURNS,
                      H.V2_ACTION_SUPPORT_SIZE],
                     [C.EPISODE_TURNS, C.SCORED_TURNS, C.ACTION_SUPPORT_SIZE]),
        "sham_briefing": (HS.SHAM_BRIEFING, C.SHAM_BRIEFING),
        "sham_goal_text": (HS.SHAM_GOAL_TEXT, C.SHAM_GOAL_TEXT),
        "sham_labels": (list(HS.SHAM_LABELS), list(C.SHAM_LABELS)),
        "action_system_prompt": (HMC.ACTION_SYSTEM_PROMPT,
                                 R.ACTION_SYSTEM_PROMPT),
        "forecast_system_prompt": (HMC.FORECAST_SYSTEM_PROMPT,
                                   R.FORECAST_SYSTEM_PROMPT),
        "return_schema": (HMC.RETURN_SCHEMA_ACTION_ONLY,
                          R.RETURN_SCHEMA_ACTION_ONLY),
        "objective_menu": (HV2.objective_menu(), R.objective_menu()),
        "objective_system_prompt": (HV2.OBJECTIVE_SYSTEM_PROMPT,
                                    R.OBJECTIVE_SYSTEM_PROMPT),
        "instruction_variants": (HV2.suite()["prompt_variants"]["variants"],
                                 R.INSTRUCTION_VARIANTS),
        "instruction_variant_map": (HFS.variant_assignment(HV2.suite()),
                                    R.instruction_variant_map()),
    }
    for key, (a, b) in pairs.items():
        t.equal(f"constant_{key}", a, b, key)
    for agent in C.AGENTS:
        t.equal("action_space", H.v2_action_space(agent), C.action_space(agent))
    for seed in range(2000):
        for mode in ("permutation", "independent"):
            t.equal("assign_objectives",
                    H.assign_objectives(seed, H.AGENTS, mode),
                    C.assign_objectives(seed, C.AGENTS, mode))


def lookahead_checks(m, t: Tally):
    HP, P = m["HP"], m["P"]
    for scenario in m["C"].SCENARIO_IDS:
        for seed in (2700, 2705):
            h_traj = HP.reference_trajectory(seed, scenario)
            p_traj = P.reference_trajectory(seed, scenario)
            t.equal("reference_trajectory", h_traj, p_traj)
            for state in p_traj:
                for agent in m["C"].AGENTS:
                    t.equal("action_values", HP.action_values(state, agent),
                            P.action_values(state, agent))


# ------------------------------------------------------- recorded episodes

def recorded_sets(harness: Path, tmp: Path):
    """Yield (set label, episode record) from the studied campaigns."""
    for path in sorted((harness / PHASE2_S0).glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        yield "phase2_s0", record.get("result", record)
    for label, rel in STORES:
        source = harness / rel
        target = tmp / label
        target.mkdir(parents=True)
        for item in source.glob("index.sqlite*"):
            shutil.copy2(item, target / item.name)
        con = sqlite3.connect(target / "index.sqlite")
        try:
            rows = con.execute(
                "SELECT cell_id, result_json FROM cells WHERE "
                "status='COMMITTED' AND result_json IS NOT NULL "
                "ORDER BY cell_id").fetchall()
        finally:
            con.close()
        for _cell, text in rows:
            record = json.loads(text)
            if isinstance(record, dict) and "final_state" in record:
                yield label, record


def recorded_checks(m, t: Tally, harness: Path, tmp: Path, readers_every: int):
    H, C, R = m["H"], m["C"], m["R"]
    summary: dict[str, dict] = {}
    example = None
    for label, record in recorded_sets(harness, tmp):
        final = record["final_state"]
        eid = record.get("eid")
        row = summary.setdefault(label, {"episodes": 0, "sham": 0,
                                         "endpoint_equal": 0,
                                         "actions_match_log": 0,
                                         "scenarios": {}})
        row["episodes"] += 1
        row["scenarios"][final["scenario_id"]] = (
            row["scenarios"].get(final["scenario_id"], 0) + 1)
        h_states = H.replay_states(final)
        p_states = C.replay_states(final)
        t.equal("recorded_replay_states", h_states, p_states, f"{label} {eid}")
        endpoint_ok = canonical(p_states[max(p_states)]) == canonical(final)
        t.check("recorded_endpoint", endpoint_ok, f"{label} {eid}")
        row["endpoint_equal"] += int(endpoint_ok)
        logged = [(a["turn"], a["agent"], a["action"])
                  for a in record.get("actions", [])]
        replayed = [(e["turn"], e["actor"], e["action"])
                    for e in final["events"]
                    if C.is_applied_action(e["actor"], e["action"])]
        replayed = replayed[2:]  # the two scenario opening events
        actions_ok = logged == replayed
        t.check("recorded_actions_match_log", actions_ok, f"{label} {eid}")
        row["actions_match_log"] += int(actions_ok)
        display = None
        if record.get("arm") == "sham":
            display = record["rendered_objectives"]
            row["sham"] += 1
        seats = record.get("seats") or {}
        for turn, state in sorted(p_states.items()):
            if turn >= int(final["turn"]):
                continue
            readers = (display is None and turn < C.SCORED_TURNS
                       and turn % readers_every == 0)
            keys = {}
            for agent in C.AGENTS:
                key = seats.get(agent, TRIO[0])
                try:
                    R.instruction_variant_for(key, agent, final["scenario_id"])
                except KeyError:
                    key = TRIO[0]
                keys[agent] = key
            compare_state_views(
                m, t, state, model_key=keys,
                position_id=f"{label}-{eid}-{turn}", readers=readers,
                sham_display=display)
        if eid == "p3xsrc000":
            state = p_states[1]
            shared = state["shared"]
            example = {
                "episode": eid, "set": label, "turn": 1,
                "mover": "agent_1",
                "public_role": state["agents"]["agent_1"]["public_role"],
                "supplies": shared["supplies"],
                "network_health": shared["network_health"],
                "printer_health": shared["printer_health"],
                "kitchen_cleanliness": shared["kitchen_cleanliness"],
                "open_commitments": len([c for c in shared["open_commitments"]
                                         if c["status"] == "open"]),
                "inspection_evidence": shared["inspection_evidence"],
                "recorded_action": next(
                    (a["action"] for a in record["actions"]
                     if a["turn"] == 1 and a["agent"] == "agent_1"), None),
            }
    return summary, example


FIXTURES = (
    # (set label, episode id, fixture file name)
    ("phase2_s0", "ep000", "recorded_phase2_s0_ep000.json"),
    ("phase3_s5a", "p3xsrc000", "recorded_phase3_s5a_p3xsrc000.json"),
    ("phase3_s0b", None, "recorded_phase3_s0b_first_sham.json"),
)


def export_fixtures(m, harness: Path, tmp: Path) -> list[str]:
    """Write three recorded episodes to tests/fixtures, each with prompt
    digests computed by the HARNESS implementation."""
    H, HV2 = m["H"], m["HV2"]
    wanted = {(label, eid): name for label, eid, name in FIXTURES}
    written = []
    for label, record in recorded_sets(harness, tmp):
        eid = record.get("eid")
        name = wanted.get((label, eid))
        if name is None and (label, None) in wanted and record.get("arm") == "sham":
            name = wanted.pop((label, None))
        if name is None:
            continue
        wanted.pop((label, eid), None)
        final = record["final_state"]
        display = (record.get("rendered_objectives")
                   if record.get("arm") == "sham" else None)
        seats = record.get("seats") or {}
        states = H.replay_states(final)
        actor, reader = {}, {}
        for turn in range(int(final["turn"])):
            state = states[turn]
            actor[str(turn)] = {}
            for agent in H.AGENTS:
                msgs = (h_sham_actor(m, state, agent, display) if display
                        else HV2.acting_messages(state, agent))
                actor[str(turn)][agent] = digest(msgs)
            if display is None and turn < H.V2_SCORED_TURNS:
                reader[str(turn)] = {}
                for agent in H.AGENTS:
                    views = h_reader_views(
                        m, state, agent, seats.get(agent, TRIO[0]),
                        f"{label}-{eid}-{turn}")
                    reader[str(turn)][agent] = {v: digest(x)
                                                for v, x in views.items()}
        fixture = {
            "_note": ("A recorded Commons episode from the studied campaigns. "
                      "final_state is the recorded terminal state; actions is "
                      "the recorded per-seat action log; harness_digests are "
                      "SHA-256 digests of canonical JSON of the actor and "
                      "reader messages, computed by the original harness. "
                      "Model reasoning traces and raw replies are not "
                      "included."),
            "set": label,
            "eid": eid,
            "arm": record.get("arm", "active"),
            "seed": final["seed"],
            "scenario_id": final["scenario_id"],
            "objectives": {a: s["hidden_objective"]
                           for a, s in final["agents"].items()},
            "sham_display": display,
            "seats": seats,
            "actions": [{k: a[k] for k in ("turn", "agent", "action")}
                        for a in record.get("actions", [])],
            "final_state": final,
            "harness_digests": {"actor": actor, "reader": reader,
                                "reader_seed_context": {
                                    "campaign": "commons_v2_campaign",
                                    "snapshot_id": f"{label}-{eid}-<turn>",
                                    "forecast_suite_version": SUITE_VERSION}},
        }
        path = REPO / "tests" / "fixtures" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fixture, indent=1, ensure_ascii=False) + "\n",
                        encoding="utf-8", newline="\n")
        written.append(name)
        if not wanted:
            break
    return written


GOLDEN_TEXT = (
    # (file name, scenario, seed, turn, agent, view); play is uniform random
    # from random.Random(f"golden:{scenario}:{seed}"), as in golden_episode.
    ("actor_delivery_pressure_s2700_t0_agent_1.txt", "delivery_pressure",
     2700, 0, "agent_1", "actor"),
    ("reader_ordered_delivery_pressure_s2700_t3_agent_2.txt",
     "delivery_pressure", 2700, 3, "agent_2", "ordered"),
    ("reader_tally_delivery_pressure_s2700_t3_agent_2.txt",
     "delivery_pressure", 2700, 3, "agent_2", "tally"),
    ("goal_ordered_delivery_pressure_s2700_t3_agent_2.txt",
     "delivery_pressure", 2700, 3, "agent_2", "goal-ordered"),
)


def write_golden_text(m) -> list[str]:
    """Full prompt texts rendered by the HARNESS, for byte-exact tests."""
    H, HV2 = m["H"], m["HV2"]
    written = []
    for name, scenario, seed, turn, agent, view in GOLDEN_TEXT:
        env = H.ConcordiaPredictiveCommonsV2Env()
        env.reset(seed, H.V2_CONDITION, scenario)
        rng = random.Random(f"golden:{scenario}:{seed}")
        for _ in range(turn):
            env.step({a: rng.choice([x["id"] for x in H.v2_action_space(a)])
                      for a in H.AGENTS})
        state = env.serialize_state()
        if view == "actor":
            messages = HV2.acting_messages(state, agent)
        else:
            messages = h_reader_views(m, state, agent, TRIO[seed % 3],
                                      f"golden-{scenario}-{seed}-{turn}")[view]
        text = "".join(f"===== {msg['role']} =====\n{msg['content']}\n\n"
                       for msg in messages)
        path = REPO / "tests" / "golden" / name
        path.write_bytes(text.encode("utf-8"))
        written.append(name)
    return written


def git_info(harness: Path) -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", str(harness), *cmd],
                              capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", *SOURCE_FILES)
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "source_files": list(SOURCE_FILES),
            "source_files_differ_from_head": dirty.splitlines()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--harness", required=True,
                    help="path to the research harness checkout (read only)")
    ap.add_argument("--n-seeds", type=int, default=12)
    ap.add_argument("--seed-start", type=int, default=2700)
    ap.add_argument("--readers-every", type=int, default=1,
                    help="compare reader prompts at every k-th scored turn "
                         "of recorded episodes")
    ap.add_argument("--report", default=None)
    ap.add_argument("--write-golden", action="store_true")
    ap.add_argument("--golden-text-only", action="store_true",
                    help="only (re)write the tests/golden/*.txt prompt texts")
    ap.add_argument("--fixtures-only", action="store_true",
                    help="only (re)write tests/fixtures/recorded_*.json")
    args = ap.parse_args()

    harness = Path(args.harness).resolve()
    started = time.time()
    m = load(harness)
    if args.golden_text_only:
        print(write_golden_text(m))
        return 0
    if args.fixtures_only:
        tmp = Path(tempfile.mkdtemp(prefix="commons_fixtures_"))
        try:
            print(export_fixtures(m, harness, tmp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        return 0
    t = Tally()
    constant_checks(m, t)
    goldens = synthetic_episodes(m, t, args)
    sham_checks(m, t)
    lookahead_checks(m, t)
    tmp = Path(tempfile.mkdtemp(prefix="commons_equivalence_"))
    try:
        recorded, example = recorded_checks(m, t, harness, tmp,
                                            args.readers_every)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "tool": "scripts/check_equivalence.py",
        "harness": git_info(harness),
        "python": sys.version.split()[0],
        "comparisons": dict(sorted(t.counts.items())),
        "total_comparisons": sum(t.counts.values()),
        "failures": len(t.failures),
        "failure_examples": t.failures[:20],
        "recorded_episodes": recorded,
        "recorded_episode_total": sum(r["episodes"] for r in recorded.values()),
        "paper_example_reconstruction": example,
        "runtime_s": round(time.time() - started, 1),
    }
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8", newline="\n")
    if args.write_golden:
        write_golden_reference(goldens, report["harness"]["commit"])
    print(json.dumps({k: report[k] for k in (
        "total_comparisons", "failures", "recorded_episode_total",
        "runtime_s")}))
    for failure in t.failures[:20]:
        print("FAIL", failure)
    return 1 if t.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
