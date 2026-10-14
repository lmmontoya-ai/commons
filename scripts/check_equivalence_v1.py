"""Check that ``commons.v1`` reproduces the original research harness exactly.

This is a provenance tool for maintainers, not part of the test suite. It
needs a checkout of the private research harness that Commons v1 was
extracted from (repository ``self-other-prediction-smoke-tests``), including
its recorded data under ``data/outputs/predictive_commons_m4_3`` and the
tracked manifests under ``docs/``. The harness script ``commons_m44.py``
also imports a checkout of the public LEDGER repository from the path
written in that script. Run it with an interpreter that has the harness's
own dependencies installed (the harness's virtual environment), for example:

    <harness>/.venv/Scripts/python.exe -B scripts/check_equivalence_v1.py \
        --harness <harness checkout> \
        --report docs/provenance/v1_equivalence_report.json

``--harness`` may also be given through the environment variable
``COMMONS_HARNESS``. There is no default. The script stops with a clear
message if the harness or any recorded input is missing.

The harness is only read. Bytecode writing is disabled, and the recorded run
store is copied to a temporary directory before it is opened.

What is compared:

1. Constants: briefing, scenarios, roles, goals, menus, acknowledgement
   texts, system prompts, return schema, instruction variants.
2. Seeded synthetic trajectories (all scenarios, uniform and supply-heavy
   play, acknowledgements on and off): reset states, step events, rewards
   and states, ``state_id``, every observation, actor prompts (JSON and
   markdown) for every seat, reader prompts for every view, format and
   instruction variant, and replay from the final state. Harness versus
   this package.
3. The 90 trajectories of the M4.3 turn-8 bank, rebuilt from the frozen
   inputs in ``src/commons/v1/data``: the frozen inputs equal what the
   recorded snapshots contain; the rebuilt turn-8 state equals the recorded
   state (``state_sha256``, ``state_id``); every intermediate state equals
   the harness replay; the bank's three reader observations match
   ``snapshot_manifest.json``; the suite's three reader observations match
   ``prompt_fixture_manifest.json``; actor prompts for every seat and turn
   and the target's reader prompts at every turn equal the harness.
4. The 48 M4.4 positions: the position list and states equal the harness
   sampler; prompts equal the harness; prompt digests equal every recorded
   outbound message in the M4.4 run store (960 cells), the nine-forecaster
   panel manifest (864 cells) and the perspective-study manifest (480 cells;
   its "imagine" arm is the full-record prompt plus the study's perspective
   paragraph, read from the harness script); the source report files hash
   to the values those manifests pin.

``--write-data`` regenerates the frozen inputs from the harness records.
``--write-golden`` writes full prompt texts rendered by the HARNESS to
``tests/golden/v1_*.txt``.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse  # noqa: E402
import ast  # noqa: E402
import contextlib  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import shutil  # noqa: E402
import sqlite3  # noqa: E402
import subprocess  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "src" / "commons" / "v1" / "data"
GOLDEN = REPO / "tests" / "golden"

# Harness inputs, relative to the harness root.
SOURCE_FILES = (
    "src/self_other_prediction/envs/social_sim.py",
    "src/self_other_prediction/envs/base.py",
    "src/self_other_prediction/events.py",
    "src/self_other_prediction/prompt_templates.py",
    "src/self_other_prediction/prompt_render_md.py",
    "src/self_other_prediction/model_clients.py",
    "src/self_other_prediction/arms.py",
    "src/self_other_prediction/m4_forecast_suite.py",
    "scripts/commons_m44.py",
    "scripts/ordinary_prediction_panel.py",
    "scripts/self_simulation_prediction.py",
    "config/predictive_commons_m4_3_forecast_suite_m4_3_1.json",
)
M43 = "data/outputs/predictive_commons_m4_3"
SNAPSHOT_MANIFEST = f"{M43}/snapshot_manifest.json"
FIXTURE_MANIFEST = f"{M43}/forecast_suites/m4.3-fs1/prompt_fixture_manifest.json"
BANK_DIR = f"{M43}/workspace/data/bank/concordia_predictive_commons"
M44_DIR = f"{M43}/m44"
M44_STORE = f"{M44_DIR}/_store"
PANEL_MANIFEST = "docs/ordinary_prediction_panel/manifests/commons.json"
PERSPECTIVE_MANIFEST = (
    "docs/self_simulation_prediction/manifests/ordinary/manifests/commons.json")
PERSPECTIVE_SCRIPT = "scripts/self_simulation_prediction.py"
REQUIRED = (SNAPSHOT_MANIFEST, FIXTURE_MANIFEST, BANK_DIR, M44_STORE,
            f"{M44_DIR}/m44_report.json", f"{M44_DIR}/targets_summary.json",
            PANEL_MANIFEST, PERSPECTIVE_MANIFEST, *SOURCE_FILES)

VIEWS = ("board", "ordered", "shuffled")
HARNESS_VIEW = {"board": "board_only", "ordered": "ordered",
                "shuffled": "shuffled"}
HARNESS_ARM = {"board": "state_only", "ordered": "full_history",
               "shuffled": "shuffled_history"}
PANEL_VIEW = {"situation": "board", "full_record": "ordered"}
FEEDBACK = "action_feedback"


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def require(harness: Path) -> None:
    missing = [rel for rel in REQUIRED if not (harness / rel).exists()]
    if missing:
        raise SystemExit(
            "this check needs the private research harness checkout and its "
            f"recorded data; missing under {harness}:\n  "
            + "\n  ".join(missing))


def load(harness: Path) -> dict:
    sys.path.insert(0, str(harness / "scripts"))
    sys.path.insert(0, str(harness / "src"))
    sys.path.insert(0, str(REPO / "src"))
    with contextlib.redirect_stdout(io.StringIO()):
        import commons_m44 as HM
        from self_other_prediction import arms as HA
        from self_other_prediction import m4_forecast_suite as HFS
        from self_other_prediction import model_clients as HMC
        from self_other_prediction import prompt_templates as HPT
        from self_other_prediction.envs import social_sim as HE
    import commons.render as R
    import commons.v1 as V
    return dict(HM=HM, HA=HA, HFS=HFS, HMC=HMC, HPT=HPT, HE=HE, R=R, V=V)


def perspective_paragraph(harness: Path) -> str:
    """The perspective study's IMAGINE constant, read without importing the
    study script (it pulls in run infrastructure)."""
    tree = ast.parse((harness / PERSPECTIVE_SCRIPT).read_text(encoding="utf-8"))
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and getattr(node.targets[0], "id", None) == "IMAGINE"):
            return ast.literal_eval(node.value)
    raise SystemExit(f"IMAGINE not found in {PERSPECTIVE_SCRIPT}")


def append_paragraph(messages, paragraph: str):
    """The perspective study's ``_append``."""
    out = json.loads(json.dumps(messages))
    out[-1]["content"] = out[-1]["content"].rstrip() + "\n\n" + paragraph + "\n"
    return out


# ------------------------------------------------------------ harness side

def h_env(m, state):
    env = m["HE"].ConcordiaPredictiveCommonsEnv()
    env.restore(state)
    return env


def h_actor(m, state, agent, prompt_format="json"):
    env = h_env(m, state)
    env.set_probe_target(agent)
    return m["HMC"].build_action_messages(
        env.observe(agent), env.action_space(agent), action_only=True,
        prompt_format=prompt_format)


def h_reader_observations(m, state, target, variant, seed_context):
    """Harness reader observations, built the way the forecast suite does."""
    HA = m["HA"]
    env = h_env(m, state)
    env.set_probe_target(target)
    context = HA.ProbeContext(
        env=env, target_id=target, predictor_role="observer_only",
        predictor_agent_id=None, probe_visibility="public",
        condition=str(state["condition"]), shuffle_seed_context=seed_context)
    instruction = m["HPT"].M4_1_INSTRUCTION_VARIANTS[variant]
    full = HA.prediction_observation(context)
    shuffled = HA.shuffled_history_observation(context)
    env.history_horizon = 0
    try:
        board = HA.prediction_observation(context)
    finally:
        env.history_horizon = 24
    out = {}
    for view, obs in (("board", board), ("ordered", full),
                      ("shuffled", shuffled)):
        obs = json.loads(json.dumps(obs))
        obs["instruction"] = instruction
        out[view] = obs
    return out


def h_reader(m, state, target, variant, seed_context, prompt_format="json"):
    menu = h_env(m, state).action_space(target)
    return {view: m["HMC"].build_forecast_messages(
                obs, menu, action_only=True, prompt_format=prompt_format)
            for view, obs in h_reader_observations(
                m, state, target, variant, seed_context).items()}


def h_replay(m, state, snapshot_id="eq", action_feedback=True):
    snap = {"env_name": "concordia_predictive_commons", "seed": state["seed"],
            "condition": state["condition"], "scenario_id": state["scenario_id"],
            "rollout": {"action_feedback": action_feedback}, "state": state,
            "turn": state["turn"], "snapshot_id": snapshot_id}
    with contextlib.redirect_stdout(io.StringIO()):
        states = m["HM"].replay_states(snap)
    return [states[t] for t in range(int(state["turn"]) + 1)]


# ---------------------------------------------------------- comparisons

def constant_checks(m, t: Tally) -> None:
    HE, HMC, HPT, HFS, V, R = (m["HE"], m["HMC"], m["HPT"], m["HFS"], m["V"],
                               m["R"])
    env = HE.ConcordiaPredictiveCommonsEnv()
    t.equal("constant_env_name", [env.name, env.world, env.env_version],
            [V.ENV_NAME, V.WORLD, V.ENV_VERSION])
    t.equal("constant_briefing", HE.WORLD_BRIEFINGS["concordia"], V.BRIEFING)
    t.equal("constant_history_horizon", env.history_horizon,
            V.CommonsV1Env.history_horizon)
    t.equal("constant_scenarios",
            {k: {"shared": v["shared"], "events": [list(e) for e in v["events"]]}
             for k, v in HE.V8_SCENARIOS.items()},
            {k: {"shared": v["shared"], "events": [list(e) for e in v["events"]]}
             for k, v in V.SCENARIOS.items()})
    t.equal("constant_scenario_ids", list(HE.V8_SCENARIO_IDS), list(V.SCENARIO_IDS))
    t.equal("constant_private_goals", HE.V8_PRIVATE_GOALS, V.PRIVATE_GOALS)
    t.equal("constant_public_roles",
            {a: p["public_role"] for a, p in HE.CONCORDIA_PROFILES.items()},
            V.PUBLIC_ROLES)
    t.equal("constant_fixed_actions", [list(a) for a in HE.V8_FIXED_ACTIONS],
            [list(a) for a in V.FIXED_ACTIONS])
    env.reset(1600, "tradeoff_v8", "delivery_pressure")
    for agent in V.AGENTS:
        t.equal("constant_action_space", env.action_space(agent),
                V.action_space(agent), agent)
    families = sorted({V.action_family(a) for a in V.legal_action_ids("agent_1")})
    t.equal("constant_feedback", {f: env._action_feedback(a) for f, a in (
        (V.action_family(a), a) for a in V.legal_action_ids("agent_1"))},
            {f: V.ACTION_FEEDBACK[f] for f in families})
    t.equal("constant_action_system_prompt", HPT.ACTION_SYSTEM_PROMPT,
            m["R"].ACTION_SYSTEM_PROMPT)
    t.equal("constant_forecast_system_prompt", HPT.FORECAST_SYSTEM_PROMPT,
            m["R"].FORECAST_SYSTEM_PROMPT)
    t.equal("constant_return_schema", HMC.RETURN_SCHEMA_ACTION_ONLY,
            m["R"].RETURN_SCHEMA_ACTION_ONLY)
    t.equal("constant_protocol_version", HMC.PROMPT_PROTOCOL_VERSION,
            V.PROMPT_PROTOCOL_VERSION)
    t.equal("constant_perspective", HPT.NEUTRAL_FORECAST_PERSPECTIVE,
            m["R"].NEUTRAL_FORECAST_PERSPECTIVE)
    t.equal("constant_instruction_variants", HPT.M4_1_INSTRUCTION_VARIANTS,
            R.INSTRUCTION_VARIANTS)
    suite = m["HM"].suite()
    t.equal("constant_suite_variants", suite["prompt_variants"]["variants"],
            R.INSTRUCTION_VARIANTS)
    t.equal("constant_suite_version", suite["suite"]["forecast_suite_version"],
            V.FORECAST_SUITE_VERSION)
    t.equal("constant_variant_map", HFS.variant_assignment(suite),
            R.instruction_variant_map())


def compare_state(m, t: Tally, state, *, variant: str, seed_context,
                  formats=("json", "markdown")) -> None:
    """Observations and every prompt at one state, for every seat."""
    V = m["V"]
    h = h_env(m, state)
    p = V.CommonsV1Env()
    p.restore(state)
    t.equal("public_state", h.public_state(), p.public_state())
    where = f"seed={state['seed']} turn={state['turn']}"
    for agent in V.AGENTS:
        h.set_probe_target(agent)
        t.equal("observe", h.observe(agent), p.observe(agent), f"{where} {agent}")
        t.equal("public_observation", h.public_observation(agent),
                p.public_observation(agent), f"{where} {agent}")
        for fmt in formats:
            t.equal(f"actor_messages_{fmt}", h_actor(m, state, agent, fmt),
                    V.actor_messages(state, agent, prompt_format=fmt),
                    f"{where} {agent}")
            hr = h_reader(m, state, agent, variant, seed_context, fmt)
            for view in VIEWS:
                t.equal(f"reader_{view}_{fmt}", hr[view],
                        V.reader_messages(state, agent, view,
                                          instruction_variant=variant,
                                          shuffle_seed_context=seed_context,
                                          prompt_format=fmt),
                        f"{where} {agent} {view}")


def synthetic_episodes(m, t: Tally, args) -> None:
    HE, V = m["HE"], m["V"]
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    variants = ("V1", "V2", "V3")
    for scenario in V.SCENARIO_IDS:
        for seed in seeds:
            for play in ("uniform", "take"):
                for feedback in (True, False):
                    if not feedback and seed % 3:
                        continue
                    h = HE.ConcordiaPredictiveCommonsEnv()
                    h.action_feedback = feedback
                    p = V.CommonsV1Env(action_feedback=feedback)
                    t.equal("reset_state", h.reset(seed, "tradeoff_v8", scenario),
                            p.reset(seed, "tradeoff_v8", scenario),
                            f"{scenario} {seed}")
                    rng = random.Random(f"v1-equivalence:{scenario}:{seed}:{play}")
                    variant = variants[seed % 3]
                    for turn in range(V.EPISODE_TURNS + 1):
                        state = p.serialize_state()
                        t.equal("state_id", h.state_id(), p.state_id())
                        if feedback:
                            compare_state(
                                m, t, state, variant=variant,
                                seed_context={"campaign": "v1-equivalence",
                                              "snapshot_id": f"{scenario}-{seed}-{turn}",
                                              "forecast_suite_version": "m4.3-fs1"})
                        if turn == V.EPISODE_TURNS:
                            break
                        moves = {}
                        for agent in V.AGENTS:
                            menu = V.legal_action_ids(agent)
                            if play == "take" and rng.random() < 0.6:
                                moves[agent] = rng.choice(("TAKE_3", "TAKE_1"))
                            else:
                                moves[agent] = rng.choice(menu)
                        hr = h.step(dict(moves))
                        pr = p.step(dict(moves))
                        t.equal("step_events", hr.events, pr.events)
                        t.equal("step_rewards", hr.rewards, pr.rewards)
                        t.equal("step_state", h.serialize_state(), p.serialize_state(),
                                f"{scenario} {seed} {play} turn {turn}")
                    final = p.serialize_state()
                    t.equal("replay_states", h_replay(m, final, action_feedback=feedback),
                            V.replay_states(final), f"{scenario} {seed} {play}")


# ------------------------------------------------------------- recorded data

def bank_snapshots(harness: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((harness / SNAPSHOT_MANIFEST).read_text(encoding="utf-8"))
    snaps = []
    for row in manifest["rows"]:
        path = harness / BANK_DIR / row["snapshot_id"] / "snapshot.json"
        snaps.append(json.loads(path.read_text(encoding="utf-8")))
    return manifest, snaps


def recorded_actions(m, snap) -> list[list[str]]:
    """Seat-ordered action labels per turn, read from a recorded snapshot."""
    HE = m["HE"]
    per_turn: dict[int, list[tuple[str, str]]] = {}
    for event in snap["state"]["events"]:
        if event["action"] == FEEDBACK:
            continue
        per_turn.setdefault(int(event["turn"]), []).append(
            (event["actor"], event["action"]))
    opening = [(a, x) for a, x, _ in HE.V8_SCENARIOS[snap["scenario_id"]]["events"]]
    if per_turn[0][:2] != opening:
        raise SystemExit(f"{snap['snapshot_id']}: unexpected opening events")
    per_turn[0] = per_turn[0][2:]
    out = []
    for turn in range(int(snap["turn"])):
        pairs = per_turn.get(turn, [])
        if [a for a, _ in pairs] != ["agent_1", "agent_2", "agent_3"]:
            raise SystemExit(f"{snap['snapshot_id']}: turn {turn} is not in "
                             "seat order; the frozen format cannot hold it")
        out.append([x for _, x in pairs])
    return out


def open_store(harness: Path, tmp: Path) -> list[dict]:
    """Every committed cell of the M4.4 run store with the digest of its
    recorded outbound messages. The store is copied before it is opened."""
    source = harness / M44_STORE
    target = tmp / "m44_store"
    target.mkdir(parents=True)
    for item in source.iterdir():
        if item.is_file():
            shutil.copy2(item, target / item.name)
    pack = (target / "blobs.pack").read_bytes()
    index = json.loads((target / "blobs_pack.json").read_text(encoding="utf-8"))
    con = sqlite3.connect(target / "index.sqlite")
    try:
        rows = con.execute(
            "SELECT c.cell_id, c.result_json, c.blob_hash, b.codec FROM cells c "
            "JOIN blobs b ON b.blob_hash = c.blob_hash "
            "WHERE c.status='COMMITTED' ORDER BY c.cell_id").fetchall()
    finally:
        con.close()
    zstd = None
    cells = []
    for cell_id, result_json, blob_hash, codec in rows:
        offset, length = index[blob_hash]
        data = pack[offset:offset + length]
        if codec == "zstd":
            if zstd is None:
                import zstandard
                zstd = zstandard.ZstdDecompressor()
            raw = zstd.decompress(data)
        else:
            raw = gzip.decompress(data)
        if hashlib.sha256(raw).hexdigest() != blob_hash:
            raise SystemExit(f"store blob {blob_hash} failed its hash check")
        blob = json.loads(raw)
        result = json.loads(result_json)
        cells.append({"cell_id": cell_id, "position_id": result["position_id"],
                      "kind": "forecast" if "view" in result else "target",
                      "view": result.get("view"),
                      "rendering": result.get("rendering", "json"),
                      "prompt_sha256": hashlib.sha256(
                          canonical(blob["messages"]).encode("utf-8")).hexdigest()})
    return cells


def recorded_position_digests(harness: Path, store: list[dict]
                              ) -> tuple[dict, dict]:
    """Per position: the recorded prompt digests, one value per key, from
    the run store and the panel manifest. Raises if cells disagree."""
    panel = json.loads((harness / PANEL_MANIFEST).read_text(encoding="utf-8"))
    out: dict[str, dict] = {}

    def put(pid, fmt, key, value, source):
        slot = out.setdefault(pid, {}).setdefault(fmt, {})
        if key in slot and slot[key] != value:
            raise SystemExit(f"{pid} {fmt} {key}: {source} disagrees")
        slot[key] = value

    for cell in store:
        if cell["kind"] == "target":
            put(cell["position_id"], "json", "actor", cell["prompt_sha256"], "store")
        else:
            view = {"board_only": "board", "ordered": "ordered",
                    "shuffled": "shuffled"}[cell["view"]]
            put(cell["position_id"], cell["rendering"], view,
                cell["prompt_sha256"], "store")
    for cell in panel["cells"]:
        put(cell["position_id"], "json", PANEL_VIEW[cell["view"]],
            cell["prompt_sha256"], "panel manifest")
    counts = {"store_cells": len(store), "panel_cells": len(panel["cells"])}
    return out, counts


def bank_checks(m, t: Tally, harness: Path) -> dict:
    V, HFS, HM = m["V"], m["HFS"], m["HM"]
    manifest, snaps = bank_snapshots(harness)
    fixtures = {row["snapshot_id"]: row for row in json.loads(
        (harness / FIXTURE_MANIFEST).read_text(encoding="utf-8"))["rows"]}
    rows = {row["snapshot_id"]: row for row in V.trajectories()}
    suite = HM.suite()
    t.equal("bank_ids", sorted(rows), sorted(r["snapshot_id"] for r in manifest["rows"]))
    for mrow, snap in zip(manifest["rows"], snaps):
        sid = snap["snapshot_id"]
        row = rows[sid]
        t.check("bank_snapshot_sha256", HFS.sha256_json(snap) == mrow["snapshot_sha256"], sid)
        t.check("bank_record_state_sha256",
                HFS.sha256_json(snap["state"]) == mrow["state_sha256"], sid)
        t.equal("bank_frozen_inputs", {
            "campaign_id": snap["campaign_id"], "seed": snap["seed"],
            "scenario_id": snap["scenario_id"], "target_id": snap["probe_target_id"],
            "instruction_variant": HFS.variant_for_snapshot(snap, suite),
            "actions": recorded_actions(m, snap), "state_sha256": mrow["state_sha256"],
            "state_id": snap["state_id"], "snapshot_sha256": mrow["snapshot_sha256"],
            "observation_sha256_by_arm": mrow["observation_sha256_by_arm"],
            "suite_observation_sha256_by_arm": {
                arm: fixtures[sid]["observation_sha256_by_arm"][arm]
                for arm in HARNESS_ARM.values()},
        }, {k: row[k] for k in (
            "campaign_id", "seed", "scenario_id", "target_id",
            "instruction_variant", "actions", "state_sha256", "state_id",
            "snapshot_sha256", "observation_sha256_by_arm",
            "suite_observation_sha256_by_arm")}, sid)
        t.check("bank_variant_public_map", m["R"].instruction_variant_for(
            snap["target_model_key"], snap["probe_target_id"], snap["scenario_id"])
            == row["instruction_variant"], sid)
        t.check("bank_fixture_variant", fixtures[sid]["variant"]
                == row["instruction_variant"], sid)
        t.equal("bank_action_space", snap["action_space"],
                V.action_space(row["target_id"]), sid)
        states = V.trajectory_states(row)
        t.equal("bank_turn8_state", states[-1], snap["state"], sid)
        t.check("bank_state_sha256", V.state_sha256(states[-1]) == mrow["state_sha256"], sid)
        t.check("bank_state_id", V.state_id(states[-1]) == snap["state_id"], sid)
        with contextlib.redirect_stdout(io.StringIO()):
            hstates = HM.replay_states(snap)
        for turn, state in enumerate(states):
            t.equal("bank_state_vs_harness_replay", hstates[turn], state, f"{sid} t{turn}")
        bank_obs = V.bank_observations(row)
        for arm, obs in bank_obs.items():
            t.check("bank_observation_vs_manifest",
                    V.observation_sha256(obs) == mrow["observation_sha256_by_arm"][arm],
                    f"{sid} {arm}")
        hctx = {"campaign": manifest["campaign_id"], "snapshot_id": sid}
        hobs = h_reader_observations(m, snap["state"], row["target_id"], "V1", hctx)
        for view, arm in HARNESS_ARM.items():
            t.equal("bank_observation_vs_harness", hobs[view], bank_obs[arm],
                    f"{sid} {arm}")
        hsuite = HFS.observations_for_snapshot(snap, suite)
        for arm, obs in V.suite_observations(row).items():
            t.check("suite_observation_vs_fixture_manifest",
                    V.observation_sha256(obs)
                    == fixtures[sid]["observation_sha256_by_arm"][arm], f"{sid} {arm}")
            t.equal("suite_observation_vs_harness", hsuite[arm], obs, f"{sid} {arm}")
        # Every turn of the trajectory: actor prompts for every seat, and the
        # target's reader and acting prompts as the M4.4 path renders them.
        traj = {"snap": snap, "states": hstates}
        for turn, state in enumerate(states):
            for agent in V.AGENTS:
                t.equal("bank_actor_messages", h_actor(m, state, agent),
                        V.actor_messages(state, agent), f"{sid} t{turn} {agent}")
            hpos = HM.make_position(traj, turn)
            obs = HM.observations(hpos)
            for view in VIEWS:
                t.equal("bank_reader_messages", HM.forecast_messages(
                    hpos, HARNESS_VIEW[view], "json", obs),
                    V.reader_messages(state, row["target_id"], view,
                                      instruction_variant=row["instruction_variant"],
                                      shuffle_seed_context=V.suite_seed_context(
                                          row, hpos["position_id"])),
                    f"{sid} t{turn} {view}")
    return {"trajectories": len(snaps), "states": len(snaps) * 9,
            "snapshot_manifest_sha256": sha256_file(harness / SNAPSHOT_MANIFEST),
            "prompt_fixture_manifest_sha256": sha256_file(harness / FIXTURE_MANIFEST)}


def position_checks(m, t: Tally, harness: Path, tmp: Path) -> dict:
    V, HM = m["V"], m["HM"]
    with contextlib.redirect_stdout(io.StringIO()):
        hpositions = HM._positions()
    ports = {p["position_id"]: p for p in V.positions()}
    t.equal("position_list", [
        {"position_id": p["position_id"], "trajectory_id": p["trajectory_id"],
         "turn": p["turn"], "target_id": p["target_agent"]} for p in hpositions],
        [{k: p[k] for k in ("position_id", "trajectory_id", "turn", "target_id")}
         for p in V.positions()])
    store = open_store(harness, tmp)
    recorded, counts = recorded_position_digests(harness, store)
    paragraph = perspective_paragraph(harness)
    for hp in hpositions:
        pid = hp["position_id"]
        pos = ports[pid]
        state = V.position_state(pos)
        t.equal("position_state", hp["snap"]["state"], state, pid)
        t.check("position_state_sha256",
                V.state_sha256(state) == pos["state_sha256"], pid)
        t.equal("position_legal_actions", HM.legal_actions(hp), V.legal_actions(pos), pid)
        t.equal("position_frozen_digests", recorded[pid], pos["prompt_sha256"], pid)
        obs = HM.observations(hp)
        for fmt in ("json", "markdown"):
            for view in VIEWS:
                messages = V.position_reader_messages(pos, view, prompt_format=fmt)
                t.equal(f"position_reader_{fmt}_vs_harness",
                        HM.forecast_messages(hp, HARNESS_VIEW[view], fmt, obs),
                        messages, f"{pid} {view}")
                t.check(f"position_reader_{fmt}_vs_recorded",
                        V.messages_sha256(messages) == recorded[pid][fmt][view],
                        f"{pid} {view}")
        actor = V.position_actor_messages(pos)
        t.equal("position_actor_vs_harness", HM.acting_messages(hp), actor, pid)
        t.check("position_actor_vs_recorded",
                V.messages_sha256(actor) == recorded[pid]["json"]["actor"], pid)

    # Cell-level comparisons against every recorded digest.
    for cell in store:
        pos = ports[cell["position_id"]]
        if cell["kind"] == "target":
            messages = V.position_actor_messages(pos)
        else:
            messages = V.position_reader_messages(
                pos, cell["view"], prompt_format=cell["rendering"])
        t.check("store_cell_prompt", V.messages_sha256(messages)
                == cell["prompt_sha256"], cell["cell_id"])
    panel = json.loads((harness / PANEL_MANIFEST).read_text(encoding="utf-8"))
    for cell in panel["cells"]:
        messages = V.position_reader_messages(ports[cell["position_id"]],
                                              PANEL_VIEW[cell["view"]])
        t.check("panel_cell_prompt", V.messages_sha256(messages)
                == cell["prompt_sha256"], cell["cell_id"])
    perspective = json.loads((harness / PERSPECTIVE_MANIFEST).read_text(encoding="utf-8"))
    for cell in perspective["cells"]:
        messages = V.position_reader_messages(ports[cell["position_id"]], "ordered")
        if cell["view"] == "imagine":
            messages = append_paragraph(messages, paragraph)
        t.check(f"perspective_cell_prompt_{cell['view']}",
                V.messages_sha256(messages) == cell["prompt_sha256"], cell["cell_id"])
    for manifest in (panel, perspective):
        source = manifest["source"]
        t.check("source_m44_report_sha256", sha256_file(
            harness / M44_DIR / "m44_report.json") == source["m44_report_sha256"])
        t.check("source_targets_summary_sha256", sha256_file(
            harness / M44_DIR / "targets_summary.json")
            == source["targets_summary_sha256"])
    return {"positions": len(hpositions),
            "trajectories": len({p["trajectory_id"] for p in hpositions}),
            "turn8_positions": sum(1 for p in hpositions if p["turn"] == 8),
            **counts, "perspective_cells": len(perspective["cells"]),
            "panel_manifest_sha256": panel["manifest_sha256"],
            "perspective_manifest_sha256": perspective["manifest_sha256"]}


# ------------------------------------------------------------ data writers

BANK_NOTE = (
    "Frozen inputs of the 90 trajectories of the M4.3 turn-8 replay bank "
    "(Commons v1, condition tradeoff_v8, prompt protocol v8, action feedback "
    "on). actions[t] lists the moves of turn t in seat order (agent_1, "
    "agent_2, agent_3); replaying them from (seed, scenario_id) rebuilds every "
    "state. The digests are the values recorded by the research harness. "
    "instruction_variant is the reader instruction paraphrase the forecast "
    "suite assigned to the trajectory. Exported by "
    "scripts/check_equivalence_v1.py --write-data.")
POSITIONS_NOTE = (
    "Frozen definitions of the 48 decision positions of the M4.4 battery: "
    "trajectory (a row of m43_turn8_bank.json), turn, and target seat. "
    "prompt_sha256 holds the digests of the recorded outbound prompts: json "
    "actor (the target's acting prompt, M4.4 run store), json board and "
    "ordered (nine-forecaster panel manifest, equal to the run store), json "
    "shuffled and every markdown prompt (M4.4 run store). state_sha256 was "
    "computed by the harness replay. Exported by "
    "scripts/check_equivalence_v1.py --write-data.")
DIGEST_RECIPES = {
    "state_sha256": "sha256 of json.dumps(state, sort_keys=True, "
                    "separators=(',', ':'), default=str)",
    "observation_sha256": "sha256 of json.dumps(observation, sort_keys=True, "
                          "separators=(',', ':'), default=str)",
    "state_id": "'sha256:' + sha256 of repr(sorted(state.items()))",
    "prompt_sha256": "sha256 of json.dumps(messages, sort_keys=True, "
                     "ensure_ascii=False, separators=(',', ':')) as UTF-8",
}


def write_rows(path: Path, meta: dict, key: str, rows: list[dict]) -> None:
    """Top-level fields indented, one row per line."""
    lines = ["{"]
    for name, value in meta.items():
        lines.append(f" {json.dumps(name)}: "
                     f"{json.dumps(value, ensure_ascii=False)},")
    lines.append(f" {json.dumps(key)}: [")
    for i, row in enumerate(rows):
        lines.append("  " + json.dumps(row, ensure_ascii=False)
                     + ("," if i < len(rows) - 1 else ""))
    lines.append(" ]")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def write_data(m, harness: Path, tmp: Path) -> list[str]:
    V, HFS, HM = m["V"], m["HFS"], m["HM"]
    manifest, snaps = bank_snapshots(harness)
    fixtures = {row["snapshot_id"]: row for row in json.loads(
        (harness / FIXTURE_MANIFEST).read_text(encoding="utf-8"))["rows"]}
    suite = HM.suite()
    rows = []
    for mrow, snap in zip(manifest["rows"], snaps):
        sid = snap["snapshot_id"]
        rows.append({
            "snapshot_id": sid, "campaign_id": snap["campaign_id"],
            "seed": snap["seed"], "scenario_id": snap["scenario_id"],
            "target_id": snap["probe_target_id"], "turn": snap["turn"],
            "instruction_variant": HFS.variant_for_snapshot(snap, suite),
            "actions": recorded_actions(m, snap),
            "state_sha256": mrow["state_sha256"], "state_id": snap["state_id"],
            "snapshot_sha256": mrow["snapshot_sha256"],
            "observation_sha256_by_arm": {
                arm: mrow["observation_sha256_by_arm"][arm]
                for arm in HARNESS_ARM.values()},
            "suite_observation_sha256_by_arm": {
                arm: fixtures[sid]["observation_sha256_by_arm"][arm]
                for arm in HARNESS_ARM.values()},
        })
    rows.sort(key=lambda r: r["snapshot_id"])
    DATA.mkdir(parents=True, exist_ok=True)
    write_rows(DATA / "m43_turn8_bank.json", {
        "_note": BANK_NOTE,
        "schema": "commons.v1.m43_turn8_bank/1",
        "env_name": V.ENV_NAME, "env_version": V.ENV_VERSION,
        "condition": V.CONDITION,
        "prompt_protocol_version": V.PROMPT_PROTOCOL_VERSION,
        "return_schema": "action_only", "action_feedback": True,
        "bank_manifest_campaign": manifest["campaign_id"],
        "forecast_suite_version": suite["suite"]["forecast_suite_version"],
        "sources": {
            "snapshot_manifest": {"path": SNAPSHOT_MANIFEST,
                                  "sha256": sha256_file(harness / SNAPSHOT_MANIFEST)},
            "prompt_fixture_manifest": {
                "path": FIXTURE_MANIFEST,
                "sha256": sha256_file(harness / FIXTURE_MANIFEST)},
        },
        "digest_recipes": {k: DIGEST_RECIPES[k] for k in (
            "state_sha256", "observation_sha256", "state_id")},
    }, "trajectories", rows)

    with contextlib.redirect_stdout(io.StringIO()):
        hpositions = HM._positions()
    recorded, _ = recorded_position_digests(harness, open_store(harness, tmp))
    panel = json.loads((harness / PANEL_MANIFEST).read_text(encoding="utf-8"))
    perspective = json.loads((harness / PERSPECTIVE_MANIFEST).read_text(encoding="utf-8"))
    prows = [{
        "position_id": p["position_id"], "trajectory_id": p["trajectory_id"],
        "turn": p["turn"], "target_id": p["target_agent"],
        "state_sha256": HFS.sha256_json(p["snap"]["state"]),
        "prompt_sha256": recorded[p["position_id"]],
    } for p in hpositions]
    write_rows(DATA / "m44_positions.json", {
        "_note": POSITIONS_NOTE,
        "schema": "commons.v1.m44_positions/1",
        "sources": {
            "run_store": M44_STORE,
            "m44_report_sha256": panel["source"]["m44_report_sha256"],
            "targets_summary_sha256": panel["source"]["targets_summary_sha256"],
            "panel_manifest": {"path": PANEL_MANIFEST,
                               "manifest_sha256": panel["manifest_sha256"]},
            "perspective_manifest": {"path": PERSPECTIVE_MANIFEST,
                                     "manifest_sha256": perspective["manifest_sha256"]},
        },
        "digest_recipes": {"state_sha256": DIGEST_RECIPES["state_sha256"],
                           "prompt_sha256": DIGEST_RECIPES["prompt_sha256"]},
    }, "positions", prows)
    return ["src/commons/v1/data/m43_turn8_bank.json",
            "src/commons/v1/data/m44_positions.json"]


GOLDEN_POSITION = "concordia_predictive_commons-seed1600-turn8-agent_1-3356000a8932@t8"


def write_golden(m) -> list[str]:
    """Full prompt texts rendered by the HARNESS at one frozen position."""
    HM = m["HM"]
    with contextlib.redirect_stdout(io.StringIO()):
        hp = next(p for p in HM._positions() if p["position_id"] == GOLDEN_POSITION)
    texts = {
        "v1_actor_seed1600_agent_1_t8.txt": HM.acting_messages(hp),
        "v1_reader_board_seed1600_agent_1_t8.txt":
            HM.forecast_messages(hp, "board_only", "json"),
        "v1_reader_ordered_seed1600_agent_1_t8.txt":
            HM.forecast_messages(hp, "ordered", "json"),
    }
    written = []
    for name, messages in texts.items():
        text = "".join(f"===== {msg['role']} =====\n{msg['content']}\n\n"
                       for msg in messages)
        (GOLDEN / name).write_bytes(text.encode("utf-8"))
        written.append(f"tests/golden/{name}")
    return written


def git_info(harness: Path) -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", str(harness), *cmd],
                              capture_output=True, text=True).stdout.strip()
    dirty = run("status", "--porcelain", "--", *SOURCE_FILES)
    return {"commit": run("rev-parse", "HEAD"),
            "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "source_files": list(SOURCE_FILES),
            "source_files_differ_from_head": dirty.splitlines(),
            "social_sim_last_commit": run(
                "log", "-1", "--format=%h %ad", "--date=short", "--",
                "src/self_other_prediction/envs/social_sim.py")}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n")[0],
        epilog="Needs the private research harness checkout and its recorded "
               "data (pass --harness or set COMMONS_HARNESS); run it with the "
               "harness's own Python environment.")
    ap.add_argument("--harness", default=os.environ.get("COMMONS_HARNESS"),
                    help="research harness checkout, read only (or set "
                         "COMMONS_HARNESS)")
    ap.add_argument("--n-seeds", type=int, default=8)
    ap.add_argument("--seed-start", type=int, default=5100)
    ap.add_argument("--report", default=None)
    ap.add_argument("--write-data", action="store_true",
                    help="regenerate src/commons/v1/data from the harness records")
    ap.add_argument("--write-golden", action="store_true",
                    help="write harness-rendered prompt texts to tests/golden/v1_*.txt")
    args = ap.parse_args()
    if not args.harness:
        ap.error("the private research harness is required: pass --harness "
                 "or set COMMONS_HARNESS")
    harness = Path(args.harness).resolve()
    require(harness)
    started = time.time()
    m = load(harness)
    tmp = Path(tempfile.mkdtemp(prefix="commons_v1_equivalence_"))
    try:
        if args.write_data or args.write_golden:
            written = []
            if args.write_data:
                written += write_data(m, harness, tmp)
            if args.write_golden:
                written += write_golden(m)
            print("\n".join(written))
            return 0
        t = Tally()
        constant_checks(m, t)
        synthetic_episodes(m, t, args)
        bank = bank_checks(m, t, harness)
        positions = position_checks(m, t, harness, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    report = {
        "tool": "scripts/check_equivalence_v1.py",
        "date": date.today().isoformat(),
        "harness": git_info(harness),
        "python": sys.version.split()[0],
        "comparisons": dict(sorted(t.counts.items())),
        "total_comparisons": sum(t.counts.values()),
        "failures": len(t.failures),
        "failure_examples": t.failures[:20],
        "m43_bank": bank,
        "m44_positions": positions,
        "synthetic": {"scenarios": 3, "seeds": args.n_seeds,
                      "seed_start": args.seed_start},
        "runtime_s": round(time.time() - started, 1),
    }
    text = json.dumps(report, indent=1, ensure_ascii=False) + "\n"
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps({k: report[k] for k in (
        "total_comparisons", "failures", "runtime_s")}))
    for failure in t.failures[:20]:
        print("FAIL", failure)
    return 1 if t.failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
