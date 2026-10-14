# Provenance

This package was extracted on 2026-10-14 from the private research harness used for the paper (repository `self-other-prediction-smoke-tests`, branch `main`, commit `e5673cdcb6c5ff4acbb8d519734a750ff215c0a4`). None of the harness source files listed below differed between that commit and the working tree at extraction time.

## Module map

| This package | Harness source | What was kept |
|---|---|---|
| `commons/core/events.py` | `src/self_other_prediction/events.py` | `make_event`, `view_events`, `shuffled_events`, visibility scopes. Not ported: identity anonymization and turn stripping (research probes only). |
| `commons/core/base.py` | `src/self_other_prediction/envs/base.py` | `StepResult`, snapshot and event-log plumbing. Not ported: the multi-environment protocol, probe-target routing, the omniscient `prompt_state`, `fallback_action`. |
| `commons/core/env.py` | `src/self_other_prediction/envs/commons_v2.py` | The whole environment (class `ConcordiaPredictiveCommonsV2Env` renamed `CommonsEnv`; module constants renamed without the `V2_` prefix), `replay_states`, `is_applied_action`. `with_objectives` comes from `src/self_other_prediction/commons_v2_precheck.py`. Not ported: `setup_diagnostics`, `action_feedback` echo events (always off), prediction-awareness notices (refused instead). |
| `commons/core/sham.py` | `src/self_other_prediction/envs/commons_v2_sham.py` | The whole sham arm (class `ShamObjectiveCommonsV2` renamed `ShamCommonsEnv`), `restore_sham`, `assert_inert`. |
| `commons/render/markdown.py` | `src/self_other_prediction/prompt_render_md.py` | The markdown user-message renderer, unchanged except the unused debrief task line. |
| `commons/render/messages.py` | `src/self_other_prediction/model_clients.py`, `src/self_other_prediction/prompt_templates.py` | System-message composition, actor and forecast message builders (markdown format, action-only return schema), the system task sentences, the reader perspective label and the three instruction paraphrases. Not ported: the JSON prompt format, debrief, and other probe templates. |
| `commons/render/views.py` | `src/self_other_prediction/arms.py`, `src/self_other_prediction/m4_forecast_suite.py`, `scripts/commons_v2_campaign.py`, `scripts/commons_v3_campaign.py` | Actor messages (`acting_messages`, sham `episode_acting_messages`); reader views board, ordered, shuffled (`prediction_observation`, `shuffled_history_observation`, `observations_for_snapshot`); tally (`_counts_only`); goal naming (`objective_messages`, `objective_menu`). |
| `commons/data/reader_instruction_variants.v1.json` | `config/predictive_commons_m4_3_forecast_suite_m4_3_1.json` | The frozen reader-instruction variant map (108 cells), exported through `m4_forecast_suite.variant_assignment`. |
| `commons/policies/lookahead.py` | `src/self_other_prediction/commons_v2_precheck.py` | `action_values`, `softmax`, `best_actions`, `jsd`, `entropy_bits`, `reference_trajectory`. |
| `commons/policies/scripted.py` | new | Seat policies built on the lookahead. |
| `commons/runtime/parse.py` | `src/self_other_prediction/model_clients.py`, `scripts/commons_v3_campaign.py` | `parse_json_object`, `resolve_action_choice`, `effective_action_id` (lenient parser, phase 2); the strict parser (phase 3). |
| `commons/runtime/*` (other) | shape of the LEDGER and FORGE runtimes; `scripts/commons_m44.py` | OpenAI-compatible client, cost meter, agents, episode runner, logs, policy replay, offline fake client. The request body reproduces the harness wire client (`CommonsWireClient`); the draw policy reproduces `_draw` (phase 2) and `checked_draw` (phase 3). |

## Equivalence check

`scripts/check_equivalence.py` imports the harness implementation read-only and compares it with this package. The result of the run at extraction time is in [`equivalence_report.json`](equivalence_report.json): 130,911 comparisons with 0 failures, including 321 recorded episodes from the studied campaigns, each replayed to its recorded endpoint.

The test suite does not need the harness. It checks this package against digests and prompt texts the script exported from the harness side (`tests/golden/`) and against three recorded episodes with harness-computed prompt digests (`tests/fixtures/`).

To rerun (maintainers with a harness checkout and its virtual environment):

```
<harness>/.venv/Scripts/python.exe -B scripts/check_equivalence.py --harness <harness checkout> --report docs/provenance/equivalence_report.json --write-golden
```

## Commons v1

`commons.v1` was added on 2026-10-14 from the same harness commit. It is the earlier Office environment that the report (report v4) used for its Office results on ordinary play; see [`../V1.md`](../V1.md). None of the harness source files listed below differed between that commit and the working tree. The environment code has been unchanged in behavior since 2026-07-14 (later commits to `social_sim.py` changed comments only), before the M4.3 bank was collected.

| This package | Harness source | What was kept |
|---|---|---|
| `commons/v1/env.py` | `src/self_other_prediction/envs/social_sim.py` | The protocol-v8 path of `StructuredSocialEnv` and `ConcordiaPredictiveCommonsEnv` (renamed `CommonsV1Env`): `v8_action_space`, `V8_FIXED_ACTIONS`, `V8_SCENARIOS`, `V8_PRIVATE_GOALS`, the briefing, the reset (including the seeded draws the v8 reset overwrites), the v8 transitions and rewards, the action acknowledgements, `state_id`. Replay from action labels and from an event log follows `scripts/commons_m44.py` `replay_states`. Not ported: the pasture world, older conditions, prediction-awareness notices, `setup_diagnostics`. |
| `commons/v1/render.py` | `src/self_other_prediction/model_clients.py`, `arms.py`, `m4_forecast_suite.py`, `scripts/commons_m44.py` | The JSON user message (`_user_content` with `prompt_format="json"`), actor messages, reader views board, ordered and shuffled (`prediction_observation`, `shuffled_history_observation`, `observations_for_snapshot`), the digest recipes. Shares the system sentences, instruction paraphrases and markdown renderer with `commons/render`. |
| `commons/v1/decisions.py`, `commons/v1/data/*.json` | `data/outputs/predictive_commons_m4_3/` (bank snapshots, `snapshot_manifest.json`, `prompt_fixture_manifest.json`, `m44/_store`), `docs/ordinary_prediction_panel/manifests/commons.json`, `scripts/commons_m44.py` (`_positions`, `make_position`, `forecast_messages`, `acting_messages`, `legal_actions`) | Seeds, scenario ids, target seats, campaign ids, instruction variants and seat-ordered action labels of the 90 bank trajectories; ids, turns and target seats of the 48 M4.4 positions; the recorded state, observation and prompt digests. No model text, replies, provider records, costs or model identities. |

`scripts/check_equivalence_v1.py` compares `commons.v1` with the harness and with the recorded digests. The result is in [`v1_equivalence_report.json`](v1_equivalence_report.json): 26,457 comparisons with 0 failures. The same script regenerates the frozen inputs (`--write-data`) and the prompt texts `tests/golden/v1_*.txt` (`--write-golden`).

```
<harness>/.venv/Scripts/python.exe -B scripts/check_equivalence_v1.py --harness <harness checkout> --report docs/provenance/v1_equivalence_report.json
```

Study prompts that are not part of the environment were not ported: the plain-text acting prompt of the "being watched" study (`scripts/commons_e12.py`), the plain-text reader prompts of the reasoning and model-name studies (`scripts/commons_l4s1.py`, `scripts/commons_e13.py`), and the perspective paragraph of the perspective study (`scripts/self_simulation_prediction.py`; the equivalence check appends it from the harness source to verify those 240 prompts).
