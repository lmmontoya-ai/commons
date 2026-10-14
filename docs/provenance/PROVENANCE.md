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
