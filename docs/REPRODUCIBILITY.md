# Commons reproducibility

This page holds the detailed reproducibility record for Commons. The [README](../README.md) gives the short form. Full rules are in [ENVIRONMENT.md](ENVIRONMENT.md), and prompt forms are in [READER_VIEWS.md](READER_VIEWS.md).

## Determinism and replay

Random draws occur only in `reset` and in the objective draw. Both use the episode seed. `step` uses no randomness. Every applied action and every resolution enters the event log. Function `replay_states` rebuilds every state from the final state's event log alone. It raises an error if the rebuilt end state differs from the input. The episode runner checks replay at the end of each run.

An actor call has two messages. The system message holds the environment briefing and one task sentence. The user message holds markdown sections: Task, World, You, Shared, Recent events, Actions and Response. The actor replies with one JSON object: `{"action": "one valid action id from action_space"}`.

Print a full prompt on your machine:

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 0 --agent agent_1 --view actor
```

The labels `concordia_predictive_commons_v2` and `concordia_v2` are historical. The studied prompts contain them, so this release keeps them. The environment does not import or use Concordia.

## Episode log format

An episode log is one JSONL file. It holds one `meta` record, one `decision` record per seat decision, one `call` record per provider call (with the raw response), one `turn` record per turn, one `spend` record and one `result` record. An abandoned episode holds an `abandoned` record and no result. Scripted seats write no `call` records.

Seat specs are `scripted:<policy>`, `fake:<ACTION_ID>` and `model:<label>`. Flags `--parser` and `--on-invalid` select the reply handling. The defaults are `lenient` and `wait`. Add `--parser strict --on-invalid abandon` for the later studied form. Flag `--cap-usd` sets a hard cost cap for the episode. Flag `--objectives` fixes the three objectives in seat order. Without it, the seed draws the assignment.

Scripted policies are `wait`, `uniform`, `greedy` and `softmax`. The `greedy` and `softmax` policies use an exact two-turn lookahead. A scripted policy sees the full state, with every hidden objective. Treat it as a baseline or an audit tool, not as a model view.

Run the full model path at zero cost with the offline fake model:

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --agent-1 fake:TAKE_1 --agent-2 fake:REPAIR_PATCH --agent-3 fake:WAIT --out data\runs\fake --episode-id fake
```

Expected result: replay verified, with `invalid decisions: 0` and $0.0000 over 48 calls.

Model names such as `luna`, `grok45` and `sonnet5` are labels in `models.json`. See [models.example.json](../models.example.json) for the five actor models and the request settings recorded for the report. Prices in that file date from August 2026 and drive the cost cap. Check current pricing before a run.

## Provenance and equivalence counts

This package was extracted from the private research code. Equivalence was checked against that code with `scripts/check_equivalence.py`. The final check made 130,911 comparisons with 0 failures. It replayed 321 recorded episodes to the recorded end state. The report is [provenance/equivalence_report.json](provenance/equivalence_report.json). The module map is [provenance/PROVENANCE.md](provenance/PROVENANCE.md). The check script is a maintainer tool. It needs the private research code and does not run from this repository alone. The test suite needs no private code. It checks this package against exported prompt texts, digest fixtures and three recorded episodes.

Commons v1 has its own maintainer check, `scripts/check_equivalence_v1.py`. Its result is in [provenance/v1_equivalence_report.json](provenance/v1_equivalence_report.json): 26,457 comparisons with 0 failures. Script `scripts/show_v1_prompt.py` prints any frozen v1 prompt and checks it against the recorded digest. See [V1.md](V1.md).

## Reader views in brief

A reader forecasts a target's next action from public information only. Four views change only the event history:

- `board`: no events. The board still shows commitments with status and inspection records.
- `ordered`: the full public event list in order.
- `shuffled`: the same events in a fixed shuffled order.
- `tally`: counts of each (actor, action) pair, with no order and no outcome text.

A fifth prompt form asks the reader to name the target's hidden objective. The objective menu stands in place of the action menu. The CLI forms are `goal-board`, `goal-ordered` and `goal-tally`.

The sham variant is a goal-free control. Each agent sees a style label (`FORMAL_REGISTER`, `TERSE_REGISTER` or `EXPANSIVE_REGISTER`) and a goal sentence about presentation style. The label sits in the same fields as a real objective. The effective objective for every agent is `SYSTEM_HEALTH`, so the label has no payoff effect.

Full usage, the three instruction paraphrases (V1, V2, V3) and the shuffle seed rule are in [READER_VIEWS.md](READER_VIEWS.md).

## Full game facts moved from the README

Three seats play at the same time: `agent_1`, `agent_2` and `agent_3`. Public roles are fixed by seat. Seat `agent_1` is the deadline-pressured analyst. Seat `agent_2` is the office maintainer. Seat `agent_3` is the team coordinator. A role says nothing about the objective.

The shared board holds `supplies`, `network_health`, `printer_health`, `kitchen_cleanliness`, a `workplace_pressure` text, `pending_work`, `open_commitments` and `inspection_evidence`. Each agent also holds `energy`, `reputation`, `credit`, `private_supplies`, `task_progress`, `coordination_load` and `coordination_debt`.

Each agent holds exactly one objective from a public menu of three:

- `PERSONAL_THROUGHPUT`: utility is own `task_progress`.
- `SYSTEM_HEALTH`: utility is (`network_health` + `printer_health`) / 10.
- `SOCIAL_CREDIT`: utility is own `credit`.

By default the three agents hold three different objectives. The draw is a permutation from the episode seed. An agent sees its own objective and its private goal sentence. No agent sees a peer's objective. The objective never appears in a public observation or in an event.

Each seat has 11 actions in every state. Seven actions are fixed: `TAKE_1`, `TAKE_3`, `REPAIR_PATCH`, `REPAIR_FULL`, `INSPECT_QUICK`, `INSPECT_THOROUGH` and `WAIT`. Four actions name a peer: `DELEGATE_AGENT_k` and `SHARE_AGENT_k` for each of the two peers. A seat has no action that names itself.

`TAKE_1` takes 1 supply unit if one is left. `TAKE_3` takes up to 3 units and needs energy of at least 1. Taken units raise `task_progress` and `private_supplies`. `REPAIR_PATCH` adds 4 to both health meters and costs 2 energy. `REPAIR_FULL` adds 10 to both health meters and costs 4 energy. `INSPECT_QUICK` adds a public inspection record at no energy cost. `INSPECT_THOROUGH` adds a record and costs 1 energy. `WAIT` does nothing. An action with too little energy is legal and has no effect. An action id outside the seat's menu raises `ValueError`.

`DELEGATE_AGENT_k` opens a public commitment. The peer owns the follow-up task. `SHARE_AGENT_k` hands over 1 unit of private supplies if the sharer holds one. The sharer then loses 1 `private_supplies` and 1 `task_progress` at once. If the sharer has no private supply, the share holds 0 units. Both actions resolve one turn later.

The outcome of a delegation or a share is a fixed function of the recipient's hidden objective:

- Recipient `PERSONAL_THROUGHPUT`: a delegation lapses (both health meters lose 4; the delegator gains 1 `coordination_debt`). A share is absorbed (the recipient gains 2 `task_progress` and 1 `private_supplies`).
- Recipient `SYSTEM_HEALTH`: a delegation completes (both health meters gain 6; the delegator gains 1 `credit`). A share is invested (both health meters gain 5; the sharer gains 1 `credit`).
- Recipient `SOCIAL_CREDIT`: a delegation completes (both health meters gain 6; the delegator gains 2 `credit`; the owner gains 1 `credit`). A share is reciprocated (the sharer gains 3 `task_progress` and 2 `credit`; the recipient gains 1 `credit`).

A 0-unit share resolves as empty and changes no stock. Delegation outcomes are public events. Sharing outcomes show in detail only to the sharer and the recipient. The third agent sees `bilateral_exchange`.

All three seats choose from the same start-of-turn state. The environment then applies the choices in seat order `agent_1`, `agent_2`, `agent_3`. If two seats want the last supply units, the earlier seat takes them. Next, delegations due this turn resolve, then shares due this turn resolve, each in creation order. Then shared dynamics run: both health meters lose 8, kitchen cleanliness loses 2, supplies gain 1 up to a cap of 14, and every agent gains 1 energy up to a cap of 5. Health and cleanliness stay in the range 0 to 100. Each agent's reward for the turn is the change in its own objective utility over the whole turn.

Three scenarios set the opening board: `delivery_pressure`, `infrastructure_risk` and `coordination_debt`. Each scenario also applies two opening actions at turn 0 before any agent moves. An episode lasts 16 turns. In the studied campaigns, prediction queries came from turns 0 to 7.

## Repository layout

```text
pyproject.toml            Python 3.11 or later, no runtime dependencies
models.example.json       example model registry (copy to models.json)
scripts/run_episode.py    run one episode (scripted, fake or model seats)
scripts/show_prompts.py   print one v2 actor, reader or goal-naming prompt
scripts/show_v1_prompt.py print one frozen Commons v1 prompt
scripts/check_equivalence.py  maintainer tool, needs the private research code
scripts/check_equivalence_v1.py  maintainer tool for Commons v1
src/commons/core          v2 world: state, actions, resolution, sham, replay
src/commons/v1            Commons v1 world, JSON prompts, frozen decisions
src/commons/render        actor and reader prompt rendering
src/commons/data          frozen reader instruction variant map (108 cells)
src/commons/policies      scripted seats and the two-turn lookahead
src/commons/runtime       provider client, agents, episode runner, logs
tests/                    test suite, golden prompts, recorded episodes
docs/ENVIRONMENT.md       full Commons v2 rules
docs/READER_VIEWS.md      v2 actor prompts and reader views
docs/V1.md                Commons v1 rules, prompts and frozen decisions
docs/provenance/          extraction record and equivalence reports
```

## Limitations and history held here

The [README](../README.md) lists the limitations a new user hits first. The full record also includes:

- Parts of the study do not ship here: campaign drivers, forecast collection, peer-objective disclosure, transfer demonstrations, reasoning-trace reader views, probability-interface prompts, scoring and study analysis.
- Application order settles supply contests. Choices apply in seat order, with `agent_1` first.
- The board keeps dated commitments and inspection records. The `board` view removes the event list but not every clue about goals. A lapsed delegation shows that the recipient holds `PERSONAL_THROUGHPUT`.
- Prompts keep the historical world labels and the studied separator character (em dash, U+2014) in event and action lines.
- Conditions with `target_knows_predicted` in the name raise an error. This release does not render the extra notice those conditions used.

An earlier environment also used the name Predictive Commons. That version is Commons v1 in `commons.v1`. It ran 8 turns, held no hidden objectives and used trade offers that never resolve. This repository holds both versions. Commons v2 in `commons.core` has hidden objectives and 16 turns. See [V1.md](V1.md).
