# Commons

Commons is a research environment for language-model agents. Three agents share one workplace for 16 turns. Each agent holds one private objective. The environment records every choice and outcome, so any episode replays from its final state. In the accompanying paper, Commons is called Office.

Commons exists to study prediction between agents. A reader sees public information about a target agent and forecasts the target's next action. The target's true objective stays hidden from the reader. Delegation and sharing outcomes depend on the recipient's hidden objective, so public play carries clues about private goals.

## The game in about a minute

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

Full rules are in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

## What an agent sees

An actor call has two messages. The system message holds the environment briefing and one task sentence. The user message holds markdown sections: Task, World, You, Shared, Recent events, Actions and Response. The actor replies with one JSON object: `{"action": "one valid action id from action_space"}`.

The block below is real output. It is part of the user message for `agent_1` at turn 0 with seed 2700 in scenario `delivery_pressure`. The copy stops before Recent events. The full messages also hold the system message, the event list, the action menu and the response schema. The test suite checks the full bytes against [tests/golden/actor_delivery_pressure_s2700_t0_agent_1.txt](tests/golden/actor_delivery_pressure_s2700_t0_agent_1.txt).

```text
## Task
Choose your next action.

## World
env name: concordia_predictive_commons_v2
world: concordia_v2
turn: 0

## You
agent id: agent_1
coordination debt: 0
coordination load: 0
credit: 0
energy: 4
hidden objective: PERSONAL_THROUGHPUT
private goal: maximise your personal throughput: the quantity of your own assignments carried to completion during this shift
private supplies: 2
public role: deadline-pressured analyst
reputation: 50
task progress: 3

## Shared
kitchen cleanliness: 54
network health: 54
open commitments: (none)
printer health: 51
supplies: 8
workplace pressure: A client revision is due this shift while two shared-system warnings remain unresolved.

### Inspection evidence
- actor=agent_2, finding=one additional symptom recorded, scope=quick, turn=0

### Pending work
- owner=agent_1, task=client_revision, urgency=3
- owner=none, task=system_warning, urgency=2
```

Print the full prompt on your machine:

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 0 --agent agent_1 --view actor
```

The labels `concordia_predictive_commons_v2` and `concordia_v2` are historical. The studied prompts contain them, so this release keeps them. The environment does not import or use Concordia.

## Quick start

You need Python 3.11 or later. Work from a checkout of this repository. The environment lives in `.venv` at the repository root. The package has no runtime dependencies. The dev extra adds `pytest` and `ruff`.

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check src scripts tests
```

On Linux or macOS, use `.venv/bin/python` in place of `.venv\Scripts\python`. This release was tested on Windows only. On a fresh install the test command reports 114 passed and the ruff command reports no errors.

Run a zero-cost scripted episode. It needs no key and no network:

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --agent-1 scripted:greedy --agent-2 scripted:softmax --agent-3 scripted:uniform --out data\runs\smoke --episode-id smoke
```

Expected result: `episode smoke (active arm), 16 turns, replay verified: True`. Final utilities are 45 for `agent_1`, 10.5 for `agent_2` and 16 for `agent_3`. Cost is $0.0000 over 0 calls. The log is `data\runs\smoke\smoke.jsonl`. Scripted policies are `wait`, `uniform`, `greedy` and `softmax`. The `greedy` and `softmax` policies use an exact two-turn lookahead. A scripted policy sees the full state, with every hidden objective. Treat it as a baseline or an audit tool, not as a model view.

Run the full model path at zero cost with the offline fake model:

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --agent-1 fake:TAKE_1 --agent-2 fake:REPAIR_PATCH --agent-3 fake:WAIT --out data\runs\fake --episode-id fake
```

Expected result: replay verified, with `invalid decisions: 0` and $0.0000 over 48 calls.

Run with a language model through OpenRouter. Copy `models.example.json` to `models.json` first. The key comes from the `OPENROUTER_API_KEY` environment variable. Set it in your shell. Do not put a key in any file.

```powershell
Copy-Item models.example.json models.json
$env:OPENROUTER_API_KEY = "your key here"
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --objectives PERSONAL_THROUGHPUT,SYSTEM_HEALTH,SOCIAL_CREDIT --models models.json --agent-1 model:luna --agent-2 model:grok45 --agent-3 model:sonnet5 --cap-usd 2 --out data\runs\pilot
```

This command spends money. Without a key, the runner stops before any call with `cannot run model seats: no API key: pass api_key or set OPENROUTER_API_KEY` and exit code 1. Model names such as `luna`, `grok45` and `sonnet5` are labels in `models.json`. See [models.example.json](models.example.json) for the five actor models and the request settings recorded for the accompanying paper. Prices in that file date from August 2026 and drive the cost cap. Check current pricing before a run.

Seat specs are `scripted:<policy>`, `fake:<ACTION_ID>` and `model:<label>`. Flags `--parser` and `--on-invalid` select the reply handling. The defaults are `lenient` and `wait`. Add `--parser strict --on-invalid abandon` for the later studied form. Flag `--cap-usd` sets a hard cost cap for the episode. Flag `--objectives` fixes the three objectives in seat order. Without it, the seed draws the assignment.

## Sham variant and reader views

The sham variant is a goal-free control. Each agent sees a style label (`FORMAL_REGISTER`, `TERSE_REGISTER` or `EXPANSIVE_REGISTER`) and a goal sentence about presentation style. The label sits in the same fields as a real objective. The effective objective for every agent is `SYSTEM_HEALTH`, so the label has no payoff effect. Run the sham arm at zero cost:

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 3001 --scenario infrastructure_risk --sham FORMAL_REGISTER,TERSE_REGISTER,EXPANSIVE_REGISTER --agent-1 scripted:greedy --agent-2 scripted:greedy --agent-3 scripted:greedy --out data\runs\sham --episode-id sham
```

Expected result: `episode sham (sham arm), 16 turns, replay verified: True`.

A reader forecasts a target's next action from public information only. Four views change only the event history:

- `board`: no events. The board still shows commitments with status and inspection records.
- `ordered`: the full public event list in order.
- `shuffled`: the same events in a fixed shuffled order.
- `tally`: counts of each (actor, action) pair, with no order and no outcome text.

A fifth prompt form asks the reader to name the target's hidden objective. The objective menu stands in place of the action menu. The CLI forms are `goal-board`, `goal-ordered` and `goal-tally`.

Print each form with `scripts/show_prompts.py`. For example:

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view ordered
.venv\Scripts\python scripts\show_prompts.py --log data\runs\smoke\smoke.jsonl --turn 3 --agent agent_2 --view ordered
```

Without `--log`, the script reaches the turn by scripted play with `--policy` (default `greedy`) for every seat. With `--log`, it replays the logged episode. Full usage, the three instruction paraphrases (V1, V2, V3) and the shuffle seed rule are in [docs/READER_VIEWS.md](docs/READER_VIEWS.md).

## Determinism and replay

Random draws occur only in `reset` and in the objective draw. Both use the episode seed. `step` uses no randomness. Every applied action and every resolution enters the event log. Function `replay_states` rebuilds every state from the final state's event log alone. It raises an error if the rebuilt end state differs from the input. The episode runner checks replay at the end of each run.

This package was extracted from the private research code. Equivalence was checked against that code with `scripts/check_equivalence.py`. The final check made 130,911 comparisons with 0 failures. It replayed 321 recorded episodes to the recorded end state. The report is [docs/provenance/equivalence_report.json](docs/provenance/equivalence_report.json). The module map is [docs/provenance/PROVENANCE.md](docs/provenance/PROVENANCE.md). The check script is a maintainer tool. It needs the private research code and does not run from this repository alone. The test suite needs no private code. It checks this package against exported prompt texts, digest fixtures and three recorded episodes.

An episode log is one JSONL file. It holds one `meta` record, one `decision` record per seat decision, one `call` record per provider call (with the raw response), one `turn` record per turn, one `spend` record and one `result` record. An abandoned episode holds an `abandoned` record and no result. Scripted seats write no `call` records.

## Repository layout

```text
pyproject.toml            Python 3.11 or later, no runtime dependencies
models.example.json       example model registry (copy to models.json)
scripts/run_episode.py    run one episode (scripted, fake or model seats)
scripts/show_prompts.py   print one actor, reader or goal-naming prompt
scripts/check_equivalence.py  maintainer tool, needs the private research code
src/commons/core          world: state, actions, resolution, sham, replay
src/commons/render        actor and reader prompt rendering
src/commons/data          frozen reader instruction variant map (108 cells)
src/commons/policies      scripted seats and the two-turn lookahead
src/commons/runtime       provider client, agents, episode runner, logs
tests/                    test suite, golden prompts, recorded episodes
docs/ENVIRONMENT.md       full environment rules
docs/READER_VIEWS.md      actor prompts and reader views
docs/provenance/          extraction record and equivalence report
```

## Documentation index

- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): full rules, state, actions, resolution, observations and runtime.
- [docs/READER_VIEWS.md](docs/READER_VIEWS.md): actor prompts, the four reader views, goal naming and sham prompts.
- [docs/provenance/PROVENANCE.md](docs/provenance/PROVENANCE.md): where each module came from and what stayed private.
- [docs/provenance/equivalence_report.json](docs/provenance/equivalence_report.json): counts from the equivalence check.

## Known limitations

- The runtime runs seats and calls one after another. It has no parallel lanes.
- A shuffled reader prompt from the studied campaigns needs the original seed context (campaign id and position id). Position ids do not ship in this repository. Without that context the shuffle order differs.
- Parts of the study do not ship here: campaign drivers, forecast collection, peer-objective disclosure, transfer demonstrations, reasoning-trace reader views, probability-interface prompts, scoring and paper analysis.
- This release was tested on Windows only. Linux and macOS paths use `.venv/bin/python` and were not tested.
- Application order settles supply contests. Choices apply in seat order, with `agent_1` first.
- The board keeps dated commitments and inspection records. The `board` view removes the event list but not every clue about goals. A lapsed delegation shows that the recipient holds `PERSONAL_THROUGHPUT`.
- Model results do not repeat exactly. Models sample with randomness, and model access, serving providers and prices change over time.
- Prompts keep the historical world labels and the studied separator character (em dash, U+2014) in event and action lines.
- Conditions with `target_knows_predicted` in the name raise an error. This release does not render the extra notice those conditions used.

## History

An earlier environment also used the name Predictive Commons. That version ran 8 turns, held no hidden objectives and used consequence-free trade offers. It is a different environment. This repository holds the later version with hidden objectives and 16 turns.

## Related environments

This environment is one of three native environments in the accompanying paper. LEDGER, called Contracting in the accompanying paper, is public at [https://github.com/lmmontoya-ai/ledger](https://github.com/lmmontoya-ai/ledger). FORGE, called Engineering in the accompanying paper, is not published yet. Commons, called Office in the accompanying paper, is not published yet.

## Citation

Placeholder. The owner will add the citation for the accompanying paper here.

```bibtex
% TODO(owner): replace with the paper citation.
```

## License

Not yet chosen. The owner will add a LICENSE file before public release.
