# Commons

Commons is a research environment for language-model agents. Three agents share one workplace for 16 turns, and each agent holds one private objective. You can use it to study whether one agent can predict what another agent will do next from public play. In the accompanying paper, Commons is called Office.

## How the game works

Full rules are in [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md).

- Three seats, `agent_1`, `agent_2`, and `agent_3`, choose at the same time each turn. A game lasts 16 turns.
- Each seat has a public role, such as analyst or maintainer. The role says nothing about the objective.
- Each agent holds one of three objectives: finish its own tasks, keep shared systems healthy, or gain credit from others. By default, the three agents hold three different objectives.
- An agent sees its own objective. No agent sees a peer's objective.
- Each turn, an agent picks 1 of 11 actions. It can take supplies, repair shared systems, inspect, wait, delegate work to a peer, or share supplies with a peer.
- A delegation or a share takes effect one turn later. What it does then depends on the recipient's hidden objective, so public play gives clues about private goals.
- Delegation outcomes are public. Share outcomes show in detail only to the two agents involved.
- Shared systems get worse each turn unless agents repair them. Supplies and energy refill a little each turn.
- An agent's reward each turn is the change in its own objective.
- Three opening scenarios set the starting board.

## Quick start

You need Python 3.11 or later.

```powershell
git clone https://github.com/lmmontoya-ai/commons
cd commons
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m pytest -q
.venv\Scripts\python -m ruff check src scripts tests
```

The clone makes a folder named `commons`. The next lines make a `.venv` folder for this project and install Commons and the test tools into it. The test line reports 114 passed, and the ruff line reports no errors. On Linux or macOS, use `.venv/bin/python` in place of `.venv\Scripts\python`. Run a free scripted game with the next command.

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --agent-1 scripted:greedy --agent-2 scripted:softmax --agent-3 scripted:uniform --out data\runs\smoke --episode-id smoke
```

Expect `episode smoke (active arm), 16 turns, replay verified: True` and $0.0000 over 0 calls. It needs no key and no network.

Run with a model through OpenRouter. Copy `models.example.json` to `models.json` first. The key comes from the `OPENROUTER_API_KEY` environment variable. Do not put a key in any file.

```powershell
Copy-Item models.example.json models.json
$env:OPENROUTER_API_KEY = "your key here"
.venv\Scripts\python scripts\run_episode.py --seed 2700 --scenario delivery_pressure --objectives PERSONAL_THROUGHPUT,SYSTEM_HEALTH,SOCIAL_CREDIT --models models.json --agent-1 model:luna --agent-2 model:grok45 --agent-3 model:sonnet5 --cap-usd 2 --out data\runs\pilot
```

This command spends money. Without a key, the runner stops before any call with `cannot run model seats: no API key: pass api_key or set OPENROUTER_API_KEY`.

## What an agent sees

An agent reads its own record, the shared board, recent events, and its action menu, then replies with one action name. The block below is real output. It is part of the prompt for `agent_1` at turn 0, cut after the hidden objective line. The full prompt is in [tests/golden/actor_delivery_pressure_s2700_t0_agent_1.txt](tests/golden/actor_delivery_pressure_s2700_t0_agent_1.txt). The world labels in the prompt are historical and are kept as studied.

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
```

## Use in the accompanying paper

The paper used the active objectives above as the main form. A reader saw public information only and forecast the target's next action, or named the target's hidden objective. As a control, the sham variant shows style labels with no payoff effect in place of real objectives. Full usage for reader views and the sham arm is in [docs/READER_VIEWS.md](docs/READER_VIEWS.md).

Run the sham control at no cost:

```powershell
.venv\Scripts\python scripts\run_episode.py --seed 3001 --scenario infrastructure_risk --sham FORMAL_REGISTER,TERSE_REGISTER,EXPANSIVE_REGISTER --agent-1 scripted:greedy --agent-2 scripted:greedy --agent-3 scripted:greedy --out data\runs\sham --episode-id sham
```

Expect `episode sham (sham arm), 16 turns, replay verified: True`.

## Reproducibility

Play is deterministic apart from the seeded start. Any recorded game replays exactly from its final state, and the runner checks replay at the end of each run. This release was checked against the original research code, with zero mismatches. All counts and log details are in [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Documentation

- [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md): full rules, state, actions, and runtime.
- [docs/READER_VIEWS.md](docs/READER_VIEWS.md): actor prompts, reader views, goal naming, and sham prompts.
- [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md): determinism, replay, log format, equivalence counts, and repository layout.
- [docs/provenance/PROVENANCE.md](docs/provenance/PROVENANCE.md): where each module came from.
- [docs/provenance/equivalence_report.json](docs/provenance/equivalence_report.json): counts from the equivalence check.

## Known limitations

- This release was tested on Windows only. Linux and macOS paths were not tested.
- A shuffled reader prompt from the studied runs needs seed context that does not ship here. Without it, the shuffle order differs.
- Model results do not repeat exactly. Models sample with randomness, and model access and prices change over time.
- The board keeps dated commitments and inspection records. Even with no event list, the board can give clues about goals.
- The runtime runs seats and calls one after another. It has no parallel lanes.

## Related environments

All three native environments in the accompanying paper are public:

- LEDGER, called Contracting in the accompanying paper: [https://github.com/lmmontoya-ai/ledger](https://github.com/lmmontoya-ai/ledger).
- FORGE, called Engineering in the accompanying paper: [https://github.com/lmmontoya-ai/forge](https://github.com/lmmontoya-ai/forge).
- Commons, called Office in the accompanying paper: [https://github.com/lmmontoya-ai/commons](https://github.com/lmmontoya-ai/commons).

## Citation

Placeholder. The owner will add the citation for the accompanying paper here.

```bibtex
% TODO(owner): replace with the paper citation.
```

## License

Not yet chosen. The owner will add a LICENSE file.
