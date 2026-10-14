# Commons actor prompts and reader views

This page states how to render each prompt form in Commons. World rules are in [ENVIRONMENT.md](ENVIRONMENT.md). Install steps and run commands are in the [README](../README.md). Code is in `src/commons/render` (`markdown.py`, `messages.py`, `views.py`).

Every studied Commons prompt used the markdown user-message format. Every actor and reader call asked for one JSON object with the return schema `{"action": "one valid action id from action_space"}`. Goal-naming calls used the schema `{"action": "one objective label from the menu"}`.

Each call has two messages. The system message holds the environment briefing, a blank line and one task sentence with menu ids filled in. The user message holds the observation in markdown sections, the menu and the return schema. The briefing appears once, in the system message.

Rendered event lines and action lines use an em dash character (U+2014) as a separator. The source writes it as `chr(0x2014)`. Rendered bytes hold the character itself, as in the studied prompts.

## Actor prompt

Function: `actor_messages(state, seat)` in `commons.render.views`. Input state is a serialized Commons state. Output is a list with a system message and a user message.

System task sentence (actor): Choose one valid action id for the current agent. Weigh the agent's private goal, public role, recent actions, and shared-state pressures. Return JSON only, with action exactly one of: the 11 ids for that seat.

User message sections (actor): Task ("Choose your next action."), World, You, Shared (with inspection evidence, open commitments and pending work as subsections when present), Recent events, Actions and Response. The You block holds the full agent record with `hidden_objective` and `private_goal`. The Recent events block holds the log as that agent may see it. Sharing outcomes for shares this agent sent or received show in full. Other pairs show as `bilateral_exchange`.

Print the actor prompt:

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 0 --agent agent_1 --view actor
```

The script prints both messages and a SHA-256 digest of the message list. It calls no model.

## Reader frame

Function: `reader_messages(state, target, view)` in `commons.render.views`. A reader forecasts the target's next action from public information only.

System task sentence (reader): Forecast the target agent's next action. Choose exactly one valid action id. Return JSON only, with action exactly one of: the 11 ids for the target seat.

User message sections (reader): Task ("Predict the target agent's next action."), Setup (perspective `forecasting_target` and target id), Instruction (one paraphrase, see below), Target observation, State, Actions and Response. Target observation holds the target public observation (public role only, board and events under the selected view). State holds the public state (turn, board, all public roles and events under the selected view). Actions holds the target seat menu.

No reader prompt holds a hidden objective, a private goal sentence or private stocks for any seat.

Three instruction paraphrases exist:

- V1: Use everything shown about the target and shared state to forecast the target's next action.
- V2: Based on the information provided about the target and the shared situation, forecast the target's next action.
- V3: Forecast which action the target will take next, given what is shown about the target and the shared state.

Parameter `instruction_variant` selects the paraphrase. The CLI flag is `--variant` with default V1.

In the studied campaigns a frozen map chose the paraphrase by target model key, target seat and scenario. The map holds 108 cells. It ships at `src/commons/data/reader_instruction_variants.v1.json`. Function `instruction_variant_for(model_key, seat, scenario)` reads it. Keys use studied model keys such as `openrouter_gemini37_flash`. A key outside the map raises `KeyError`. Function `instruction_variant_map` returns the full map.

## The four reader views

The views share one frame. They differ only in the event history. View names for the API are `board`, `ordered`, `shuffled` and `tally`. Aliases `board_only` and `state_only` mean `board`. Alias `full_history` means `ordered`. Alias `shuffled_history` means `shuffled`. Alias `counts` means `tally`. Any other name raises `ValueError`.

### board

No events. The render sets the history horizon to 0. Recent events render as `(none)`. The board still shows commitments with status and turn numbers and inspection records with turn numbers. Pending shares do not show.

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view board
```

### ordered

The full public event list in order. Sharing outcomes show as `bilateral_exchange`, as they do for any outside viewer.

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view ordered
```

### shuffled

The same public events in a fixed shuffled order. Every rendered event list in the prompt uses the same shuffle seed.

Without an explicit seed context, the render uses a default context with the state seed, scenario, turn and target, plus version label `m4.3-fs1`. To match a studied shuffled prompt byte for byte, pass the original context with keys `campaign`, `snapshot_id` (the position id) and `forecast_suite_version` set to `m4.3-fs1`. Position ids do not ship in this repository.

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view shuffled
```

The CLI has no flag for the seed context. It always uses the default context. Use the API form below to pass an original context.

### tally

Counts in place of events. Each event list becomes counts by (actor, action), sorted by actor then action. Counts cover the public event list as an outside viewer sees it. Outcome names count under the names the outside viewer sees (for example `follow_up_completed` and `bilateral_exchange`). No order and no outcome text remain. The briefing history sentence changes to: recent_event_counts reports how many times each agent took each action, with no order and no outcome text. If the briefing lacks the history sentence, the tally render raises `ValueError` and refuses to render.

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view tally
```

## Goal-naming prompt

Function: `goal_naming_messages(state, target, view)` in `commons.render.views`. The reader names the target's hidden objective. The three-item objective menu stands in place of the action menu. Each menu entry shows the objective label and its private goal sentence as a description.

System task sentence (goal naming): Identify the target agent's hidden objective. Every agent holds exactly one objective from a fixed three-item menu, and the three agents hold three different objectives. Choose exactly one. Return JSON only, with action exactly one of: PERSONAL_THROUGHPUT, SYSTEM_HEALTH, SOCIAL_CREDIT.

Instruction sentence: Use everything shown about the target and shared state to identify which hidden objective the target was assigned. The V1 to V3 paraphrase does not apply here. The render replaces the reader instruction with this sentence.

CLI view names are `goal-board`, `goal-ordered` and `goal-tally`. Each name pairs the goal task with the reader view of the same suffix.

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view goal-board
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view goal-ordered
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view goal-tally
```

## How show_prompts reaches a state

Script `scripts/show_prompts.py` renders locally and calls no model. View names: `actor`, `board`, `ordered`, `shuffled`, `tally`, `goal-board`, `goal-ordered`, `goal-tally`.

Two state sources exist:

- Without `--log`: reset with `--seed`, `--scenario` and optional `--objectives`, then step `--turn` times with `--policy` for every seat. The default policy is `greedy`. Other values are `wait`, `uniform` and `softmax`. The prompt shows the state at the start of the requested turn. Turn 0 is the reset state.
- With `--log`: replay the episode log and take the state at `--turn`. Flag `--agent` names the actor seat or the reader target. A turn outside the log stops the script with an error message.

Examples. Reader view at a turn reached by scripted play, with paraphrase V2:

```powershell
.venv\Scripts\python scripts\show_prompts.py --seed 2700 --scenario delivery_pressure --turn 3 --agent agent_2 --view ordered --variant V2
```

Reader view from a logged episode (run the smoke episode in the README first):

```powershell
.venv\Scripts\python scripts\show_prompts.py --log data\runs\smoke\smoke.jsonl --turn 3 --agent agent_2 --view ordered
```

The two commands can show different states at the same turn number. The first state comes from `greedy` play by all seats. The second state comes from the logged seat policies.

## Sham actor prompts

Sham actor prompts use the actor form with sham text in three places: the briefing menu sentence, `hidden_objective` and `private_goal`. See the sham section in [ENVIRONMENT.md](ENVIRONMENT.md) for labels and sentences.

Render a sham actor prompt from a logged sham episode. Run the sham episode in the README first, then:

```powershell
.venv\Scripts\python scripts\show_prompts.py --log data\runs\sham\sham.jsonl --turn 0 --agent agent_1 --view actor
```

The script reads the display map from the log meta record and renders the sham labels. Without `--log`, the script renders active-arm actor prompts only. It has no sham flag.

In code, pass the display map to the actor render: `actor_messages(state, seat, sham_display=display)`. The state must hold `SYSTEM_HEALTH` for the seat. Else the render raises `ShamObjectiveError`.

## Render in code

Run this code from the repository root with the project environment. It picks the studied paraphrase for one cell, renders the ordered view, then renders the shuffled view with an explicit seed context:

```python
import sys
sys.path.insert(0, "src")

from commons.core import CONDITION, CommonsEnv
from commons.render import instruction_variant_for, reader_messages

env = CommonsEnv()
env.reset(2700, CONDITION, "delivery_pressure")
state = env.serialize_state()
variant = instruction_variant_for(
    "openrouter_gemini37_flash", "agent_1", "delivery_pressure")
messages = reader_messages(state, "agent_1", "ordered",
                             instruction_variant=variant)
print(variant, [m["role"] for m in messages])

shuffled = reader_messages(
    state, "agent_1", "shuffled",
    shuffle_seed_context={"campaign": "test-campaign",
                          "snapshot_id": "test-position",
                          "forecast_suite_version": "m4.3-fs1"})
print("shuffled roles:", [m["role"] for m in shuffled])
print("shuffled differs from ordered:", shuffled != messages)
```

The first print reads `V3 ['system', 'user']`. The seed context values in this example are test values. They prove the call form only. Studied prompts need the original campaign id and position id in those slots.

Related functions in `commons.render`:

- `actor_observation(state, seat)`: the actor observation dict. Pass `sham_display` for the sham form.
- `reader_observation(state, target, view)`: the reader observation dict before message building. It takes `instruction_variant` and `shuffle_seed_context`.
- `counts_only(observation)`: the tally transform on an ordered observation.
- `objective_menu()`: the three-item menu for goal naming.
- `default_shuffle_seed_context(state, target)`: the default shuffle context.
- `canonical_view(view)`: map a view name or alias to `board`, `ordered`, `shuffled` or `tally`.
- `messages_digest(messages)`: SHA-256 over the canonical JSON of a message list. Episode logs store this digest per decision and per call.

All render functions are deterministic. The same input gives the same bytes on each call.

## Golden prompt files

Four full prompt texts ship under `tests/golden` as `.txt` files: one actor prompt, one ordered reader prompt, one tally reader prompt and one ordered goal-naming prompt. File names hold the scenario, seed, turn and seat. Tests in `tests/test_render.py` check byte equality with these files. File `tests/golden/harness_reference.jsonl` holds digest prefixes for 12 seeded episodes. Tests in `tests/test_golden_reference.py` check this package against those digests. Golden files hold studied prompt bytes, with the separator character in event and action lines. Treat them as data. Do not edit them to match local style rules.
