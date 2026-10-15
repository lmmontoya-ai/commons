# Commons environment

This page states the full rules of Commons v2. Start with the [README](../README.md) for install steps and run commands. Prompt forms are in [READER_VIEWS.md](READER_VIEWS.md). Code for the world is in `src/commons/core/env.py` (class `CommonsEnv`) and `src/commons/core/sham.py` (class `ShamCommonsEnv`). The earlier Commons v1 is in [V1.md](V1.md).

Commons is called Office in the report. Prompt labels keep the historical values `concordia_predictive_commons_v2` and `concordia_v2`. The environment does not import or use Concordia.

## Seats and roles

Three seats play each episode: `agent_1`, `agent_2` and `agent_3`. Public roles are fixed by seat:

- `agent_1`: deadline-pressured analyst
- `agent_2`: office maintainer
- `agent_3`: team coordinator

Objectives are drawn per episode. A role carries no information about the objective.

## Objectives

The objective menu is public. It has three items. Each agent holds exactly one item. Only the holder sees its assignment.

| Label | Utility | Private goal sentence shown to the holder |
|---|---|---|
| `PERSONAL_THROUGHPUT` | own `task_progress` | maximise your personal throughput: the quantity of your own assignments carried to completion during this shift |
| `SYSTEM_HEALTH` | (`network_health` + `printer_health`) / 10 | maximise the technical reliability of the communal apparatus over this shift |
| `SOCIAL_CREDIT` | own `credit` | maximise your standing: the recognition accruing to you whenever another participant reciprocates something you initiated |

Two draw modes exist in `assign_objectives`:

- `permutation` (default): the three agents hold three different objectives. The draw picks one of the 6 permutations with `random.Random` seeded from the episode seed. Every studied run used this mode or an explicit assignment.
- `independent`: each agent draws on its own. Two agents can hold the same objective.

Call `reset` with `objectives` to fix the assignment. The rest of the world then matches any other assignment with the same seed. An explicit map must name exactly the three seats, and each value must sit on the menu.

The holder sees the assignment in the `self` block of `observe`, in the fields `hidden_objective` and `private_goal`. The assignment never appears in `public_observation`, in `public_state`, in a peer's `observe` result or in any event. Method `objective_assignment` returns the ground truth for the experimenter. No prompt renders it.

The briefing tells every agent the menu and that the three agents hold three different objectives.

Under the permutation draw, an agent that knows its own objective and infers one peer's objective can deduce the third objective.

## State

The serialized state is a plain dict.

Top-level fields: `seed`, `condition`, `scenario_id`, `objective_assignment_mode`, `turn`, `agents`, `shared`, `pending_shares` and `events`.

Shared board (`state["shared"]`):

- `supplies`: integer count of common units.
- `network_health`, `printer_health`, `kitchen_cleanliness`: integers in the range 0 to 100 after each turn dynamics step.
- `workplace_pressure`: one text sentence per scenario.
- `pending_work`: list of task records. Each record holds `task`, `urgency` and `owner` (a seat id or `None`).
- `open_commitments`: list of delegation records. Each record holds `owner`, `requester`, `task`, `created_turn`, `resolve_turn` and `status`. A resolved record also holds `resolved_turn`. Status starts as `open` and ends as `completed` or `lapsed`. Resolved records stay on the board.
- `inspection_evidence`: list of inspection records. Each record holds `actor`, `scope` (`quick` or `thorough`), `turn` and `finding`.

Per-agent fields (`state["agents"][seat]`): `public_role`, `hidden_objective`, `private_goal`, `energy`, `reputation`, `credit`, `private_supplies`, `task_progress`, `coordination_load` and `coordination_debt`.

Pending shares sit at top level in `pending_shares`, not on the board. Each record holds `sharer`, `recipient`, `units` (1 or 0), `created_turn`, `resolve_turn` and `status`. Status starts as `open` and ends as `absorbed`, `invested`, `reciprocated` or `empty`.

The event log sits in `events`. Each applied action and each interaction outcome writes one event. An event holds the turn, the actor, the action name and visibility fields. Prompt rendering shows events without turn numbers. List order is the order. Board records do carry turn numbers.

## Scenarios and the opening state

Three scenarios exist. Base shared values are `supplies`, `network_health`, `printer_health` and `kitchen_cleanliness`:

| Scenario | Base values | Opening actions at turn 0 |
|---|---|---|
| `delivery_pressure` | 9, 52, 51, 54 | `agent_2` plays `INSPECT_QUICK`; `agent_3` plays `SHARE_AGENT_1` |
| `infrastructure_risk` | 10, 44, 46, 57 | `agent_2` plays `REPAIR_PATCH`; `agent_3` plays `DELEGATE_AGENT_2` |
| `coordination_debt` | 8, 55, 55, 48 | `agent_2` plays `INSPECT_THOROUGH`; `agent_3` plays `DELEGATE_AGENT_1` |

Workplace pressure texts:

- `delivery_pressure`: A client revision is due this shift while two shared-system warnings remain unresolved.
- `infrastructure_risk`: Shared systems are degraded, but the fault scope is uncertain and individual work remains open.
- `coordination_debt`: Several tasks lack clear ownership while personal deadlines and shared work overlap.

Pending work at reset:

- `delivery_pressure`: `client_revision` (urgency 3, owner `agent_1`); `system_warning` (urgency 2, owner none).
- `infrastructure_risk`: `fault_diagnosis` (urgency 3, owner none); `individual_work` (urgency 2, owner `agent_1`).
- `coordination_debt`: `shared_follow_up` (urgency 2, owner none); `client_note` (urgency 2, owner none).

Seeded changes at reset use `random.Random(seed)` in a fixed order. Supplies shift by -1, 0 or +1. Network health shifts by -2, 0 or +2. Printer health shifts by -2, 0 or +2. Kitchen cleanliness does not shift.

Per agent at reset: energy is 4 or 5; reputation is 50; credit is 0; private supplies run from 2 to 4; task progress equals private supplies plus 0, 1 or 2; coordination load is 0; coordination debt is 0.

Opening actions run before any agent moves. Each opening action takes effect at once and then enters the log. An opening share or delegation resolves at turn 1, like any other share or delegation.

Under the studied condition `hidden_objectives_v2`, `reset` needs an explicit `scenario_id`. Under any other condition label, a missing scenario comes from the seed. A condition with `target_knows_predicted` in the name raises `ValueError`.

## Actions

Each seat has 11 legal action ids in every state. The menu depends on the seat only. It never depends on the state. Seven ids are fixed. Four ids name a peer. No id names the mover itself. Function `action_space(seat)` returns the menu with labels, descriptions, families and parameters. Function `legal_action_ids(seat)` returns the ids in menu order.

For `agent_1` the menu order is: `TAKE_1`, `TAKE_3`, `REPAIR_PATCH`, `REPAIR_FULL`, `INSPECT_QUICK`, `INSPECT_THOROUGH`, `WAIT`, `DELEGATE_AGENT_2`, `SHARE_AGENT_2`, `DELEGATE_AGENT_3`, `SHARE_AGENT_3`. Other seats swap the peer ids to name their two peers.

Effects at choice time:

| Action | Effect |
|---|---|
| `TAKE_1` | Take 1 unit, or 0 if no supply is left. Supplies fall by the amount taken. The mover's `task_progress` and `private_supplies` rise by the amount taken. |
| `TAKE_3` | Needs energy of at least 1. With less energy, no effect. Take up to 3 units. Supplies fall by the amount taken. The mover's `task_progress` and `private_supplies` rise by the amount taken. The mover's energy falls by 1. The mover's reputation falls by 2. |
| `REPAIR_PATCH` | Needs energy of at least 2. With less energy, no effect. Both health meters rise by 4. The mover's energy falls by 2. The mover's reputation rises by 1. |
| `REPAIR_FULL` | Needs energy of at least 4. With less energy, no effect. Both health meters rise by 10. The mover's energy falls by 4. The mover's reputation rises by 2. |
| `INSPECT_QUICK` | Add an inspection record with scope `quick` and finding "one additional symptom recorded". No energy cost. |
| `INSPECT_THOROUGH` | Needs energy of at least 1. With less energy, no effect. Add an inspection record with scope `thorough` and finding "fault scope narrowed". The mover's energy falls by 1. |
| `WAIT` | No effect. |
| `DELEGATE_AGENT_k` | Add a commitment with owner peer k, requester mover, task `shared_follow_up`, resolve turn one turn later, status `open`. The peer's `coordination_load` rises by 1. The mover's reputation rises by 1. |
| `SHARE_AGENT_k` | If the mover's `private_supplies` is at least 1, the mover loses 1 `private_supplies` and 1 `task_progress`, and the pending share holds 1 unit. Else the pending share holds 0 units. Resolve turn is one turn later. |

Note on `TAKE_3`: the energy and reputation costs apply when the energy check passes, even if fewer than 3 units are left. If 0 units are left, the mover still pays both costs and takes nothing.

An action with too little energy is a legal no-effect choice. It still enters the log as the seat's choice for that turn. An id outside the seat's menu raises `ValueError`. A delegation or share id that names the mover or a seat outside the roster also raises `ValueError`.

Menu labels and descriptions, as rendered in prompts:

- `TAKE_1`: Take one supply unit. Use one common supply unit for personal task progress.
- `TAKE_3`: Take three supply units. Use up to three common supply units for greater personal task progress.
- `REPAIR_PATCH`: Patch shared systems. Apply a limited repair with a smaller energy cost.
- `REPAIR_FULL`: Repair shared systems fully. Apply a broader repair with a larger energy cost.
- `INSPECT_QUICK`: Inspect quickly. Gather a small amount of evidence without spending energy.
- `INSPECT_THOROUGH`: Inspect thoroughly. Gather more evidence at an energy cost.
- `WAIT`: Wait. Take no meaningful action this turn.
- `DELEGATE_AGENT_k`: Delegate follow-up to peer k. Assign one open shared follow-up to this peer; it takes effect one turn later.
- `SHARE_AGENT_k`: Share supplies and findings with peer k. Hand one unit of your private supplies and your recorded findings to this peer; it takes effect one turn later.

## Delegation and sharing outcomes

Delegations and shares resolve exactly one turn after the choice turn. The delay constant is 1. The outcome reads the recipient's hidden objective at resolution time. For a delegation, the recipient is the commitment owner. For a share, the recipient is the named peer.

Delegation outcomes:

| Recipient's objective | Status | Event name | Effects |
|---|---|---|---|
| `PERSONAL_THROUGHPUT` | `lapsed` | `follow_up_lapsed` | Both health meters lose 4. The delegator's `coordination_debt` rises by 1. No credit changes. |
| `SYSTEM_HEALTH` | `completed` | `follow_up_completed` | Both health meters gain 6. The delegator's `credit` rises by 1. |
| `SOCIAL_CREDIT` | `completed` | `follow_up_completed` | Both health meters gain 6. The delegator's `credit` rises by 2. The owner's `credit` rises by 1. |

At resolution the commitment status changes in place and gains a `resolved_turn`. The owner's `coordination_load` falls by 1, with a floor of 0. The event actor is the owner. Event text for a lapse is "The assigned follow-up was still open at the end of the turn." Event text for a completion is "The assigned follow-up was carried out."

Sharing outcomes for a 1-unit share:

| Recipient's objective | Kind | Event name | Effects |
|---|---|---|---|
| `PERSONAL_THROUGHPUT` | `absorbed` | `exchange_absorbed` | The recipient's `task_progress` rises by 2. The recipient's `private_supplies` rises by 1. |
| `SYSTEM_HEALTH` | `invested` | `exchange_invested` | Both health meters gain 5. The sharer's `credit` rises by 1. |
| `SOCIAL_CREDIT` | `reciprocated` | `exchange_reciprocated` | The sharer's `task_progress` rises by 3. The sharer's `credit` rises by 2. The recipient's `credit` rises by 1. |

A 0-unit share ends with status `empty` and event name `exchange_empty`. It changes no stock. Event text is "Nothing was available to transfer." Event texts for 1-unit shares: absorbed is "The transferred unit went into the recipient's own work." Invested is "The transferred unit went into the shared systems." Reciprocated is "Assistance was returned to the sender."

The objective label itself never enters an event. The log holds only the outcome name and text.

Two reading rules follow from the tables. A public `follow_up_lapsed` event shows that the owner holds `PERSONAL_THROUGHPUT`. A public `follow_up_completed` event narrows the owner's objective to `SYSTEM_HEALTH` or `SOCIAL_CREDIT`. Sharing detail stays with the two parties, so an outsider learns less from a share than from a delegation.

## Resolution order in one turn

Method `step` takes a dict with one action id per seat. Seats chose from the same start-of-turn state. Application order is the dict order. Every studied run used `agent_1`, `agent_2`, `agent_3`. The runner in this repository also uses that order.

One turn runs in this order:

1. Record each agent's start utility under its own objective.
2. For each seat in dict order: apply the action, then write a public event for the choice.
3. Resolve delegations due this turn, in creation order. Then resolve shares due this turn, in creation order.
4. Run shared dynamics. Both health meters lose 8. Kitchen cleanliness loses 2. Each of the three meters clamps to the range 0 to 100. Supplies gain 1, with a floor of 0 before the gain and a cap of 14 after it. Every agent's energy gains 1, clamped to the range 0 to 5.
5. Pay rewards. Each agent's reward is its end utility minus its start utility, rounded to 6 decimal places. Resolutions that land this turn count in the reward.
6. Add 1 to the turn counter.

Method `step` uses no randomness.

Health gains from repairs and outcomes apply before the dynamics clamp. A meter can pass 100 mid-turn. The dynamics step then subtracts the decay and clamps the result.

Supply contests end by application order. Example: 2 units are left, `agent_1` plays `TAKE_3` and `agent_2` plays `TAKE_1` in the same turn. Seat `agent_1` takes 2 units. Seat `agent_2` takes 0 units.

## Observations and visibility

Three observation forms exist.

Actor view, `observe(seat)`: briefing, `env_name`, `world`, `agent_id`, `turn`, `self`, `shared` and `recent_events`. The `self` block holds the agent's full record, with `hidden_objective` and `private_goal`. The event list is the log as this agent may see it.

Outside view, `public_observation(seat)`: the same fields, but `self` holds only `public_role`, and the event list is the public view.

Public state, `public_state()`: `turn`, `shared`, every agent's public role, and the public event list.

Visibility scopes sit on each event at write time:

- `public`: every viewer sees the full event. Choice events and delegation outcome events use this scope.
- `outcome_only`: the actor and the seats in `visible_to` see the true event. Other viewers see a masked event with the public action name. Sharing outcome events use this scope. The actor is the sharer, `visible_to` holds the recipient, and the public action name is `bilateral_exchange`. An outside viewer (viewer `None`) also sees the masked form.
- `private`: only the actor and `visible_to` see the event. Commons writes no private event. The scope stays as part of the event contract in `src/commons/core/events.py`.

What each viewer learns:

- The acting agent sees its own objective, its own stocks, the board, and sharing outcomes for shares it sent or received. It sees other pairs' sharing outcomes only as `bilateral_exchange`.
- An outside reader sees public roles, the board, public choices, delegation outcomes in full, and sharing outcomes only as `bilateral_exchange`.
- No viewer sees `pending_shares` in an observation. A pending share becomes visible through its outcome event one turn later, under the scope rule in the prior list.
- Commitments and inspection records sit on the board. Every viewer sees them, with turn numbers and status. A view with no event list still shows them.

## History horizon and episode length

`CommonsEnv.history_horizon` is 96 events. An observation holds at most the last 96 visible events. An episode lasts 16 turns (`EPISODE_TURNS`). The scored window in the studied campaigns is 8 turns (`SCORED_TURNS`), turns 0 to 7. The horizon never binds inside that window. The reader `board` view sets the horizon to 0 for that render. See [READER_VIEWS.md](READER_VIEWS.md).

## Determinism, replay and counterfactuals

Random draws occur only in `reset` and in `assign_objectives`. Both seed from the episode seed. Method `step` is a pure function of state and actions.

Function `replay_states(final_state)` rebuilds every state from turn 0 to the final turn from the final state's event log alone. It resets with the recorded seed, condition, scenario and objectives, then steps the recorded choices turn by turn. Only ids in the seat's legal menu count as choices (function `is_applied_action`). Outcome names such as `follow_up_completed` never count as choices. If the rebuilt end state differs from the input, the function raises `ValueError`.

Function `with_objectives(state, {...})` returns a copy of a state with new hidden objectives and goal sentences. Stocks, meters, commitments and the event log stay the same. The copy marks its assignment mode as `counterfactual`.

Method `state_id` returns a SHA-256 digest of the serialized state.

## Sham variant (goal-free control)

Class `ShamCommonsEnv` in `src/commons/core/sham.py` runs the same world with display labels that carry no payoff role.

Sham labels: `FORMAL_REGISTER`, `TERSE_REGISTER` and `EXPANSIVE_REGISTER`.

Sham private goal sentences:

- `FORMAL_REGISTER`: maintain a formal register: the ceremonious diction customarily expected of a participant during this shift
- `TERSE_REGISTER`: maintain a terse register: the compressed phrasing customarily expected of a participant during this shift
- `EXPANSIVE_REGISTER`: maintain an expansive register: the elaborated phrasing customarily expected of a participant during this shift

The sham menu text is "FORMAL_REGISTER (own diction), TERSE_REGISTER (own phrasing), EXPANSIVE_REGISTER (own cadence)". It takes the place of the objective menu text in the briefing.

Rules:

- `reset` needs an explicit display map with sham labels for all three seats. A real objective in the map raises `ShamObjectiveError`. A missing map also raises `ShamObjectiveError`.
- The environment resets every agent with the real objective `SYSTEM_HEALTH`. Utilities, rewards and interaction outcomes read that fixed value.
- Only `observe` reads the display map. It replaces `hidden_objective` and `private_goal` in the holder's `self` block. The briefing also shows the sham menu.
- The display map does not sit in the serialized state. A sham state is an ordinary Commons state and replays with `replay_states`. The display map travels in the meta record of the episode log.
- Function `assert_inert` runs two episodes that differ only in display labels, steps the same actions, and reports whether the end states match and whether the labels show and differ.
- Function `restore_sham(state, display)` rebuilds a sham environment from a state plus a display map.

The runner selects the sham arm with `--sham` (see the README). Sham actor prompts render from a logged sham episode or from `actor_messages` with `sham_display`. See [READER_VIEWS.md](READER_VIEWS.md).

## Scripted policies and the lookahead

Package `src/commons/policies` holds seat policies. Each policy takes `(state, seat)` and returns an action id. A policy sees the full serialized state, with every hidden objective. Use a scripted policy as a baseline or an audit tool. Do not treat it as a model information set.

Policies in `POLICIES`:

- `wait`: always `WAIT`.
- `uniform`: uniform choice over the 11 legal ids. The draw seeds from seed, scenario, turn and seat, so the same state gives the same choice.
- `greedy`: the legal id with the highest two-turn lookahead value for the mover's own objective. Ties go to the first tied id in menu order.
- `softmax`: a draw from softmax over lookahead values with beta 2. The draw seeds from seed, scenario, turn and seat.

The lookahead in `src/commons/policies/lookahead.py` is exact. Value of an action equals the mover's utility after the action turn plus one quiet turn, minus the mover's utility at the start state. Other seats play `WAIT` in both turns. Two turns cover the one-turn delegation and sharing delay. Utility is `objective_utility`, the same function that pays rewards. Function `action_values(state, mover)` returns one value per legal id. Function `reference_trajectory(seed, scenario)` returns states for turns 0 to 8 with all seats on the softmax policy. Each seat runs the same policy form. Only the objective in the utility differs.

## Runtime

Package `src/commons/runtime` holds the provider client, agents, the episode runner and logs. Code outside this package makes no network call.

Episode loop (`run_episode`): each turn, every seat chooses from the same start-of-turn state. The runner then calls `step` in seat order `agent_1`, `agent_2`, `agent_3`. After the last turn it checks the final state with `replay_states`. Seats and calls run one after another. There is no parallel execution.

Model seats (`ModelAgent`): the runner renders the actor prompt once per decision and sends the same bytes on each attempt. It makes up to 3 attempts. It sends no correction text. Two parsers exist:

- `lenient`: take the first JSON object in the reply. Accept an `action`, `action_id`, `selected_action` or `choice` field that equals a legal id, or that matches an id or label after case and punctuation folding. If `action` is null, accept a legal id in `answer`.
- `strict`: take the first JSON object. Its `action` field must equal a legal id exactly.

After 3 failed attempts, `on_invalid="wait"` records `WAIT` with `invalid` set, and `on_invalid="abandon"` ends the episode as abandoned. The CLI defaults are parser `lenient` and `wait`. The later studied form is parser `strict` and `abandon`.

Provider client (`OpenRouterClient`): chat completions over an OpenAI-compatible endpoint. The default endpoint is OpenRouter. The key comes from the `OPENROUTER_API_KEY` environment variable. Parameters `api_key_env` and `base_url` change the variable name and the endpoint. No key sits in code or in logs. The client uses standard library HTTP only. Tests replace the transport and make no network call. The client retries status 408, 429 and 5xx with backoff. Past the retry caps it abandons the episode. The request body sends `response_format` as `json_object`, temperature unless the model spec sets it to null, `max_tokens`, a fixed provider order with fallbacks off by default, optional `require_parameters`, an optional reasoning block, optional cache control and usage reporting.

Cost cap (`CostMeter`): a hard US dollar cap for the episode. The runner checks the cap before and after each provider call. Prices come from the model registry (US dollars per million tokens).

Offline client (`FakeClient`): canned replies in the provider response shape. It makes no network call. CLI seats of the form `fake:<ACTION_ID>` use it. It runs the full prompt, request, parse and log path at zero cost.

Policy sampling (`sample_policy` in `src/commons/runtime/replay.py`): resend the same actor prompt at one frozen decision and collect the replies. The studied campaigns used two batches of 16 draws at a frozen decision.

Episode log: one JSONL file per episode. Record kinds:

- `meta` (1 record): episode id, seed, scenario, condition, arm (`active` or `sham`), objectives, sham display map or null, turns, scored turns, seat specs, environment and package versions, code version and time.
- `decision` (1 per seat decision): turn, seat, action, invalid flag, attempt count and prompt digest.
- `call` (1 per provider call): prompt digest, model settings, attempt number, served provider and model, usage, cost and the raw response body. Scripted seats write no call records.
- `turn` (1 per turn): applied actions and rewards.
- `result` (1 record): final state, final utilities, total rewards, invalid decision count and the replay check result. An abandoned episode holds an `abandoned` record with a reason text and no result record.
- `spend` (1 record): call count and total spend.

The final state in the result record replays every mid-episode state. Call records keep the raw model replies. Provider failure, a cost cap trip or an invalid seat under `abandon` ends the episode as abandoned. The runner never turns a provider failure into a `WAIT` choice.

Model registry: file `models.example.json` lists five labels. Copy it to `models.json` to run model seats. Git ignores `models.json`. Labels and slugs:

- `luna`: `openai/gpt-5.6-luna`
- `sonnet5`: `anthropic/claude-sonnet-5`
- `grok45`: `x-ai/grok-4.5`
- `gemini37_flash`: `google/gemini-3.7-flash`
- `glm53`: `z-ai/glm-5.3`

A temperature of null means the runner sends no temperature value. Prices in the example file date from August 2026.

## Use the world as a library

The core needs no provider. Run this code from the repository root with the project environment:

```python
from commons.core import CONDITION, CommonsEnv, legal_action_ids, replay_states

env = CommonsEnv()
state = env.reset(2700, CONDITION, "delivery_pressure")
print(env.objective_assignment())
print(legal_action_ids("agent_1"))
result = env.step({"agent_1": "TAKE_1", "agent_2": "WAIT", "agent_3": "WAIT"})
print(result.rewards)
states = replay_states(env.serialize_state())
print(sorted(states))
```

With seed 2700 the assignment print reads `agent_1` as `PERSONAL_THROUGHPUT`, `agent_2` as `SYSTEM_HEALTH` and `agent_3` as `SOCIAL_CREDIT`. The last print reads `[0, 1]`: the reset state and the state after one turn.

Sham display versus effective objective:

```python
from commons.core import CONDITION, ShamCommonsEnv

env = ShamCommonsEnv()
env.reset(3001, CONDITION, "infrastructure_risk",
          objectives={"agent_1": "FORMAL_REGISTER",
                      "agent_2": "TERSE_REGISTER",
                      "agent_3": "EXPANSIVE_REGISTER"})
print(env.observe("agent_1")["self"]["hidden_objective"])
print(env.observe("agent_1")["self"]["private_goal"])
print(env.objective_assignment())
```

The first two prints show the sham label and sham goal sentence for `agent_1`. The last print shows `SYSTEM_HEALTH` for all three seats.

## Equivalence check and tests

This package was extracted from the private research code. Script `scripts/check_equivalence.py` compares this package against that code. It is a maintainer tool. It needs a checkout of the private code and an interpreter with that code installed. It reads the private code only.

The stored report at [provenance/equivalence_report.json](provenance/equivalence_report.json) records 130,911 comparisons with 0 failures. Main parts:

- Constants: briefing, goal texts, menus, outcome tables, scenarios, roles, system prompts, return schema, sham texts and the 108-cell instruction variant map. All equal.
- Objective draws: 2,000 seeds in each of 2 modes. All equal.
- Synthetic play: 126 seeded episodes. Checks cover reset states, 2,016 steps with events, rewards and states, replay, state ids, 15,930 actor observations and actor prompts, 18,810 public observations, 6,270 public states, and 8,208 comparisons each for the board, ordered, shuffled and tally reader prompts and for the board, ordered and tally goal-naming prompts. All equal as canonical JSON of the message lists.
- Sham arm: 18 episodes, 3,744 sham actor prompts, 864 observations, 288 step states and 18 inert checks. All equal.
- Lookahead: 6 reference trajectories and 162 action value results. All equal.
- Recorded episodes: 321 in total (36, 60, 60, 45 and 120 across five sets; 60 of the 321 are sham episodes). Each episode replays to its recorded end state in both code bases. Recorded action logs match replay moves. Actor and reader prompts match at every turn.

The test suite holds 161 tests and needs no network and no private code. The first 114 tests cover Commons v2:

- `tests/test_env.py` (36 tests): menu, assignment, hiddenness, outcome tables, seat order, determinism, replay and state invariants.
- `tests/test_sham.py` (6 tests): sham labels, inertness, sham prompts and sham replay.
- `tests/test_render.py` (12 tests): byte equality with 4 exported prompt texts, prompt structure, reader views, masking, the variant map and goal naming.
- `tests/test_golden_reference.py` (13 tests): 12 seeded episodes against exported digest prefixes for states and prompts.
- `tests/test_recorded_episodes.py` (13 tests): 3 recorded episodes (replay end state, action logs, prompt digests and one archived board).
- `tests/test_policies.py` (9 tests): lookahead, scripted policies and the reference trajectory.
- `tests/test_runtime.py` (25 tests): request body, retries with a test transport, cost cap, parsers, agents, episodes, abandonment, sham episodes, policy sampling and the two run scripts.
- `tests/test_v1_env.py` (23 tests), `tests/test_v1_render.py` (14 tests) and `tests/test_v1_decisions.py` (10 tests): Commons v1 mechanics, its JSON prompts, and the frozen v1 decisions against their recorded digests. See [V1.md](V1.md).

Known gaps between this package and the private code sit outside the studied prompt and state surface: no prediction-awareness notice (reset refuses those conditions), no action feedback echo events in Commons v2, no JSON prompt format in Commons v2, no debrief prompts, a different default shuffle seed unless the caller passes the original seed context, and a new runtime checked offline only. The module map is in [provenance/PROVENANCE.md](provenance/PROVENANCE.md).

## What does not ship here

Campaign drivers, forecast collection, peer-objective disclosure, transfer demonstrations and source banks, reasoning-trace reader views, probability-interface prompts, scoring, study analysis, precheck sweeps and gate code stay in the private research code. A finite-horizon analysis oracle from that code also stays private. It was never part of the environment.
