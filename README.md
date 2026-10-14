# Commons

Commons is a three-agent shared-workplace environment for language-model agents, with hidden objectives and a delegation and sharing channel whose outcomes depend on the recipient's hidden objective. It is the environment called "Office" in the paper "Predicting Language-Model Agents from Behavioral Evidence".

This README is a placeholder. Full documentation is in preparation.

## Quick start

```
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"     # on Linux or macOS: .venv/bin/python
.venv/Scripts/python -m pytest
.venv/Scripts/python scripts/run_episode.py --seed 2700 --scenario delivery_pressure --agent-1 scripted:greedy --agent-2 scripted:softmax --agent-3 scripted:uniform --out data/runs/smoke
.venv/Scripts/python scripts/show_prompts.py --seed 2700 --scenario delivery_pressure --turn 0 --agent agent_1 --view actor
```

Running seats with a language model needs an `OPENROUTER_API_KEY` environment variable and a model registry (copy `models.example.json` to `models.json`); see the docstring of `scripts/run_episode.py`.

Provenance and the equivalence check against the original research code are described in [`docs/provenance/PROVENANCE.md`](docs/provenance/PROVENANCE.md).

No license has been chosen yet.
