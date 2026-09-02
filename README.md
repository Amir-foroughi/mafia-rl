# Mafia RL

A configurable, headless Mafia game engine intended for reinforcement-learning
experiments. The current implementation contains the deterministic engine,
seeded random agents, and a batch simulation runner. It deliberately has no UI
or learning-framework dependency.

## Run the tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

## Run random games

```python
from mafia_rl.simulation import run_batch

print(run_batch(1_000, first_seed=0))
```

The approved rules and staged development plan are in `docs/plan.md`.
