# Mafia RL

A configurable, headless Mafia game engine intended for reinforcement-learning
experiments. The current implementation contains the deterministic engine,
seeded random agents, a batch simulation runner, and an optional masked
focal-Citizen Gymnasium adapter. The core has no learning-framework dependency.

## Run the tests

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests -v
```

## Run random games

The runner supports a single episode or an aggregated batch:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m mafia_rl.runner episode --seed 11
python -m mafia_rl.runner batch --games 1000 --first-seed 0
```

Game rules can be changed from the command line. For example:

```powershell
python -m mafia_rl.runner batch --games 50 --mafia 2 --citizens 4 --max-days 8
```

After installing the project, the same commands are available through the
`mafia-rl` executable (for example, `mafia-rl episode --seed 11`). The Python
API remains available as well:

```python
from mafia_rl.simulation import run_batch

print(run_batch(1_000, first_seed=0))
```

The approved rules and staged development plan are in `docs/plan.md`.

## First RL experiment

Install the optional dependencies with `pip install -e ".[rl]"`, then use
`mafia_rl.rl_env.FocalCitizenEnv`. It exposes a fixed discrete catalog and a
MaskablePPO-compatible `action_masks()` method. Random opponents advance
automatically, including after the focal Citizen dies, and the only nonzero
reward is the terminal faction result.
