"""Training and held-out evaluation for focal or shared Citizen policies."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import GameConfig
from .rl_env import FocalCitizenEnv, SharedCitizenEnv


CITIZEN_MODES = ("focal", "shared")


def _citizen_env(mode: str, config: GameConfig | None = None):
    if mode == "focal":
        return FocalCitizenEnv(config)
    if mode == "shared":
        return SharedCitizenEnv(config)
    raise ValueError(f"unknown citizen mode: {mode!r}")


class MaskedPolicy(Protocol):
    def predict(self, observation, *, action_masks, deterministic: bool): ...


@dataclass(frozen=True)
class EvaluationResult:
    games: int
    citizen_wins: int
    mafia_wins: int
    draws: int

    @property
    def citizen_win_rate(self) -> float:
        return self.citizen_wins / self.games


def train_citizen(
    model_path: str | Path,
    *,
    total_timesteps: int = 20_000,
    seed: int = 0,
    config: GameConfig | None = None,
    verbose: int = 1,
    resume_from: str | Path | None = None,
    citizen_mode: str = "focal",
) -> Path:
    """Train a new policy or continue a saved MaskablePPO checkpoint."""
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install mafia-rl[rl] to train a policy") from exc

    destination = Path(model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    env = _citizen_env(citizen_mode, config)
    if resume_from is None:
        model = MaskablePPO(
            "MlpPolicy",
            env,
            seed=seed,
            verbose=verbose,
            n_steps=256,
            batch_size=64,
            gamma=0.99,
            policy_kwargs={"net_arch": [128, 128]},
        )
        reset_num_timesteps = True
    else:
        checkpoint = _existing_checkpoint(resume_from)
        model = MaskablePPO.load(str(checkpoint), env=env, verbose=verbose)
        model.set_random_seed(seed)
        reset_num_timesteps = False
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        progress_bar=False,
    )
    model.save(str(destination))
    env.close()
    return destination if destination.suffix == ".zip" else Path(f"{destination}.zip")


def load_citizen_policy(
    model_path: str | Path,
    config: GameConfig | None = None,
    *,
    citizen_mode: str = "focal",
):
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:  # pragma: no cover
        raise ImportError("install mafia-rl[rl] to evaluate a policy") from exc
    return MaskablePPO.load(
        str(_existing_checkpoint(model_path)), env=_citizen_env(citizen_mode, config)
    )


def _existing_checkpoint(model_path: str | Path) -> Path:
    path = Path(model_path)
    candidates = (path, Path(f"{path}.zip")) if path.suffix != ".zip" else (path,)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Citizen checkpoint not found: {path}")


def _action_details(action) -> tuple[str, tuple[int, ...], str]:
    action_type = type(action).__name__
    if hasattr(action, "targets"):
        targets = tuple(int(item) for item in action.targets)
    else:
        targets = (int(action.target),)
    target_text = "none" if not targets else "|".join(str(item) for item in targets)
    return action_type, targets, f"{action_type}:{target_text}"


class EvaluationRecorder:
    """Collect compact aggregates and write only a capped number of traces."""

    def __init__(self, label: str, output_dir: Path, trace_games: int, citizen_mode: str = "focal") -> None:
        self.label = label
        self.output_dir = output_dir
        self.trace_games = trace_games
        self.citizen_mode = citizen_mode
        self.frequencies: Counter[tuple[str, str, int, str]] = Counter()
        self.phase_totals: Counter[str] = Counter()
        self.phase_legal_totals: Counter[str] = Counter()
        self.phase_target_totals: Counter[str] = Counter()
        self.phase_self_targets: Counter[str] = Counter()
        self.kpi_selections: Counter[str] = Counter()
        self.kpi_mafia_selections: Counter[str] = Counter()
        self.kpi_decisions: Counter[str] = Counter()
        self.kpi_decisions_with_mafia: Counter[str] = Counter()
        self.total_days = 0
        self.last_citizen_games: deque[dict] = deque(maxlen=5)
        self._game_number = 0
        self._current_seed = 0
        self._trace: dict | None = None

    def start_game(self, env: FocalCitizenEnv, seed: int) -> None:
        self._game_number += 1
        self._current_seed = seed
        if self._game_number <= self.trace_games:
            self._trace = {
                "schema_version": 1,
                "policy": self.label,
                "game_number": self._game_number,
                "seed": seed,
                "focal_player": env.focal_player,
                "focal_role": "citizen",
                "citizen_mode": self.citizen_mode,
                "controlled_citizens": list(getattr(env, "controlled_players", (env.focal_player,))),
                "true_roles": {
                    str(pid): player.role.value
                    for pid, player in env.engine.state.players.items()
                },
                "role_information_note": (
                    "Post-game analysis only; true roles were not exposed to the acting policy."
                ),
                "decisions": [],
            }
        else:
            self._trace = None

    def record_decision(self, env: FocalCitizenEnv, action_index: int) -> None:
        focal = getattr(env, "acting_player", None)
        if focal is None:
            focal = env.focal_player
        assert focal is not None
        action = env.catalog[action_index]
        action_type, targets, signature = _action_details(action)
        phase = env.engine.state.phase.value
        legal_count = int(env.action_masks().sum())
        self.frequencies[(phase, action_type, len(targets), signature)] += 1
        self.phase_totals[phase] += 1
        self.phase_legal_totals[phase] += legal_count
        self.phase_target_totals[phase] += len(targets)
        self.phase_self_targets[phase] += int(focal in targets)
        kpi_name = {
            "day_target": "day_targets",
            "nomination": "nominations",
            "defence_vote": "defence_votes",
        }.get(phase)
        mafia_targets = [
            target
            for target in targets
            if env.engine.state.players[target].alignment.value == "mafia"
        ]
        if kpi_name is not None:
            self.kpi_selections[kpi_name] += len(targets)
            self.kpi_mafia_selections[kpi_name] += len(mafia_targets)
            self.kpi_decisions[kpi_name] += 1
            self.kpi_decisions_with_mafia[kpi_name] += int(bool(mafia_targets))
        if self._trace is not None:
            observation = env.engine.observe(focal)
            self._trace["decisions"].append(
                {
                    "day": observation.day,
                    "actor": focal,
                    "phase": phase,
                    "action_index": action_index,
                    "action_type": action_type,
                    "targets": list(targets),
                    "ground_truth_mafia_targets": mafia_targets,
                    "legal_action_count": legal_count,
                    "living_players": [pid for pid, alive in observation.alive.items() if alive],
                    "announced_eliminated": observation.announced_eliminated,
                    "announced_mafia": observation.announced_mafia,
                    "revealed_alignments": {
                        str(pid): alignment.value
                        for pid, alignment in observation.revealed_alignments.items()
                    },
                }
            )

    def finish_game(self, env: FocalCitizenEnv) -> None:
        self.total_days += env.engine.state.day
        self.last_citizen_games.append(self._citizen_action_history(env))
        if self._trace is None:
            return
        self._trace["result"] = {
            "winner": env.engine.state.winner.value if env.engine.state.winner else None,
            "terminated": env.engine.state.terminated,
            "truncated": env.engine.state.truncated,
            "days": env.engine.state.day,
        }
        self._trace["public_events"] = [
            {
                "sequence": event.sequence,
                "day": event.day,
                "type": event.event_type,
                "actor": event.actor,
                "payload": dict(event.payload),
            }
            for event in env.engine.observe(env.focal_player).events
            if event.visibility.value == "public"
        ]
        trace_dir = self.output_dir / "traces"
        trace_dir.mkdir(parents=True, exist_ok=True)
        prefix = "game" if self.label == "trained" else f"{self.label}_game"
        path = trace_dir / f"{prefix}_{self._game_number:06d}.json"
        path.write_text(json.dumps(self._trace, indent=2), encoding="utf-8")

    def _citizen_action_history(self, env: FocalCitizenEnv) -> dict:
        action_events = {
            "day_targets": "DayTarget",
            "nomination_vote": "NominationVote",
            "runoff_vote": "RunoffVote",
            "defence_vote": "DefenceVote",
            "protection_selected": "Protect",
            "investigation_result": "Investigate",
        }
        focal = env.focal_player
        players = {}
        for pid, player in env.engine.state.players.items():
            if player.alignment.value != "citizen":
                continue
            actions = []
            for event in env.engine.state.events:
                if event.actor != pid or event.event_type not in action_events:
                    continue
                targets = event.payload.get("targets")
                if targets is None:
                    targets = (event.payload["target"],)
                actions.append(
                    {
                        "sequence": event.sequence,
                        "day": event.day,
                        "action_type": action_events[event.event_type],
                        "targets": [int(target) for target in targets],
                    }
                )
            players[str(pid)] = {
                "role": player.role.value,
                "is_trained_focal_citizen": (
                    self.label == "trained" and self.citizen_mode == "focal" and pid == focal
                ),
                "uses_shared_trained_policy": (
                    self.label == "trained" and self.citizen_mode == "shared"
                    and player.role.value == "citizen"
                ),
                "actions": actions,
            }
        return {
            "game_number": self._game_number,
            "seed": self._current_seed,
            "focal_player": focal,
            "citizen_mode": self.citizen_mode,
            "true_roles": {
                str(pid): player.role.value
                for pid, player in env.engine.state.players.items()
            },
            "result": {
                "winner": env.engine.state.winner.value if env.engine.state.winner else None,
                "truncated": env.engine.state.truncated,
                "days": env.engine.state.day,
            },
            "citizen_aligned_players": players,
        }


def evaluate_citizen(
    policy: MaskedPolicy | None,
    *,
    games: int = 1_000,
    first_seed: int = 100_000,
    config: GameConfig | None = None,
    random_seed: int = 0,
    recorder: EvaluationRecorder | None = None,
    citizen_mode: str = "focal",
) -> EvaluationResult:
    """Evaluate a learned policy, or a uniform-random focal baseline."""
    if games < 1:
        raise ValueError("games must be positive")
    env = _citizen_env(citizen_mode, config)
    rng = np.random.default_rng(random_seed)
    citizen_wins = mafia_wins = draws = 0
    for offset in range(games):
        episode_seed = first_seed + offset
        observation, _ = env.reset(seed=episode_seed)
        if recorder is not None:
            recorder.start_game(env, episode_seed)
        terminated = truncated = False
        while not (terminated or truncated):
            mask = env.action_masks()
            if policy is None:
                action = int(rng.choice(np.flatnonzero(mask)))
            else:
                action, _ = policy.predict(
                    observation, action_masks=mask, deterministic=True
                )
                action = int(action)
            if recorder is not None:
                recorder.record_decision(env, action)
            observation, _, terminated, truncated, _ = env.step(action)
        if recorder is not None:
            recorder.finish_game(env)
        if truncated:
            draws += 1
        elif env.engine.state.winner.value == "citizen":
            citizen_wins += 1
        else:
            mafia_wins += 1
    env.close()
    return EvaluationResult(games, citizen_wins, mafia_wins, draws)


def evaluate_with_artifacts(
    policy: MaskedPolicy,
    *,
    games: int = 1_000,
    first_seed: int = 100_000,
    config: GameConfig | None = None,
    random_seed: int = 0,
    output_dir: str | Path = "artifacts/evaluation",
    trace_games: int = 10,
    citizen_mode: str = "focal",
) -> tuple[EvaluationResult, EvaluationResult, Path]:
    """Run paired evaluation and write compact, human-readable analysis files."""
    if trace_games < 0:
        raise ValueError("trace_games cannot be negative")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    trace_dir = destination / "traces"
    if trace_dir.is_dir():
        for pattern in ("game_*.json", "random_game_*.json"):
            for old_trace in trace_dir.glob(pattern):
                old_trace.unlink()
    trained_recorder = EvaluationRecorder("trained", destination, trace_games, citizen_mode)
    random_recorder = EvaluationRecorder("random", destination, trace_games, citizen_mode)
    trained = evaluate_citizen(
        policy,
        games=games,
        first_seed=first_seed,
        config=config,
        recorder=trained_recorder,
        citizen_mode=citizen_mode,
    )
    baseline = evaluate_citizen(
        None,
        games=games,
        first_seed=first_seed,
        config=config,
        random_seed=random_seed,
        recorder=random_recorder,
        citizen_mode=citizen_mode,
    )
    _write_action_frequencies(destination, (trained_recorder, random_recorder))
    _write_phase_summary(destination, (trained_recorder, random_recorder))
    _write_mafia_kpis(destination, (trained_recorder, random_recorder))
    _write_strategy_summary(
        destination, trained, baseline, trained_recorder, random_recorder, first_seed,
        citizen_mode,
    )
    _write_last_citizen_games(destination, trained_recorder)
    return trained, baseline, destination


def _write_action_frequencies(output_dir: Path, recorders) -> None:
    path = output_dir / "action_frequencies.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("policy", "phase", "action_type", "target_count", "action", "count", "phase_percent"))
        for recorder in recorders:
            for (phase, action_type, target_count, signature), count in sorted(recorder.frequencies.items()):
                percentage = count / recorder.phase_totals[phase] if recorder.phase_totals[phase] else 0.0
                writer.writerow((recorder.label, phase, action_type, target_count, signature, count, f"{percentage:.6f}"))


def _write_phase_summary(output_dir: Path, recorders) -> None:
    path = output_dir / "phase_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("policy", "phase", "decisions", "avg_legal_actions", "avg_targets", "self_target_rate"))
        for recorder in recorders:
            for phase, decisions in sorted(recorder.phase_totals.items()):
                writer.writerow(
                    (
                        recorder.label,
                        phase,
                        decisions,
                        f"{recorder.phase_legal_totals[phase] / decisions:.3f}",
                        f"{recorder.phase_target_totals[phase] / decisions:.3f}",
                        f"{recorder.phase_self_targets[phase] / decisions:.6f}",
                    )
                )


def _recorder_summary(recorder: EvaluationRecorder) -> dict:
    phases = {}
    by_phase: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for (phase, _action_type, _target_count, signature), count in recorder.frequencies.items():
        by_phase[phase].append((count, signature))
    for phase, decisions in sorted(recorder.phase_totals.items()):
        phases[phase] = {
            "decisions": decisions,
            "average_legal_actions": recorder.phase_legal_totals[phase] / decisions,
            "average_targets": recorder.phase_target_totals[phase] / decisions,
            "self_target_rate": recorder.phase_self_targets[phase] / decisions,
            "most_common_actions": [
                {"action": signature, "count": count}
                for count, signature in sorted(by_phase[phase], reverse=True)[:5]
            ],
        }
    return {
        "average_game_days": recorder.total_days / recorder._game_number,
        "mafia_identification_kpis": _mafia_kpi_summary(recorder),
        "phases": phases,
    }


def _mafia_kpi_summary(recorder: EvaluationRecorder) -> dict:
    result = {}
    for name in ("day_targets", "nominations", "defence_votes"):
        selections = recorder.kpi_selections[name]
        mafia_selections = recorder.kpi_mafia_selections[name]
        decisions = recorder.kpi_decisions[name]
        decisions_with_mafia = recorder.kpi_decisions_with_mafia[name]
        result[name] = {
            "total_selections": selections,
            "mafia_selections": mafia_selections,
            "mafia_selection_rate": mafia_selections / selections if selections else None,
            "total_decisions": decisions,
            "decisions_with_at_least_one_mafia": decisions_with_mafia,
            "decision_hit_rate": decisions_with_mafia / decisions if decisions else None,
        }
    return result


def _write_mafia_kpis(output_dir: Path, recorders) -> None:
    path = output_dir / "mafia_kpis.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "policy",
                "kpi",
                "total_selections",
                "mafia_selections",
                "mafia_selection_rate",
                "total_decisions",
                "decisions_with_mafia",
                "decision_hit_rate",
            )
        )
        for recorder in recorders:
            for name, values in _mafia_kpi_summary(recorder).items():
                selection_rate = values["mafia_selection_rate"]
                hit_rate = values["decision_hit_rate"]
                writer.writerow(
                    (
                        recorder.label,
                        name,
                        values["total_selections"],
                        values["mafia_selections"],
                        "" if selection_rate is None else f"{selection_rate:.6f}",
                        values["total_decisions"],
                        values["decisions_with_at_least_one_mafia"],
                        "" if hit_rate is None else f"{hit_rate:.6f}",
                    )
                )


def _result_dict(result: EvaluationResult) -> dict:
    return {
        "games": result.games,
        "citizen_wins": result.citizen_wins,
        "mafia_wins": result.mafia_wins,
        "draws": result.draws,
        "citizen_win_rate": result.citizen_win_rate,
    }


def _write_strategy_summary(
    output_dir, trained, baseline, trained_recorder, random_recorder, first_seed,
    citizen_mode="focal",
) -> None:
    summary = {
        "schema_version": 1,
        "citizen_mode": citizen_mode,
        "evaluation": {
            "first_seed": first_seed,
            "last_seed": first_seed + trained.games - 1,
            "paired_seeds": True,
            "trained": _result_dict(trained),
            "random": _result_dict(baseline),
            "citizen_win_rate_difference": trained.citizen_win_rate - baseline.citizen_win_rate,
        },
        "behavior": {
            "trained": _recorder_summary(trained_recorder),
            "random": _recorder_summary(random_recorder),
        },
        "interpretation_note": (
            "These are observed action frequencies, not symbolic rules or causal explanations of the neural network."
        ),
    }
    (output_dir / "strategy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def _write_last_citizen_games(output_dir: Path, recorder: EvaluationRecorder) -> None:
    payload = {
        "schema_version": 1,
        "policy": recorder.label,
        "scope": (
            "The final five trained-policy evaluation games. Citizen-aligned players "
            "include normal Citizens, the Doctor, and the Detective."
        ),
        "privacy_note": (
            "Roles and private night actions are post-game analysis data and were not "
            "available to the Citizen policy while acting."
        ),
        "games": list(recorder.last_citizen_games),
    }
    (output_dir / "last_5_trained_games_all_citizen_actions.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
