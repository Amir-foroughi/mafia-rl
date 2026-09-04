"""Command-line runner for the Mafia engine's executable workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from .config import GameConfig
from .simulation import run_batch, run_episode


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return number


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = GameConfig()
    parser.add_argument("--mafia", type=_positive_int, default=defaults.mafia_count)
    parser.add_argument("--citizens", type=int, default=defaults.citizen_count)
    parser.add_argument("--doctors", type=int, default=defaults.doctor_count)
    parser.add_argument("--detectives", type=int, default=defaults.detective_count)
    parser.add_argument("--max-days", type=_positive_int, default=defaults.max_days)


def _add_citizen_mode_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--citizen-mode",
        choices=("focal", "shared"),
        default="focal",
        help="focal trains one normal Citizen; shared trains one policy through all normal Citizens",
    )


def _config_from_args(args: argparse.Namespace) -> GameConfig:
    config = GameConfig(
        mafia_count=args.mafia,
        citizen_count=args.citizens,
        doctor_count=args.doctors,
        detective_count=args.detectives,
        max_days=args.max_days,
    )
    config.validate()
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mafia-rl",
        description="Run seeded Mafia games with the built-in random agents.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    episode = commands.add_parser("episode", help="run one game")
    episode.add_argument("--seed", type=int, default=0)
    _add_config_arguments(episode)

    batch = commands.add_parser("batch", help="run games and aggregate results")
    batch.add_argument("--games", type=_positive_int, default=100)
    batch.add_argument("--first-seed", type=int, default=0)
    _add_config_arguments(batch)

    train = commands.add_parser("train-citizen", help="train focal or shared Citizen MaskablePPO")
    train.add_argument("--timesteps", type=_positive_int, default=20_000)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--model-path", default="artifacts/citizen_maskable_ppo")
    train.add_argument(
        "--resume-from",
        help="continue this checkpoint instead of creating a new policy",
    )
    _add_citizen_mode_argument(train)
    _add_config_arguments(train)

    evaluate = commands.add_parser("evaluate-citizen", help="evaluate against random opponents")
    evaluate.add_argument("--games", type=_positive_int, default=1_000)
    evaluate.add_argument("--first-seed", type=int, default=100_000)
    evaluate.add_argument("--random-seed", type=int, default=0)
    evaluate.add_argument("--model-path", default="artifacts/citizen_maskable_ppo")
    evaluate.add_argument("--analysis-dir", default="artifacts/evaluation")
    evaluate.add_argument(
        "--trace-games",
        type=_nonnegative_int,
        default=10,
        help="number of detailed trained and random game traces to save",
    )
    _add_citizen_mode_argument(evaluate)
    _add_config_arguments(evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = _config_from_args(args)
    except ValueError as error:
        parser.error(str(error))

    if args.command == "episode":
        result = run_episode(config, seed=args.seed)
        winner = result.winner.value if result.winner is not None else "draw"
        status = "truncated" if result.truncated else "finished"
        print(
            f"Episode seed={result.seed}: {status}, winner={winner}, "
            f"days={result.days}, events={result.event_count}"
        )
        return 0

    if args.command == "batch":
        stats = run_batch(args.games, config, first_seed=args.first_seed)
        print(f"Games: {stats.games}")
        print(f"Mafia wins: {stats.mafia_wins}")
        print(f"Citizen wins: {stats.citizen_wins}")
        print(f"Draws: {stats.draws}")
        print(f"Average days: {stats.average_days:.2f}")
        return 0

    from .training import evaluate_with_artifacts, load_citizen_policy, train_citizen

    if args.command == "train-citizen":
        saved = train_citizen(
            args.model_path,
            total_timesteps=args.timesteps,
            seed=args.seed,
            config=config,
            resume_from=args.resume_from,
            citizen_mode=args.citizen_mode,
        )
        print(f"Saved Citizen policy: {saved}")
        return 0

    policy = load_citizen_policy(
        args.model_path, config, citizen_mode=args.citizen_mode
    )
    trained, baseline, analysis_dir = evaluate_with_artifacts(
        policy,
        games=args.games,
        first_seed=args.first_seed,
        config=config,
        random_seed=args.random_seed,
        output_dir=args.analysis_dir,
        trace_games=args.trace_games,
        citizen_mode=args.citizen_mode,
    )
    print(f"Held-out seeds: {args.first_seed}..{args.first_seed + args.games - 1}")
    print(f"Trained Citizen wins: {trained.citizen_wins}/{trained.games} ({trained.citizen_win_rate:.1%})")
    print(f"Random Citizen wins: {baseline.citizen_wins}/{baseline.games} ({baseline.citizen_win_rate:.1%})")
    print(f"Difference: {trained.citizen_win_rate - baseline.citizen_win_rate:+.1%}")
    print(f"Trained Mafia wins/draws: {trained.mafia_wins}/{trained.draws}")
    print(f"Random Mafia wins/draws: {baseline.mafia_wins}/{baseline.draws}")
    print(f"Strategy analysis: {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
