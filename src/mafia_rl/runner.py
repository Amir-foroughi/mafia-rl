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


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    defaults = GameConfig()
    parser.add_argument("--mafia", type=_positive_int, default=defaults.mafia_count)
    parser.add_argument("--citizens", type=int, default=defaults.citizen_count)
    parser.add_argument("--doctors", type=int, default=defaults.doctor_count)
    parser.add_argument("--detectives", type=int, default=defaults.detective_count)
    parser.add_argument("--max-days", type=_positive_int, default=defaults.max_days)


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

    stats = run_batch(args.games, config, first_seed=args.first_seed)
    print(f"Games: {stats.games}")
    print(f"Mafia wins: {stats.mafia_wins}")
    print(f"Citizen wins: {stats.citizen_wins}")
    print(f"Draws: {stats.draws}")
    print(f"Average days: {stats.average_days:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
