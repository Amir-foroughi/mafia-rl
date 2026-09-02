"""Headless episode runner and aggregate statistics."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

from .agents import Agent, RandomAgent
from .config import GameConfig
from .engine import GameEngine
from .models import Alignment, PlayerId


AgentFactory = Callable[[PlayerId, int], Agent]


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    winner: Alignment | None
    terminated: bool
    truncated: bool
    days: int
    event_count: int


@dataclass(frozen=True)
class BatchStats:
    games: int
    mafia_wins: int
    citizen_wins: int
    draws: int
    average_days: float


def run_episode(
    config: GameConfig | None = None,
    *,
    seed: int = 0,
    agent_factory: AgentFactory | None = None,
) -> EpisodeResult:
    engine = GameEngine(config)
    engine.reset(seed)
    factory = agent_factory or (lambda player_id, episode_seed: RandomAgent(
        seed=(episode_seed + 1) * 1_000_003 + player_id
    ))
    agents = {
        pid: factory(pid, seed)
        for pid in engine.state.players
    }
    while not (engine.state.terminated or engine.state.truncated):
        requests = engine.current_requests()
        actions = {
            pid: agents[pid].act(engine.observe(pid), request)
            for pid, request in requests.items()
        }
        engine.step(actions)
    return EpisodeResult(
        seed=seed,
        winner=engine.state.winner,
        terminated=engine.state.terminated,
        truncated=engine.state.truncated,
        days=engine.state.day,
        event_count=len(engine.state.events),
    )


def run_batch(
    games: int,
    config: GameConfig | None = None,
    *,
    first_seed: int = 0,
) -> BatchStats:
    if games < 1:
        raise ValueError("games must be positive")
    results = [
        run_episode(config, seed=first_seed + offset)
        for offset in range(games)
    ]
    winners = Counter(result.winner for result in results)
    return BatchStats(
        games=games,
        mafia_wins=winners[Alignment.MAFIA],
        citizen_wins=winners[Alignment.CITIZEN],
        draws=winners[None],
        average_days=sum(result.days for result in results) / games,
    )
