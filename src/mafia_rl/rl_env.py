"""Gymnasium adapters for focal and shared-policy Citizen experiments.

Install the ``rl`` project extra to use this module.  The game engine itself
intentionally remains free of learning-framework dependencies.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Callable

try:
    import gymnasium as gym
    import numpy as np
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError("install mafia-rl[rl] to use FocalCitizenEnv") from exc

from .agents import Agent, RandomAgent
from .config import GameConfig
from .engine import GameEngine
from .models import (
    Action,
    Alignment,
    DayTarget,
    DefenceVote,
    Investigate,
    MafiaKillVote,
    NominationVote,
    Phase,
    Protect,
    Role,
    RunoffVote,
)


OpponentFactory = Callable[[int, int], Agent]


def build_action_catalog(player_count: int) -> tuple[Action, ...]:
    """Return the fixed, deterministic catalog shared by every decision kind."""
    seats = tuple(range(player_count))
    actions: list[Action] = []
    actions.extend(
        DayTarget(combo)
        for size in range(4)
        for combo in itertools.combinations(seats, size)
    )
    actions.extend(NominationVote(combo) for combo in itertools.combinations(seats, 2))
    actions.extend(RunoffVote(combo) for size in (1, 2) for combo in itertools.combinations(seats, size))
    actions.extend(DefenceVote(seat) for seat in seats)
    actions.extend(MafiaKillVote(seat) for seat in seats)
    actions.extend(Protect(seat) for seat in seats)
    actions.extend(Investigate(seat) for seat in seats)
    return tuple(actions)


class FocalCitizenEnv(gym.Env):
    """Single-agent masked environment with seeded random opponents.

    Other players are advanced automatically.  A dead focal player remains an
    episode participant: the simulation runs to completion and returns the
    Citizen faction's terminal result.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        opponent_factory: OpponentFactory | None = None,
    ) -> None:
        super().__init__()
        self.config = config or GameConfig()
        self.config.validate()
        if self.config.citizen_count < 1:
            raise ValueError("FocalCitizenEnv requires a normal Citizen")
        self.engine = GameEngine(self.config)
        self.catalog = build_action_catalog(self.config.player_count)
        self._catalog_index = {action: index for index, action in enumerate(self.catalog)}
        self.action_space = gym.spaces.Discrete(len(self.catalog))
        # Scalars + seat features + public action-history matrices by Day.
        n = self.config.player_count
        self._observation_size = 4 + 4 + len(Phase) + 7 * n + self.config.max_days * 4 * n * n
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self._observation_size,), dtype=np.float32
        )
        self._opponent_factory = opponent_factory
        self._opponents: dict[int, Agent] = {}
        self.focal_player: int | None = None
        self._episode_seed = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        del options
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode_seed = int(seed)
        self.engine.reset(self._episode_seed)
        citizens = [
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN
        ]
        # Independent seeded choice makes the focal seat random among normal Citizens.
        self.focal_player = random.Random(self._episode_seed ^ 0xC1712E).choice(citizens)
        factory = self._opponent_factory or (
            lambda pid, episode_seed: RandomAgent((episode_seed + 1) * 1_000_003 + pid)
        )
        self._opponents = {
            pid: factory(pid, self._episode_seed)
            for pid in self.engine.state.players if pid != self.focal_player
        }
        self._advance_to_focal()
        return self._encode_observation(), self._info()

    def step(self, action: int):
        if self.focal_player is None:
            raise RuntimeError("reset() must be called before step()")
        requests = self.engine.current_requests()
        request = requests.get(self.focal_player)
        if request is None:
            raise RuntimeError("the environment is not awaiting a focal action")
        chosen = self.catalog[int(action)] if self.action_space.contains(action) else None
        if chosen not in request.legal_actions:
            raise ValueError(f"masked or out-of-range action: {action}")
        actions = self._random_actions(requests)
        actions[self.focal_player] = chosen
        self.engine.step(actions)
        self._advance_to_focal()
        terminated = self.engine.state.terminated
        truncated = self.engine.state.truncated
        reward = 0.0
        if terminated:
            reward = 1.0 if self.engine.state.winner is Alignment.CITIZEN else -1.0
        return self._encode_observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        """MaskablePPO-compatible boolean legal-action mask."""
        mask = np.zeros(len(self.catalog), dtype=np.bool_)
        if self.focal_player is None:
            return mask
        request = self.engine.current_requests().get(self.focal_player)
        if request is not None:
            for action in request.legal_actions:
                mask[self._catalog_index[action]] = True
        return mask

    def _random_actions(self, requests):
        return {
            pid: self._opponents[pid].act(self.engine.observe(pid), request)
            for pid, request in requests.items() if pid != self.focal_player
        }

    def _advance_to_focal(self) -> None:
        assert self.focal_player is not None
        while not (self.engine.state.terminated or self.engine.state.truncated):
            requests = self.engine.current_requests()
            if self.engine.state.players[self.focal_player].alive and self.focal_player in requests:
                return
            self.engine.step(self._random_actions(requests))

    def _info(self) -> dict:
        return {
            "action_mask": self.action_masks(),
            "focal_player": self.focal_player,
            "winner": self.engine.state.winner.value if self.engine.state.winner else None,
        }

    def _encode_observation(self) -> np.ndarray:
        assert self.focal_player is not None
        return self._encode_player_observation(self.focal_player)

    def _encode_player_observation(self, player_id: int) -> np.ndarray:
        """Encode one player's information using the common policy input shape."""
        observation = self.engine.observe(player_id)
        n = self.config.player_count
        values: list[float] = [
            min(observation.day, self.config.max_days) / self.config.max_days,
            observation.public_living_mafia / self.config.mafia_count,
            observation.announced_eliminated / n,
            observation.announced_mafia / self.config.mafia_count,
        ]
        values.extend(float(observation.own_role is role) for role in Role)
        values.extend(float(observation.phase is phase) for phase in Phase)
        values.extend(float(observation.player_id == pid) for pid in range(n))
        values.extend(float(observation.alive[pid]) for pid in range(n))
        values.extend(float(observation.revealed_alignments.get(pid) is Alignment.MAFIA) for pid in range(n))
        values.extend(float(observation.revealed_alignments.get(pid) is Alignment.CITIZEN) for pid in range(n))
        values.extend(float(pid in observation.mafia_teammates) for pid in range(n))
        values.extend(float(observation.investigation_results.get(pid) is Alignment.MAFIA) for pid in range(n))
        values.extend(float(observation.investigation_results.get(pid) is Alignment.CITIZEN) for pid in range(n))

        event_kinds = ("day_targets", "nomination_vote", "runoff_vote", "defence_vote")
        history = np.zeros((self.config.max_days, 4, n, n), dtype=np.float32)
        for event in observation.events:
            if event.event_type not in event_kinds or event.actor is None:
                continue
            kind = event_kinds.index(event.event_type)
            targets = event.payload.get("targets")
            if targets is None:
                targets = (event.payload["target"],)
            for target in targets:
                history[event.day - 1, kind, event.actor, int(target)] = 1.0
        values.extend(history.ravel().tolist())
        return np.asarray(values, dtype=np.float32)


class SharedCitizenEnv(FocalCitizenEnv):
    """Turn-based adapter in which all normal Citizens share one policy.

    A Gym step collects one normal Citizen's action.  When every living normal
    Citizen requested in the current engine phase has acted, their buffered
    actions are submitted jointly with seeded random actions for all other
    roles.  Consequently every normal-Citizen decision becomes a transition in
    the single shared policy's rollout.
    """

    def __init__(
        self,
        config: GameConfig | None = None,
        *,
        opponent_factory: OpponentFactory | None = None,
    ) -> None:
        super().__init__(config, opponent_factory=opponent_factory)
        self.controlled_players: tuple[int, ...] = ()
        self.acting_player: int | None = None
        self._pending_players: list[int] = []
        self._pending_actions: dict[int, Action] = {}

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        gym.Env.reset(self, seed=seed)
        del options
        if seed is None:
            seed = int(self.np_random.integers(0, 2**31 - 1))
        self._episode_seed = int(seed)
        self.engine.reset(self._episode_seed)
        self.controlled_players = tuple(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        # Kept for compatibility with existing recorders and analysis files.
        self.focal_player = self.controlled_players[0]
        factory = self._opponent_factory or (
            lambda pid, episode_seed: RandomAgent((episode_seed + 1) * 1_000_003 + pid)
        )
        self._opponents = {
            pid: factory(pid, self._episode_seed)
            for pid in self.engine.state.players if pid not in self.controlled_players
        }
        self._pending_players = []
        self._pending_actions = {}
        self.acting_player = None
        self._advance_to_shared_decision()
        return self._encode_observation(), self._info()

    def step(self, action: int):
        if self.acting_player is None:
            raise RuntimeError("reset() must be called before step()")
        actor = self.acting_player
        request = self.engine.current_requests().get(actor)
        if request is None:
            raise RuntimeError("the environment is not awaiting the acting Citizen")
        chosen = self.catalog[int(action)] if self.action_space.contains(action) else None
        if chosen not in request.legal_actions:
            raise ValueError(f"masked or out-of-range action: {action}")
        self._pending_actions[actor] = chosen
        self._pending_players.pop(0)
        if self._pending_players:
            self.acting_player = self._pending_players[0]
        else:
            requests = self.engine.current_requests()
            actions = self._random_actions(requests)
            actions.update(self._pending_actions)
            self.engine.step(actions)
            self._pending_actions = {}
            self.acting_player = None
            self._advance_to_shared_decision()
        terminated = self.engine.state.terminated
        truncated = self.engine.state.truncated
        reward = 0.0
        if terminated:
            reward = 1.0 if self.engine.state.winner is Alignment.CITIZEN else -1.0
        return self._encode_observation(), reward, terminated, truncated, self._info()

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(len(self.catalog), dtype=np.bool_)
        if self.acting_player is None:
            return mask
        request = self.engine.current_requests().get(self.acting_player)
        if request is not None:
            for action in request.legal_actions:
                mask[self._catalog_index[action]] = True
        return mask

    def _random_actions(self, requests):
        return {
            pid: self._opponents[pid].act(self.engine.observe(pid), request)
            for pid, request in requests.items()
            if pid not in self.controlled_players
        }

    def _advance_to_shared_decision(self) -> None:
        while not (self.engine.state.terminated or self.engine.state.truncated):
            requests = self.engine.current_requests()
            controlled = [
                pid for pid in self.controlled_players
                if self.engine.state.players[pid].alive and pid in requests
            ]
            if controlled:
                self._pending_players = controlled
                self.acting_player = controlled[0]
                return
            self.engine.step(self._random_actions(requests))
        self._pending_players = []
        self.acting_player = None

    def _info(self) -> dict:
        return {
            "action_mask": self.action_masks(),
            "acting_player": self.acting_player,
            "controlled_players": self.controlled_players,
            "citizen_mode": "shared",
            "winner": self.engine.state.winner.value if self.engine.state.winner else None,
        }

    def _encode_observation(self) -> np.ndarray:
        # Gymnasium requires a valid terminal observation.  The first controlled
        # seat is used only after termination, when no further action is taken.
        player_id = self.acting_player
        if player_id is None:
            player_id = self.controlled_players[0]
        return self._encode_player_observation(player_id)
