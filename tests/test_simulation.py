from __future__ import annotations

import unittest

from mafia_rl.config import GameConfig
from mafia_rl.engine import GameEngine
from mafia_rl.models import MafiaKillVote, Phase, Protect, Role
from mafia_rl.simulation import run_batch, run_episode


class SimulationTest(unittest.TestCase):
    def test_random_episode_completes(self) -> None:
        result = run_episode(seed=11)
        self.assertTrue(result.terminated or result.truncated)
        self.assertLessEqual(result.days, 10)

    def test_random_episode_is_reproducible(self) -> None:
        self.assertEqual(run_episode(seed=23), run_episode(seed=23))

    def test_batch_counts_every_game(self) -> None:
        stats = run_batch(20, first_seed=100)
        self.assertEqual(
            stats.mafia_wins + stats.citizen_wins + stats.draws,
            stats.games,
        )

    def test_max_day_is_reported_as_truncation(self) -> None:
        engine = GameEngine(GameConfig(max_days=1))
        engine.reset(seed=5)
        engine.state.phase = Phase.NIGHT
        mafia = [
            pid for pid, player in engine.state.players.items()
            if player.role is Role.MAFIA
        ]
        doctor = next(
            pid for pid, player in engine.state.players.items()
            if player.role is Role.DOCTOR
        )
        target = next(
            pid for pid, player in engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        actions = {pid: MafiaKillVote(target) for pid in mafia}
        actions[doctor] = Protect(target)
        detective = next(
            pid for pid, player in engine.state.players.items()
            if player.role is Role.DETECTIVE
        )
        actions[detective] = engine.legal_actions(detective)[0]
        transition = engine.step(actions)
        self.assertTrue(transition.truncated)
        self.assertFalse(transition.terminated)
        self.assertEqual(transition.phase_after, Phase.TERMINAL)
        self.assertTrue(all(reward == 0.0 for reward in transition.rewards.values()))


if __name__ == "__main__":
    unittest.main()
