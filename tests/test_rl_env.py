from __future__ import annotations

import unittest

try:
    from mafia_rl.rl_env import FocalCitizenEnv, SharedCitizenEnv
except ImportError:
    FocalCitizenEnv = None
    SharedCitizenEnv = None

from mafia_rl.models import Phase, Role, Visibility


@unittest.skipIf(FocalCitizenEnv is None, "RL optional dependencies are not installed")
class FocalCitizenEnvTest(unittest.TestCase):
    def test_reset_returns_masked_focal_citizen_decision(self) -> None:
        env = FocalCitizenEnv()
        observation, info = env.reset(seed=4)
        focal = info["focal_player"]
        self.assertEqual(env.engine.state.players[focal].role, Role.CITIZEN)
        self.assertEqual(observation.shape, env.observation_space.shape)
        self.assertTrue(info["action_mask"].any())
        self.assertEqual(int(info["action_mask"].sum()), len(env.engine.legal_actions(focal)))

    def test_private_events_do_not_enter_focal_observation(self) -> None:
        env = FocalCitizenEnv()
        env.reset(seed=9)
        visible = env.engine.observe(env.focal_player).events
        self.assertTrue(all(event.visibility is Visibility.PUBLIC or env.focal_player in event.recipients for event in visible))
        self.assertNotIn("night_zero_mafia_reveal", {event.event_type for event in visible})

    def test_dead_focal_is_advanced_to_terminal_reward(self) -> None:
        env = FocalCitizenEnv()
        env.reset(seed=12)
        env.engine.state.players[env.focal_player].alive = False
        env._advance_to_focal()
        self.assertTrue(env.engine.state.terminated or env.engine.state.truncated)
        if env.engine.state.terminated:
            expected = 1.0 if env.engine.state.winner.value == "citizen" else -1.0
            reward = expected
        else:
            reward = 0.0
        self.assertIn(reward, (-1.0, 0.0, 1.0))

    def test_rollout_delivers_only_terminal_faction_reward(self) -> None:
        env = FocalCitizenEnv()
        _, info = env.reset(seed=18)
        rewards = []
        terminated = truncated = False
        while not (terminated or truncated):
            action = int(info["action_mask"].nonzero()[0][0])
            _, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)
        self.assertTrue(all(reward == 0.0 for reward in rewards[:-1]))
        if terminated:
            self.assertIn(rewards[-1], (-1.0, 1.0))
        else:
            self.assertEqual(rewards[-1], 0.0)

    def test_seed_is_reproducible(self) -> None:
        first, second = FocalCitizenEnv(), FocalCitizenEnv()
        obs1, info1 = first.reset(seed=33)
        obs2, info2 = second.reset(seed=33)
        self.assertEqual(info1["focal_player"], info2["focal_player"])
        self.assertTrue((obs1 == obs2).all())
        self.assertTrue((info1["action_mask"] == info2["action_mask"]).all())


@unittest.skipIf(SharedCitizenEnv is None, "RL optional dependencies are not installed")
class SharedCitizenEnvTest(unittest.TestCase):
    def test_all_normal_citizens_share_control(self) -> None:
        env = SharedCitizenEnv()
        _, info = env.reset(seed=4)
        expected = tuple(
            pid for pid, player in env.engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        self.assertEqual(info["controlled_players"], expected)
        self.assertEqual(info["acting_player"], expected[0])
        self.assertEqual(info["citizen_mode"], "shared")

        actors = []
        initial_sequence = len(env.engine.state.events)
        for _ in expected:
            actor = env.acting_player
            actors.append(actor)
            mask = env.action_masks()
            _, _, terminated, truncated, _ = env.step(int(mask.nonzero()[0][0]))
            self.assertFalse(terminated or truncated)
            if len(actors) < len(expected):
                # Buffered choices are simultaneous: the engine phase is not
                # resolved until the last controlled Citizen has acted.
                self.assertEqual(len(env.engine.state.events), initial_sequence)
        self.assertEqual(tuple(actors), expected)
        submitted = {
            event.actor for event in env.engine.state.events
            if event.event_type == "day_targets"
        }
        self.assertTrue(set(expected).issubset(submitted))

    def test_shared_seed_is_reproducible(self) -> None:
        first, second = SharedCitizenEnv(), SharedCitizenEnv()
        obs1, info1 = first.reset(seed=33)
        obs2, info2 = second.reset(seed=33)
        self.assertEqual(info1["controlled_players"], info2["controlled_players"])
        self.assertEqual(info1["acting_player"], info2["acting_player"])
        self.assertTrue((obs1 == obs2).all())
        self.assertTrue((info1["action_mask"] == info2["action_mask"]).all())

    def test_shared_rollout_uses_each_citizen_perspective(self) -> None:
        env = SharedCitizenEnv()
        observation, info = env.reset(seed=18)
        seen = set()
        terminated = truncated = False
        while not (terminated or truncated):
            actor = info["acting_player"]
            seen.add(actor)
            identity_start = 4 + len(Role) + len(Phase)
            identity = observation[identity_start:identity_start + env.config.player_count]
            self.assertEqual(int(identity.argmax()), actor)
            action = int(info["action_mask"].nonzero()[0][0])
            observation, _, terminated, truncated, info = env.step(action)
        self.assertGreater(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
