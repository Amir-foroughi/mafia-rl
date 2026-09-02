from __future__ import annotations

import random
import unittest
from collections import Counter

from mafia_rl.config import GameConfig
from mafia_rl.engine import GameEngine
from mafia_rl.models import (
    Alignment,
    DayTarget,
    DeathPhase,
    DefenceVote,
    Investigate,
    InvalidActionError,
    MafiaKillVote,
    NominationVote,
    Phase,
    Protect,
    Role,
    RunoffVote,
)


class EngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = GameEngine()
        self.engine.reset(seed=7)

    def test_default_roster_and_night_zero(self) -> None:
        counts = Counter(player.role for player in self.engine.state.players.values())
        self.assertEqual(counts[Role.MAFIA], 3)
        self.assertEqual(counts[Role.DOCTOR], 1)
        self.assertEqual(counts[Role.DETECTIVE], 1)
        self.assertEqual(counts[Role.CITIZEN], 5)
        self.assertEqual(self.engine.state.phase, Phase.DAY_TARGET)
        self.assertEqual(self.engine.state.day, 1)
        announcement = self.engine.state.events[-1]
        self.assertEqual(announcement.event_type, "day_announcement")
        self.assertEqual(announcement.payload["cumulative_eliminated"], 0)

    def test_same_seed_assigns_same_roles(self) -> None:
        other = GameEngine()
        other.reset(seed=7)
        self.assertEqual(
            [p.role for p in self.engine.state.players.values()],
            [p.role for p in other.state.players.values()],
        )

    def test_day_target_allows_zero_three_and_self(self) -> None:
        requests = self.engine.current_requests()
        for player_id, request in requests.items():
            self.assertTrue(request.legal_actions)
            self.assertIn(DayTarget(()), request.legal_actions)
            self.assertIn(DayTarget((player_id,)), request.legal_actions)
            other = tuple(pid for pid in self.engine.state.players if pid != player_id)
            self.assertIn(DayTarget(tuple(sorted((player_id, *other[:2])))), request.legal_actions)
            for action in request.legal_actions:
                self.assertIsInstance(action, DayTarget)
                self.assertGreaterEqual(len(action.targets), 0)
                self.assertLessEqual(len(action.targets), 3)
                self.assertEqual(len(action.targets), len(set(action.targets)))

    def test_day_target_allows_mafia_teammate(self) -> None:
        mafia = [pid for pid, p in self.engine.state.players.items() if p.role is Role.MAFIA]
        self.assertIn(DayTarget((mafia[1],)), self.engine.legal_actions(mafia[0]))

    def test_day_target_rejects_dead_duplicate_and_too_many(self) -> None:
        actor = 0
        dead = 1
        self.engine.state.players[dead].alive = False
        legal = self.engine.legal_actions(actor)
        self.assertNotIn(DayTarget((dead,)), legal)
        self.assertNotIn(DayTarget((actor, actor)), legal)
        self.assertNotIn(DayTarget((0, 2, 3, 4)), legal)

    def test_invalid_joint_actor_set_is_rejected(self) -> None:
        requests = self.engine.current_requests()
        actions = {pid: req.legal_actions[0] for pid, req in requests.items()}
        actions.pop(next(iter(actions)))
        with self.assertRaises(InvalidActionError):
            self.engine.step(actions)

    def test_nomination_is_independent_and_excludes_self(self) -> None:
        self.engine.step({pid: DayTarget(()) for pid in self.engine.current_requests()})
        self.assertEqual(self.engine.state.phase, Phase.NOMINATION)
        legal = self.engine.legal_actions(2)
        self.assertIn(NominationVote((0, 1)), legal)
        self.assertIn(NominationVote((8, 9)), legal)
        self.assertTrue(all(2 not in action.targets for action in legal))
        self.assertTrue(all(len(action.targets) == len(set(action.targets)) == 2 for action in legal))

    def test_final_vote_tie_eliminates_nobody(self) -> None:
        self.engine.state.defendants = (0, 1)
        self.engine.state.phase = Phase.DEFENCE_VOTE
        voters = list(self.engine.current_requests())
        self.assertEqual(len(voters), 8)
        actions = {
            pid: DefenceVote(0 if index < 4 else 1)
            for index, pid in enumerate(voters)
        }
        self.engine.step(actions)
        self.assertTrue(self.engine.state.players[0].alive)
        self.assertTrue(self.engine.state.players[1].alive)
        self.assertEqual(self.engine.state.phase, Phase.NIGHT)

    def test_runoff_uses_stable_ranking(self) -> None:
        self.engine.state.phase = Phase.RUNOFF
        self.engine.state.locked_defendants = (0,)
        self.engine.state.runoff_candidates = (1, 2, 3)
        self.engine.state.runoff_open_seats = 1
        self.engine.state.original_nomination_totals = {1: 3, 2: 3, 3: 3}
        self.engine.state.day_targets = {
            0: (1,), 1: (2,), 2: (1,), 3: (2,), 4: (1,),
            5: (1,), 6: (1,), 7: (2,), 8: (2,), 9: (2,),
        }
        # Candidates and locked defendant do not vote. Split six votes 3-3.
        requests = self.engine.current_requests()
        actions = {}
        for index, pid in enumerate(requests):
            actions[pid] = RunoffVote((1 if index < 3 else 2,))
        self.engine.step(actions)
        # Vote totals tie; player 1 has more incoming targets and wins.
        self.assertEqual(self.engine.state.defendants, (0, 1))

    def test_runoff_choices_are_not_limited_by_targeting(self) -> None:
        self.engine.state.phase = Phase.RUNOFF
        self.engine.state.locked_defendants = (0,)
        self.engine.state.runoff_candidates = (1, 2, 3)
        self.engine.state.runoff_open_seats = 1
        self.engine.state.day_targets = {4: (9,)}
        self.assertEqual(
            set(self.engine.legal_actions(4)),
            {RunoffVote((1,)), RunoffVote((2,)), RunoffVote((3,))},
        )

    def test_self_target_counts_in_runoff_tiebreak(self) -> None:
        self.engine.state.phase = Phase.RUNOFF
        self.engine.state.locked_defendants = (0,)
        self.engine.state.runoff_candidates = (1, 2, 3)
        self.engine.state.runoff_open_seats = 1
        self.engine.state.original_nomination_totals = {1: 3, 2: 3, 3: 3}
        self.engine.state.day_targets = {1: (1,)}
        requests = self.engine.current_requests()
        actions = {
            pid: RunoffVote((1 if index % 2 == 0 else 2,))
            for index, pid in enumerate(requests)
        }
        self.engine.step(actions)
        self.assertEqual(self.engine.state.defendants, (0, 1))

    def test_mafia_observation_does_not_leak_to_citizen(self) -> None:
        mafia = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.MAFIA
        )
        citizen = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        self.assertTrue(self.engine.observe(mafia).mafia_teammates)
        self.assertEqual(self.engine.observe(citizen).mafia_teammates, ())
        citizen_types = {event.event_type for event in self.engine.observe(citizen).events}
        self.assertNotIn("night_zero_mafia_reveal", citizen_types)
        self.assertNotIn("episode_seed", citizen_types)

    def test_night_save_and_private_investigation(self) -> None:
        self.engine.state.phase = Phase.NIGHT
        mafia = [
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.MAFIA
        ]
        doctor = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.DOCTOR
        )
        detective = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.DETECTIVE
        )
        target = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        actions = {pid: MafiaKillVote(target) for pid in mafia}
        actions[doctor] = Protect(target)
        actions[detective] = Investigate(mafia[0])
        self.engine.step(actions)
        self.assertTrue(self.engine.state.players[target].alive)
        self.assertEqual(
            self.engine.observe(detective).investigation_results[mafia[0]],
            Alignment.MAFIA,
        )
        citizen = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN and pid != target
        )
        citizen_types = {event.event_type for event in self.engine.observe(citizen).events}
        self.assertNotIn("investigation_result", citizen_types)
        self.assertNotIn("protection_selected", citizen_types)
        self.assertNotIn("mafia_kill_ballots", citizen_types)

    def test_cumulative_announcement_reveals_inferable_alignments(self) -> None:
        mafia = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.MAFIA
        )
        citizen = next(
            pid for pid, player in self.engine.state.players.items()
            if player.role is Role.CITIZEN
        )
        self.engine._eliminate(mafia, DeathPhase.DAY)
        self.engine._eliminate(citizen, DeathPhase.NIGHT)
        self.engine.state.day = 2
        self.engine._announce_day()
        self.assertEqual(self.engine.state.announced_eliminated, 2)
        self.assertEqual(self.engine.state.announced_mafia, 1)
        self.assertEqual(
            self.engine.state.revealed_alignments,
            {mafia: Alignment.MAFIA, citizen: Alignment.CITIZEN},
        )

    def test_submission_order_does_not_change_public_resolution(self) -> None:
        other = GameEngine()
        other.reset(seed=7)
        requests = self.engine.current_requests()
        actions = {pid: request.legal_actions[0] for pid, request in requests.items()}
        self.engine.step(actions)
        other.step(dict(reversed(list(actions.items()))))
        self.assertEqual(self.engine.state.day_targets, other.state.day_targets)
        self.assertEqual(self.engine.state.events, other.state.events)

    def test_config_rejects_initial_parity(self) -> None:
        with self.assertRaises(ValueError):
            GameConfig(mafia_count=2, citizen_count=2, doctor_count=0,
                       detective_count=0).validate()


if __name__ == "__main__":
    unittest.main()
