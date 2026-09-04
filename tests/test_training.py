from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

try:
    from mafia_rl.training import evaluate_citizen, evaluate_with_artifacts
except ImportError:
    evaluate_citizen = None


@unittest.skipIf(evaluate_citizen is None, "RL optional dependencies are not installed")
class TrainingTest(unittest.TestCase):
    def test_random_baseline_counts_every_game_and_is_reproducible(self) -> None:
        first = evaluate_citizen(None, games=5, first_seed=500, random_seed=8)
        second = evaluate_citizen(None, games=5, first_seed=500, random_seed=8)
        self.assertEqual(first, second)
        self.assertEqual(first.citizen_wins + first.mafia_wins + first.draws, 5)

    def test_shared_random_baseline_counts_every_game_and_is_reproducible(self) -> None:
        first = evaluate_citizen(
            None, games=5, first_seed=500, random_seed=8, citizen_mode="shared"
        )
        second = evaluate_citizen(
            None, games=5, first_seed=500, random_seed=8, citizen_mode="shared"
        )
        self.assertEqual(first, second)
        self.assertEqual(first.citizen_wins + first.mafia_wins + first.draws, 5)

    def test_evaluation_writes_traces_and_summaries(self) -> None:
        class FirstLegalPolicy:
            def predict(self, observation, *, action_masks, deterministic):
                del observation, deterministic
                return action_masks.nonzero()[0][0], None

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            trained, baseline, destination = evaluate_with_artifacts(
                FirstLegalPolicy(),
                games=2,
                first_seed=700,
                random_seed=9,
                output_dir=output,
                trace_games=1,
            )
            trace = output / "traces" / "game_000001.json"
            self.assertTrue(trace.is_file())
            self.assertTrue((output / "traces" / "random_game_000001.json").is_file())
            self.assertTrue((output / "action_frequencies.csv").is_file())
            self.assertTrue((output / "phase_summary.csv").is_file())
            self.assertTrue((output / "mafia_kpis.csv").is_file())
            self.assertTrue((output / "strategy_summary.json").is_file())
            citizen_actions = output / "last_5_trained_games_all_citizen_actions.json"
            self.assertTrue(citizen_actions.is_file())
            payload = json.loads(trace.read_text(encoding="utf-8"))
            self.assertEqual(payload["seed"], 700)
            self.assertEqual(len(payload["true_roles"]), 10)
            self.assertTrue(payload["decisions"])
            self.assertIn("ground_truth_mafia_targets", payload["decisions"][0])
            self.assertIn("public_events", payload)
            citizen_payload = json.loads(citizen_actions.read_text(encoding="utf-8"))
            self.assertEqual(len(citizen_payload["games"]), 2)
            first_game_players = citizen_payload["games"][0]["citizen_aligned_players"]
            self.assertTrue(first_game_players)
            self.assertTrue(any(item["actions"] for item in first_game_players.values()))
            summary = json.loads(
                (output / "strategy_summary.json").read_text(encoding="utf-8")
            )
            kpis = summary["behavior"]["trained"]["mafia_identification_kpis"]
            self.assertEqual(set(kpis), {"day_targets", "nominations", "defence_votes"})
            self.assertEqual(trained.games, baseline.games, 2)
            self.assertEqual(destination, output)


if __name__ == "__main__":
    unittest.main()
