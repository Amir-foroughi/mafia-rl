from __future__ import annotations

import contextlib
import io
import unittest

from mafia_rl.runner import main


class RunnerTest(unittest.TestCase):
    def run_command(self, *arguments: str) -> str:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            result = main(arguments)
        self.assertEqual(result, 0)
        return output.getvalue()

    def test_episode_command(self) -> None:
        output = self.run_command("episode", "--seed", "11")
        self.assertIn("Episode seed=11:", output)
        self.assertIn("winner=", output)

    def test_batch_command(self) -> None:
        output = self.run_command("batch", "--games", "3", "--first-seed", "20")
        self.assertIn("Games: 3", output)
        self.assertIn("Average days:", output)

    def test_training_commands_parse(self) -> None:
        parser = __import__("mafia_rl.runner", fromlist=["build_parser"]).build_parser()
        train = parser.parse_args(("train-citizen", "--timesteps", "10"))
        evaluate = parser.parse_args(("evaluate-citizen", "--games", "10"))
        self.assertEqual(train.command, "train-citizen")
        self.assertEqual(evaluate.command, "evaluate-citizen")
        self.assertEqual(train.citizen_mode, "focal")
        self.assertEqual(evaluate.citizen_mode, "focal")

    def test_shared_citizen_mode_parses(self) -> None:
        parser = __import__("mafia_rl.runner", fromlist=["build_parser"]).build_parser()
        train = parser.parse_args(("train-citizen", "--citizen-mode", "shared"))
        evaluate = parser.parse_args(("evaluate-citizen", "--citizen-mode", "shared"))
        self.assertEqual(train.citizen_mode, "shared")
        self.assertEqual(evaluate.citizen_mode, "shared")

    def test_resume_and_analysis_arguments_parse(self) -> None:
        parser = __import__("mafia_rl.runner", fromlist=["build_parser"]).build_parser()
        train = parser.parse_args(
            ("train-citizen", "--resume-from", "old_model", "--model-path", "new_model")
        )
        evaluate = parser.parse_args(
            ("evaluate-citizen", "--analysis-dir", "reports", "--trace-games", "0")
        )
        self.assertEqual(train.resume_from, "old_model")
        self.assertEqual(train.model_path, "new_model")
        self.assertEqual(evaluate.analysis_dir, "reports")
        self.assertEqual(evaluate.trace_games, 0)


if __name__ == "__main__":
    unittest.main()
