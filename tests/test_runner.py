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


if __name__ == "__main__":
    unittest.main()
