from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from scripts import test as test_script
from scripts.aws.support import AwsError


class SshTestScriptTests(unittest.TestCase):
    def test_main_returns_zero_when_ssh_works(self) -> None:
        stdout = StringIO()
        with (
            patch.object(
                test_script,
                "check_connection",
                return_value="SSH ok",
            ),
            redirect_stdout(stdout),
        ):
            status = test_script.main([])

        self.assertEqual(status, 0)
        output = stdout.getvalue()
        self.assertIn("SSH communication test", output)
        self.assertIn("PASS  ssh", output)
        self.assertIn("SSH ok", output)

    def test_main_returns_one_when_ssh_fails(self) -> None:
        stderr = StringIO()
        with (
            patch.object(
                test_script,
                "check_connection",
                side_effect=AwsError(
                    "Shared instance i-abc is stopped.",
                    "Run `python scripts/aws/instance.py start`.",
                ),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            status = test_script.main([])

        self.assertEqual(status, 1)
        output = stderr.getvalue()
        self.assertIn("FAIL  ssh", output)
        self.assertIn("Problem: Shared instance i-abc is stopped.", output)
        self.assertIn("scripts/aws/instance.py start", output)

    def test_verbose_prints_a_traceback_on_failure(self) -> None:
        stderr = StringIO()
        with (
            patch.object(
                test_script,
                "check_connection",
                side_effect=AwsError("no instance", "run setup"),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            status = test_script.main(["--verbose"])

        self.assertEqual(status, 1)
        self.assertIn("AwsError", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
