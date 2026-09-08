from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch
import unittest

from scripts import test as test_script
from scripts.aws.support import AwsError, InstanceConfig


CONFIG = InstanceConfig(
    instance_id="i-abc",
    region="us-west-2",
    instance_type="c7a.16xlarge",
    security_group_id="sg-123",
    default_cpus=12,
)


class SshTestScriptTests(unittest.TestCase):
    def test_main_reports_all_three_checks_in_order_on_success(self) -> None:
        stdout = StringIO()
        with (
            patch.object(
                test_script,
                "check_ssh_key",
                return_value="available at /tmp/key",
            ),
            patch.object(
                test_script,
                "check_instance",
                return_value=("i-abc is running", {"InstanceId": "i-abc"}),
            ),
            patch.object(
                test_script,
                "check_ssh_connection",
                return_value="connected to ubuntu@1.2.3.4",
            ),
            redirect_stdout(stdout),
        ):
            status = test_script.main([])

        self.assertEqual(status, 0)
        output = stdout.getvalue()
        self.assertIn("Workshop SSH tests", output)
        self.assertIn("PASS  ssh-key", output)
        self.assertIn("available at /tmp/key", output)
        self.assertIn("PASS  instance", output)
        self.assertIn("i-abc is running", output)
        self.assertIn("PASS  connection", output)
        self.assertIn("connected to ubuntu@1.2.3.4", output)
        self.assertLess(output.index("ssh-key"), output.index("instance"))
        self.assertLess(output.index("instance"), output.index("connection"))

    def test_main_skips_connection_when_the_instance_is_not_running(self) -> None:
        stdout = StringIO()
        with (
            patch.object(
                test_script,
                "check_ssh_key",
                return_value="available at /tmp/key",
            ),
            patch.object(
                test_script,
                "check_instance",
                side_effect=AwsError(
                    "Shared instance i-abc is stopped.",
                    "Run `python scripts/aws/instance.py start`.",
                ),
            ),
            patch.object(test_script, "check_ssh_connection") as connection,
            redirect_stdout(stdout),
        ):
            status = test_script.main([])

        self.assertEqual(status, 1)
        output = stdout.getvalue()
        self.assertIn("PASS  ssh-key", output)
        self.assertIn("FAIL  instance", output)
        self.assertIn("Problem: Shared instance i-abc is stopped.", output)
        self.assertIn("scripts/aws/instance.py start", output)
        self.assertIn("SKIP  connection", output)
        self.assertIn("instance is not running", output)
        self.assertLess(output.index("ssh-key"), output.index("instance"))
        self.assertLess(output.index("instance"), output.index("connection"))
        connection.assert_not_called()

    def test_main_still_checks_later_steps_when_the_key_fails(self) -> None:
        stdout = StringIO()
        with (
            patch.object(
                test_script,
                "check_ssh_key",
                side_effect=AwsError(
                    "The workshop SSH key is missing.",
                    "An instructor must run publish.",
                ),
            ),
            patch.object(
                test_script,
                "check_instance",
                return_value=("i-abc is running", {"InstanceId": "i-abc"}),
            ),
            patch.object(
                test_script,
                "check_ssh_connection",
                return_value="connected to ubuntu@1.2.3.4",
            ) as connection,
            redirect_stdout(stdout),
        ):
            status = test_script.main([])

        self.assertEqual(status, 1)
        output = stdout.getvalue()
        self.assertIn("FAIL  ssh-key", output)
        self.assertIn("PASS  instance", output)
        self.assertIn("PASS  connection", output)
        connection.assert_called_once()

    def test_verbose_prints_a_traceback_on_failure(self) -> None:
        stderr = StringIO()
        with (
            patch.object(
                test_script,
                "check_ssh_key",
                side_effect=AwsError("no key", "publish the key"),
            ),
            patch.object(
                test_script,
                "check_instance",
                side_effect=AwsError("no instance", "run setup"),
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            status = test_script.main(["--verbose"])

        self.assertEqual(status, 1)
        self.assertIn("AwsError", stderr.getvalue())

    def test_check_ssh_key_uses_the_cached_identity(self) -> None:
        with patch.object(
            test_script,
            "cached_workshop_ssh_identity",
            return_value=Path("/tmp/workshop-id_ed25519"),
        ):
            detail = test_script.check_ssh_key()

        self.assertEqual(detail, "available at /tmp/workshop-id_ed25519")

    def test_check_instance_fails_when_stopped(self) -> None:
        with (
            patch.object(test_script, "load_instance_config", return_value=CONFIG),
            patch.object(test_script, "ec2_client"),
            patch.object(
                test_script,
                "require_live_instance",
                return_value={"State": {"Name": "stopped"}, "InstanceId": "i-abc"},
            ),
            patch.object(test_script, "instance_state", return_value="stopped"),
        ):
            with self.assertRaises(AwsError) as raised:
                test_script.check_instance()

        self.assertIn("stopped", raised.exception.problem)
        self.assertIn("scripts/aws/instance.py start", raised.exception.resolution)
        self.assertNotIn("setup", raised.exception.resolution.lower())

    def test_check_instance_does_not_suggest_setup_when_unavailable(self) -> None:
        with patch.object(
            test_script,
            "load_instance_config",
            side_effect=AwsError(
                "No shared AWS instance is registered.",
                "An instructor must run `python scripts/aws/instance.py setup`.",
            ),
        ):
            with self.assertRaises(AwsError) as raised:
                test_script.check_instance()

        self.assertIn("No shared AWS instance", raised.exception.problem)
        self.assertNotIn("setup", raised.exception.resolution.lower())

    def test_check_instance_does_not_suggest_setup_for_other_states(self) -> None:
        with (
            patch.object(test_script, "load_instance_config", return_value=CONFIG),
            patch.object(test_script, "ec2_client"),
            patch.object(
                test_script,
                "require_live_instance",
                return_value={"State": {"Name": "pending"}, "InstanceId": "i-abc"},
            ),
            patch.object(test_script, "instance_state", return_value="pending"),
        ):
            with self.assertRaises(AwsError) as raised:
                test_script.check_instance()

        self.assertIn("pending", raised.exception.problem)
        self.assertNotIn("setup", raised.exception.resolution.lower())

    def test_check_instance_returns_the_running_description(self) -> None:
        description = {
            "State": {"Name": "running"},
            "InstanceId": "i-abc",
            "PublicIpAddress": "1.2.3.4",
        }
        with (
            patch.object(test_script, "load_instance_config", return_value=CONFIG),
            patch.object(test_script, "ec2_client"),
            patch.object(
                test_script,
                "require_live_instance",
                return_value=description,
            ),
            patch.object(test_script, "instance_state", return_value="running"),
        ):
            detail, returned = test_script.check_instance()

        self.assertEqual(detail, "i-abc is running")
        self.assertIs(returned, description)

    def test_check_ssh_connection_reports_the_ssh_target(self) -> None:
        session = Mock()
        session.target = "ubuntu@1.2.3.4"
        with (
            patch.object(
                test_script,
                "open_ssh_session",
                return_value=_context(session),
            ),
            patch.object(
                test_script,
                "run_ssh",
                return_value=Mock(returncode=0, stdout="", stderr=""),
            ),
        ):
            detail = test_script.check_ssh_connection({"InstanceId": "i-abc"})

        self.assertEqual(detail, "connected to ubuntu@1.2.3.4")


def _context(value: object):
    class _Manager:
        def __enter__(self) -> object:
            return value

        def __exit__(self, *_exc: object) -> None:
            return None

    return _Manager()


if __name__ == "__main__":
    unittest.main()
