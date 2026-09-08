from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from scripts.aws import remote as aws_remote
from scripts.aws.support import AwsError, InstanceConfig


class AwsRemoteTests(unittest.TestCase):
    def test_parser_requires_a_run_command(self) -> None:
        check = aws_remote.parse_args(["check"])
        self.assertEqual(check.action, "check")
        run = aws_remote.parse_args(
            [
                "run",
                "--fallback-local",
                "--",
                "python",
                "scripts/simulate.py",
                "latent_network",
            ]
        )
        self.assertTrue(run.fallback_local)
        self.assertEqual(
            aws_remote.normalize_command(run.command),
            ["python", "scripts/simulate.py", "latent_network"],
        )

    def test_normalize_command_rejects_an_empty_command(self) -> None:
        with self.assertRaises(AwsError):
            aws_remote.normalize_command(["--"])

    def test_run_falls_back_locally_when_the_instance_record_is_missing(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    aws_remote,
                    "load_instance_config",
                    side_effect=AwsError("No shared AWS instance.", "setup"),
                ),
                patch.object(
                    aws_remote.subprocess,
                    "run",
                    return_value=Mock(returncode=0),
                ) as local,
            ):
                status = aws_remote.run_with_optional_fallback(
                    ["python", "scripts/simulate.py", "latent_network"],
                    fallback_local=True,
                    root=root,
                    announce=lambda _message: None,
                )

        self.assertEqual(status, 0)
        local.assert_called_once()
        self.assertEqual(
            local.call_args.args[0],
            ["python", "scripts/simulate.py", "latent_network"],
        )

    def test_run_does_not_fall_back_without_the_flag(self) -> None:
        with TemporaryDirectory() as directory:
            with patch.object(
                aws_remote,
                "load_instance_config",
                side_effect=AwsError("No shared AWS instance.", "setup"),
            ):
                with self.assertRaises(AwsError):
                    aws_remote.run_with_optional_fallback(
                        ["python", "scripts/simulate.py", "latent_network"],
                        fallback_local=False,
                        root=Path(directory),
                        announce=lambda _message: None,
                    )

    def test_remote_script_failure_is_not_a_local_fallback(self) -> None:
        with self.assertRaises(aws_remote.RemoteExecutionError):
            with patch.object(
                aws_remote,
                "run_on_instance",
                side_effect=aws_remote.RemoteExecutionError(
                    "remote failed",
                    "inspect output",
                ),
            ):
                aws_remote.run_with_optional_fallback(
                    ["python", "scripts/simulate.py", "latent_network"],
                    fallback_local=True,
                    announce=lambda _message: None,
                )

    def test_check_does_not_start_a_stopped_instance(self) -> None:
        config = InstanceConfig(
            instance_id="i-abc",
            region="us-west-2",
            instance_type="c7a.16xlarge",
            security_group_id="sg-123",
            default_cpus=12,
        )
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(
                    aws_remote,
                    "load_instance_config",
                    return_value=config,
                ),
                patch.object(aws_remote, "ec2_client") as ec2_client,
                patch.object(
                    aws_remote,
                    "ensure_instance_running",
                    side_effect=AwsError(
                        "Shared instance i-abc is stopped.",
                        "Run start.",
                    ),
                ) as ensure,
            ):
                with self.assertRaises(AwsError) as raised:
                    aws_remote.check_connection(
                        root=root,
                        announce=lambda _message: None,
                    )

        self.assertIn("stopped", raised.exception.problem)
        ensure.assert_called_once()
        self.assertFalse(ensure.call_args.kwargs["start_if_stopped"])
        ec2_client.assert_called_once_with("us-west-2")
