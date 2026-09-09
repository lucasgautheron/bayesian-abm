from __future__ import annotations

from pathlib import Path
import shlex
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

    def _run_on_instance(
        self,
        command: list[str],
        *,
        root: Path,
        run_remote: Mock,
        copy_back: Mock,
    ) -> None:
        config = InstanceConfig(
            instance_id="i-abc",
            region="us-west-2",
            instance_type="c7a.16xlarge",
            security_group_id="sg-123",
            default_cpus=12,
        )
        with (
            patch.object(aws_remote, "load_instance_config", return_value=config),
            patch.object(aws_remote, "ec2_client"),
            patch.object(
                aws_remote,
                "ensure_instance_running",
                return_value={"InstanceId": "i-abc", "PublicIpAddress": "1.2.3.4"},
            ),
            patch.object(
                aws_remote,
                "open_ssh_session",
                return_value=_context(Mock()),
            ),
            patch.object(aws_remote, "rsync_to_remote"),
            patch.object(aws_remote, "run_remote_command", run_remote),
            patch.object(aws_remote, "rsync_results_back", copy_back),
            patch.object(
                aws_remote,
                "remote_workdir",
                return_value="/home/ubuntu/users/alice-ab12",
            ),
        ):
            aws_remote.run_on_instance(
                command,
                root=root,
                announce=lambda _message: None,
            )

    def test_run_on_instance_copies_results_after_remote_failure(self) -> None:
        run_remote = Mock(
            side_effect=aws_remote.RemoteExecutionError("failed", "inspect")
        )
        copy_back = Mock()
        with TemporaryDirectory() as directory:
            with self.assertRaises(aws_remote.RemoteExecutionError):
                self._run_on_instance(
                    ["python", "scripts/simulate.py", "latent_network"],
                    root=Path(directory),
                    run_remote=run_remote,
                    copy_back=copy_back,
                )

        copy_back.assert_called_once()

    def test_run_on_instance_keeps_remote_failure_when_copy_back_fails(
        self,
    ) -> None:
        run_remote = Mock(
            side_effect=aws_remote.RemoteExecutionError("failed", "inspect")
        )
        copy_back = Mock(side_effect=AwsError("copy failed", "retry"))
        with TemporaryDirectory() as directory:
            with self.assertRaises(aws_remote.RemoteExecutionError) as raised:
                self._run_on_instance(
                    ["python", "scripts/simulate.py", "latent_network"],
                    root=Path(directory),
                    run_remote=run_remote,
                    copy_back=copy_back,
                )

        self.assertEqual(raised.exception.problem, "failed")
        copy_back.assert_called_once()

    def test_run_on_instance_rewrites_paths_strips_show_and_copies_extra(
        self,
    ) -> None:
        run_remote = Mock()
        copy_back = Mock()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._run_on_instance(
                [
                    "python",
                    "scripts/simulate.py",
                    "latent_network",
                    "--show",
                    "--output",
                    str(root / "output" / "model" / "simulations.png"),
                    "--report-dir",
                    str(root / "artifacts" / "custom"),
                ],
                root=root,
                run_remote=run_remote,
                copy_back=copy_back,
            )

        sent = run_remote.call_args.args[2]
        self.assertNotIn("--show", sent)
        self.assertIn("--output", sent)
        self.assertIn("output/model/simulations.png", sent)
        self.assertIn("artifacts/custom", sent)
        self.assertIn("--cpus", sent)
        self.assertEqual(copy_back.call_args.kwargs["extra"], ["artifacts/custom"])

    def test_rsync_to_remote_quotes_the_ssh_transport(self) -> None:
        session = Mock()
        session.identity = Path("/tmp/my key/id")
        session.known_hosts = Path("/tmp/known hosts")
        session.target = "ubuntu@1.2.3.4"
        with (
            patch.object(
                aws_remote,
                "run_ssh",
                return_value=Mock(returncode=0, stdout="", stderr=""),
            ),
            patch.object(
                aws_remote,
                "run_logged",
                return_value=Mock(returncode=0, stdout="", stderr=""),
            ) as logged,
        ):
            aws_remote.rsync_to_remote(session, Path("/tmp/src"), "/home/ubuntu/users/x")

        args = logged.call_args.args[0]
        transport = args[args.index("-e") + 1]
        parts = shlex.split(transport)
        self.assertEqual(parts[0], "ssh")
        self.assertIn("/tmp/my key/id", parts)
        self.assertIn("UserKnownHostsFile=/tmp/known hosts", parts)

    def test_rsync_results_back_copies_extra_file_destinations(self) -> None:
        session = Mock()
        session.identity = Path("/tmp/id")
        session.known_hosts = Path("/tmp/known_hosts")
        session.target = "ubuntu@1.2.3.4"

        def fake_ssh(_session: object, command: str, **_kwargs: object) -> Mock:
            if command.startswith("test -d ") and "output" in command:
                return Mock(returncode=1, stdout="", stderr="")
            if command.startswith("test -d ") and "reports" in command:
                return Mock(returncode=1, stdout="", stderr="")
            if command.startswith("test -e "):
                return Mock(returncode=0, stdout="", stderr="")
            if command.startswith("test -d "):
                return Mock(returncode=1, stdout="", stderr="")
            return Mock(returncode=1, stdout="", stderr="")

        with TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(aws_remote, "run_ssh", side_effect=fake_ssh),
                patch.object(
                    aws_remote,
                    "run_logged",
                    return_value=Mock(returncode=0, stdout="", stderr=""),
                ) as logged,
            ):
                aws_remote.rsync_results_back(
                    session,
                    "/home/ubuntu/users/alice-ab12",
                    root,
                    extra=("artifacts/custom.png",),
                )

        args = logged.call_args.args[0]
        self.assertIn(
            "ubuntu@1.2.3.4:/home/ubuntu/users/alice-ab12/artifacts/custom.png",
            args,
        )
        self.assertIn(str(root / "artifacts" / "custom.png"), args)


def _context(value: object):
    class _Manager:
        def __enter__(self) -> object:
            return value

        def __exit__(self, *_exc: object) -> None:
            return None

    return _Manager()
