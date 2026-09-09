from __future__ import annotations

import io
from pathlib import Path
import shlex
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import Mock, patch

from scripts.aws import instance as aws_instance
from scripts.aws import support as aws_support


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, **kwargs: Any) -> None:
        body = kwargs["Body"]
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.objects[(kwargs["Bucket"], kwargs["Key"])] = body

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = (kwargs["Bucket"], kwargs["Key"])
        if key not in self.objects:
            raise RuntimeError("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[key])}

    def delete_object(self, **kwargs: Any) -> None:
        self.objects.pop((kwargs["Bucket"], kwargs["Key"]), None)

    def head_bucket(self, **kwargs: Any) -> None:
        return None

    def create_bucket(self, **kwargs: Any) -> None:
        return None

    def put_public_access_block(self, **kwargs: Any) -> None:
        return None


class AwsSupportTests(unittest.TestCase):
    def _write_dallingerconfig(
        self,
        directory: Path,
        access_key: str = "AKIADALLINGER",
        secret: str = "dallinger-secret",
    ) -> Path:
        path = directory / aws_support.DALLINGER_CONFIG_NAME
        path.write_text(
            "[AWS Access]\n"
            f"aws_access_key_id = {access_key}\n"
            f"aws_secret_access_key = {secret}\n"
            "aws_region = us-east-1\n",
            encoding="utf-8",
        )
        return path

    def test_load_aws_credentials_reads_dallingerconfig(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_dallingerconfig(Path(directory))

            credentials = aws_support.load_aws_credentials(path=path, environ={})

        self.assertEqual(
            credentials,
            {
                "aws_access_key_id": "AKIADALLINGER",
                "aws_secret_access_key": "dallinger-secret",
            },
        )

    def test_load_aws_credentials_prefers_environment_override(self) -> None:
        with TemporaryDirectory() as directory:
            path = self._write_dallingerconfig(Path(directory))

            credentials = aws_support.load_aws_credentials(
                path=path,
                environ={
                    "AWS_ACCESS_KEY_ID": "AKIAENV",
                    "AWS_SECRET_ACCESS_KEY": "env-secret",
                },
            )

        self.assertEqual(
            credentials,
            {
                "aws_access_key_id": "AKIAENV",
                "aws_secret_access_key": "env-secret",
            },
        )

    def test_load_aws_credentials_accepts_environment_without_a_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / aws_support.DALLINGER_CONFIG_NAME
            credentials = aws_support.load_aws_credentials(
                path=path,
                environ={
                    "AWS_ACCESS_KEY_ID": "AKIAENV",
                    "AWS_SECRET_ACCESS_KEY": "env-secret",
                },
            )

        self.assertEqual(
            credentials,
            {
                "aws_access_key_id": "AKIAENV",
                "aws_secret_access_key": "env-secret",
            },
        )

    def test_load_aws_credentials_explains_a_missing_file(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / aws_support.DALLINGER_CONFIG_NAME
            with self.assertRaises(aws_support.AwsError) as raised:
                aws_support.load_aws_credentials(path=path, environ={})

        self.assertIn(str(path), raised.exception.problem)
        self.assertIn(str(path), raised.exception.resolution)
        self.assertIn("[AWS Access]", raised.exception.resolution)

    def test_load_aws_credentials_explains_missing_keys(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / aws_support.DALLINGER_CONFIG_NAME
            path.write_text("[AWS Access]\naws_region = us-east-1\n", encoding="utf-8")
            with self.assertRaises(aws_support.AwsError) as raised:
                aws_support.load_aws_credentials(path=path, environ={})

        self.assertIn(str(path), raised.exception.problem)
        self.assertIn("aws_access_key_id", raised.exception.resolution)

    def test_ec2_and_s3_clients_pass_loaded_credentials(self) -> None:
        credentials = {
            "aws_access_key_id": "AKIACLIENT",
            "aws_secret_access_key": "client-secret",
        }
        boto3 = Mock()
        with (
            patch.object(aws_support, "load_aws_credentials", return_value=credentials),
            patch.object(aws_support, "import_boto3", return_value=boto3),
        ):
            aws_support.ec2_client("us-west-2")
            aws_support.s3_client()

        boto3.client.assert_any_call(
            "ec2",
            region_name="us-west-2",
            aws_access_key_id="AKIACLIENT",
            aws_secret_access_key="client-secret",
        )
        boto3.client.assert_any_call(
            "s3",
            region_name=aws_support.WORKSHOP_S3_REGION,
            aws_access_key_id="AKIACLIENT",
            aws_secret_access_key="client-secret",
        )

    def test_translate_boto_error_points_at_dallingerconfig(self) -> None:
        error = aws_support.translate_boto_error(
            RuntimeError("Unable to locate credentials")
        )

        self.assertIn("[AWS Access]", error.resolution)
        self.assertIn(str(aws_support.dallinger_config_path()), error.resolution)

    def test_remote_user_rejects_an_empty_local_username(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(aws_support.getpass, "getuser", return_value="***"):
                with self.assertRaises(aws_support.AwsError) as raised:
                    aws_support.remote_user(root=root)

        self.assertIn(str(aws_support.remote_user_id_path(root)), raised.exception.resolution)

    def test_remote_user_reuses_the_gitignored_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = aws_support.remote_user_id_path(root)
            path.parent.mkdir()
            path.write_text("lucasgautheron-a3f2\n", encoding="utf-8")

            self.assertEqual(
                aws_support.remote_user(root=root),
                "lucasgautheron-a3f2",
            )

    def test_remote_user_creates_a_stable_id_once(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(
                aws_support.getpass,
                "getuser",
                return_value="lucasgautheron",
            ):
                first = aws_support.remote_user(root=root)
                second = aws_support.remote_user(root=root)

            self.assertEqual(first, second)
            self.assertRegex(first, r"^lucasgautheron-[0-9a-f]{4}$")
            self.assertEqual(
                aws_support.remote_user_id_path(root).read_text(encoding="utf-8").strip(),
                first,
            )

    def test_remote_user_rejects_an_empty_stored_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = aws_support.remote_user_id_path(root)
            path.parent.mkdir()
            path.write_text("***\n", encoding="utf-8")

            with self.assertRaises(aws_support.AwsError) as raised:
                aws_support.remote_user(root=root)

            self.assertIn(str(path), raised.exception.problem)

    def test_rsync_exclude_args_cover_the_planned_paths(self) -> None:
        flags = aws_support.rsync_exclude_args()

        self.assertIn("--exclude=.venv/", flags)
        self.assertIn("--exclude=/output/", flags)
        self.assertIn("--exclude=/reports/", flags)
        self.assertIn("--exclude=.git/", flags)
        self.assertIn("--exclude=slides/", flags)
        self.assertIn("--exclude=.cursor/", flags)
        self.assertIn("--exclude=.config/aws-ssh/", flags)
        self.assertIn("--exclude=.config/aws-remote-user", flags)
        self.assertIn("--exclude=tests/", flags)

    def test_rsync_ssh_transport_quotes_paths_with_spaces(self) -> None:
        transport = aws_support.rsync_ssh_transport(
            Path("/tmp/my key/id"),
            Path("/tmp/known hosts"),
        )
        parts = shlex.split(transport)

        self.assertEqual(parts[0], "ssh")
        self.assertIn("/tmp/my key/id", parts)
        self.assertIn("UserKnownHostsFile=/tmp/known hosts", parts)

    def test_prepare_remote_command_strips_show_and_relativizes_paths(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rewritten, extra, dropped = aws_support.prepare_remote_command(
                [
                    "python",
                    "scripts/simulate.py",
                    "latent_network",
                    "--show",
                    "--output",
                    str(root / "output" / "model" / "simulations.png"),
                    "--report-dir=artifacts/custom",
                ],
                root,
            )

        self.assertTrue(dropped)
        self.assertEqual(
            rewritten,
            [
                "python",
                "scripts/simulate.py",
                "latent_network",
                "--output",
                "output/model/simulations.png",
                "--report-dir=artifacts/custom",
            ],
        )
        self.assertEqual(extra, ["artifacts/custom"])

    def test_prepare_remote_command_rejects_paths_outside_the_repo(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(aws_support.AwsError) as raised:
                aws_support.prepare_remote_command(
                    [
                        "python",
                        "scripts/simulate.py",
                        "latent_network",
                        "--output",
                        "/tmp/out.png",
                    ],
                    root,
                )

        self.assertIn("outside the repository", raised.exception.problem)

    def test_with_remote_cpus_does_not_override_an_explicit_flag(self) -> None:
        command = ["python", "scripts/simulate.py", "latent_network", "--cpus", "3"]

        self.assertEqual(aws_support.with_remote_cpus(command), command)
        self.assertEqual(
            aws_support.with_remote_cpus(
                ["python", "scripts/simulate.py", "latent_network"]
            ),
            [
                "python",
                "scripts/simulate.py",
                "latent_network",
                "--cpus",
                "16",
            ],
        )
        self.assertTrue(aws_support.command_has_cpus(["prog", "--cpus=8"]))

    def test_load_instance_config_explains_a_missing_s3_object(self) -> None:
        with self.assertRaises(aws_support.AwsError) as raised:
            aws_support.load_instance_config(s3=FakeS3())

        self.assertIn("setup", raised.exception.resolution)
        self.assertIn("s3://", raised.exception.problem)

    def test_save_and_load_round_trip_the_s3_instance_record(self) -> None:
        config = aws_support.InstanceConfig(
            instance_id="i-abc",
            region="us-west-2",
            instance_type="c7a.16xlarge",
            security_group_id="sg-123",
            default_cpus=12,
        )
        s3 = FakeS3()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            uri = aws_support.save_instance_config(config, root, s3=s3)
            self.assertTrue(uri.startswith("s3://"))
            self.assertFalse(aws_support.instance_config_path(root).exists())
            loaded = aws_support.load_instance_config(root, s3=s3)
            self.assertEqual(loaded, config)
            aws_support.clear_instance_config(root, s3=s3)
            self.assertIsNone(aws_support.load_instance_config_optional(s3=s3))

    def test_fetch_workshop_ssh_identity_creates_cache_and_writes_key(
        self,
    ) -> None:
        s3 = FakeS3()
        s3.put_object(
            Bucket=aws_support.workshop_s3_bucket(),
            Key=aws_support.WORKSHOP_SSH_S3_KEY,
            Body=b"PRIVATE",
        )
        s3.put_object(
            Bucket=aws_support.workshop_s3_bucket(),
            Key=aws_support.WORKSHOP_SSH_PUB_S3_KEY,
            Body=b"ssh-ed25519 AAAA workshop\n",
        )
        with TemporaryDirectory() as directory:
            cache = Path(directory) / ".config" / "aws-ssh"
            self.assertFalse(cache.exists())
            identity = aws_support.fetch_workshop_ssh_identity(
                cache,
                s3=s3,
            )
            self.assertTrue(cache.is_dir())
            self.assertEqual(identity.read_bytes(), b"PRIVATE")
            self.assertIn(
                "ssh-ed25519 AAAA workshop",
                aws_support.public_key_path(identity).read_text(encoding="utf-8"),
            )

    def test_cached_workshop_ssh_identity_skips_s3_when_present(self) -> None:
        s3 = FakeS3()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            identity = aws_support.workshop_ssh_identity(root)
            identity.parent.mkdir(parents=True)
            identity.write_bytes(b"CACHED")
            aws_support.public_key_path(identity).write_text(
                "ssh-ed25519 AAAA cached\n",
                encoding="utf-8",
            )
            loaded = aws_support.cached_workshop_ssh_identity(root=root, s3=s3)
            self.assertEqual(loaded, identity)
            self.assertEqual(loaded.read_bytes(), b"CACHED")
            self.assertEqual(s3.objects, {})

    def test_instance_parser_accepts_lifecycle_commands(self) -> None:
        setup = aws_instance.parse_args(
            ["setup", "--region", "us-west-2", "--instance-type", "c7a.12xlarge"]
        )
        self.assertEqual(setup.command, "setup")
        self.assertEqual(setup.region, "us-west-2")
        self.assertEqual(setup.instance_type, "c7a.12xlarge")
        self.assertEqual(setup.cpus, 16)
        self.assertEqual(aws_instance.parse_args(["start"]).command, "start")
        self.assertEqual(aws_instance.parse_args(["stop"]).command, "stop")
        self.assertEqual(aws_instance.parse_args(["destroy"]).command, "destroy")
        self.assertEqual(
            aws_instance.parse_args(["provision"]).command,
            "provision",
        )
        self.assertEqual(
            aws_instance.parse_args(["publish"]).command,
            "publish",
        )

    def test_user_data_installs_the_workshop_public_key(self) -> None:
        script = aws_instance.user_data_script(
            "ssh-ed25519 AAAA bayesian-modelling-workshop"
        )

        self.assertIn("ssh-ed25519 AAAA bayesian-modelling-workshop", script)
        self.assertIn("/home/ubuntu/.ssh/authorized_keys", script)
        self.assertIn("KERAS_BACKEND=jax", script)
        self.assertNotIn("conda env config vars set", script)

    def test_remote_command_uses_the_env_python(self) -> None:
        command = aws_support.conda_run_command(["python", "scripts/simulate.py"])

        self.assertIn("KERAS_BACKEND=jax", command)
        self.assertIn(aws_support.CONDA_PYTHON, command)
        self.assertNotIn("conda run", command)
        probe = aws_support.ssh_probe_command()
        self.assertIn(aws_support.CONDA_PYTHON, probe)
        self.assertIn("-c", probe)
        self.assertNotIn("conda run", probe)

    def test_resolve_ubuntu_ami_picks_the_newest_canonical_image(self) -> None:
        ec2 = Mock()
        ec2.describe_images.return_value = {
            "Images": [
                {"ImageId": "ami-old", "CreationDate": "2024-01-01T00:00:00.000Z"},
                {"ImageId": "ami-new", "CreationDate": "2026-08-01T00:00:00.000Z"},
            ]
        }

        self.assertEqual(aws_instance.resolve_ubuntu_ami(ec2), "ami-new")
        ec2.describe_images.assert_called_once()
        owners = ec2.describe_images.call_args.kwargs["Owners"]
        self.assertEqual(owners, [aws_instance.CANONICAL_OWNER_ID])

    def test_ensure_instance_running_refuses_to_start_when_checking(self) -> None:
        config = aws_support.InstanceConfig(
            instance_id="i-abc",
            region="us-west-2",
            instance_type="c7a.16xlarge",
            security_group_id="sg-123",
            default_cpus=12,
        )
        ec2 = Mock()
        with patch.object(
            aws_support,
            "describe_instance",
            return_value={"State": {"Name": "stopped"}, "InstanceId": "i-abc"},
        ):
            with self.assertRaises(aws_support.AwsError) as raised:
                aws_support.ensure_instance_running(
                    ec2,
                    config,
                    start_if_stopped=False,
                )

        self.assertIn("scripts/aws/instance.py start", raised.exception.resolution)
        ec2.start_instances.assert_not_called()
