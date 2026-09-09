"""Shared helpers for the workshop EC2 instance and remote runner."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
import configparser
from dataclasses import dataclass
import getpass
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, TypedDict


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INSTANCE_CONFIG_NAME = "aws-instance.json"
REMOTE_USER_ID_NAME = "aws-remote-user"
DEFAULT_REGION = "us-west-2"
DEFAULT_INSTANCE_TYPE = "c7a.16xlarge"
DEFAULT_REMOTE_CPUS = 16
DEFAULT_LOCAL_CPUS = 4
WORKSHOP_NAME = "bayesian-modelling-workshop"
WORKSHOP_S3_BUCKET = "bayesian-modelling-workshop-292651677991"
WORKSHOP_S3_REGION = DEFAULT_REGION
INSTANCE_CONFIG_S3_KEY = "workshop/instance.json"
WORKSHOP_SSH_S3_KEY = "workshop/ssh/id_ed25519"
WORKSHOP_SSH_PUB_S3_KEY = "workshop/ssh/id_ed25519.pub"
DALLINGER_CONFIG_NAME = ".dallingerconfig"
DALLINGER_AWS_SECTION = "AWS Access"
SSH_USER = "ubuntu"
CONDA_ROOT = "/opt/miniconda3"
CONDA_ENV = "bayesian-modelling"
CONDA_PYTHON = f"{CONDA_ROOT}/envs/{CONDA_ENV}/bin/python"
REMOTE_USERS_ROOT = "/home/ubuntu/users"
SENTINEL_MINICONDA = "/opt/workshop/.miniconda-ready"
SENTINEL_PROVISIONED = "/opt/workshop/.provisioned"
RSYNC_EXCLUDES = (
    ".git/",
    ".venv/",
    ".venv*/",
    "__pycache__/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    ".DS_Store",
    "*.pyc",
    "/output/",
    "/reports/",
    "slides/",
    ".cursor/",
    ".config/aws-ssh/",
    ".config/aws-remote-user",
    "tests/",
)
REMOTE_PATH_FLAGS = ("--output", "--diagnostics-dir", "--report-dir")
STANDARD_RESULT_ROOTS = ("output", "reports")
LIVE_INSTANCE_STATES = frozenset(
    {"pending", "running", "stopping", "stopped", "shutting-down"}
)


class InstanceConfig(TypedDict):
    instance_id: str
    region: str
    instance_type: str
    security_group_id: str
    default_cpus: int


class AwsCredentials(TypedDict):
    aws_access_key_id: str
    aws_secret_access_key: str


class AwsError(RuntimeError):
    """An AWS or SSH failure with a concrete next action."""

    def __init__(self, problem: str, resolution: str) -> None:
        super().__init__(problem)
        self.problem = problem
        self.resolution = resolution


def instance_config_path(root: Path = ROOT) -> Path:
    """Return the optional local cache of the S3 instance record."""

    return root / ".config" / INSTANCE_CONFIG_NAME


def workshop_s3_bucket(environ: Mapping[str, str] | None = None) -> str:
    """Return the workshop S3 bucket that stores instance metadata and SSH."""

    env = os.environ if environ is None else environ
    return str(env.get("AWS_WORKSHOP_S3_BUCKET") or WORKSHOP_S3_BUCKET)


def remote_user_id_path(root: Path = ROOT) -> Path:
    """Return the gitignored per-workstation remote folder id."""

    return root / ".config" / REMOTE_USER_ID_NAME


def sanitize_remote_name(raw: str) -> str:
    """Return a filesystem-safe remote folder name."""

    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw).strip("._-")


def remote_user(*, root: Path = ROOT) -> str:
    """Return the isolated remote folder name for this workstation."""

    path = remote_user_id_path(root)
    if path.is_file():
        stored = sanitize_remote_name(path.read_text(encoding="utf-8").strip())
        if not stored:
            raise AwsError(
                f"{path} does not contain a usable remote user id.",
                "Replace the file with a simple id such as alice-a3f2.",
            )
        return stored
    user = sanitize_remote_name(getpass.getuser())
    if not user:
        raise AwsError(
            "Could not derive a remote user folder name.",
            f"Write a simple id such as alice-a3f2 to {path}.",
        )
    identity = f"{user}-{secrets.token_hex(2)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{identity}\n", encoding="utf-8")
    return identity


def remote_workdir(
    user: str | None = None,
    *,
    root: Path = ROOT,
) -> str:
    """Return the per-user repository directory on the instance."""

    return f"{REMOTE_USERS_ROOT}/{user or remote_user(root=root)}"


def rsync_exclude_args(
    excludes: Sequence[str] = RSYNC_EXCLUDES,
) -> list[str]:
    """Return rsync ``--exclude`` flags for a local-to-remote mirror."""

    return [f"--exclude={pattern}" for pattern in excludes]


def command_has_cpus(command: Sequence[str]) -> bool:
    """Return whether ``command`` already sets ``--cpus``."""

    return any(
        argument == "--cpus" or argument.startswith("--cpus=")
        for argument in command
    )


def with_remote_cpus(
    command: Sequence[str],
    cpus: int = DEFAULT_REMOTE_CPUS,
) -> list[str]:
    """Append ``--cpus`` for a remote run when the user did not set it."""

    if command_has_cpus(command):
        return list(command)
    return [*command, "--cpus", str(cpus)]


def relativize_repo_path(value: str, root: Path) -> str:
    """Return ``value`` as a POSIX path inside ``root``, or raise."""

    if not value:
        raise AwsError(
            "A remote path flag was empty.",
            "Pass a repository-relative path such as output/model/simulations.png.",
        )
    path = Path(value)
    resolved_root = root.resolve()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise AwsError(
            f"{value} is outside the repository and cannot be used remotely.",
            "Pass a path inside the repository, such as "
            "output/<model>/simulations.png.",
        ) from exc
    return relative.as_posix()


def extra_result_paths(relative_paths: Sequence[str]) -> list[str]:
    """Return destinations that are not under output/ or reports/."""

    extras: list[str] = []
    seen: set[str] = set()
    for relative in relative_paths:
        parts = Path(relative).parts
        if not parts or parts[0] in STANDARD_RESULT_ROOTS:
            continue
        if relative in seen:
            continue
        seen.add(relative)
        extras.append(relative)
    return extras


def prepare_remote_command(
    command: Sequence[str],
    root: Path,
) -> tuple[list[str], list[str], bool]:
    """Strip ``--show``, relativize path flags, and list extra copy-back paths."""

    without_show: list[str] = []
    show_dropped = False
    for argument in command:
        if argument == "--show" or argument.startswith("--show="):
            show_dropped = True
            continue
        without_show.append(argument)

    rewritten: list[str] = []
    destinations: list[str] = []
    index = 0
    while index < len(without_show):
        argument = without_show[index]
        matched = False
        for flag in REMOTE_PATH_FLAGS:
            if argument == flag:
                if index + 1 >= len(without_show):
                    raise AwsError(
                        f"{flag} is missing a path.",
                        "Pass a repository-relative path after the flag.",
                    )
                relative = relativize_repo_path(without_show[index + 1], root)
                rewritten.extend([flag, relative])
                destinations.append(relative)
                index += 2
                matched = True
                break
            prefix = f"{flag}="
            if argument.startswith(prefix):
                relative = relativize_repo_path(argument[len(prefix) :], root)
                rewritten.append(f"{flag}={relative}")
                destinations.append(relative)
                index += 1
                matched = True
                break
        if not matched:
            rewritten.append(argument)
            index += 1
    return rewritten, extra_result_paths(destinations), show_dropped


def rsync_ssh_transport(identity: Path, known_hosts: Path) -> str:
    """Return a quoted ``rsync -e`` SSH command."""

    return shlex.join(["ssh", *ssh_options(identity, known_hosts)])


def parse_instance_config(payload: Mapping[str, Any], *, source: str) -> InstanceConfig:
    """Validate instance metadata from S3 or a leftover local file."""

    required = (
        "instance_id",
        "region",
        "instance_type",
        "security_group_id",
        "default_cpus",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise AwsError(
            f"{source} is missing {', '.join(missing)}.",
            "Rerun `python scripts/aws/instance.py setup`.",
        )
    return InstanceConfig(
        instance_id=str(payload["instance_id"]),
        region=str(payload["region"]),
        instance_type=str(payload["instance_type"]),
        security_group_id=str(payload["security_group_id"]),
        default_cpus=int(payload["default_cpus"]),
    )


def load_local_instance_config(root: Path = ROOT) -> InstanceConfig | None:
    """Read a leftover local instance file used only to bootstrap S3."""

    path = instance_config_path(root)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AwsError(
            f"{path} is not valid JSON.",
            "Remove the local file and rerun setup, or fix the JSON.",
        ) from exc
    if not isinstance(payload, Mapping):
        raise AwsError(
            f"{path} is not a JSON object.",
            "Remove the local file and rerun setup.",
        )
    return parse_instance_config(payload, source=str(path))


def s3_object_uri(key: str, *, bucket: str | None = None) -> str:
    """Return an ``s3://`` URI for a workshop object."""

    return f"s3://{bucket or workshop_s3_bucket()}/{key}"


def is_missing_s3_object(exc: BaseException) -> bool:
    """Return whether ``exc`` is a missing S3 object or bucket."""

    name = type(exc).__name__.lower()
    text = str(exc).lower()
    markers = (
        "nosuchkey",
        "nosuchbucket",
        "not found",
        "404",
        "no such key",
        "the specified key does not exist",
        "the specified bucket does not exist",
    )
    return any(marker in name or marker in text for marker in markers)


def load_instance_config(
    root: Path = ROOT,
    s3: Any | None = None,
) -> InstanceConfig:
    """Read the live instance record from S3."""

    config = load_instance_config_optional(s3=s3)
    if config is not None:
        return config
    raise AwsError(
        f"No shared AWS instance is registered at "
        f"{s3_object_uri(INSTANCE_CONFIG_S3_KEY)}.",
        "An instructor must run `python scripts/aws/instance.py setup`.",
    )


def load_instance_config_optional(s3: Any | None = None) -> InstanceConfig | None:
    """Return the S3 instance record, or ``None`` if it has not been published."""

    client = s3 if s3 is not None else s3_client()
    try:
        response = client.get_object(
            Bucket=workshop_s3_bucket(),
            Key=INSTANCE_CONFIG_S3_KEY,
        )
        payload = json.loads(response["Body"].read().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise AwsError(
            f"{s3_object_uri(INSTANCE_CONFIG_S3_KEY)} is not valid JSON.",
            "Rerun `python scripts/aws/instance.py setup`.",
        ) from exc
    except Exception as exc:
        if is_missing_s3_object(exc):
            return None
        translated = translate_boto_error(exc)
        raise AwsError(translated.problem, translated.resolution) from exc
    if not isinstance(payload, Mapping):
        raise AwsError(
            f"{s3_object_uri(INSTANCE_CONFIG_S3_KEY)} is not a JSON object.",
            "Rerun `python scripts/aws/instance.py setup`.",
        )
    return parse_instance_config(
        payload,
        source=s3_object_uri(INSTANCE_CONFIG_S3_KEY),
    )


def save_instance_config(
    config: InstanceConfig,
    root: Path = ROOT,
    s3: Any | None = None,
) -> str:
    """Publish instance metadata to S3 so students do not need a repo update."""

    del root
    client = s3 if s3 is not None else s3_client()
    body = json.dumps(config, indent=2) + "\n"
    try:
        client.put_object(
            Bucket=workshop_s3_bucket(),
            Key=INSTANCE_CONFIG_S3_KEY,
            Body=body.encode("utf-8"),
            ServerSideEncryption="AES256",
            ContentType="application/json",
        )
    except Exception as exc:
        translated = translate_boto_error(exc)
        raise AwsError(translated.problem, translated.resolution) from exc
    return s3_object_uri(INSTANCE_CONFIG_S3_KEY)


def clear_instance_config(root: Path = ROOT, s3: Any | None = None) -> None:
    """Remove the S3 instance record after destroy."""

    path = instance_config_path(root)
    if path.is_file():
        path.unlink()
    client = s3 if s3 is not None else s3_client()
    try:
        client.delete_object(
            Bucket=workshop_s3_bucket(),
            Key=INSTANCE_CONFIG_S3_KEY,
        )
    except Exception as exc:
        if is_missing_s3_object(exc):
            return
        translated = translate_boto_error(exc)
        raise AwsError(translated.problem, translated.resolution) from exc


def dallinger_config_path(home: Path | None = None) -> Path:
    """Return the path to Dallinger's user config file."""

    return (home if home is not None else Path.home()) / DALLINGER_CONFIG_NAME


def _aws_credential_resolution(path: Path) -> str:
    return (
        f"Add aws_access_key_id and aws_secret_access_key under "
        f"[{DALLINGER_AWS_SECTION}] in {path}."
    )


def load_aws_credentials(
    path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AwsCredentials:
    """Read AWS keys from the environment, then ~/.dallingerconfig."""

    env = os.environ if environ is None else environ
    config_path = path if path is not None else dallinger_config_path()
    access_key = str(env.get("AWS_ACCESS_KEY_ID") or "").strip()
    secret_key = str(env.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if not access_key or not secret_key:
        if not config_path.is_file():
            raise AwsError(
                f"AWS credentials were not found at {config_path}.",
                _aws_credential_resolution(config_path),
            )
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(config_path, encoding="utf-8")
        if parser.has_section(DALLINGER_AWS_SECTION):
            section = parser[DALLINGER_AWS_SECTION]
            if not access_key:
                access_key = section.get("aws_access_key_id", "").strip()
            if not secret_key:
                secret_key = section.get("aws_secret_access_key", "").strip()
    if not access_key or not secret_key:
        raise AwsError(
            f"AWS credentials are missing from {config_path}.",
            _aws_credential_resolution(config_path),
        )
    return {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
    }


def import_boto3() -> Any:
    """Import boto3 or explain that it is part of the project environment."""

    try:
        import boto3  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise AwsError(
            "Required Python module 'boto3' is not importable.",
            "Confirm that the project environment is active, then run "
            "`python -m pip install -r requirements.txt`.",
        ) from exc
    return boto3


def _boto_client(service: str, region: str) -> Any:
    credentials = load_aws_credentials()
    return import_boto3().client(
        service,
        region_name=region,
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
    )


def ec2_client(region: str) -> Any:
    """Return an EC2 client for ``region`` using ~/.dallingerconfig credentials."""

    return _boto_client("ec2", region)


def s3_client(region: str = WORKSHOP_S3_REGION) -> Any:
    """Return an S3 client for the workshop bucket region."""

    return _boto_client("s3", region)


def ensure_workshop_bucket(s3: Any | None = None) -> str:
    """Create the workshop bucket if needed and keep it private."""

    client = s3 if s3 is not None else s3_client()
    bucket = workshop_s3_bucket()
    try:
        client.head_bucket(Bucket=bucket)
    except Exception:
        try:
            client.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": WORKSHOP_S3_REGION},
            )
        except Exception as exc:
            lowered = str(exc).lower()
            if (
                "bucketalreadyownedbyyou" not in lowered
                and "bucketalreadyexists" not in lowered
            ):
                translated = translate_boto_error(exc)
                raise AwsError(translated.problem, translated.resolution) from exc
    try:
        client.put_public_access_block(
            Bucket=bucket,
            PublicAccessBlockConfiguration={
                "BlockPublicAcls": True,
                "IgnorePublicAcls": True,
                "BlockPublicPolicy": True,
                "RestrictPublicBuckets": True,
            },
        )
    except Exception as exc:
        translated = translate_boto_error(exc)
        raise AwsError(translated.problem, translated.resolution) from exc
    return bucket


def describe_instance(ec2: Any, instance_id: str) -> dict[str, Any] | None:
    """Return the instance description, or ``None`` if it no longer exists."""

    try:
        response = ec2.describe_instances(InstanceIds=[instance_id])
    except Exception as exc:
        name = type(exc).__name__
        if "InvalidInstanceID" in name or "InvalidInstanceID" in str(exc):
            return None
        raise
    reservations = response.get("Reservations") or []
    for reservation in reservations:
        instances = reservation.get("Instances") or []
        if instances:
            return instances[0]
    return None


def instance_state(description: Mapping[str, Any] | None) -> str:
    """Return the EC2 instance-state name."""

    if description is None:
        return "not-found"
    state = description.get("State") or {}
    return str(state.get("Name") or "unknown")


def public_ip_address(description: Mapping[str, Any]) -> str:
    """Return the public IPv4 address or explain that it is missing."""

    address = description.get("PublicIpAddress")
    if not address:
        raise AwsError(
            "The shared instance has no public IP address yet.",
            "Wait a few seconds and rerun the command, or check that the "
            "subnet assigns public IPv4 addresses.",
        )
    return str(address)


def require_live_instance(ec2: Any, config: InstanceConfig) -> dict[str, Any]:
    """Return the instance description when it still exists."""

    description = describe_instance(ec2, config["instance_id"])
    state = instance_state(description)
    if description is None or state == "terminated":
        raise AwsError(
            f"Shared instance {config['instance_id']} no longer exists.",
            "An instructor must run `python scripts/aws/instance.py setup`.",
        )
    return description


def wait_for_state(
    ec2: Any,
    instance_id: str,
    waiter_name: str,
    *,
    delay: int = 5,
    max_attempts: int = 80,
) -> dict[str, Any]:
    """Wait for an EC2 waiter, then return the latest description."""

    ec2.get_waiter(waiter_name).wait(
        InstanceIds=[instance_id],
        WaiterConfig={"Delay": delay, "MaxAttempts": max_attempts},
    )
    description = describe_instance(ec2, instance_id)
    if description is None:
        raise AwsError(
            f"Instance {instance_id} disappeared while waiting for {waiter_name}.",
            "Inspect the instance in the AWS console and rerun setup if needed.",
        )
    return description


def wait_for_public_ip(
    ec2: Any,
    instance_id: str,
    *,
    timeout_seconds: float = 180,
    pause_seconds: float = 5,
) -> dict[str, Any]:
    """Wait until the instance reports a public IPv4 address."""

    deadline = time.monotonic() + timeout_seconds
    description: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        description = describe_instance(ec2, instance_id)
        if description and description.get("PublicIpAddress"):
            return description
        time.sleep(pause_seconds)
    raise AwsError(
        f"Timed out waiting for a public IP on {instance_id}.",
        "Confirm that the subnet maps public IPv4 addresses on launch.",
    )


def ensure_instance_running(
    ec2: Any,
    config: InstanceConfig,
    *,
    start_if_stopped: bool,
    announce: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Return a running instance, optionally starting a stopped one."""

    description = require_live_instance(ec2, config)
    state = instance_state(description)
    instance_id = config["instance_id"]
    if state == "pending":
        announce(f"Waiting for {instance_id} to finish starting.")
        description = wait_for_state(ec2, instance_id, "instance_running")
        return wait_for_public_ip(ec2, instance_id)
    if state == "running":
        if description.get("PublicIpAddress"):
            return description
        return wait_for_public_ip(ec2, instance_id)
    if state == "stopping":
        announce(f"Waiting for {instance_id} to finish stopping.")
        wait_for_state(ec2, instance_id, "instance_stopped")
        state = "stopped"
    if state == "stopped":
        if not start_if_stopped:
            raise AwsError(
                f"Shared instance {instance_id} is stopped.",
                "Run `python scripts/aws/instance.py start` from the "
                "repository root.",
            )
        announce(f"Starting shared instance {instance_id}.")
        ec2.start_instances(InstanceIds=[instance_id])
        wait_for_state(ec2, instance_id, "instance_running")
        return wait_for_public_ip(ec2, instance_id)
    raise AwsError(
        f"Shared instance {instance_id} is {state}.",
        "Wait for the current transition to finish, or rerun setup.",
    )


def workshop_ssh_identity(root: Path = ROOT) -> Path:
    """Return the gitignored shared workshop SSH key path."""

    return root / ".config" / "aws-ssh" / "workshop-id_ed25519"


def ensure_workshop_ssh_key(
    root: Path = ROOT,
    s3: Any | None = None,
) -> tuple[Path, str]:
    """Reuse the shared workshop key from disk or S3, or create it once."""

    identity = workshop_ssh_identity(root)
    public_path = public_key_path(identity)
    if identity.is_file() and public_path.is_file():
        return identity, public_path.read_text(encoding="utf-8")
    try:
        fetch_workshop_ssh_identity(identity.parent, s3=s3)
        return identity, public_path.read_text(encoding="utf-8")
    except AwsError as exc:
        if "missing from" not in exc.problem:
            raise
    public_key = generate_ed25519_key(
        identity,
        comment="bayesian-modelling-workshop",
    )
    identity.chmod(0o600)
    return identity, public_key


def upload_workshop_ssh_key(
    identity: Path,
    public_key: str,
    s3: Any | None = None,
) -> str:
    """Store the workshop private key in S3 for anyone with AWS credentials."""

    client = s3 if s3 is not None else s3_client()
    ensure_workshop_bucket(client)
    try:
        client.put_object(
            Bucket=workshop_s3_bucket(),
            Key=WORKSHOP_SSH_S3_KEY,
            Body=identity.read_bytes(),
            ServerSideEncryption="AES256",
            ContentType="application/octet-stream",
        )
        client.put_object(
            Bucket=workshop_s3_bucket(),
            Key=WORKSHOP_SSH_PUB_S3_KEY,
            Body=public_key.encode("utf-8"),
            ServerSideEncryption="AES256",
            ContentType="text/plain",
        )
    except Exception as exc:
        translated = translate_boto_error(exc)
        raise AwsError(translated.problem, translated.resolution) from exc
    return s3_object_uri(WORKSHOP_SSH_S3_KEY)


def fetch_workshop_ssh_identity(
    directory: Path,
    s3: Any | None = None,
) -> Path:
    """Download the shared workshop SSH key into ``directory``."""

    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise AwsError(
            f"Could not create the SSH key cache directory {directory}: {exc}",
            "Confirm that the repository is writable, then rerun the command.",
        ) from exc
    client = s3 if s3 is not None else s3_client()
    identity = directory / "workshop-id_ed25519"
    public_path = public_key_path(identity)
    try:
        private = client.get_object(
            Bucket=workshop_s3_bucket(),
            Key=WORKSHOP_SSH_S3_KEY,
        )["Body"].read()
        public = client.get_object(
            Bucket=workshop_s3_bucket(),
            Key=WORKSHOP_SSH_PUB_S3_KEY,
        )["Body"].read()
    except Exception as exc:
        translated = translate_boto_error(exc)
        if is_missing_s3_object(exc):
            raise AwsError(
                f"The workshop SSH key is missing from "
                f"{s3_object_uri(WORKSHOP_SSH_S3_KEY)}.",
                "An instructor must run `python scripts/aws/instance.py "
                "publish` or `setup`.",
            ) from exc
        raise AwsError(translated.problem, translated.resolution) from exc
    try:
        identity.write_bytes(private)
        identity.chmod(0o600)
        public_path.write_bytes(public)
        public_path.chmod(0o644)
    except OSError as exc:
        raise AwsError(
            f"Could not cache the workshop SSH key in {directory}: {exc}",
            "Confirm that the repository is writable, then rerun the command.",
        ) from exc
    return identity


def cached_workshop_ssh_identity(
    root: Path = ROOT,
    s3: Any | None = None,
) -> Path:
    """Reuse the gitignored workshop key, downloading it from S3 once."""

    identity = workshop_ssh_identity(root)
    if identity.is_file() and public_key_path(identity).is_file():
        return identity
    return fetch_workshop_ssh_identity(identity.parent, s3=s3)


def public_key_path(identity: Path) -> Path:
    """Return the ``.pub`` sibling of an SSH identity file."""

    return identity.with_name(identity.name + ".pub")


def generate_ed25519_key(
    identity: Path,
    *,
    comment: str = "bayesian-modelling-workshop",
) -> str:
    """Create an SSH key and return the public-key text."""

    identity.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-f",
            str(identity),
            "-N",
            "",
            "-C",
            comment,
            "-q",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    public_path = identity.with_name(identity.name + ".pub")
    if result.returncode != 0 or not public_path.is_file():
        detail = (result.stderr or result.stdout or "ssh-keygen failed").strip()
        raise AwsError(
            f"Could not create an ephemeral SSH key: {detail}",
            "Install OpenSSH (`ssh-keygen`) and rerun the command.",
        )
    return public_path.read_text(encoding="utf-8")


def ssh_options(identity: Path, known_hosts: Path) -> list[str]:
    """Return SSH options that do not persist host keys in the user config."""

    return [
        "-i",
        str(identity),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=30",
    ]


@dataclass(frozen=True)
class SshSession:
    """SSH credentials for the shared instance."""

    host: str
    identity: Path
    known_hosts: Path
    public_key: str
    instance_id: str

    @property
    def target(self) -> str:
        return f"{SSH_USER}@{self.host}"

    def ssh_command(self, remote_command: str, *, tty: bool) -> list[str]:
        flag = "-t" if tty else "-T"
        return [
            "ssh",
            flag,
            *ssh_options(self.identity, self.known_hosts),
            self.target,
            remote_command,
        ]


@contextmanager
def open_ssh_session(
    description: Mapping[str, Any],
    *,
    root: Path = ROOT,
    s3: Any | None = None,
) -> Iterator[SshSession]:
    """Open SSH with the shared workshop key from S3."""

    from tempfile import TemporaryDirectory

    host = public_ip_address(description)
    instance_id = str(description["InstanceId"])
    workshop = cached_workshop_ssh_identity(root=root, s3=s3)
    with TemporaryDirectory(prefix="bayesian-aws-ssh-") as temporary:
        known_hosts = Path(temporary) / "known_hosts"
        known_hosts.touch()
        yield SshSession(
            host=host,
            identity=workshop,
            known_hosts=known_hosts,
            public_key=public_key_path(workshop).read_text(encoding="utf-8"),
            instance_id=instance_id,
        )


def run_logged(
    args: Sequence[str],
    *,
    capture: bool = False,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess, optionally capturing text output."""

    return subprocess.run(
        list(args),
        text=True,
        capture_output=capture,
        check=check,
    )


def run_ssh(
    session: SshSession,
    remote_command: str,
    *,
    tty: bool = False,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one SSH command on the shared instance."""

    return run_logged(
        session.ssh_command(remote_command, tty=tty),
        capture=capture,
    )


def wait_for_ssh(
    session: SshSession,
    *,
    timeout_seconds: float = 300,
    pause_seconds: float = 5,
) -> None:
    """Retry SSH until the instance accepts the workshop key."""

    deadline = time.monotonic() + timeout_seconds
    last_error = "SSH did not become ready."
    while time.monotonic() < deadline:
        result = run_ssh(session, "true", capture=True)
        if result.returncode == 0:
            return
        last_error = (result.stderr or result.stdout or last_error).strip()
        time.sleep(pause_seconds)
    raise AwsError(
        f"Timed out waiting for SSH: {last_error}",
        "Confirm that port 22 is open and that the instance finished booting.",
    )


def wait_for_remote_file(
    session: SshSession,
    path: str,
    *,
    timeout_seconds: float = 900,
    pause_seconds: float = 10,
) -> None:
    """Retry until ``path`` exists on the instance."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        result = run_ssh(session, f"test -f {shlex.quote(path)}", capture=True)
        if result.returncode == 0:
            return
        time.sleep(pause_seconds)
    raise AwsError(
        f"Timed out waiting for {path} on the shared instance.",
        "Inspect instance console output; setup installs Miniconda during "
        "first boot and may take several minutes.",
    )


def conda_run_command(args: Sequence[str]) -> str:
    """Return a remote shell command that runs ``args`` with the env Python."""

    parts = list(args)
    if parts and parts[0] == "python":
        parts[0] = CONDA_PYTHON
    elif not parts or parts[0] != CONDA_PYTHON:
        parts = [CONDA_PYTHON, *parts]
    quoted = " ".join(shlex.quote(part) for part in parts)
    return f"KERAS_BACKEND=jax {quoted}"


def ssh_probe_command() -> str:
    """Return a cheap remote probe used by ``remote.py check``."""

    return conda_run_command(["python", "-c", "print('ok')"])


def translate_boto_error(exc: BaseException) -> AwsError:
    """Turn a boto3/botocore failure into workshop guidance."""

    name = type(exc).__name__
    message = str(exc)
    lowered = message.lower()
    if "unable to locate credentials" in lowered or name == "NoCredentialsError":
        return AwsError(
            "AWS credentials were not found.",
            _aws_credential_resolution(dallinger_config_path()),
        )
    if "unauthorizedoperation" in lowered or "accessdenied" in lowered:
        return AwsError(
            f"AWS denied the request: {message}",
            "Ask an instructor to grant the IAM actions listed in "
            "scripts/aws/README.md.",
        )
    if "invalidinstanceid" in lowered:
        return AwsError(
            "The configured instance ID was not found in this AWS account.",
            "Confirm the instance record in S3 and that your credentials "
            "belong to the workshop account.",
        )
    return AwsError(
        f"{name}: {message}",
        "Rerun the command after checking AWS credentials, region, and "
        "instance state.",
    )
