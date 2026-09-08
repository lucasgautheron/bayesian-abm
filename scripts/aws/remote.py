"""Run workshop commands on the shared instance, with a local fallback."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws.support import (
    AwsError,
    InstanceConfig,
    SshSession,
    conda_run_command,
    ec2_client,
    ensure_instance_running,
    load_instance_config,
    open_ssh_session,
    remote_workdir,
    rsync_exclude_args,
    run_logged,
    run_ssh,
    ssh_options,
    ssh_probe_command,
    translate_boto_error,
    with_remote_cpus,
)


class RemoteExecutionError(AwsError):
    """The workshop command failed after SSH was already working."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check SSH to the shared instance, or run a workshop command "
            "there with an optional local fallback."
        )
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser(
        "check",
        help="probe SSH and the remote conda environment",
    )
    run = subparsers.add_parser(
        "run",
        help="mirror this repo, run a command remotely, and copy results back",
    )
    run.add_argument(
        "--fallback-local",
        action="store_true",
        help="run the command locally if SSH is unavailable",
    )
    run.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="workshop command, usually after --",
    )
    return parser.parse_args(argv)


def normalize_command(command: Sequence[str]) -> list[str]:
    parts = list(command)
    if parts and parts[0] == "--":
        parts = parts[1:]
    if not parts:
        raise AwsError(
            "No command was given to run remotely.",
            "Pass a workshop command after `--`, for example "
            "`python scripts/aws/remote.py run --fallback-local -- "
            "python scripts/simulate.py latent_network`.",
        )
    return parts


def remote_default_cpus(config: InstanceConfig) -> int:
    cpus = int(config["default_cpus"])
    if cpus < 1:
        raise AwsError(
            "default_cpus must be a positive integer.",
            "Rerun `python scripts/aws/instance.py setup`.",
        )
    return cpus


def rsync_to_remote(
    session: SshSession,
    source: Path,
    destination: str,
) -> None:
    mkdir = run_ssh(session, f"mkdir -p {shlex.quote(destination)}", capture=True)
    if mkdir.returncode != 0:
        detail = (mkdir.stderr or mkdir.stdout or "mkdir failed").strip()
        raise AwsError(
            f"Could not create {destination} on the instance: {detail}",
            "Confirm SSH works and rerun the command.",
        )
    result = run_logged(
        [
            "rsync",
            "-az",
            "--delete",
            *rsync_exclude_args(),
            "-e",
            "ssh " + " ".join(ssh_options(session.identity, session.known_hosts)),
            f"{source}/",
            f"{session.target}:{destination}/",
        ],
        capture=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "rsync failed").strip()
        raise AwsError(
            f"Could not copy the local repository to the instance: {detail}",
            "Install rsync and OpenSSH, then rerun the command.",
        )


def rsync_results_back(
    session: SshSession,
    source: str,
    root: Path,
) -> None:
    for relative in ("output", "reports"):
        exists = run_ssh(
            session,
            f"test -d {shlex.quote(f'{source}/{relative}')}",
            capture=True,
        )
        if exists.returncode != 0:
            continue
        local = root / relative
        local.mkdir(parents=True, exist_ok=True)
        result = run_logged(
            [
                "rsync",
                "-az",
                "-e",
                "ssh "
                + " ".join(ssh_options(session.identity, session.known_hosts)),
                f"{session.target}:{source}/{relative}/",
                f"{local}/",
            ],
            capture=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "rsync failed").strip()
            raise AwsError(
                f"Could not copy {relative}/ back from the instance: {detail}",
                "The remote command may have finished; copy the files manually "
                "or rerun the command.",
            )


def run_remote_command(
    session: SshSession,
    workdir: str,
    command: Sequence[str],
    *,
    tty: bool,
) -> None:
    remote = (
        f"cd {shlex.quote(workdir)} && {conda_run_command(command)}"
    )
    result = run_ssh(session, remote, tty=tty, capture=False)
    if result.returncode != 0:
        raise RemoteExecutionError(
            "The remote workshop command failed.",
            "Inspect the remote output above. This is not an SSH outage, so "
            "the command was not rerun locally.",
        )


def check_connection(
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> str:
    """Probe SSH without starting a stopped instance or syncing the repo."""

    config = load_instance_config(root)
    ec2 = ec2_client(config["region"])
    description = ensure_instance_running(
        ec2,
        config,
        start_if_stopped=False,
        announce=announce,
    )
    with open_ssh_session(description, root=root) as session:
        result = run_ssh(session, ssh_probe_command(), capture=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "probe failed").strip()
        raise AwsError(
            f"SSH reached the instance, but the remote Python probe failed: "
            f"{detail}",
            "An instructor may need to rerun `python scripts/aws/instance.py "
            "setup` so the remote environment is complete.",
        )
    return "SSH ok"


def run_on_instance(
    command: Sequence[str],
    *,
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> None:
    config = load_instance_config(root)
    remote_command = with_remote_cpus(command, remote_default_cpus(config))
    ec2 = ec2_client(config["region"])
    description = ensure_instance_running(
        ec2,
        config,
        start_if_stopped=True,
        announce=announce,
    )
    workdir = remote_workdir()
    announce(f"Mirroring the repository to {workdir}.")
    with open_ssh_session(description, root=root) as session:
        rsync_to_remote(session, root, workdir)
        announce("Running the command on the shared instance.")
        run_remote_command(
            session,
            workdir,
            remote_command,
            tty=sys.stdin.isatty(),
        )
        announce("Copying output/ and reports/ back.")
        rsync_results_back(session, workdir, root)


def run_locally(
    command: Sequence[str],
    *,
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> int:
    announce("Running the command locally.")
    result = subprocess.run(list(command), cwd=root)
    return result.returncode


def run_with_optional_fallback(
    command: Sequence[str],
    *,
    fallback_local: bool,
    root: Path = ROOT,
    announce: Callable[[str], None] = print,
) -> int:
    try:
        run_on_instance(command, root=root, announce=announce)
        return 0
    except RemoteExecutionError:
        raise
    except AwsError as exc:
        if not fallback_local:
            raise
        announce(f"Remote run unavailable: {exc.problem}")
        announce(f"Next step was: {exc.resolution}")
        return run_locally(command, root=root, announce=announce)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.action == "check":
            detail = check_connection()
            print(detail)
            return 0
        command = normalize_command(args.command)
        return run_with_optional_fallback(
            command,
            fallback_local=args.fallback_local,
        )
    except AwsError as exc:
        print(f"error: {exc.problem}", file=sys.stderr)
        print(f"Next step: {exc.resolution}", file=sys.stderr)
        return 1
    except Exception as exc:
        translated = translate_boto_error(exc)
        print(f"error: {translated.problem}", file=sys.stderr)
        print(f"Next step: {translated.resolution}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
