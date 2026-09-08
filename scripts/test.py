"""Probe workshop SSH key access, instance state, and SSH connectivity."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws.support import (
    AwsError,
    cached_workshop_ssh_identity,
    ec2_client,
    instance_state,
    load_instance_config,
    open_ssh_session,
    require_live_instance,
    run_ssh,
)


CHECK_NAMES = ("ssh-key", "instance", "connection")
NAME_WIDTH = max(len(name) for name in CHECK_NAMES)


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    elapsed: float
    detail: str = ""
    problem: str = ""
    resolution: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test workshop SSH key access, instance state, and SSH "
            "connectivity."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the full traceback when a check fails",
    )
    return parser.parse_args(argv)


def check_ssh_key(*, root: Path = ROOT) -> str:
    """Return a detail string when the workshop SSH key can be read."""

    identity = cached_workshop_ssh_identity(root=root)
    return f"available at {identity}"


def check_instance(*, root: Path = ROOT) -> tuple[str, dict[str, Any]]:
    """Return a detail string and description when the instance is running."""

    try:
        config = load_instance_config(root)
        ec2 = ec2_client(config["region"])
        description = require_live_instance(ec2, config)
    except AwsError as exc:
        raise AwsError(exc.problem, _instance_check_resolution(exc)) from exc
    state = instance_state(description)
    instance_id = config["instance_id"]
    if state != "running":
        if state in {"stopped", "stopping"}:
            resolution = (
                "Run `python scripts/aws/instance.py start` from the "
                "repository root."
            )
        else:
            resolution = "Wait for the current transition to finish."
        raise AwsError(
            f"Shared instance {instance_id} is {state}.",
            resolution,
        )
    return f"{instance_id} is running", description


def _instance_check_resolution(exc: AwsError) -> str:
    """Keep `/test` from recommending instance creation."""

    if "setup" in exc.resolution.lower():
        return (
            "Ask an instructor to restore the shared instance. "
            "`/test` does not create it."
        )
    return exc.resolution


def check_ssh_connection(
    description: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> str:
    """Return a detail string when SSH to the running instance works."""

    with open_ssh_session(description, root=root) as session:
        result = run_ssh(session, "true", capture=True)
        target = session.target
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "SSH failed").strip()
        raise AwsError(
            f"Could not connect over SSH: {detail}",
            "Confirm that port 22 is open and that the instance finished "
            "booting.",
        )
    return f"connected to {target}"


def _run_check(
    name: str,
    callback: Callable[[], str],
    *,
    verbose: bool,
) -> CheckResult:
    started = time.perf_counter()
    try:
        detail = callback()
    except AwsError as exc:
        if verbose:
            traceback.print_exc()
        return CheckResult(
            name=name,
            status="FAIL",
            elapsed=time.perf_counter() - started,
            problem=exc.problem,
            resolution=exc.resolution,
        )
    return CheckResult(
        name=name,
        status="PASS",
        elapsed=time.perf_counter() - started,
        detail=detail,
    )


def _report(result: CheckResult) -> None:
    extra = f"  {result.detail}" if result.detail else ""
    print(
        f"{result.status}  {result.name:<{NAME_WIDTH}} "
        f"{result.elapsed:5.1f}s{extra}",
        flush=True,
    )
    if result.status == "FAIL":
        print(f"Problem: {result.problem}", flush=True)
        print(f"Next step: {result.resolution}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print("Workshop SSH tests", flush=True)

    key_result = _run_check(
        "ssh-key",
        lambda: check_ssh_key(),
        verbose=args.verbose,
    )
    _report(key_result)

    description: dict[str, Any] | None = None
    started = time.perf_counter()
    try:
        detail, description = check_instance()
    except AwsError as exc:
        if args.verbose:
            traceback.print_exc()
        instance_result = CheckResult(
            name="instance",
            status="FAIL",
            elapsed=time.perf_counter() - started,
            problem=exc.problem,
            resolution=exc.resolution,
        )
    else:
        instance_result = CheckResult(
            name="instance",
            status="PASS",
            elapsed=time.perf_counter() - started,
            detail=detail,
        )
    _report(instance_result)

    if description is None:
        connection_result = CheckResult(
            name="connection",
            status="SKIP",
            elapsed=0.0,
            detail="instance is not running",
        )
    else:
        connection_result = _run_check(
            "connection",
            lambda: check_ssh_connection(description),
            verbose=args.verbose,
        )
    _report(connection_result)

    results = (key_result, instance_result, connection_result)
    return 0 if all(result.status != "FAIL" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
