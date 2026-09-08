"""Probe SSH to the shared workshop instance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.aws.remote import check_connection
from scripts.aws.support import AwsError


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test SSH communication with the shared workshop instance.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the full traceback when SSH fails",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.perf_counter()
    print("SSH communication test")
    try:
        detail = check_connection()
    except AwsError as exc:
        elapsed = time.perf_counter() - started
        print(f"FAIL  ssh               {elapsed:5.1f}s", file=sys.stderr)
        print(f"Problem: {exc.problem}", file=sys.stderr)
        print(f"Next step: {exc.resolution}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - started
    print(f"PASS  ssh               {elapsed:5.1f}s  {detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
