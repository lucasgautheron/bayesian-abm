"""Run a fast, self-contained smoke test of the local workflow.

The test uses an in-memory model and observations. It does not require the
workshop datasets or ``.config/summary.ini``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_PYTHON_MIN = (3, 11)
SUPPORTED_PYTHON_MAX = (3, 13)
REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scipy",
    "networkx",
    "pyarrow",
    "tqdm",
    "pymc",
    "jax",
    "keras",
    "bayesflow",
)


@dataclass(frozen=True)
class Diagnostic:
    """A concise failure explanation and a concrete next action."""

    problem: str
    resolution: str


class SmokeTestFailure(RuntimeError):
    """A smoke-test failure with stage and remediation context."""

    def __init__(
        self,
        stage: str,
        problem: str,
        resolution: str,
    ) -> None:
        super().__init__(problem)
        self.stage = stage
        self.problem = problem
        self.resolution = resolution


def python_version_diagnostic(
    version: tuple[int, int],
) -> Diagnostic | None:
    """Return guidance when the interpreter is outside the supported range."""

    if SUPPORTED_PYTHON_MIN <= version <= SUPPORTED_PYTHON_MAX:
        return None
    return Diagnostic(
        problem=(
            f"Python {version[0]}.{version[1]} is unsupported; this project "
            "requires Python 3.11, 3.12, or 3.13."
        ),
        resolution=(
            "Create and activate a Python 3.12 environment, then reinstall "
            "the project requirements. See README.md under Installation."
        ),
    )


def missing_module_diagnostic(name: str) -> Diagnostic:
    """Return the installation guidance for one missing dependency."""

    if name == "jax":
        command = "python -m pip install jax"
    else:
        command = "python -m pip install -r requirements.txt"
    return Diagnostic(
        problem=f"Required Python module {name!r} is not importable.",
        resolution=(
            "Confirm that the project environment is active, then run "
            f"`{command}` from the repository root."
        ),
    )


def exception_diagnostic(exc: BaseException) -> Diagnostic:
    """Classify common local scientific-Python failures."""

    if isinstance(exc, ModuleNotFoundError):
        name = (exc.name or "unknown").split(".", maxsplit=1)[0]
        if name in {"tensorflow", "torch"}:
            return Diagnostic(
                problem=(
                    f"Keras tried to load the unavailable {name!r} backend."
                ),
                resolution=(
                    "Select JAX before starting Python: "
                    "`export KERAS_BACKEND=jax`, then rerun the test."
                ),
            )
        return missing_module_diagnostic(name)

    message = str(exc)
    lowered = message.lower()
    if "keras" in lowered and "backend" in lowered:
        return Diagnostic(
            problem=message,
            resolution=(
                "Select JAX before starting Python: "
                "`export KERAS_BACKEND=jax`, then rerun the test."
            ),
        )
    if any(token in lowered for token in ("dlopen", "wrong architecture")):
        return Diagnostic(
            problem=message,
            resolution=(
                "A compiled dependency is incompatible with this Python "
                "environment. Recreate a Python 3.12 environment and install "
                "`requirements.txt` plus `jax` again."
            ),
        )
    if isinstance(exc, MemoryError):
        return Diagnostic(
            problem="The smoke test exhausted available memory.",
            resolution=(
                "Close memory-intensive applications and rerun with the CPU "
                "environment described in README.md."
            ),
        )
    return Diagnostic(
        problem=f"{type(exc).__name__}: {message}",
        resolution=(
            "Rerun with `python scripts/test.py --verbose` and inspect the "
            "first failing project frame in the traceback."
        ),
    )


def probe_modules(names: Sequence[str]) -> dict[str, str]:
    """Import dependencies together out of process and report native crashes."""

    code = f"""
import importlib

for name in {tuple(names)!r}:
    print(f"PROBE {{name}}", flush=True)
    module = importlib.import_module(name)
    detail = module.backend.backend() if name == "keras" else "ok"
    print(f"PASS {{name}} {{detail}}", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode == 0:
        details = {}
        for line in result.stdout.splitlines():
            if line.startswith("PASS "):
                _, name, detail = line.split(maxsplit=2)
                details[name] = detail
        return details

    probed = [
        line.removeprefix("PROBE ")
        for line in result.stdout.splitlines()
        if line.startswith("PROBE ")
    ]
    completed = {
        line.split(maxsplit=2)[1]
        for line in result.stdout.splitlines()
        if line.startswith("PASS ")
    }
    failed_name = next(
        (name for name in reversed(probed) if name not in completed),
        "unknown dependency",
    )

    if result.returncode < 0:
        try:
            signal_name = signal.Signals(-result.returncode).name
        except ValueError:
            signal_name = f"signal {-result.returncode}"
        raise SmokeTestFailure(
            "environment",
            f"Importing {failed_name!r} crashed Python with {signal_name}.",
            (
                "A compiled dependency is incompatible with this Python "
                "environment. Recreate a Python 3.12 environment and install "
                "`requirements.txt` plus `jax` again."
            ),
        )

    missing_match = re.search(
        r"No module named ['\"]([^'\"]+)['\"]",
        result.stderr,
    )
    if missing_match is not None:
        missing_name = missing_match.group(1).split(".", maxsplit=1)[0]
        if failed_name == "keras" and missing_name in {"tensorflow", "torch"}:
            diagnostic = Diagnostic(
                problem=(
                    f"Keras tried to load the unavailable "
                    f"{missing_name!r} backend."
                ),
                resolution=(
                    "Select JAX before starting Python: "
                    "`export KERAS_BACKEND=jax`, then rerun the test."
                ),
            )
        else:
            diagnostic = missing_module_diagnostic(missing_name)
    else:
        message = result.stderr.strip().splitlines()
        diagnostic = Diagnostic(
            problem=(
                message[-1]
                if message
                else f"Importing {name!r} exited with status {result.returncode}."
            ),
            resolution=(
                "Rerun with `python scripts/test.py --verbose`, then verify "
                "the active environment against README.md."
            ),
        )
    raise SmokeTestFailure(
        "environment",
        diagnostic.problem,
        diagnostic.resolution,
    )


def probe_module(name: str) -> str:
    """Import one dependency out of process."""

    return probe_modules((name,))[name]


def check_environment() -> str:
    """Import the local stack and verify its required backend."""

    version = (sys.version_info.major, sys.version_info.minor)
    diagnostic = python_version_diagnostic(version)
    if diagnostic is not None:
        raise SmokeTestFailure(
            "environment",
            diagnostic.problem,
            diagnostic.resolution,
        )

    # Headless plotting makes the command safe in terminals and remote shells.
    os.environ.setdefault("MPLBACKEND", "Agg")
    backend = probe_modules(REQUIRED_MODULES)["keras"]
    if backend != "jax":
        raise SmokeTestFailure(
            "environment",
            f"Keras is using the {backend!r} backend instead of 'jax'.",
            (
                "Set `KERAS_BACKEND=jax` before Python starts (or configure "
                "that variable in the Conda environment), reactivate the "
                "environment, and rerun the test."
            ),
        )
    return f"Python {version[0]}.{version[1]}; all imports; Keras backend jax"


def make_dummy_components() -> tuple[Any, Any, tuple[str, ...]]:
    """Build an unregistered model and deterministic observed summaries."""

    import numpy as np

    pm = importlib.import_module("pymc")

    from base.model import Model
    from base.observations import Observations
    from base.summaries import compute_scalar_summaries

    class DummyModel(Model):
        """A scalar location prior observed through short Gaussian samples."""

        name = "_workflow_smoke_test"
        dataset = "_workflow_smoke_data"
        inference_variables = ("location",)

        def build_prior(self, **context: Any) -> pm.Model:
            del context
            with pm.Model() as prior:
                pm.Normal("location", mu=0.0, sigma=1.0)
            return prior

        def simulate(
            self,
            parameters: Any,
            rng: np.random.Generator,
            **context: Any,
        ) -> dict[str, np.ndarray[Any, Any]]:
            location = float(np.asarray(parameters["location"]).reshape(-1)[0])
            count = int(context["sample_size"])
            return {
                "value": np.asarray(
                    rng.normal(location, 0.35, size=count),
                    dtype=np.float32,
                )
            }

        def validate_simulation(
            self,
            simulation: Any,
            **context: Any,
        ) -> dict[str, np.ndarray[Any, Any]]:
            if set(simulation) != {"value"}:
                raise ValueError("dummy simulation must contain only 'value'")
            values = np.asarray(simulation["value"], dtype=np.float32)
            if values.shape != (int(context["sample_size"]),):
                raise ValueError("dummy simulation has the wrong shape")
            if not np.all(np.isfinite(values)):
                raise ValueError("dummy simulation must be finite")
            return {"value": values}

        def summarize(
            self,
            simulation: Any,
            summaries: Any,
            **context: Any,
        ) -> dict[str, np.ndarray[Any, Any]]:
            data = self.validate_simulation(simulation, **context)
            return compute_scalar_summaries(
                data,
                summaries,
                label="dummy summary",
            )

    context = {"sample_size": 8}
    summaries = {
        "sample_mean": lambda data: np.mean(data["value"]),
        "sample_scale": lambda data: np.std(data["value"]),
    }
    observed_data = {
        "value": np.asarray(
            [-0.15, 0.05, 0.21, 0.38, 0.44, 0.51, 0.65, 0.71],
            dtype=np.float32,
        )
    }
    observed_summaries = compute_scalar_summaries(
        observed_data,
        summaries,
        label="dummy summary",
    )
    observations = Observations(
        dataset=DummyModel.dataset,
        context=context,
        summaries=summaries,
        conditions={
            name: values[None, ...]
            for name, values in observed_summaries.items()
        },
        observation_ids=np.asarray(["dummy"]),
    )
    return DummyModel(), observations, tuple(summaries)


def check_dummy_contract(model: Any, observations: Any) -> str:
    """Check seeded model behavior and the in-memory data contract."""

    import numpy as np

    first_parameters, first_data = model.sample(
        seed=7,
        **observations.context,
    )
    second_parameters, second_data = model.sample(
        seed=7,
        **observations.context,
    )
    np.testing.assert_array_equal(
        first_parameters["location"],
        second_parameters["location"],
    )
    np.testing.assert_array_equal(first_data["value"], second_data["value"])
    if first_data["value"].shape != (observations.context["sample_size"],):
        raise AssertionError("dummy model returned an unexpected data shape")
    if not np.all(np.isfinite(first_data["value"])):
        raise AssertionError("dummy model returned non-finite data")
    if observations.count != 1:
        raise AssertionError("dummy observations must contain one dataset")
    return "seeded prior and simulation; finite dummy observations"


def run_simulation_stage(
    model: Any,
    observations: Any,
    summary_names: Sequence[str],
    destination: Path,
) -> str:
    """Exercise the production prior-predictive workflow."""

    from unittest.mock import patch

    import matplotlib.pyplot as plt

    import scripts.simulate as simulate_script

    output = destination / "simulations.png"
    with (
        patch.object(simulate_script, "resolve_model", return_value=model),
        patch.object(
            simulate_script,
            "load_observations",
            return_value=observations,
        ),
    ):
        figure = simulate_script.run_simulations(
            model.name,
            output_path=output,
            runs=12,
            seed=11,
            cpus=1,
            summary_names=summary_names,
        )
    try:
        if not output.is_file() or output.stat().st_size == 0:
            raise AssertionError("simulation figure was not written")
        if len(figure.axes) != len(summary_names) ** 2:
            raise AssertionError("simulation pair plot has missing panels")
    finally:
        plt.close(figure)
    return f"12 simulations; {len(summary_names)} summaries; figure verified"


def run_inference_stage(
    model: Any,
    observations: Any,
    summary_names: Sequence[str],
    destination: Path,
) -> str:
    """Exercise minimal production training and posterior prediction."""

    from unittest.mock import patch

    import matplotlib.pyplot as plt

    import scripts.inference as inference_script

    posterior_path = destination / "posterior.png"
    summary_path = destination / "training_simulations.png"
    with (
        patch.object(inference_script, "resolve_model", return_value=model),
        patch.object(
            inference_script,
            "load_observations",
            return_value=observations,
        ),
    ):
        plots = inference_script.run_inference(
            model.name,
            output_path=posterior_path,
            summary_output_path=summary_path,
            epochs=1,
            num_simulations=32,
            batch_size=8,
            posterior_draws=16,
            predictive_runs=8,
            diagnostic_datasets=0,
            diagnostic_draws=8,
            observation_batch_size=8,
            seed=13,
            cpus=1,
            summary_names=summary_names,
        )
    try:
        expected_plots = {"simulations", "posterior", "posterior_predictive"}
        missing = expected_plots.difference(plots)
        if missing:
            raise AssertionError(
                f"inference did not return plots: {', '.join(sorted(missing))}"
            )
        expected_files = (
            posterior_path,
            posterior_path.with_name("posterior_predictive.png"),
            summary_path,
        )
        missing_files = [
            path.name
            for path in expected_files
            if not path.is_file() or path.stat().st_size == 0
        ]
        if missing_files:
            raise AssertionError(
                "inference did not write: " + ", ".join(missing_files)
            )
    finally:
        for figure in plots.values():
            plt.close(figure)
    return "32 training simulations; posterior and predictive figures verified"


def run_stage(
    name: str,
    action: Callable[[], str],
    emit: Callable[[str], None],
) -> None:
    """Run one timed stage and wrap unexpected failures."""

    started = time.perf_counter()
    try:
        detail = action()
    except SmokeTestFailure:
        raise
    except BaseException as exc:
        diagnostic = exception_diagnostic(exc)
        raise SmokeTestFailure(
            name,
            diagnostic.problem,
            diagnostic.resolution,
        ) from exc
    elapsed = time.perf_counter() - started
    emit(f"PASS  {name:<18} {elapsed:5.1f}s  {detail}")


def run_smoke_test(emit: Callable[[str], None] = print) -> None:
    """Run every local workflow smoke-test stage."""

    run_stage("environment", check_environment, emit)
    model, observations, summary_names = make_dummy_components()
    run_stage(
        "dummy model/data",
        lambda: check_dummy_contract(model, observations),
        emit,
    )
    with TemporaryDirectory(prefix="bayesian-workflow-test-") as temporary:
        destination = Path(temporary)
        run_stage(
            "simulation",
            lambda: run_simulation_stage(
                model,
                observations,
                summary_names,
                destination,
            ),
            emit,
        )
        run_stage(
            "inference",
            lambda: run_inference_stage(
                model,
                observations,
                summary_names,
                destination,
            ),
            emit,
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test the local simulation and inference workflow using "
            "only an in-memory dummy model and data."
        )
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="show the full traceback when a stage fails",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command and return a process exit status."""

    args = parse_args(argv)
    started = time.perf_counter()
    print("Local workflow smoke test")
    try:
        run_smoke_test()
    except SmokeTestFailure as exc:
        print(f"FAIL  {exc.stage}", file=sys.stderr)
        print(f"Problem: {exc.problem}", file=sys.stderr)
        print(f"Next step: {exc.resolution}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1
    elapsed = time.perf_counter() - started
    print(f"PASS  complete            {elapsed:5.1f}s  local workflow is ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
