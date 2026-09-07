from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from subprocess import CompletedProcess
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import test as test_script


class DiagnosticTests(unittest.TestCase):
    def test_supported_python_range_is_inclusive(self) -> None:
        self.assertIsNone(test_script.python_version_diagnostic((3, 11)))
        self.assertIsNone(test_script.python_version_diagnostic((3, 13)))
        for version in ((3, 10), (3, 14)):
            with self.subTest(version=version):
                diagnostic = test_script.python_version_diagnostic(version)
                self.assertIsNotNone(diagnostic)
                assert diagnostic is not None
                self.assertIn("Python 3.12 environment", diagnostic.resolution)

    def test_missing_jax_has_its_separate_install_command(self) -> None:
        jax = test_script.missing_module_diagnostic("jax")
        numpy = test_script.missing_module_diagnostic("numpy")

        self.assertIn("python -m pip install jax", jax.resolution)
        self.assertIn(
            "python -m pip install -r requirements.txt",
            numpy.resolution,
        )

    def test_missing_default_keras_backend_suggests_jax(self) -> None:
        result = CompletedProcess(
            args=[],
            returncode=1,
            stdout="PROBE keras\n",
            stderr="ModuleNotFoundError: No module named 'tensorflow'",
        )
        with patch.object(test_script.subprocess, "run", return_value=result):
            with self.assertRaises(test_script.SmokeTestFailure) as raised:
                test_script.probe_module("keras")

        self.assertIn("KERAS_BACKEND=jax", raised.exception.resolution)

    def test_module_probe_reports_missing_transitive_dependency(self) -> None:
        result = CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="ModuleNotFoundError: No module named 'jax'",
        )
        with patch.object(test_script.subprocess, "run", return_value=result):
            with self.assertRaises(test_script.SmokeTestFailure) as raised:
                test_script.probe_module("bayesflow")

        self.assertIn("'jax'", raised.exception.problem)
        self.assertIn("python -m pip install jax", raised.exception.resolution)

    def test_module_probe_reports_native_crash(self) -> None:
        result = CompletedProcess(
            args=[],
            returncode=-11,
            stdout="",
            stderr="",
        )
        with patch.object(test_script.subprocess, "run", return_value=result):
            with self.assertRaises(test_script.SmokeTestFailure) as raised:
                test_script.probe_module("numpy")

        self.assertIn("SIGSEGV", raised.exception.problem)
        self.assertIn("Python 3.12 environment", raised.exception.resolution)


class OrchestrationTests(unittest.TestCase):
    def test_run_stage_wraps_failure_with_stage_name(self) -> None:
        def fail() -> str:
            raise ValueError("bad shape")

        with self.assertRaises(test_script.SmokeTestFailure) as raised:
            test_script.run_stage("simulation", fail, lambda message: None)

        self.assertEqual(raised.exception.stage, "simulation")
        self.assertIn("bad shape", raised.exception.problem)
        self.assertIn("--verbose", raised.exception.resolution)

    def test_smoke_test_runs_all_stages_in_order(self) -> None:
        model = object()
        observations = SimpleNamespace()
        messages: list[str] = []

        with (
            patch.object(
                test_script,
                "check_environment",
                return_value="environment ready",
            ) as environment,
            patch.object(
                test_script,
                "make_dummy_components",
                return_value=(model, observations, ("mean", "scale")),
            ),
            patch.object(
                test_script,
                "check_dummy_contract",
                return_value="contract ready",
            ) as contract,
            patch.object(
                test_script,
                "run_simulation_stage",
                return_value="simulation ready",
            ) as simulation,
            patch.object(
                test_script,
                "run_inference_stage",
                return_value="inference ready",
            ) as inference,
        ):
            test_script.run_smoke_test(messages.append)

        environment.assert_called_once_with()
        contract.assert_called_once_with(model, observations)
        simulation.assert_called_once()
        inference.assert_called_once()
        self.assertEqual(
            [message.split()[1] for message in messages],
            ["environment", "dummy", "simulation", "inference"],
        )
        destination = simulation.call_args.args[3]
        self.assertIsInstance(destination, Path)
        self.assertEqual(inference.call_args.args[3], destination)
        self.assertFalse(destination.exists())

    def test_main_returns_zero_on_success(self) -> None:
        stdout = StringIO()
        with (
            patch.object(test_script, "run_smoke_test"),
            redirect_stdout(stdout),
        ):
            status = test_script.main([])

        self.assertEqual(status, 0)
        self.assertIn("local workflow is ready", stdout.getvalue())

    def test_main_returns_one_with_actionable_failure(self) -> None:
        failure = test_script.SmokeTestFailure(
            "environment",
            "missing dependency",
            "install dependency",
        )
        stderr = StringIO()
        with (
            patch.object(
                test_script,
                "run_smoke_test",
                side_effect=failure,
            ),
            redirect_stdout(StringIO()),
            redirect_stderr(stderr),
        ):
            status = test_script.main([])

        self.assertEqual(status, 1)
        output = stderr.getvalue()
        self.assertIn("FAIL  environment", output)
        self.assertIn("Problem: missing dependency", output)
        self.assertIn("Next step: install dependency", output)


if __name__ == "__main__":
    unittest.main()
