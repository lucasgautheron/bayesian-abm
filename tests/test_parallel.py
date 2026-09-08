from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

import scripts.parallel as parallel
from scripts.parallel import (
    concatenate_batches,
    partition_runs,
    sample_model_comparison_in_processes,
    sample_model_in_processes,
    simulate_summaries_in_processes,
)


def _count_runs_in_worker(runs):
    for _ in range(runs):
        parallel._advance_worker_progress(1)
    return {"value": np.arange(runs)}


class ParallelSimulationTests(unittest.TestCase):
    def test_partitions_runs_evenly_without_empty_workers(self) -> None:
        self.assertEqual(partition_runs(10, 3), [4, 3, 3])
        self.assertEqual(partition_runs(2, 8), [1, 1])

    def test_concatenates_fields_in_batch_order(self) -> None:
        combined = concatenate_batches(
            [
                {"x": np.array([[1], [2]]), "y": np.array([3, 4])},
                {"x": np.array([[5]]), "y": np.array([6])},
            ]
        )

        np.testing.assert_array_equal(combined["x"], [[1], [2], [5]])
        np.testing.assert_array_equal(combined["y"], [3, 4, 6])

    def test_uses_spawned_process_pool_and_reports_completed_runs(self) -> None:
        class Result:
            def __init__(self, counter, tasks):
                self.counter = counter
                self.tasks = tasks
                self.polls = 0

            def ready(self):
                return self.polls >= 2

            def wait(self, timeout):
                self.assert_timeout = timeout
                self.counter.value = (3, 5)[self.polls]
                self.polls += 1

            def get(self):
                self.counter.value = 7
                return [
                    {"value": np.full((task[1], 1), index)}
                    for index, task in enumerate(self.tasks)
                ]

        class Pool:
            def __init__(self, *, processes, counter):
                self.processes = processes
                self.counter = counter

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def map_async(self, worker, tasks):
                del worker
                self.tasks = tasks
                return Result(self.counter, tasks)

        class Counter:
            value = 0

        class Context:
            def __init__(self):
                self.pool = None

            def Value(self, typecode, initial):
                self.value_arguments = (typecode, initial)
                return Counter()

            def Pool(self, *, processes, initializer, initargs):
                self.initializer = initializer
                self.pool = Pool(processes=processes, counter=initargs[0])
                return self.pool

        context = Context()
        completed = []
        with patch("scripts.parallel.get_context", return_value=context) as get:
            simulated = sample_model_in_processes(
                "latent_network",
                runs=7,
                seed=42,
                cpus=3,
                include_parameters=True,
                summary_names=("first", "second"),
                progress=completed.append,
            )

        get.assert_called_once_with("spawn")
        self.assertEqual(context.value_arguments, ("q", 0))
        self.assertEqual(context.pool.processes, 3)
        self.assertEqual(completed, [3, 2, 2])
        self.assertEqual(simulated["value"].shape, (7, 1))
        self.assertTrue(
            all(
                task[4] == ("first", "second")
                for task in context.pool.tasks
            )
        )

    def test_posterior_predictive_tasks_slice_parameter_draws(self) -> None:
        with patch(
            "scripts.parallel._run_tasks_in_processes",
            return_value={"value": np.zeros((5, 1))},
        ) as run:
            simulate_summaries_in_processes(
                "latent_network",
                parameters={
                    "rate": np.arange(5.0),
                    "scale": np.arange(10.0, 15.0).reshape(5, 1),
                },
                summary_names=("first", "second"),
                seed=7,
                cpus=2,
            )

        tasks = run.call_args.args[1]
        self.assertEqual(run.call_args.args[0], parallel._simulate_summaries_chunk)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tasks[0][1]["rate"].shape, (3,))
        self.assertEqual(tasks[1][1]["rate"].shape, (2,))
        self.assertTrue(all(task[2] == ("first", "second") for task in tasks))
        np.testing.assert_array_equal(tasks[0][1]["rate"], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(tasks[1][1]["rate"], [3.0, 4.0])

    def test_model_comparison_tasks_include_selected_summaries(self) -> None:
        with patch(
            "scripts.parallel._run_tasks_in_processes",
            return_value={"value": np.zeros((3, 1))},
        ) as run:
            sample_model_comparison_in_processes(
                ("first_model", "second_model"),
                runs=3,
                simulator_seed=1,
                selection_seed=2,
                cpus=2,
                summary_names=("second", "first"),
            )

        tasks = run.call_args.args[1]
        self.assertTrue(
            all(task[4] == ("second", "first") for task in tasks)
        )

    def test_shared_progress_counter_works_with_spawned_processes(self) -> None:
        completed = []

        simulated = parallel._run_tasks_in_processes(
            _count_runs_in_worker,
            [3, 2],
            progress=completed.append,
        )

        self.assertEqual(sum(completed), 5)
        self.assertEqual(simulated["value"].shape, (5,))


if __name__ == "__main__":
    unittest.main()
