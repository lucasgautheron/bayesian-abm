"""Multiprocess helpers for precomputed simulation batches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from multiprocessing import get_context
from typing import Any

import numpy as np
from numpy.typing import NDArray


Seed = int | np.integer | np.random.SeedSequence
SimulationBatch = Mapping[str, NDArray[Any]]
_WORKER_PROGRESS: Any = None


def _seed_sequence(seed: Seed) -> np.random.SeedSequence:
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def validate_cpus(cpus: int) -> None:
    """Require at least one simulation worker."""

    if cpus < 1:
        raise ValueError("cpus must be positive")


def _initialize_worker(progress: Any) -> None:
    global _WORKER_PROGRESS
    _WORKER_PROGRESS = progress


def _advance_worker_progress(amount: int) -> None:
    if _WORKER_PROGRESS is None:
        return
    with _WORKER_PROGRESS.get_lock():
        _WORKER_PROGRESS.value += amount


def partition_runs(runs: int, cpus: int) -> list[int]:
    """Split runs as evenly as possible across available workers."""

    validate_cpus(cpus)
    if runs < 1:
        raise ValueError("runs must be positive")
    worker_count = min(runs, cpus)
    quotient, remainder = divmod(runs, worker_count)
    return [
        quotient + (worker < remainder)
        for worker in range(worker_count)
    ]


def concatenate_batches(
    batches: Sequence[SimulationBatch],
) -> dict[str, NDArray[Any]]:
    """Concatenate aligned simulation batches along their run axis."""

    if not batches:
        raise ValueError("at least one simulation batch is required")
    keys = tuple(batches[0])
    expected = set(keys)
    if any(set(batch) != expected for batch in batches[1:]):
        raise ValueError("simulation batches must contain the same fields")
    return {
        name: np.concatenate(
            [np.asarray(batch[name]) for batch in batches],
            axis=0,
        )
        for name in keys
    }


def _sample_model_chunk(
    task: tuple[
        str,
        int,
        np.random.SeedSequence,
        bool,
        tuple[str, ...],
    ],
) -> dict[str, NDArray[Any]]:
    """Build and sample one model inside a worker process."""

    model_name, runs, seed, include_parameters, summary_names = task

    from base.observations import load_observations
    from models import resolve_model

    model = resolve_model(model_name)
    observations = load_observations(
        model.dataset,
        summary_names=summary_names,
    )
    simulator = model.to_bayesflow_simulator(
        observations.summaries,
        seed=seed,
        include_parameters=include_parameters,
        progress=(
            _advance_worker_progress
            if _WORKER_PROGRESS is not None
            else None
        ),
        **observations.context,
    )
    return dict(simulator.sample((runs,)))


def _run_tasks_in_processes(
    worker: Callable[[Any], SimulationBatch],
    tasks: Sequence[Any],
    *,
    progress: Callable[[int], object] | None,
) -> dict[str, NDArray[Any]]:
    """Run simulation tasks and relay worker progress to the parent."""

    context = get_context("spawn")
    counter = context.Value("q", 0) if progress is not None else None
    with context.Pool(
        processes=len(tasks),
        initializer=_initialize_worker,
        initargs=(counter,),
    ) as pool:
        result = pool.map_async(worker, tasks)
        reported = 0
        while not result.ready():
            result.wait(0.1)
            if progress is not None:
                current = counter.value
                if current > reported:
                    progress(current - reported)
                    reported = current
        batches = result.get()
        if progress is not None:
            current = counter.value
            if current > reported:
                progress(current - reported)
    return concatenate_batches(batches)


def sample_model_in_processes(
    model_name: str,
    *,
    runs: int,
    seed: Seed,
    cpus: int,
    include_parameters: bool,
    summary_names: Sequence[str],
    progress: Callable[[int], object] | None = None,
) -> dict[str, NDArray[Any]]:
    """Sample independent model chunks in spawned worker processes."""

    counts = partition_runs(runs, cpus)
    seeds = _seed_sequence(seed).spawn(len(counts))
    tasks = [
        (
            model_name,
            count,
            child_seed,
            include_parameters,
            tuple(summary_names),
        )
        for count, child_seed in zip(counts, seeds)
    ]
    return _run_tasks_in_processes(
        _sample_model_chunk,
        tasks,
        progress=progress,
    )


def _parameter_draw_count(parameters: Mapping[str, NDArray[Any]]) -> int:
    """Return the shared stacked-draw length of ``parameters``."""

    counts = [int(np.asarray(values).shape[0]) for values in parameters.values()]
    if not counts:
        raise ValueError("parameter draws are required")
    if len(set(counts)) != 1:
        raise ValueError("parameter draws must share a draw axis")
    return counts[0]


def _slice_parameter_draws(
    parameters: Mapping[str, NDArray[Any]],
    start: int,
    stop: int,
) -> dict[str, NDArray[Any]]:
    """Return a contiguous slice of stacked parameter draws."""

    return {
        name: np.asarray(values)[start:stop]
        for name, values in parameters.items()
    }


def _simulate_summaries_chunk(
    task: tuple[
        str,
        dict[str, NDArray[Any]],
        tuple[str, ...],
        np.random.SeedSequence,
    ],
) -> dict[str, NDArray[Any]]:
    """Simulate summaries from one parameter chunk in a worker process."""

    model_name, parameters, summary_names, seed = task

    from base.observations import load_observations
    from models import resolve_model

    model = resolve_model(model_name)
    observations = load_observations(
        model.dataset,
        summary_names=summary_names,
    )
    return model.simulate_summaries(
        parameters,
        observations.summaries,
        seed=seed,
        progress=(
            _advance_worker_progress
            if _WORKER_PROGRESS is not None
            else None
        ),
        **observations.context,
    )


def simulate_summaries_in_processes(
    model_name: str,
    *,
    parameters: Mapping[str, NDArray[Any]],
    summary_names: Sequence[str],
    seed: Seed,
    cpus: int,
    progress: Callable[[int], object] | None = None,
) -> dict[str, NDArray[Any]]:
    """Simulate posterior-predictive summaries in spawned worker processes."""

    n_draws = _parameter_draw_count(parameters)
    counts = partition_runs(n_draws, cpus)
    seeds = _seed_sequence(seed).spawn(len(counts))
    tasks = []
    start = 0
    for count, child_seed in zip(counts, seeds):
        stop = start + count
        tasks.append(
            (
                model_name,
                _slice_parameter_draws(parameters, start, stop),
                tuple(summary_names),
                child_seed,
            )
        )
        start = stop
    return _run_tasks_in_processes(
        _simulate_summaries_chunk,
        tasks,
        progress=progress,
    )


def _sample_model_comparison_chunk(
    task: tuple[
        tuple[str, ...],
        int,
        np.random.SeedSequence,
        np.random.SeedSequence,
        tuple[str, ...],
    ],
) -> dict[str, NDArray[Any]]:
    """Build and sample one model-comparison chunk in a worker process."""

    (
        model_names,
        runs,
        simulator_seed,
        selection_seed,
        summary_names,
    ) = task

    import bayesflow as bf

    from base.model import Model
    from base.observations import load_observations
    from models import resolve_model

    models = Model.validate_collection(
        [resolve_model(name) for name in model_names]
    )
    observations = load_observations(
        models[0].dataset,
        summary_names=summary_names,
    )
    model_seeds = simulator_seed.spawn(len(models))
    simulators = [
        model.to_bayesflow_simulator(
            observations.summaries,
            seed=model_seed,
            include_parameters=False,
            progress=(
                _advance_worker_progress
                if _WORKER_PROGRESS is not None
                else None
            ),
            **observations.context,
        )
        for model, model_seed in zip(models, model_seeds)
    ]
    simulator = bf.simulators.ModelComparisonSimulator(
        simulators=simulators,
        use_mixed_batches=True,
        key_conflicts="error",
    )
    np.random.seed(
        int(selection_seed.generate_state(1, dtype=np.uint32)[0])
    )
    return dict(simulator.sample((runs,)))


def sample_model_comparison_in_processes(
    model_names: Sequence[str],
    *,
    runs: int,
    simulator_seed: Seed,
    selection_seed: Seed,
    cpus: int,
    summary_names: Sequence[str],
    progress: Callable[[int], object] | None = None,
) -> dict[str, NDArray[Any]]:
    """Sample model-comparison chunks in spawned worker processes."""

    counts = partition_runs(runs, cpus)
    simulator_seeds = _seed_sequence(simulator_seed).spawn(len(counts))
    selection_seeds = _seed_sequence(selection_seed).spawn(len(counts))
    names = tuple(model_names)
    tasks = [
        (
            names,
            count,
            simulation_seed,
            model_selection_seed,
            tuple(summary_names),
        )
        for count, simulation_seed, model_selection_seed in zip(
            counts,
            simulator_seeds,
            selection_seeds,
        )
    ]
    return _run_tasks_in_processes(
        _sample_model_comparison_chunk,
        tasks,
        progress=progress,
    )


__all__ = [
    "concatenate_batches",
    "partition_runs",
    "sample_model_comparison_in_processes",
    "sample_model_in_processes",
    "simulate_summaries_in_processes",
    "validate_cpus",
]
