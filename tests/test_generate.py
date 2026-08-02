import time

import numpy as np
from src.ml.generate import generate_grb_event, PHYSICS_MODELS, ENERGY_BINS, TIME_BINS


def test_generate_grb_event_shape_and_finite():
    rng = np.random.default_rng(seed=1)
    for physics_model in PHYSICS_MODELS:
        tensor, metadata = generate_grb_event(physics_model, rng=rng)
        assert tensor.shape == (3, ENERGY_BINS, TIME_BINS), (
            f"Error, expected shape (3, {ENERGY_BINS}, {TIME_BINS}) got {tensor.shape}"
        )
        assert not np.any(np.isnan(tensor)), f"Error, {physics_model} event contains NaN values"
        assert not np.any(np.isinf(tensor)), f"Error, {physics_model} event contains infinite values"
        assert metadata["physics_model"] == physics_model


def test_generate_grb_event_runs_under_50ms():
    rng = np.random.default_rng(seed=2)
    n_trials = 50

    for physics_model in PHYSICS_MODELS:
        start = time.perf_counter()
        for _ in range(n_trials):
            generate_grb_event(physics_model, rng=rng)
        elapsed = time.perf_counter() - start
        mean_ms = (elapsed / n_trials) * 1000

        assert mean_ms < 50.0, (
            f"Gate 1 failure: {physics_model} averaged {mean_ms:.2f} ms/event, exceeds the 50ms budget"
        )
