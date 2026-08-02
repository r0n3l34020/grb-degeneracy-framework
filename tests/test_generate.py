import time
import json

import numpy as np
from src.ml.generate import (
    generate_grb_event,
    generate_dataset_suite,
    PHYSICS_MODELS,
    ENERGY_BINS,
    TIME_BINS,
)


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


def test_generate_dataset_suite_balanced_across_classes_and_snr(tmp_path):
    out_path, metadata = generate_dataset_suite(
        events_per_class=9, snr_levels=(5.0, 15.0, 50.0),
        out_path=tmp_path / "suite.h5", n_workers=2, write_batch_size=5,
    )

    assert len(metadata) == 27  # 3 classes * 9 events_per_class
    counts_by_model = {m: sum(1 for x in metadata if x["physics_model"] == m) for m in PHYSICS_MODELS}
    assert all(count == 9 for count in counts_by_model.values())

    counts_by_snr = {snr: sum(1 for x in metadata if x["snr"] == snr) for snr in (5.0, 15.0, 50.0)}
    assert all(count == 9 for count in counts_by_snr.values())  # 3 classes * 3 events per SNR

    import h5py
    with h5py.File(out_path, "r") as f:
        assert f["tensors"].shape == (27, 3, ENERGY_BINS, TIME_BINS)
        assert f["labels"].shape == (27,)
        label_map = json.loads(f.attrs["label_map"])
        assert set(label_map.keys()) == set(PHYSICS_MODELS)
        tensors = f["tensors"][:]
        assert not np.any(np.isnan(tensors))
        assert not np.any(np.isinf(tensors))
