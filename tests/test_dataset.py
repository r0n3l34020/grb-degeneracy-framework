import numpy as np
import torch
from torch.utils.data import DataLoader
from src.ml.dataset import (
    GRBMultimodalDataset,
    N_CHANNELS,
    N_BINS,
    add_gaussian_noise,
    add_poisson_noise,
    inject_noise,
)


def test_dataset_batch_shape():
    dataset = GRBMultimodalDataset(num_samples=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch, labels = next(iter(loader))
    assert batch.shape == (4, N_CHANNELS, N_BINS), (
        f"Error, expected shape (4, {N_CHANNELS}, {N_BINS}) got {tuple(batch.shape)}"
    )
    assert not torch.any(torch.isnan(batch)), "Error, batch contains NaN values"
    assert not torch.any(torch.isinf(batch)), "Error, batch contains infinite values"


def test_gaussian_noise_std_matches_snr():
    rng = np.random.default_rng(seed=42)
    signal = np.full(20000, 100.0)  # large + constant, for stable statistics

    for snr in (2.0, 10.0, 50.0):
        noisy, theoretical_std = add_gaussian_noise(signal, snr, rng=rng)
        measured_std = np.std(noisy - signal)
        assert np.isclose(measured_std, theoretical_std, rtol=0.05), (
            f"SNR={snr}: expected noise std ~{theoretical_std}, measured {measured_std}"
        )


def test_gaussian_noise_decreases_as_snr_increases():
    rng = np.random.default_rng(seed=42)
    signal = np.full(20000, 100.0)

    noisy_low_snr, _ = add_gaussian_noise(signal, snr=2.0, rng=rng)
    noisy_high_snr, _ = add_gaussian_noise(signal, snr=50.0, rng=rng)

    std_low_snr = np.std(noisy_low_snr - signal)
    std_high_snr = np.std(noisy_high_snr - signal)
    assert std_low_snr > std_high_snr, "Lower SNR should produce larger noise amplitude"


def test_poisson_noise_variance_matches_mean():
    rng = np.random.default_rng(seed=42)
    signal = np.full(20000, 50.0)  # Poisson property: variance == mean (rate)

    noisy = add_poisson_noise(signal, rng=rng)
    assert np.isclose(np.var(noisy), 50.0, rtol=0.1)


def test_inject_noise_preserves_shape_and_is_finite():
    rng = np.random.default_rng(seed=42)
    signal = np.abs(rng.normal(loc=100.0, scale=10.0, size=(500,)))

    noisy = inject_noise(signal, snr=10.0, rng=rng)
    assert noisy.shape == signal.shape
    assert not np.any(np.isnan(noisy)), "Error, noisy signal contains NaN values"
    assert not np.any(np.isinf(noisy)), "Error, noisy signal contains infinite values"
