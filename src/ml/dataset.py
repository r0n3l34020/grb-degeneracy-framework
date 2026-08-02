import numpy as np
import torch
from torch.utils.data import Dataset

# Mock multimodal tensor schema.
# Each sample is a (3, N_BINS) tensor: 3 channels = [Spectrum, LightCurve, Polarization],
# each represented as a length-N_BINS 1D vector for now. This is a Day 1 stub only —
# real physics-derived data and the full (3, Energy_Bins, Time_Bins) shape land on Day 4.
N_CHANNELS = 3
N_BINS = 100


class GRBMultimodalDataset(Dataset):
    def __init__(self, num_samples: int = 1000, n_bins: int = N_BINS):
        self.num_samples = num_samples
        self.n_bins = n_bins

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Placeholder mock data until the physics generator (Day 4) is wired in.
        sample = torch.randn(N_CHANNELS, self.n_bins)
        label = torch.tensor(0, dtype=torch.long)
        return sample, label


# --- Synthetic observational noise injector -------------------------------
# Operates on numpy arrays (matches the physics simulation outputs in src/physics/*),
# not torch tensors — conversion to tensors happens later in the pipeline (Day 4).


def add_poisson_noise(signal: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
    """Photon counting (shot) noise: each bin is resampled from Poisson(rate=signal)."""
    rng = rng or np.random.default_rng()
    rate = np.clip(signal, a_min=0.0, a_max=None)  # Poisson rate must be non-negative
    return rng.poisson(rate).astype(float)


def add_gaussian_noise(signal: np.ndarray, snr: float, rng: np.random.Generator = None):
    """Additive Gaussian background noise, scaled so the RMS signal-to-noise ratio equals `snr`."""
    rng = rng or np.random.default_rng()
    signal_rms = np.sqrt(np.mean(signal ** 2))
    noise_std = signal_rms / snr
    noise = rng.normal(loc=0.0, scale=noise_std, size=signal.shape)
    return signal + noise, noise_std


def inject_noise(signal: np.ndarray, snr: float, include_poisson: bool = True,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Apply Poisson shot noise followed by Gaussian background noise at the given SNR."""
    rng = rng or np.random.default_rng()
    noisy = add_poisson_noise(signal, rng=rng) if include_poisson else signal.copy()
    noisy, _ = add_gaussian_noise(noisy, snr, rng=rng)
    return noisy


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = GRBMultimodalDataset(num_samples=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch, labels = next(iter(loader))
    print("Batch tensor shape:", tuple(batch.shape))
    assert batch.shape == (4, N_CHANNELS, N_BINS), (
        f"Error, expected shape (4, {N_CHANNELS}, {N_BINS}) got {tuple(batch.shape)}"
    )
    print("Day 1 dataset stub OK.")

    signal = np.full(1000, 100.0)
    for snr in (2.0, 10.0, 50.0):
        noisy, noise_std = add_gaussian_noise(signal, snr)
        measured_std = np.std(noisy - signal)
        print(f"SNR={snr:>5.1f}  theoretical_noise_std={noise_std:.3f}  measured_noise_std={measured_std:.3f}")
