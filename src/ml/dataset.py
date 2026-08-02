import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# === Day 1 ===
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


# === Day 2 ===
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


# === Day 6 ===
# --- Data validation --------------------------------------------------------
# Edge-case physics parameters (e.g. near-zero t_decay, extreme E_break) can drive
# the forward model into NaN/Inf territory. Catch that here before it silently
# corrupts a serialized dataset or a training run.


def validate_tensor(tensor: np.ndarray, name: str = "tensor") -> None:
    if np.any(np.isnan(tensor)):
        raise ValueError(f"Error, {name} contains NaN values")
    if np.any(np.isinf(tensor)):
        raise ValueError(f"Error, {name} contains infinite values")


# --- Dataset serialization ---------------------------------------------------
# Consolidates many individual events into one file for high-speed loading,
# instead of opening thousands of tiny .npy files at train time.


def save_dataset_hdf5(tensors: np.ndarray, labels: list, path) -> Path:
    """tensors: (N, 3, Energy_Bins, Time_Bins) array. labels: list[str] of length N (physics_model per event)."""
    validate_tensor(tensors, name="dataset tensor stack")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    label_map = {name: i for i, name in enumerate(sorted(set(labels)))}
    label_ids = np.array([label_map[label] for label in labels], dtype=np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("tensors", data=tensors, compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=label_ids)
        f.attrs["label_map"] = json.dumps(label_map)
    return path


def load_dataset_hdf5(path):
    with h5py.File(path, "r") as f:
        tensors = f["tensors"][:]
        label_ids = f["labels"][:]
        label_map = json.loads(f.attrs["label_map"])
    return tensors, label_ids, label_map


class HDF5GRBDataset(Dataset):
    """High-speed loader for a consolidated HDF5 event file (Day 6)."""

    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.length = f["tensors"].shape[0]
            self.label_map = json.loads(f.attrs["label_map"])
        self._file = None  # opened lazily so each DataLoader worker gets its own handle

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        tensor = torch.from_numpy(self._file["tensors"][idx])
        label = torch.tensor(int(self._file["labels"][idx]), dtype=torch.long)
        return tensor, label


def save_dataset_pt(tensors: np.ndarray, labels: list, path) -> Path:
    """Alternative to HDF5: a single torch .pt file, fully loaded into memory on read."""
    validate_tensor(tensors, name="dataset tensor stack")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    label_map = {name: i for i, name in enumerate(sorted(set(labels)))}
    label_ids = torch.tensor([label_map[label] for label in labels], dtype=torch.long)

    torch.save({"tensors": torch.from_numpy(tensors), "labels": label_ids, "label_map": label_map}, path)
    return path


def load_dataset_pt(path):
    payload = torch.load(path, weights_only=False)
    return payload["tensors"], payload["labels"], payload["label_map"]


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Day 1 demo
    dataset = GRBMultimodalDataset(num_samples=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch, labels = next(iter(loader))
    print("Batch tensor shape:", tuple(batch.shape))
    assert batch.shape == (4, N_CHANNELS, N_BINS), (
        f"Error, expected shape (4, {N_CHANNELS}, {N_BINS}) got {tuple(batch.shape)}"
    )
    print("Day 1 dataset stub OK.")

    # Day 2 demo
    signal = np.full(1000, 100.0)
    for snr in (2.0, 10.0, 50.0):
        noisy, noise_std = add_gaussian_noise(signal, snr)
        measured_std = np.std(noisy - signal)
        print(f"SNR={snr:>5.1f}  theoretical_noise_std={noise_std:.3f}  measured_noise_std={measured_std:.3f}")
