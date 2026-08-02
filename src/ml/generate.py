import sys
import json
from pathlib import Path

import numpy as np

# synchrotron.py does `from liv import ...` / `from alp import ...` (bare, not package-qualified),
# which only resolves if src/physics/ itself is on sys.path. We don't touch that file, so we shim
# sys.path here instead — this only affects imports inside this process, nothing on disk.
_PHYSICS_DIR = Path(__file__).resolve().parents[2] / "src" / "physics"
if str(_PHYSICS_DIR) not in sys.path:
    sys.path.insert(0, str(_PHYSICS_DIR))

from src.physics.synchrotron import simulate_grb_emission, fred_light_curve
from src.physics.liv import apply_liv_to_spectrum
from src.physics.alp import apply_alp_to_spectrum, photon_survival_probability
from src.ml.dataset import inject_noise, validate_tensor, save_dataset_hdf5

PHYSICS_MODELS = ("standard", "liv", "alp")
ENERGY_BINS = 100
TIME_BINS = 100
DEFAULT_SNR = 15.0


def synthetic_polarization_map(physics_model: str, energy_grid: np.ndarray, time_grid: np.ndarray,
                                exotic_params: dict) -> np.ndarray:
    """
    Physically-motivated PLACEHOLDER polarization-degree map (values in [0, 1]).
    Real GRB 221009A polarimetry doesn't exist for this event, so this channel is synthetic —
    but it's still derived from each model's own physics rather than being arbitrary noise:
      - standard: a flat baseline degree, typical of a tangled-field synchrotron shock.
      - liv: same baseline as standard — in this simplified treatment, LIV shifts photon
        arrival *time*, not polarization state. Documented limitation, not an oversight.
      - alp: derived from the ALP photon-survival probability itself, since ALP-photon
        mixing is fundamentally a polarization-basis effect — its energy-dependent
        oscillation is reused here as the polarization signature.
    """
    baseline_degree = 0.15
    if physics_model in ("standard", "liv"):
        return np.full((len(energy_grid), len(time_grid)), baseline_degree)

    survival = photon_survival_probability(
        energy_grid, exotic_params["g_ag"], exotic_params["B"], exotic_params["L"], f_E_func=1.0
    )
    degree = np.clip(baseline_degree + 0.5 * (1.0 - survival), 0.0, 1.0)
    return np.tile(degree[:, None], (1, len(time_grid)))


def generate_grb_event(physics_model: str = "standard", snr: float = DEFAULT_SNR,
                        energy_bins: int = ENERGY_BINS, time_bins: int = TIME_BINS,
                        rng: np.random.Generator = None):
    """
    Forward-simulate one synthetic GRB event and package it as a unified
    (3, Energy_Bins, Time_Bins) tensor: [Spectral-temporal flux, Temporal light-curve
    reference, Polarimetric degree map]. Returns (tensor, metadata).
    """
    rng = rng or np.random.default_rng()
    if physics_model not in PHYSICS_MODELS:
        raise ValueError(f"physics_model must be one of {PHYSICS_MODELS}, got {physics_model!r}")

    p = rng.uniform(1.5, 3.0)
    alpha = (p - 1) / 2
    beta = alpha + 0.5
    E_break = 10.0 ** rng.uniform(0.0, 6.0)
    f_break = 1.0
    t_rise = rng.uniform(1.0, 5.0)
    t_decay = rng.uniform(5.0, 30.0)
    A = 5.0

    energy_grid = np.logspace(0, 6, energy_bins)
    time_grid = np.linspace(0, 50, time_bins)

    base_matrix = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)

    exotic_params = {}
    if physics_model == "standard":
        flux_matrix = base_matrix
    elif physics_model == "liv":
        eqg = 10.0 ** rng.uniform(17.0, 19.0)
        flux_matrix = apply_liv_to_spectrum(time_grid, energy_grid, eqg, base_matrix)
        exotic_params["eqg"] = eqg
    else:  # alp
        g_ag = rng.uniform(1e-10, 1e-9)
        B = rng.uniform(1e-6, 1e-5)
        L = rng.uniform(1000.0, 10000.0)
        survival = photon_survival_probability(energy_grid, g_ag, B, L, f_E_func=1.0)
        flux_matrix = apply_alp_to_spectrum(base_matrix, survival)
        exotic_params.update(g_ag=g_ag, B=B, L=L)

    light_curve = fred_light_curve(time_grid, t_decay, t_rise, A)
    temporal_channel = np.tile(light_curve, (energy_bins, 1))
    polarization_channel = synthetic_polarization_map(physics_model, energy_grid, time_grid, exotic_params)

    noisy_flux = inject_noise(flux_matrix, snr=snr, rng=rng)
    noisy_temporal = inject_noise(temporal_channel, snr=snr, rng=rng)
    noisy_polarization = np.clip(
        polarization_channel + rng.normal(0.0, 0.1 / snr, polarization_channel.shape), 0.0, 1.0
    )

    tensor = np.stack([noisy_flux, noisy_temporal, noisy_polarization], axis=0).astype(np.float32)
    metadata = {
        "physics_model": physics_model, "snr": snr, "p": float(p), "E_break": float(E_break),
        "t_rise": float(t_rise), "t_decay": float(t_decay),
        **{k: float(v) for k, v in exotic_params.items()},
    }
    return tensor, metadata


def generate_dataset(num_samples: int = 1000, snr: float = DEFAULT_SNR,
                      energy_bins: int = ENERGY_BINS, time_bins: int = TIME_BINS,
                      out_path: Path = None, seed: int = 0):
    """
    Generate `num_samples` events (physics model chosen uniformly at random) and save
    them as one consolidated HDF5 file (Day 6) for high-speed loading, instead of
    thousands of individual .npy files.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(out_path) if out_path else (
        Path(__file__).resolve().parents[2] / "data" / "synthetic" / "events.h5"
    )

    tensors = np.empty((num_samples, 3, energy_bins, time_bins), dtype=np.float32)
    labels = []
    all_metadata = []
    for i in range(num_samples):
        physics_model = str(rng.choice(PHYSICS_MODELS))
        tensor, metadata = generate_grb_event(physics_model, snr=snr, energy_bins=energy_bins,
                                               time_bins=time_bins, rng=rng)
        validate_tensor(tensor, name=f"event {i} ({physics_model})")

        tensors[i] = tensor
        labels.append(physics_model)
        all_metadata.append(metadata)

    save_dataset_hdf5(tensors, labels, out_path)

    metadata_path = out_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(all_metadata, f, indent=2)

    return out_path, all_metadata


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    out_path, metadata = generate_dataset(num_samples=1000)
    elapsed = time.perf_counter() - start

    print(f"Generated {len(metadata)} events in {elapsed:.2f}s ({elapsed / len(metadata) * 1000:.2f} ms/event)")
    print(f"Saved consolidated dataset to {out_path}")
    counts = {m: sum(1 for x in metadata if x["physics_model"] == m) for m in PHYSICS_MODELS}
    print("Class balance:", counts)
