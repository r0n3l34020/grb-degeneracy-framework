import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Same sys.path shim as src/ml/generate.py — synchrotron.py's own `from liv import ...` only
# resolves if src/physics/ is on sys.path. We don't touch that file, so we work around it here.
_PHYSICS_DIR = Path(__file__).resolve().parents[2] / "src" / "physics"
if str(_PHYSICS_DIR) not in sys.path:
    sys.path.insert(0, str(_PHYSICS_DIR))

from src.physics.synchrotron import simulate_grb_emission
from src.physics.liv import apply_liv_to_spectrum
from src.ml.dataset import add_gaussian_noise
from src.ml.generate import ENERGY_BINS, TIME_BINS

# === Day 7 ===
# Fisher Information Matrix inversion for parameter variance estimates (Cramer-Rao bound)
# and 2D error-ellipse plotting for parameter-constraint slices.


def numerical_jacobian(model_func, theta: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Central finite-difference Jacobian of model_func(theta) -> 1D array. Shape (n_outputs, n_params)."""
    theta = np.asarray(theta, dtype=float)
    baseline = model_func(theta)
    jac = np.zeros((baseline.size, theta.size))
    for i in range(theta.size):
        step = np.zeros_like(theta)
        step[i] = eps
        plus = model_func(theta + step)
        minus = model_func(theta - step)
        jac[:, i] = (plus - minus).ravel() / (2 * eps)
    return jac


def compute_fisher_matrix(model_func, theta: np.ndarray, sigma, eps: float = 1e-3) -> np.ndarray:
    """
    Gaussian-noise Fisher Information Matrix: F_ij = sum_k (1/sigma_k^2) * dM_k/dtheta_i * dM_k/dtheta_j.
    `sigma` may be a scalar (homogeneous noise) or an array matching the model output size.
    """
    jac = numerical_jacobian(model_func, theta, eps=eps)
    sigma_arr = np.asarray(sigma, dtype=float)
    weight = np.full(jac.shape[0], 1.0 / sigma_arr ** 2) if sigma_arr.size == 1 else 1.0 / sigma_arr.ravel() ** 2
    return jac.T @ (jac * weight[:, None])


def invert_fisher_matrix(fisher_matrix: np.ndarray):
    """Cramér-Rao bound: covariance = F^-1, so sigma^2(theta_i) >= (F^-1)_ii."""
    covariance = np.linalg.inv(fisher_matrix)
    variances = np.diag(covariance)
    return covariance, variances


def plot_error_ellipse(covariance_2x2: np.ndarray, center, ax=None, n_std: float = 1.0, label=None, **kwargs):
    """Draw the n_std-sigma confidence ellipse implied by a 2x2 covariance submatrix."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_2x2)
    order = eigenvalues.argsort()[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    width, height = 2 * n_std * np.sqrt(np.clip(eigenvalues, 0, None))

    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle, fill=False, label=label, **kwargs)
    ax.add_patch(ellipse)
    ax.plot(*center, marker="+", color=ellipse.get_edgecolor())
    ax.autoscale_view()
    return ax, ellipse


def liv_spectral_index_fisher_ellipse(p0: float = 2.0, log10_eqg0: float = -13.5, snr: float = 15.0,
                                       save_path=None):
    """
    2-parameter Fisher analysis for theta = [spectral index p, log10(E_QG)] (the LIV energy scale),
    the example slice named in the Day 7 plan. Demonstrates sigma^2(theta_i) >= (F^-1)_ii and plots
    the resulting 1-sigma / 2-sigma error ellipse.

    log10_eqg0 defaults to -13.5 (matching src/ml/generate.py's LIV eqg range), not a physical
    Planck-scale value — see the NOTE in generate.py on the grid-resolution/units mismatch that
    makes eqg~1e17-1e19 produce a zero-derivative (unresolvable) effect at this grid resolution.
    """
    E_break, f_break, t_rise, t_decay, A = 1e3, 1.0, 2.0, 15.0, 5.0
    energy_grid = np.logspace(0, 6, ENERGY_BINS)
    time_grid = np.linspace(0, 50, TIME_BINS)

    def model_func(theta):
        p, log10_eqg = theta
        alpha = (p - 1) / 2
        beta = alpha + 0.5
        base = simulate_grb_emission(energy_grid, time_grid, E_break, alpha, beta, f_break, t_decay, t_rise, A)
        shifted = apply_liv_to_spectrum(time_grid, energy_grid, 10.0 ** log10_eqg, base)
        return shifted.ravel()

    theta0 = np.array([p0, log10_eqg0])
    fiducial = model_func(theta0)
    _, sigma = add_gaussian_noise(fiducial, snr=snr)  # homogeneous noise level from the fiducial signal

    fisher_matrix = compute_fisher_matrix(model_func, theta0, sigma, eps=1e-3)
    covariance, variances = invert_fisher_matrix(fisher_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    for n_std in (1, 2):
        plot_error_ellipse(covariance, theta0, ax=ax, n_std=n_std, label=f"{n_std}$\\sigma$",
                            edgecolor="tab:blue", linewidth=2.0 / n_std)
    ax.set_xlabel("Spectral index p")
    ax.set_ylabel(r"$\log_{10}(E_{QG})$")
    ax.set_title(f"Fisher-matrix constraint: p vs LIV energy scale (SNR={snr})")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return {
        "theta0": theta0, "fisher_matrix": fisher_matrix, "covariance": covariance,
        "variances": variances, "sigma_theta": np.sqrt(variances), "figure": fig,
    }


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    result = liv_spectral_index_fisher_ellipse(save_path=out_dir / "day7_fisher_ellipse.png")
    print("theta0 (p, log10 E_QG):", result["theta0"])
    print("Fisher matrix:\n", result["fisher_matrix"])
    print("1-sigma uncertainties:", result["sigma_theta"])
