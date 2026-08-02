import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless-safe backend; no display required
import matplotlib.pyplot as plt

# === Day 3 ===
# Overlay/comparison heatmaps for Standard vs LIV vs ALP spectral-temporal profiles.


def plot_spectral_temporal_heatmap(matrix_2d, energy_grid, time_grid, ax=None, title=None, cmap="inferno"):
    """Single 2D flux heatmap: log10(energy) on Y, time on X."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    mesh = ax.pcolormesh(time_grid, np.log10(energy_grid), matrix_2d, shading="auto", cmap=cmap)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\log_{10}$ Energy (eV)")
    if title:
        ax.set_title(title)
    return ax, mesh


def plot_model_comparison(standard_matrix, liv_matrix, alp_matrix, energy_grid, time_grid, save_path=None):
    """Side-by-side spectral-temporal heatmaps for the three competing physics models, shared color scale."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    models = [
        ("Standard (Synchrotron)", standard_matrix),
        ("LIV", liv_matrix),
        ("ALP-conversion", alp_matrix),
    ]
    vmax = max(np.max(m) for _, m in models)
    mesh = None
    for ax, (title, matrix) in zip(axes, models):
        _, mesh = plot_spectral_temporal_heatmap(matrix, energy_grid, time_grid, ax=ax, title=title)
        mesh.set_clim(0, vmax)
    fig.colorbar(mesh, ax=axes, label="Flux Intensity", shrink=0.85)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_overlay_contours(standard_matrix, liv_matrix, alp_matrix, energy_grid, time_grid, save_path=None):
    """Superimpose contour lines of all three models on one axis — where lines coincide, the models are degenerate."""
    fig, ax = plt.subplots(figsize=(7, 5))
    models = [
        ("Standard", standard_matrix, "cyan"),
        ("LIV", liv_matrix, "magenta"),
        ("ALP-conversion", alp_matrix, "yellow"),
    ]
    log_energy = np.log10(energy_grid)
    for label, matrix, color in models:
        levels = np.linspace(matrix.max() * 0.2, matrix.max() * 0.9, 4)
        ax.contour(time_grid, log_energy, matrix, levels=levels, colors=color, linewidths=1.2)
        ax.plot([], [], color=color, label=label)  # proxy artist so contours show in the legend
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(r"$\log_{10}$ Energy (eV)")
    ax.set_title("Standard vs LIV vs ALP — Spectral-Temporal Overlay")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


if __name__ == "__main__":
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    data_dir = repo_root / "data" / "synthetic" / "test_datasets"

    standard_matrix = np.load(sorted((data_dir / "ssc").glob("*.npy"))[0])
    liv_matrix = np.load(sorted((data_dir / "liv").glob("*.npy"))[0])
    alp_matrix = np.load(sorted((data_dir / "alp").glob("*.npy"))[0])

    energy_grid = np.logspace(0, 6, standard_matrix.shape[0])
    time_grid = np.linspace(0, 50, standard_matrix.shape[1])

    out_dir = repo_root / "data" / "synthetic" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_model_comparison(standard_matrix, liv_matrix, alp_matrix, energy_grid, time_grid,
                           save_path=out_dir / "day3_model_comparison.png")
    plot_overlay_contours(standard_matrix, liv_matrix, alp_matrix, energy_grid, time_grid,
                           save_path=out_dir / "day3_overlay_contours.png")
    print("Saved comparison and overlay figures to", out_dir)
