from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d


def map_landscape():
  data_path = Path("reports/population_eval_data.npz")
  if not data_path.exists():
    raise FileNotFoundError(
        "Run 01_evaluate_population.py first to generate telemetry data"
    )

  data = np.load(data_path)
  delta_i = data["delta_i"]
  params = data["params"]

  e_break = params[:, 0]
  g_a_gamma = params[:, 1]

  stat, x_edges, y_edges, _ = binned_statistic_2d(
      e_break, g_a_gamma, delta_i, statistic="mean", bins=60
  )

  stat_filled = np.nan_to_num(stat, nan=np.nanmean(delta_i))
  smoothed_stat = gaussian_filter(stat_filled, sigma=1.2)

  fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
  extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]

  im = ax.imshow(
      smoothed_stat.T,
      origin="lower",
      extent=extent,
      cmap="magma",
      aspect="auto",
      interpolation="bicubic",
  )

  contours = ax.contour(
      smoothed_stat.T,
      levels=7,
      extent=extent,
      colors="white",
      alpha=0.3,
      linewidths=0.7,
  )
  ax.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

  cbar = fig.colorbar(im, ax=ax, pad=0.03)
  cbar.set_label(
      r"Information Gain $\Delta I$ (nats)",
      fontsize=12,
      rotation=270,
      labelpad=16,
  )

  ax.set_title(
      "Global GRB Parameter Degeneracy Landscape",
      fontsize=14,
      fontweight="bold",
      pad=15,
  )
  ax.set_xlabel(
      r"Break Energy $\log_{10}(E_{\mathrm{break}} / \mathrm{keV})$",
      fontsize=12,
      labelpad=8,
  )
  ax.set_ylabel(
      r"Axion-Photon Coupling $\log_{10}(g_{a\gamma} / \mathrm{GeV}^{-1})$",
      fontsize=12,
      labelpad=8,
  )

  ax.grid(True, linestyle="--", alpha=0.25, color="gray")
  ax.spines["top"].set_visible(False)
  ax.spines["right"].set_visible(False)

  fig_dir = Path("reports/figures")
  fig_dir.mkdir(parents=True, exist_ok=True)
  out_path = fig_dir / "global_degeneracy_landscape.png"
  plt.savefig(out_path, dpi=300, bbox_inches="tight")
  print(
      "[SUCCESS] Saved publication-grade global degeneracy landscape to"
      f" {out_path}"
  )


if __name__ == "__main__":
  map_landscape()
