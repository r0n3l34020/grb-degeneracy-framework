from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class MultimodalGRBEncoder(nn.Module):

  def __init__(self, in_channels: int = 3, feature_dim: int = 128):
    super().__init__()
    self.conv_blocks = nn.Sequential(
        nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm2d(32),
        nn.SiLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.SiLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.SiLU(),
        nn.AdaptiveAvgPool2d((2, 2)),
    )
    self.fc = nn.Linear(128 * 2 * 2, feature_dim)

  def forward(self, x):
    if x.dim() == 3:
      x = x.unsqueeze(1)
    return F.silu(self.fc(torch.flatten(self.conv_blocks(x), 1)))


class DegeneracyMDNHead(nn.Module):

  def __init__(self, feature_dim: int = 128, param_dim: int = 2):
    super().__init__()
    self.param_dim = param_dim
    self.mu_head = nn.Linear(feature_dim, param_dim)
    self.cholesky_head = nn.Linear(
        feature_dim, (param_dim * (param_dim + 1)) // 2
    )

  def forward(self, features):
    mu = self.mu_head(features)
    cholesky_raw = self.cholesky_head(features)
    batch_size = features.size(0)
    L = torch.zeros(
        batch_size, self.param_dim, self.param_dim, device=features.device
    )
    tril_indices = torch.tril_indices(row=self.param_dim, col=self.param_dim)
    L[:, tril_indices[0], tril_indices[1]] = cholesky_raw
    diag_mask = torch.eye(self.param_dim, device=features.device).bool()
    L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-5
    return mu, torch.bmm(L, L.transpose(1, 2))


def plot_ellipse(mu, sigma, ax, label, color):
  vals, vecs = np.linalg.eigh(sigma)
  order = vals.argsort()[::-1]
  vals, vecs = vals[order], vecs[:, order]
  theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
  width, height = 2 * 2 * np.sqrt(np.maximum(vals, 1e-9))
  ell = Ellipse(
      xy=mu,
      width=width,
      height=height,
      angle=theta,
      edgecolor=color,
      fc="none",
      lw=2,
      label=label,
  )
  ax.add_patch(ell)
  ax.scatter(mu[0], mu[1], color=color, s=40)


def infer_grb221009a():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  h5_path = Path("data/processed/target_event.h5")
  ckpt_path = Path("checkpoints/grb_model_best.pt")

  print("\n[INFO] Initialising GRB 221009A empirical target inference pipeline...")

  if not h5_path.exists():
    raise FileNotFoundError(
        f"Target tensor file not found at {h5_path}. Please run 'python -m"
        " scripts.00_preprocess_raw' first, Sir."
    )

  if not ckpt_path.exists():
    raise FileNotFoundError(
        f"Model checkpoint not found at {ckpt_path}. Please verify training"
        " state, Sir."
    )

  # Extract empirical observational tensor
  with h5py.File(h5_path, "r") as f:
    raw_tensor = f["tensors"][0]  # Shape: (3, 128, 128)

  # Validate telemetry signals across channels
  ch_std = np.std(raw_tensor, axis=(1, 2), keepdims=True)
  ch_std = np.where(ch_std == 0, 1.0, ch_std)
  norm_tensor = raw_tensor / ch_std

  grb_tensor = (
      torch.from_numpy(norm_tensor).float().unsqueeze(0).to(device)
  )  # (1, 3, 128, 128)

  # Instantiate networks
  multi_enc = MultimodalGRBEncoder(in_channels=3).to(device)
  multi_head = DegeneracyMDNHead().to(device)
  spec_enc = MultimodalGRBEncoder(in_channels=1).to(device)
  spec_head = DegeneracyMDNHead().to(device)

  # Load checkpoint weights
  checkpoint = torch.load(ckpt_path, map_location=device)
  multi_enc.load_state_dict(checkpoint["multi_encoder_state"])
  multi_head.load_state_dict(checkpoint["multi_head_state"])
  spec_enc.load_state_dict(checkpoint["spec_encoder_state"])
  spec_head.load_state_dict(checkpoint["spec_head_state"])

  multi_enc.eval()
  multi_head.eval()
  spec_enc.eval()
  spec_head.eval()

  # Inference pass
  with torch.no_grad():
    for _ in tqdm(
        range(1), desc="Evaluating Empirical Telemetry", unit="pass"
    ):
      mu_m, sig_m = multi_head(multi_enc(grb_tensor))
      mu_s, sig_s = spec_head(spec_enc(grb_tensor[:, 0:1, :, :]))

  mu_m_np, sig_m_np = mu_m[0].cpu().numpy(), sig_m[0].cpu().numpy()
  mu_s_np, sig_s_np = mu_s[0].cpu().numpy(), sig_s[0].cpu().numpy()

  print(
      "\n================ GRB 221009A EMPIRICAL TARGET INFERENCE"
      " ================"
  )
  print(
      f"Spectral-Only (Fermi-GBM)          : μ = [{mu_s_np[0]:.4f},"
      f" {mu_s_np[1]:.4f}] | σ_E = {np.sqrt(sig_s_np[0,0]):.5f}, σ_g ="
      f" {np.sqrt(sig_s_np[1,1]):.5f}"
  )
  print(
      f"Multimodal (Fermi + LHAASO + Swift): μ = [{mu_m_np[0]:.4f},"
      f" {mu_m_np[1]:.4f}] | σ_E = {np.sqrt(sig_m_np[0,0]):.5f}, σ_g ="
      f" {np.sqrt(sig_m_np[1,1]):.5f}"
  )
  print(
      "=========================================================================\n"
  )

  # Plotting
  fig, ax = plt.subplots(figsize=(8, 6))
  plot_ellipse(
      mu_s_np,
      sig_s_np,
      ax,
      "Spectral-Only Posterior (Fermi-GBM)",
      "crimson",
  )
  plot_ellipse(
      mu_m_np,
      sig_m_np,
      ax,
      "Multimodal Posterior (Fermi + LHAASO + Swift)",
      "teal",
  )

  ax.set_title("Empirical Posterior Contours: GRB 221009A", fontsize=14)
  ax.set_xlabel(r"$\log_{10}(E_{\mathrm{break}}\ /\ \mathrm{MeV})$", fontsize=12)
  ax.set_ylabel(r"$\log_{10}(g_{a\gamma}\ /\ \mathrm{GeV}^{-1})$", fontsize=12)
  ax.legend(loc="upper right")
  ax.grid(True, linestyle="--", alpha=0.5)

  fig_path = Path("reports/figures/grb221009a_empirical_posterior.png")
  fig_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(fig_path, dpi=300, bbox_inches="tight")
  print(
      f"[SUCCESS] Saved GRB 221009A empirical posterior plot to {fig_path},"
      " Sir."
  )


if __name__ == "__main__":
  infer_grb221009a()
