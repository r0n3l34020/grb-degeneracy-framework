import json
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


class HDF5GRBDataset(Dataset):

  def __init__(self, path: Path, normalize_channels: bool = True):
    self.path = Path(path)
    self.normalize_channels = normalize_channels
    with h5py.File(self.path, "r") as f:
      self.length = f["tensors"].shape[0]
      if "labels" in f:
        self.param_key = "labels"
      elif "parameters" in f:
        self.param_key = "parameters"
      elif "params" in f:
        self.param_key = "params"
      else:
        raise KeyError(
            "None of 'labels', 'parameters', or 'params' found in HDF5 keys:"
            f" {list(f.keys())}"
        )

      if self.normalize_channels:
        sample_tensors = f["tensors"][: min(500, self.length)]
        self.ch_stds = np.std(sample_tensors, axis=(0, 2, 3), keepdims=True)
        self.ch_stds = np.where(self.ch_stds == 0, 1.0, self.ch_stds)
      else:
        self.ch_stds = None

    self._file = None

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if self._file is None:
      self._file = h5py.File(self.path, "r")
    raw_tensor = self._file["tensors"][idx]

    if self.normalize_channels and self.ch_stds is not None:
      raw_tensor = raw_tensor / self.ch_stds[0]

    tensor = torch.from_numpy(raw_tensor).float()
    raw_val = self._file[self.param_key][idx]

    if np.isscalar(raw_val) or np.ndim(raw_val) == 0:
      val_float = float(raw_val)
      transformed = np.array([val_float, val_float], dtype=np.float32)
    else:
      arr = np.array(raw_val, dtype=np.float32)
      transformed = np.sign(arr) * np.log10(np.abs(arr) + 1e-12)
      if len(transformed) == 1:
        transformed = np.repeat(transformed, 2)
      else:
        transformed = transformed[:2]

    params = torch.from_numpy(transformed).float()
    return tensor, params


class MultimodalGRBEncoder(nn.Module):

  def __init__(self, in_channels: int = 3, feature_dim: int = 128):
    super().__init__()
    self.in_channels = in_channels
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


def run_ablation():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  h5_path = Path("data/synthetic/events_suite.h5")

  dataset = HDF5GRBDataset(h5_path, normalize_channels=True)
  train_size = int(0.8 * len(dataset))
  _, test_ds = random_split(
      dataset,
      [train_size, len(dataset) - train_size],
      generator=torch.Generator().manual_seed(42),
  )
  test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

  multi_enc = MultimodalGRBEncoder(in_channels=3).to(device)
  multi_head = DegeneracyMDNHead().to(device)
  checkpoint = torch.load(
      "checkpoints/grb_model_best.pt", map_location=device
  )
  multi_enc.load_state_dict(checkpoint["multi_encoder_state"])
  multi_head.load_state_dict(checkpoint["multi_head_state"])
  multi_enc.eval()
  multi_head.eval()

  ablation_masks = {
      "Prompt Spectral Only": torch.tensor([1.0, 0.0, 0.0], device=device),
      "Spectral + Afterglow": torch.tensor([1.0, 1.0, 0.0], device=device),
      "Full Multimodal": torch.tensor([1.0, 1.0, 1.0], device=device),
  }

  results = {}

  with torch.no_grad():
    for name, mask in ablation_masks.items():
      traces = []
      for tensors, _ in tqdm(
          test_loader, desc=f"Ablating {name}", leave=False, unit="batch"
      ):
        tensors = tensors.to(device)
        masked_tensors = tensors * mask.view(1, 3, 1, 1)
        _, sigma = multi_head(multi_enc(masked_tensors))
        batch_trace = (
            torch.diagonal(sigma, dim1=1, dim2=2).sum(dim=-1).cpu().numpy()
        )
        traces.extend(batch_trace)
      results[name] = float(np.mean(traces))

  baseline_trace = results["Prompt Spectral Only"]
  rel_gains = {
      k: ((baseline_trace - v) / baseline_trace) * 100
      for k, v in results.items()
  }

  print("\n================ MODALITY ABLATION RESULTS ================")
  print(
      f"{'Modality Configuration':<25} | {'Mean Trace Tr(Σ)':<18} | {'Rel. Variance Gain':<20}"
  )
  print("-" * 68)
  for config, trace_val in results.items():
    rel = rel_gains[config]
    print(f"{config:<25} | {trace_val:<18.8f} | {rel:>+6.4f}%")
  print("==========================================================")

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

  configs = list(results.keys())
  traces = list(results.values())
  gains = list(rel_gains.values())

  min_tr, max_tr = min(traces), max(traces)
  margin = (max_tr - min_tr) * 0.25 if max_tr != min_tr else min_tr * 0.01

  ax1.bar(configs, traces, color=["#d9534f", "#f0ad4e", "#5cb85c"], width=0.45)
  ax1.set_ylabel(r"Mean Posterior Variance $\mathrm{Tr}(\Sigma)$", fontsize=11)
  ax1.set_title("Absolute Posterior Uncertainty", fontsize=12)
  ax1.set_ylim(min_tr - margin, max_tr + margin)
  ax1.grid(axis="y", linestyle="--", alpha=0.7)
  ax1.tick_params(axis="x", rotation=10)

  ax2.bar(configs, gains, color=["#777777", "#f0ad4e", "#5cb85c"], width=0.45)
  ax2.set_ylabel("Variance Reduction vs. Prompt-Only (%)", fontsize=11)
  ax2.set_title("Relative Observational Gain", fontsize=12)
  ax2.grid(axis="y", linestyle="--", alpha=0.7)
  ax2.tick_params(axis="x", rotation=10)

  plt.tight_layout()
  fig_path = Path("reports/figures/ablation_sufficiency.png")
  fig_path.parent.mkdir(parents=True, exist_ok=True)
  plt.savefig(fig_path, dpi=300, bbox_inches="tight")
  print(f"[SUCCESS] Saved enhanced ablation plot to {fig_path}")


if __name__ == "__main__":
  run_ablation()
