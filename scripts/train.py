import json
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class HDF5GRBDataset(Dataset):
    def __init__(self, path: Path, log_transform_params: bool = True):
        self.path = Path(path)
        self.log_transform = log_transform_params
        
        if not self.path.exists():
            raise FileNotFoundError(f"Target HDF5 dataset not found at: {self.path}")
            
        with h5py.File(self.path, "r") as f:
            self.length = f["tensors"].shape[0]
            self.label_map = json.loads(f.attrs["label_map"]) if "label_map" in f.attrs else {}
            
            self.param_key = None
            for key in ["parameters", "params", "targets"]:
                if key in f:
                    self.param_key = key
                    break
            self.has_params = self.param_key is not None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        with h5py.File(self.path, "r") as f:
            raw_tensor = f["tensors"][idx]
            tensor = torch.from_numpy(raw_tensor).float()

            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            
            if "labels" in f:
                label = torch.tensor(int(f["labels"][idx]), dtype=torch.long)
            else:
                label = torch.tensor(0, dtype=torch.long)
            
            if self.has_params:
                raw_params = f[self.param_key][idx]
                if self.log_transform:
                    params_processed = np.sign(raw_params) * np.log10(np.abs(raw_params) + 1e-12)
                else:
                    params_processed = raw_params
                params = torch.from_numpy(params_processed).float()
            else:
                params = torch.zeros(2, dtype=torch.float32)
            
        return tensor, label, params


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
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(128 * 2 * 2, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.conv_blocks(x)
        out = torch.flatten(out, 1)
        return F.silu(self.fc(out))


class DegeneracyMDNHead(nn.Module):
    def __init__(self, feature_dim: int = 128, param_dim: int = 2):
        super().__init__()
        self.param_dim = param_dim
        self.mu_head = nn.Linear(feature_dim, param_dim)
        self.cholesky_dim = (param_dim * (param_dim + 1)) // 2
        self.cholesky_head = nn.Linear(feature_dim, self.cholesky_dim)

    def forward(self, features: torch.Tensor):
        mu = self.mu_head(features)
        cholesky_raw = self.cholesky_head(features)
        
        batch_size = features.size(0)
        L = torch.zeros(batch_size, self.param_dim, self.param_dim, device=features.device)
        
        tril_indices = torch.tril_indices(row=self.param_dim, col=self.param_dim)
        L[:, tril_indices[0], tril_indices[1]] = cholesky_raw

        diag_mask = torch.eye(self.param_dim, device=features.device).bool()
        L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-5
        
        sigma = torch.bmm(L, L.transpose(1, 2))
        return mu, sigma


def multivariate_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return -dist.log_prob(y).mean()


def train_model(encoder, head, dataloader, optimizer, device, epochs=8):
    encoder.train()
    head.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            tensors, labels, params = batch

            if encoder.in_channels == 1:
                tensors = tensors[:, 0:1, :, :]
                
            tensors, params = tensors.to(device), params[:, :2].to(device)
            
            optimizer.zero_grad()
            features = encoder(tensors)
            mu, sigma = head(features)
            
            loss = multivariate_nll_loss(mu, sigma, params)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(head.parameters()), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1:02d}/{epochs:02d} - NLL Loss: {total_loss/len(dataloader):.4f}")


def compute_information_gain(sigma_spectral, sigma_multimodal):
    det_spec = np.linalg.det(sigma_spectral)
    det_multi = np.linalg.det(sigma_multimodal)
    delta_i = 0.5 * np.log(np.maximum(det_spec, 1e-12) / (np.maximum(det_multi, 1e-12)))
    return delta_i


def plot_degeneracy_ellipse(mu, sigma, ax, label, color):
    vals, vecs = np.linalg.eigh(sigma)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    width, height = 2 * 2 * np.sqrt(np.maximum(vals, 1e-9))
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta, edgecolor=color, fc='none', lw=2, label=label)
    ax.add_patch(ell)
    ax.scatter(mu[0], mu[1], color=color, s=30)


def run_experiment(h5_path: Path, checkpoint_path: Path, fig_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Executing training pipeline on: {device}")
    
    dataset = HDF5GRBDataset(h5_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    print("\n=== Training Spectral-Only Model ===")
    spec_encoder = MultimodalGRBEncoder(in_channels=1).to(device)
    spec_head = DegeneracyMDNHead().to(device)
    opt_spec = torch.optim.Adam(list(spec_encoder.parameters()) + list(spec_head.parameters()), lr=1e-3)
    train_model(spec_encoder, spec_head, train_loader, opt_spec, device, epochs=8)

    print("\n=== Training Full Multimodal Model ===")
    multi_encoder = MultimodalGRBEncoder(in_channels=3).to(device)
    multi_head = DegeneracyMDNHead().to(device)
    opt_multi = torch.optim.Adam(list(multi_encoder.parameters()) + list(multi_head.parameters()), lr=1e-3)
    train_model(multi_encoder, multi_head, train_loader, opt_multi, device, epochs=8)

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[INFO] Preserving trained weights to: {checkpoint_path}")
    torch.save({
        'spec_encoder_state': spec_encoder.state_dict(),
        'spec_head_state': spec_head.state_dict(),
        'multi_encoder_state': multi_encoder.state_dict(),
        'multi_head_state': multi_head.state_dict(),
    }, checkpoint_path)

    spec_encoder.eval(); spec_head.eval()
    multi_encoder.eval(); multi_head.eval()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    with torch.no_grad():
        for batch in test_loader:
            tensors, labels, params = batch
            tensors = tensors.to(device)
            
            mu_s, sig_s = spec_head(spec_encoder(tensors[:, 0:1, :, :]))
            mu_m, sig_m = multi_head(multi_encoder(tensors))
            
            sample_mu_s, sample_sig_s = mu_s[0].cpu().numpy(), sig_s[0].cpu().numpy()
            sample_mu_m, sample_sig_m = mu_m[0].cpu().numpy(), sig_m[0].cpu().numpy()
            gt_p = params[0, :2].numpy()
            
            plot_degeneracy_ellipse(sample_mu_s, sample_sig_s, ax, "Spectral Only (High Ambiguity)", "crimson")
            plot_degeneracy_ellipse(sample_mu_m, sample_sig_m, ax, "Full Multimodal (Resolved)", "green")
            ax.scatter(gt_p[0], gt_p[1], color='black', marker='*', s=100, label="True Parameter Ground Truth")
            
            info_gain = compute_information_gain(sample_sig_s, sample_sig_m)
            print(f"\n[RESULT] Information Gain (Delta I) for sample event: {info_gain:.3f} nats")
            break
            
    ax.set_title("GRB Parameter Degeneracy Contours (95% CI)")
    ax.set_xlabel("Parameter 1 (log E_break)")
    ax.set_ylabel("Parameter 2 (log g_aγ)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=300)
    print(f"[INFO] Saved degeneracy contour plot to {fig_path}")


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
    h5_path = project_root / "data" / "synthetic" / "events_suite.h5"
    checkpoint_path = project_root / "checkpoints" / "grb_model_best.pt"
    fig_path = project_root / "reports" / "figures" / "degeneracy_landscape.png"

    if not h5_path.exists():
        raise FileNotFoundError(f"Missing required HDF5 archive at '{h5_path}'. Please run step 4 (pack_hdf5.py) first.")

    print(f"[INFO] Target dataset located at: {h5_path}")
    print("[INFO] Initiating training pipeline...")
    
    start = time.perf_counter()
    run_experiment(h5_path, checkpoint_path, fig_path)
    print(f"\n[SUCCESS] Pipeline completed successfully in {time.perf_counter() - start:.1f}s")
