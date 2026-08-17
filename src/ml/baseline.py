import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.ml.dataset import HDF5GRBDataset
from src.ml.models import MultimodalGRBEncoder, DegeneracyMDNHead, multivariate_nll_loss

def train_model(encoder, head, dataloader, optimizer, device, epochs=10):
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
    delta_i = 0.5 * np.log(det_spec / (det_multi + 1e-9))
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

def run_experiment(h5_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing training pipeline on: {device}")
    
    dataset = HDF5GRBDataset(h5_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

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

    spec_encoder.eval(); spec_head.eval()
    multi_encoder.eval(); multi_head.eval()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    with torch.no_grad():
        for tensors, labels, params in test_loader:
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
            print(f"\nInformation Gain (Delta I) for sample event: {info_gain:.3f} nats")
            break
            
    ax.set_title("GRB 221009A Parameter Degeneracy Contours (95% CI)")
    ax.set_xlabel("Parameter 1 (e.g. Break Energy E_break)")
    ax.set_ylabel("Parameter 2 (e.g. Axion Coupling g_aγ)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.savefig("degeneracy_landscape.png", dpi=300)
    print("Saved degeneracy contour plot to degeneracy_landscape.png")

if __name__ == "__main__":
    h5_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "events_suite.h5"
    if not h5_path.exists():
        h5_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "events.h5"
    
    if not h5_path.exists():
        print(f"[ERROR] HDF5 dataset not found at {h5_path}, Sir.")
    else:
        print(f"Using dataset: {h5_path}")
        start = time.perf_counter()
        run_experiment(h5_path)
        print(f"Pipeline completed in {time.perf_counter() - start:.1f}s")
