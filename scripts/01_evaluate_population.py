import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import h5py
from tqdm import tqdm

class HDF5GRBDataset(Dataset):
    def __init__(self, path):
        self.path = Path(path)
        with h5py.File(self.path, "r") as f:
            self.length = f["tensors"].shape[0]
            if "labels" in f:
                self.param_key = "labels"
            elif "parameters" in f:
                self.param_key = "parameters"
            elif "params" in f:
                self.param_key = "params"
            else:
                raise KeyError(f"None of 'labels', 'parameters', or 'params' found in HDF5 keys: {list(f.keys())}")
        self._file = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        tensor = torch.from_numpy(self._file["tensors"][idx]).float()
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
            nn.BatchNorm2d(32), nn.SiLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.SiLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.SiLU(), nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(128 * 2 * 2, feature_dim)

    def forward(self, x):
        if x.dim() == 3: x = x.unsqueeze(1)
        out = self.conv_blocks(x)
        return F.silu(self.fc(torch.flatten(out, 1)))

class DegeneracyMDNHead(nn.Module):
    def __init__(self, feature_dim: int = 128, param_dim: int = 2):
        super().__init__()
        self.param_dim = param_dim
        self.mu_head = nn.Linear(feature_dim, param_dim)
        self.cholesky_head = nn.Linear(feature_dim, (param_dim * (param_dim + 1)) // 2)

    def forward(self, features):
        mu = self.mu_head(features)
        cholesky_raw = self.cholesky_head(features)
        batch_size = features.size(0)
        L = torch.zeros(batch_size, self.param_dim, self.param_dim, device=features.device)
        tril_indices = torch.tril_indices(row=self.param_dim, col=self.param_dim)
        L[:, tril_indices[0], tril_indices[1]] = cholesky_raw
        diag_mask = torch.eye(self.param_dim, device=features.device).bool()
        L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-5
        return mu, torch.bmm(L, L.transpose(1, 2))

def evaluate_population():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h5_path = Path("data/synthetic/events_suite.h5")
    ckpt_path = Path("checkpoints/grb_model_best.pt")

    dataset = HDF5GRBDataset(h5_path)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    _, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    spec_enc = MultimodalGRBEncoder(in_channels=1).to(device)
    spec_head = DegeneracyMDNHead().to(device)
    multi_enc = MultimodalGRBEncoder(in_channels=3).to(device)
    multi_head = DegeneracyMDNHead().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    spec_enc.load_state_dict(checkpoint['spec_encoder_state'])
    spec_head.load_state_dict(checkpoint['spec_head_state'])
    multi_enc.load_state_dict(checkpoint['multi_encoder_state'])
    multi_head.load_state_dict(checkpoint['multi_head_state'])

    spec_enc.eval(); spec_head.eval()
    multi_enc.eval(); multi_head.eval()

    delta_i_list, params_list = [], []

    with torch.no_grad():
        for tensors, params in tqdm(test_loader, desc="Evaluating Population", unit="batch"):
            tensors = tensors.to(device)
            _, sig_s = spec_head(spec_enc(tensors[:, 0:1, :, :]))
            _, sig_m = multi_head(multi_enc(tensors))

            det_s = torch.det(sig_s).cpu().numpy()
            det_m = torch.det(sig_m).cpu().numpy()

            delta_i = 0.5 * np.log(det_s / (det_m + 1e-9))
            delta_i_list.extend(delta_i)
            params_list.extend(params[:, :2].numpy())

    delta_i_arr = np.array(delta_i_list)
    params_arr = np.array(params_list)

    mean_val = float(np.mean(delta_i_arr))
    median_val = float(np.median(delta_i_arr))
    std_val = float(np.std(delta_i_arr))
    ci95_margin = float(1.96 * (std_val / np.sqrt(len(delta_i_arr))))

    metrics = {
        "total_test_events": len(delta_i_arr),
        "mean_delta_i_nats": mean_val,
        "median_delta_i_nats": median_val,
        "std_dev_nats": std_val,
        "ci95_lower": mean_val - ci95_margin,
        "ci95_upper": mean_val + ci95_margin
    }

    print("\n================ POPULATION STATISTICAL METRICS ================")
    print(f"Evaluated Samples : {metrics['total_test_events']}")
    print(f"Mean ΔI           : {metrics['mean_delta_i_nats']:.4f} nats")
    print(f"Median ΔI         : {metrics['median_delta_i_nats']:.4f} nats")
    print(f"Std Deviation     : {metrics['std_dev_nats']:.4f} nats")
    print(f"95% CI Interval   : [{metrics['ci95_lower']:.4f}, {metrics['ci95_upper']:.4f}] nats")
    print("================================================================")

    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    with open(reports_dir / "population_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    np.savez(reports_dir / "population_eval_data.npz", delta_i=delta_i_arr, params=params_arr)
    print(f"[SUCCESS] Exported statistical artifacts to {reports_dir}/")

if __name__ == "__main__":
    evaluate_population()
