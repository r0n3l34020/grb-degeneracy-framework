import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

N_CHANNELS = 3
N_ENERGY_BINS = 500
N_TIME_BINS = 500

class GRBMultimodalDataset(Dataset):
    def __init__(self, num_samples: int = 1000):
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = torch.randn(N_CHANNELS, N_ENERGY_BINS, N_TIME_BINS)
        label = torch.tensor(0, dtype=torch.long)
        params = torch.tensor([5.0, 27.0], dtype=torch.float32)
        return sample, label, params
    
def validate_tensor(tensor: np.ndarray, name: str = "tensor") -> None:
    if np.any(np.isnan(tensor)):
        raise ValueError(f"Error, {name} contains NaN values")
    if np.any(np.isinf(tensor)):
        raise ValueError(f"Error, {name} contains infinite values")


def save_dataset_hdf5(tensors: np.ndarray, labels: list, path: Path, parameters: np.ndarray = None) -> Path:
    validate_tensor(tensors, name="dataset tensor stack")
    if parameters is not None:
        validate_tensor(parameters, name="physical parameters stack")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    label_map = {name: i for i, name in enumerate(sorted(set(labels)))}
    label_ids = np.array([label_map[label] for label in labels], dtype=np.int64)

    with h5py.File(path, "w") as f:
        f.create_dataset("tensors", data=tensors.astype(np.float32), compression="gzip", compression_opts=4)
        f.create_dataset("labels", data=label_ids)
        f.attrs["label_map"] = json.dumps(label_map)
        
        if parameters is not None:
            f.create_dataset("parameters", data=parameters.astype(np.float32), compression="gzip", compression_opts=4)
            
    return path


class HDF5GRBDataset(Dataset):
    def __init__(self, path, log_transform_params: bool = True):
        self.path = Path(path)
        self.log_transform = log_transform_params
        
        with h5py.File(self.path, "r") as f:
            self.length = f["tensors"].shape[0]
            self.label_map = json.loads(f.attrs["label_map"])
            self.has_params = "parameters" in f
            
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = h5py.File(self.path, "r")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()
        tensor = torch.from_numpy(self._file["tensors"][idx]).float()
        label = torch.tensor(int(self._file["labels"][idx]), dtype=torch.long)
        
        if self.has_params:
            raw_params = self._file["parameters"][idx]
            if self.log_transform:
                params_processed = np.sign(raw_params) * np.log10(np.abs(raw_params) + 1e-12)
            else:
                params_processed = raw_params
                
            params = torch.from_numpy(params_processed).float()
            return tensor, label, params
            
        return tensor, label
