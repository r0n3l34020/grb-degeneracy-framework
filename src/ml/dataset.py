import torch
from torch.utils.data import Dataset

# Mock multimodal tensor schema.
# Each sample is a (3, N_BINS) tensor: 3 channels = [Spectrum, LightCurve, Polarization],
# each represented as a length-N_BINS 1D vector for now. This is a Day 1 stub only —
# real physics-derived data and the full (3, Energy_Bins, Time_Bins) shape land on Day 4.
N_CHANNELS = 3
N_BINS = 100


class GRBMultimodalDataset(Dataset):
    def __init__(self, num_samples: int = 1000, n_bins: int = N_BINS):
        self.num_samples = num_samples
        self.n_bins = n_bins

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Placeholder mock data until the physics generator (Day 4) is wired in.
        sample = torch.randn(N_CHANNELS, self.n_bins)
        label = torch.tensor(0, dtype=torch.long)
        return sample, label


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset = GRBMultimodalDataset(num_samples=16)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    batch, labels = next(iter(loader))
    print("Batch tensor shape:", tuple(batch.shape))
    assert batch.shape == (4, N_CHANNELS, N_BINS), (
        f"Error, expected shape (4, {N_CHANNELS}, {N_BINS}) got {tuple(batch.shape)}"
    )
    print("Day 1 dataset stub OK.")
