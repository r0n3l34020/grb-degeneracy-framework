import torch
import torch.nn as nn

# Feature extractor stub for the ML baseline.
# Splits the (batch, 3, N) multimodal tensor into its Spectrum, LightCurve, and
# Polarization channels and reduces each to a fixed-size feature vector with a
# single linear layer per modality. This is a Day 3 placeholder so data can flow
# end-to-end early — the real per-modality 1D-CNN heads land on Day 10.


class FeatureExtractorStub(nn.Module):
    def __init__(self, n_bins: int = 100, feature_dim: int = 16):
        super().__init__()
        self.spectral_head = nn.Linear(n_bins, feature_dim)
        self.temporal_head = nn.Linear(n_bins, feature_dim)
        self.polarimetric_head = nn.Linear(n_bins, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum, light_curve, polarization = x[:, 0, :], x[:, 1, :], x[:, 2, :]
        spectral_features = self.spectral_head(spectrum)
        temporal_features = self.temporal_head(light_curve)
        polarimetric_features = self.polarimetric_head(polarization)
        return torch.cat([spectral_features, temporal_features, polarimetric_features], dim=1)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src.ml.dataset import GRBMultimodalDataset, N_BINS

    dataset = GRBMultimodalDataset(num_samples=8)
    loader = DataLoader(dataset, batch_size=4)
    batch, _ = next(iter(loader))

    model = FeatureExtractorStub(n_bins=N_BINS, feature_dim=16)
    features = model(batch)
    print("Feature tensor shape:", tuple(features.shape))
    assert features.shape == (4, 48), f"Error, expected shape (4, 48) got {tuple(features.shape)}"
    print("Day 3 feature extractor stub OK.")
