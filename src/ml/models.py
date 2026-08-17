import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        # Fill lower triangular entries
        tril_indices = torch.tril_indices(row=self.param_dim, col=self.param_dim)
        L[:, tril_indices[0], tril_indices[1]] = cholesky_raw
 
        diag_mask = torch.eye(self.param_dim, device=features.device).bool()
        L[:, diag_mask] = F.softplus(L[:, diag_mask]) + 1e-5
        
        sigma = torch.bmm(L, L.transpose(1, 2))
        return mu, sigma


def multivariate_nll_loss(mu: torch.Tensor, sigma: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    dist = torch.distributions.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    return -dist.log_prob(y).mean()
