import torch
from torch import nn


class InputNormalizer(nn.Module):
    """
    normalization of the input parameters, the mean and std are saved in the register buffer
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.eps = eps

    @torch.no_grad()
    def fit(self, x: torch.Tensor):
        # imput x: [N, D] so the columns will be the different features
        mean = x.mean(dim=0) # dim = 0 aggregate the lines columnwise
        std = x.std(dim=0, unbiased=False).clamp_min(self.eps) # clamp_min to dodge division by 0 
        self.mean.copy_(mean)
        self.std.copy_(std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std