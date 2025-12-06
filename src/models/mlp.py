# model.py
from typing import List, Literal, Tuple
import torch
import torch.nn as nn

# using Literal enables only the given parameters
LossType = Literal["mae", "mse", "huber"] # maybe add MAPE?

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

class STLFMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 1331,
        hidden_sizes: List[int] = [512, 256, 128],
        out_dim: int = 6,             # 6 targets
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_input_norm: bool = True,
        activation: Literal["gelu","relu","leakyrelu"] = "gelu",
    ):
        super().__init__()
        self.use_input_norm = use_input_norm
        if use_input_norm:
            self.in_norm = InputNormalizer(input_dim)

        act: nn.Module
        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "leakyrelu":
            act = nn.LeakyReLU(0.01)
        else:
            raise ValueError("Unsupported activation")

        layers: List[nn.Module] = []
        dims = [input_dim] + list(hidden_sizes)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(act)
            if use_layernorm:
                layers.append(nn.LayerNorm(dims[i + 1]))
            if dropout and dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], out_dim))
        self.net = nn.Sequential(*layers)

        # for logging
        self.target_names: List[str] = [
            "yl_t+60", "yl_t+1440",
            "yw_t+5", "yw_t+30",
            "ys_t+5", "ys_t+30",
        ]

    @torch.no_grad()
    def fit_input_normalizer_from_loader(self, loader):
        """
        need to norm the input only once before training 
        """
        if not self.use_input_norm:
            return
        xs = []
        for batch in loader:
            x = batch[0]
            xs.append(x)
        X = torch.cat(xs, dim=0)
        self.in_norm.fit(X.to(self.in_norm.mean.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_input_norm:
            x = self.in_norm(x)
        return self.net(x)

