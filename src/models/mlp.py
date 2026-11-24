from typing import Literal

import torch
from torch import nn

from normalization import InputNormalizer


class STLFMLP(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 1331,
        hidden_sizes: list[int] | None = None,
        out_dim: int = 6,             # 6 targets
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_input_norm: bool = True,
        activation: Literal["gelu","relu","leakyrelu"] = "gelu",
    ):
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        super().__init__()
        self.use_input_norm = use_input_norm
        self.input_dim = input_dim
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

        layers: list[nn.Module] = []
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
        self.target_names: list[str] = [
            "yl_t+60", "yl_t+1440",
            "yw_t+5", "yw_t+30",
            "ys_t+5", "ys_t+30",
        ]
    def _flatten_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept [B, D] or [B, T, F] and return [B, D].
        """
        if x.ndim == 2:
            return x
        elif x.ndim == 3:
            x = x.reshape(x.size(0), -1)
            return x
        else:
            raise ValueError(
                f"STLFMLP expects input with shape [B, D] or [B, T, F], got {tuple(x.shape)}"
            )
        

    def _check_input_dim(self, x: torch.Tensor):
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: model expects input_dim={self.input_dim}, "
                f"but got flattened input shape {tuple(x.shape)}. "
                f"Check your dataloader output and model config."
            )
        
    @torch.no_grad()
    def fit_input_normalizer_from_loader(self, loader):
        """
        need to norm the input only once before training 
        """
        if not self.use_input_norm:
            return
        xs = []
        device = self.in_norm.mean.device

        for batch in loader:
            x = batch[0].to(device)
            x = self._flatten_if_needed(x)
            self._check_input_dim(x)
            xs.append(x)

        X = torch.cat(xs, dim=0)  # [N, D]
        self.in_norm.fit(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_if_needed(x)
        self._check_input_dim(x)

        if self.use_input_norm:
            x = self.in_norm(x)

        return self.net(x)

