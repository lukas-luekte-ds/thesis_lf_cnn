
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from complex_methods import (
    ComplexDropout,
    create_complex_activation,
    initialize_complex_weights,
)
from normalization import InputNormalizer

ComplexInitializationMethod = Literal[
    "componentwise_pytorch_default",
    "split_relu_he_normal",
    "split_relu_he_uniform",
    "split_leakyrelu_he_normal",
    "split_leakyrelu_he_uniform",
    "split_relu_xavier_normal",
    "split_relu_xavier_uniform",
    "split_leakyrelu_xavier_normal",
    "split_leakyrelu_xavier_uniform"
]
    
class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer using real/imag split. 

    Input:
        xr, xi: [B, in_features]
    Output:
        yr, yi: [B, out_features]

    implementation like in torch.nn but 2 dimensional and with the complex matrix multiplication as output
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True, initialization: dict | None = None):
        super().__init__()

        if initialization is None:
            initialization = {
                "method": "componentwise_pytorch_default"
            }

        self.in_features = in_features
        self.out_features = out_features
        self.initialization = initialization

        self.weight_real = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_imag = nn.Parameter(torch.empty(out_features, in_features))

        if bias:
            self.bias_real = nn.Parameter(torch.empty(out_features))
            self.bias_imag = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_real", None)
            self.register_parameter("bias_imag", None)

        self.reset_parameters()

    def reset_parameters(self):
        initialize_complex_weights(
            weight_real=self.weight_real,
            weight_imag=self.weight_imag,
            initialization=self.initialization,
            fan_in=self.in_features,
            fan_out=self.out_features,
        )

        if self.bias_real is not None:
            fan_in = self.in_features
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_real, -bound, bound)
            nn.init.uniform_(self.bias_imag, -bound, bound)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if xr.shape != xi.shape:
            raise ValueError(f"xr and xi must have same shape, got {xr.shape} and {xi.shape}")
        if xr.ndim != 2:
            raise ValueError(f"ComplexLinear expects [B, D], got {xr.shape}")
        # result from the complex matrix multiplication zX = x_rW_r - x_iW_i + i(x_iW_r + x_rW_i) so we get a real y_r and img y_i
        yr = F.linear(xr, self.weight_real, self.bias_real) - F.linear(xi, self.weight_imag, None) 
        yi = F.linear(xr, self.weight_imag, self.bias_imag) + F.linear(xi, self.weight_real, None)
        return yr, yi


class ComplexLayerNorm(nn.Module):
    """
    Component-wise layer normalization for complex-valued features.

    Real and imaginary parts are normalized independently.
    """

    def __init__(self, num_features: int):
        super().__init__()

        self.real_norm = nn.LayerNorm(num_features)
        self.imag_norm = nn.LayerNorm(num_features)

    def forward(
        self,
        xr: torch.Tensor,
        xi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if xr.shape != xi.shape:
            raise ValueError(
                f"xr and xi must have same shape, "
                f"got {xr.shape} and {xi.shape}"
            )

        yr = self.real_norm(xr)
        yi = self.imag_norm(xi)

        return yr, yi

class ComplexToReal(nn.Module):
    """
    Real-valued readout from complex hidden state.

    Strategy:
        concat(real, imag) -> Linear
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.proj = nn.Linear(2 * in_features, out_features)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        if xr.shape != xi.shape:
            raise ValueError(f"xr and xi must have same shape, got {xr.shape} and {xi.shape}")
        x = torch.cat([xr, xi], dim=-1)
        return self.proj(x)

class ComplexSTLFMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 1331,
        hidden_sizes: list[int] | None = None,
        out_dim: int = 6,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        use_input_norm: bool = True,
        activation: Literal["gelu", "relu", "leakyrelu", "tanh", "fully_complex_tanh", "modrelu", "cardioid"] = "gelu",
        initialization: dict | None = None
    ):
        if hidden_sizes is None:
            hidden_sizes = [512, 256, 128]
        if initialization is None:
            initialization = {
        "method": "componentwise_pytorch_default"
        }
        super().__init__()

        self.input_dim = input_dim
        self.hidden_sizes = list(hidden_sizes)
        self.out_dim = out_dim
        self.use_layernorm = use_layernorm
        self.use_input_norm = use_input_norm
        self.initialization = initialization

        if len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must contain at least one hidden layer size")
        if use_input_norm:
            self.in_norm = InputNormalizer(input_dim)

        dims = [input_dim] + list(hidden_sizes)

        self.hidden_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.layernorms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for i in range(len(dims) - 1):
            self.hidden_layers.append(ComplexLinear(dims[i], dims[i + 1], initialization = initialization))
            self.activations.append(create_complex_activation(activation, num_features = dims[i + 1]))
            if use_layernorm:
                self.layernorms.append(
                    ComplexLayerNorm(dims[i + 1])
                )
            else:
                self.layernorms.append(nn.Identity())
            self.dropouts.append(ComplexDropout(dropout))

        self.readout = ComplexToReal(hidden_sizes[-1], out_dim)
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
            return x.reshape(x.size(0), -1)
        else:
            raise ValueError(
                f"ComplexSTLFMLP expects input with shape [B, D] or [B, T, F], "
                f"got {tuple(x.shape)}"
            )

    def _check_input_dim(self, x: torch.Tensor) -> None:
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input dimension mismatch: model expects input_dim={self.input_dim}, "
                f"but got flattened input shape {tuple(x.shape)}. "
                f"Check your dataloader output and model config."
            )

    @torch.no_grad()
    def fit_input_normalizer_from_loader(self, loader):
        """
        Fit input normalization statistics once before training.
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

        X = torch.cat(xs, dim=0)
        self.in_norm.fit(X)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._flatten_if_needed(x)
        self._check_input_dim(x)

        if self.use_input_norm:
            x = self.in_norm(x)

        xr = x
        xi = torch.zeros_like(x)

        for layer, act, norm, drop in zip(self.hidden_layers, self.activations, self.layernorms, self.dropouts):
            xr, xi = layer(xr, xi)
            xr, xi = act(xr, xi)
            if self.use_layernorm:
                xr, xi = norm(xr, xi)
            xr, xi = drop(xr, xi)

        out = self.readout(xr, xi)
        return out


