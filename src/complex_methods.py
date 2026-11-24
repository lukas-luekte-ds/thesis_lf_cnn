import math
from typing import Literal

import torch
from torch import nn


class ComplexDropout(nn.Module):
    """
    Dropout applied with the same mask to real and imaginary parts.
    """
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, xr: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.training or self.p <= 0.0:
            return xr, xi

        keep_prob = 1.0 - self.p
        mask = torch.bernoulli(torch.full_like(xr, keep_prob)) / keep_prob
        return xr * mask, xi * mask

class ModReLU(nn.Module):
    def __init__(self, num_features: int, bias_init: float = -0.1):
        super().__init__()

        self.bias = nn.Parameter(
            torch.full((num_features,), bias_init)
        )

    def forward(self, xr, xi):
        magnitude = torch.hypot(xr, xi)

        active = (magnitude + self.bias > 0) & (magnitude > 0)

        safe_magnitude = torch.where(
            active,
            magnitude,
            torch.ones_like(magnitude),
        )

        scale = torch.where(
            active,
            (magnitude + self.bias) / safe_magnitude,
            torch.zeros_like(magnitude),
        )

        return scale * xr, scale * xi

class FullyComplexTanh(nn.Module):

    def forward(self, xr, xi):

        z = torch.complex(
            xr.to(torch.float64),
            xi.to(torch.float64)
        )

        out = torch.tanh(z)

        return (
            out.real.to(xr.dtype),
            out.imag.to(xi.dtype)
        )

class Cardioid(nn.Module):
    """
    Cardioid complex activation.

    f(z) = 0.5 * (1 + cos(arg(z))) * z

    For z = 0, f(z) = 0.
    """

    def forward(
        self,
        xr: torch.Tensor,
        xi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        magnitude = torch.hypot(xr, xi)

        # Avoid division by zero at z = 0.
        safe_magnitude = torch.where(
            magnitude > 0,
            magnitude,
            torch.ones_like(magnitude),
        )

        cos_phase = xr / safe_magnitude

        scale = 0.5 * (1.0 + cos_phase)

        return scale * xr, scale * xi


class ComplexSplitActivation(nn.Module):
    """
    Applies the same real activation component-wise to real and imaginary parts.

    F(z) = f(Re(z)) + i*f(Im(z))
    """

    def __init__(
        self,
        activation: Literal[
            "relu",
            "leakyrelu",
            "gelu",
            "tanh",
        ] = "relu",
    ):
        super().__init__()

        activations = {
            "relu": nn.ReLU,
            "leakyrelu": lambda: nn.LeakyReLU(negative_slope=0.01),
            "gelu": nn.GELU,
            "tanh": nn.Tanh,
        }

        if activation not in activations:
            raise ValueError(
                f"Unknown complex activation: {activation}"
            )

        self.act = activations[activation]()

    def forward(
        self,
        xr: torch.Tensor,
        xi: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        return self.act(xr), self.act(xi)

def create_complex_activation(
    activation: str,
    num_features: int | None = None
) -> nn.Module:

    split_activations = {
        "relu",
        "leakyrelu",
        "gelu",
        "tanh",
    }

    if activation in split_activations:
        return ComplexSplitActivation(activation)

    if activation == "fully_complex_tanh":
        return FullyComplexTanh()

    if activation == "modrelu":
        if num_features is None:
            raise ValueError(
                "modReLU requires num_features."
            )

        return ModReLU(
            num_features=num_features,
            bias_init=-0.1,
        )
    
    if activation == "cardioid":
        return Cardioid()

    raise ValueError(
        f"Unknown complex activation: {activation}"
    )

def initialize_complex_weights(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
    fan_in: int,
    fan_out: int,
    initialization: dict, 
) -> None:
    method = initialization["method"]

    if method == "componentwise_pytorch_default":
        init_componentwise_pytorch_default(
            weight_real,
            weight_imag,
        )
    elif method == "split_relu_he_normal":
        init_split_relu_he_normal(
            weight_real, 
            weight_imag, 
            fan_in
        )
    elif method == "split_relu_he_uniform":
        init_split_relu_he_uniform(
            weight_real, 
            weight_imag, 
            fan_in
        )
    elif method == "split_leakyrelu_he_normal":
        init_split_leakyrelu_he_normal(
            weight_real,
            weight_imag,
            fan_in,
            negative_slope=initialization["negative_slope"]
        )
    elif method == "split_leakyrelu_he_uniform":
        init_split_leakyrelu_he_uniform(
            weight_real,
            weight_imag,
            fan_in,
            negative_slope=initialization["negative_slope"]
        )
    elif method == "split_relu_xavier_normal":
        init_split_relu_xavier_normal(
            weight_real,
            weight_imag,
            fan_in,
            fan_out
        )
    elif method == "split_relu_xavier_uniform":
        init_split_relu_xavier_uniform(
            weight_real,
            weight_imag,
            fan_in,
            fan_out
        )
    elif method == "split_leakyrelu_xavier_normal":
        init_split_leakyrelu_xavier_normal(
            weight_real,
            weight_imag,
            fan_in,
            fan_out,
            negative_slope=initialization["negative_slope"]
        )
    elif method == "split_leakyrelu_xavier_uniform":
        init_split_leakyrelu_xavier_uniform(
            weight_real,
            weight_imag,
            fan_in,
            fan_out,
            negative_slope=initialization["negative_slope"]
        )
    else:
        raise ValueError(
            f"Unsupported complex initialization: {method}"
        )

def init_componentwise_pytorch_default(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
) -> None:

    nn.init.kaiming_uniform_(
        weight_real,
        a=math.sqrt(5),
    )

    nn.init.kaiming_uniform_(
        weight_imag,
        a=math.sqrt(5),
    )

def init_split_relu_he_normal(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
    fan_in: int,
) -> None:
    std = 1.0 / math.sqrt(fan_in)

    nn.init.normal_(weight_real, mean=0.0, std=std)
    nn.init.normal_(weight_imag, mean=0.0, std=std)

def init_split_relu_he_uniform(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
    fan_in: int,
) -> None:
    bound = math.sqrt(3.0 / fan_in)

    nn.init.uniform_(weight_real, -bound, bound)
    nn.init.uniform_(weight_imag, -bound, bound)

def init_split_leakyrelu_he_normal(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
    fan_in: int,
    negative_slope: float = 0.01,
) -> None:
    std = 1.0 / math.sqrt(
        (1.0 + negative_slope**2) * fan_in
    )

    nn.init.normal_(weight_real, mean=0.0, std=std)
    nn.init.normal_(weight_imag, mean=0.0, std=std)

def init_split_leakyrelu_he_uniform(
    weight_real: torch.Tensor,
    weight_imag: torch.Tensor,
    fan_in: int,
    negative_slope: float = 0.01,
) -> None:
    bound = math.sqrt(
        3.0
        / (
            (1.0 + negative_slope**2)
            * fan_in
        )
    )

    nn.init.uniform_(weight_real, -bound, bound)
    nn.init.uniform_(weight_imag, -bound, bound)

def init_split_relu_xavier_normal(
    weight_real,
    weight_imag,
    fan_in,
    fan_out,
):
    std = math.sqrt(
        2.0 / (fan_in + fan_out)
    )

    nn.init.normal_(weight_real, mean=0.0, std=std)
    nn.init.normal_(weight_imag, mean=0.0, std=std)

def init_split_relu_xavier_uniform(
    weight_real,
    weight_imag,
    fan_in,
    fan_out,
):
    bound = math.sqrt(
        6.0 / (fan_in + fan_out)
    )

    nn.init.uniform_(weight_real, -bound, bound)
    nn.init.uniform_(weight_imag, -bound, bound)

def init_split_leakyrelu_xavier_normal(
    weight_real,
    weight_imag,
    fan_in,
    fan_out,
    negative_slope,
):
    std = math.sqrt(
        2.0
        / (
            (1.0 + negative_slope**2)
            * (fan_in + fan_out)
        )
    )

    nn.init.normal_(weight_real, mean=0.0, std=std)
    nn.init.normal_(weight_imag, mean=0.0, std=std)

def init_split_leakyrelu_xavier_uniform(
    weight_real,
    weight_imag,
    fan_in,
    fan_out,
    negative_slope,
):
    bound = math.sqrt(
        6.0
        / (
            (1.0 + negative_slope**2)
            * (fan_in + fan_out)
        )
    )

    nn.init.uniform_(weight_real, -bound, bound)
    nn.init.uniform_(weight_imag, -bound, bound)



