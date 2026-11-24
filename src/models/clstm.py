import torch
import torch.nn.functional as F
from torch import nn

from complex_methods import ComplexDropout
from normalization import InputNormalizer


class ComplexLinear(nn.Module):
    """
    Complex-valued linear transformation.

    y = (Wr + iWi)(xr + ixi) + (br + ibi)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ):
        super().__init__()

        self.real = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

        self.imag = nn.Linear(
            in_features,
            out_features,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr = x.real
        xi = x.imag

        yr = (
            F.linear(xr, self.real.weight, self.real.bias)
            - F.linear(xi, self.imag.weight, None)
        )

        yi = (
            F.linear(xr, self.imag.weight, self.imag.bias)
            + F.linear(xi, self.real.weight, None)
        )

        return torch.complex(yr, yi)

class ComplexLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # komplexe affine Transformationen
        self.x_to_g = ComplexLinear(input_size, hidden_size)
        self.h_to_g = ComplexLinear(hidden_size, hidden_size, bias=False)

        # reelle Gates
        self.x_to_i = nn.Linear(2 * input_size, hidden_size)
        self.h_to_i = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.x_to_f = nn.Linear(2 * input_size, hidden_size)
        self.h_to_f = nn.Linear(2 * hidden_size, hidden_size, bias=False)

        self.x_to_o = nn.Linear(2 * input_size, hidden_size)
        self.h_to_o = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    def _as_real_features(self, z: torch.Tensor) -> torch.Tensor:
        # concat real and imag
        return torch.cat([z.real, z.imag], dim=-1)

    def forward(self, x_t: torch.Tensor, h_prev: torch.Tensor, c_prev: torch.Tensor):
        # x_t, h_prev, c_prev: complex tensors

        x_ri = self._as_real_features(x_t)
        h_ri = self._as_real_features(h_prev)

        i_t = torch.sigmoid(self.x_to_i(x_ri) + self.h_to_i(h_ri))
        f_t = torch.sigmoid(self.x_to_f(x_ri) + self.h_to_f(h_ri))
        o_t = torch.sigmoid(self.x_to_o(x_ri) + self.h_to_o(h_ri))

        g_t = torch.tanh(self.x_to_g(x_t) + self.h_to_g(h_prev))

        # reelle Gates auf komplexen Zustand anwenden
        i_t = i_t.to(g_t.real.dtype)
        f_t = f_t.to(g_t.real.dtype)
        o_t = o_t.to(g_t.real.dtype)

        i_t = torch.complex(i_t, torch.zeros_like(i_t))
        f_t = torch.complex(f_t, torch.zeros_like(f_t))
        o_t = torch.complex(o_t, torch.zeros_like(o_t))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t

class ComplexLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = ComplexLSTMCell(input_size, hidden_size)

    def forward(self, x: torch.Tensor):
        # x: (batch, seq_len, input_size), complex
        b, s, _ = x.shape
        device = x.device
        dtype = x.dtype

        h = torch.zeros(b, self.hidden_size, device=device, dtype=dtype)
        c = torch.zeros(b, self.hidden_size, device=device, dtype=dtype)

        outputs = []
        for t in range(s):
            h, c = self.cell(x[:, t, :], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)  # (batch, seq_len, hidden_size)
        return outputs, (h, c)

class ComplexLSTMForecast(nn.Module):
    """
    Many-to-one complex-valued LSTM for forecasting.

    1) Sequential input:
       x.shape == (batch_size, seq_len, input_size)

    2) Flattened sliding window:
       x.shape == (batch_size, seq_len * input_size)

    Output:
       pred.shape == (batch_size, out_dim)
    """

    def __init__(
        self,
        seq_len: int,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        out_dim: int = 6,
        accept_flat_input: bool = True,
        use_input_norm: bool = True,
        use_last_timestep: bool = True,
    ):
        if hidden_sizes is None:
            hidden_sizes = [64]

        super().__init__()

        # Validate model configuration
        if seq_len <= 0:
            raise ValueError(f"seq_len must be > 0, got {seq_len}")
        if input_size <= 0:
            raise ValueError(f"input_size must be > 0, got {input_size}")
        if not hidden_sizes or any(size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got {out_dim}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )

        # Store model configuration
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.out_dim = out_dim
        self.dropout = ComplexDropout(dropout)
        self.accept_flat_input = accept_flat_input
        self.use_input_norm = use_input_norm
        self.use_last_timestep = use_last_timestep

        # Input normalization
        if use_input_norm:
            self.in_norm = InputNormalizer(seq_len * input_size)

        # Complex LSTM layers
        dims = [input_size] + list(hidden_sizes)

        # First layer: preserve existing parameter names
        self.rnn = ComplexLSTM(
            input_size=dims[0],
            hidden_size=dims[1],
        )

        # Additional complex LSTM layers
        self.additional_rnn_layers = nn.ModuleList()

        for i in range(1, len(dims) - 1):
            self.additional_rnn_layers.append(
                ComplexLSTM(
                    input_size=dims[i],
                    hidden_size=dims[i + 1],
                )
            )

        # Real-valued output head
        self.head = nn.Linear(
            2 * hidden_sizes[-1],
            out_dim,
        )

        # For logging
        self.target_names: list[str] = [
            "yl_t+60", "yl_t+1440",
            "yw_t+5", "yw_t+30",
            "ys_t+5", "ys_t+30",
        ]

    def _reshape_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept [B, T, F] or [B, T * F] and return [B, T, F].
        """
        if x.ndim == 3:
            b, s, f = x.shape

            if s != self.seq_len or f != self.input_size:
                raise ValueError(
                    f"3D input has wrong shape: got {tuple(x.shape)}, "
                    f"expected (batch_size, {self.seq_len}, {self.input_size})"
                )

            return x

        if x.ndim == 2:
            if not self.accept_flat_input:
                raise ValueError(
                    "Received 2D flat input, but accept_flat_input=False. "
                    f"Got shape {tuple(x.shape)}."
                )

            b, flat_dim = x.shape
            expected_flat_dim = self.seq_len * self.input_size

            if flat_dim != expected_flat_dim:
                raise ValueError(
                    f"2D input has wrong feature dimension: got {flat_dim}, "
                    f"expected seq_len * input_size = "
                    f"{self.seq_len} * {self.input_size} = {expected_flat_dim}"
                )

            return x.reshape(b, self.seq_len, self.input_size)

        raise ValueError(
            f"Input x must be 2D or 3D, "
            f"got ndim={x.ndim}, shape={tuple(x.shape)}"
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

            x = self._reshape_if_needed(x)
            x = x.reshape(x.size(0), -1)

            xs.append(x)

        X = torch.cat(xs, dim=0)
        self.in_norm.fit(X)

    def _to_complex(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert real-valued input to complex-valued input
        with zero imaginary component.
        """
        return torch.complex(x, torch.zeros_like(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_if_needed(x)

        if self.use_input_norm:
            batch_size = x.size(0)

            # Position-wise normalization as in the MLP
            x = x.reshape(batch_size, -1)
            x = self.in_norm(x)
            x = x.reshape(batch_size, self.seq_len, self.input_size)

        # Convert real input to complex input
        x = self._to_complex(x)

        # First complex LSTM layer
        x, (_h_n, _c_n) = self.rnn(x)

        # Dropout between LSTM layers
        if len(self.additional_rnn_layers) > 0:
            xr, xi = self.dropout(x.real, x.imag)
            x = torch.complex(xr, xi)

        # Additional complex LSTM layers
        for i, rnn_layer in enumerate(self.additional_rnn_layers):
            x, (_h_n, _c_n) = rnn_layer(x)

            # Dropout only between LSTM layers
            if i < len(self.additional_rnn_layers) - 1:
                xr, xi = self.dropout(x.real, x.imag)
                x = torch.complex(xr, xi)

        if self.use_last_timestep:
            features = x[:, -1, :]
        else:
            features = x.mean(dim=1)

        # Convert complex features to real-valued features
        features_ri = torch.cat(
            [features.real, features.imag],
            dim=-1,
        )

        pred = self.head(features_ri)
        return pred