import torch
from torch import nn

from normalization import InputNormalizer


class LSTMForecast(nn.Module):
    def __init__(
        self,
        seq_len: int,
        input_size: int,
        hidden_sizes: list[int] | None = None,
        dropout: float = 0.0,
        out_dim: int = 6,
        bidirectional: bool = False,
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
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        # Store model configuration
        self.seq_len = seq_len
        self.input_size = input_size
        self.hidden_sizes = list(hidden_sizes)
        self.out_dim = out_dim
        self.bidirectional = bidirectional
        self.accept_flat_input = accept_flat_input
        self.use_input_norm = use_input_norm
        self.use_last_timestep = use_last_timestep

        # Input normalization
        if use_input_norm:
            self.in_norm = InputNormalizer(seq_len * input_size)

        # LSTM layers
        direction_factor = 2 if bidirectional else 1

        self.lstm_layers = nn.ModuleList()

        for i, hidden_size in enumerate(hidden_sizes):
            layer_input_size = (
                input_size
                if i == 0
                else hidden_sizes[i - 1] * direction_factor
            )

            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    num_layers=1,
                    bidirectional=bidirectional,
                    batch_first=True,
                )
            )

        # Dropout between LSTM layers
        self.dropout = nn.Dropout(dropout)

        # Output head
        self.head = nn.Linear(
            hidden_sizes[-1] * direction_factor,
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._reshape_if_needed(x)

        if self.use_input_norm:
            batch_size = x.size(0)

            # Position-wise normalization as in the MLP
            x = x.reshape(batch_size, -1)
            x = self.in_norm(x)
            x = x.reshape(batch_size, self.seq_len, self.input_size)

        # LSTM forward pass
        for i, lstm_layer in enumerate(self.lstm_layers):
            x, (_h_n, _c_n) = lstm_layer(x)

            # Dropout only between LSTM layers
            if i < len(self.lstm_layers) - 1:
                x = self.dropout(x)

        if self.use_last_timestep:
            features = x[:, -1, :]
        else:
            features = x.mean(dim=1)

        pred = self.head(features)
        return pred