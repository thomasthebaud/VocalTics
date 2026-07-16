"""Frame-level tic segmentation models."""

from torch import nn


def _prepare_input(x, input_dim):
    """Convert supported input layouts to [batch, features, time]."""
    if x.ndim == 4 and x.shape[1] == 1:
        x = x.squeeze(1)
    if x.ndim == 2:
        x = x.unsqueeze(-1)
    if x.ndim != 3:
        raise ValueError("Expected input with 2, 3, or 4 dimensions")
    if x.shape[1] == input_dim:
        return x
    if x.shape[2] == input_dim:
        return x.transpose(1, 2)
    raise ValueError(
        f"Expected {input_dim} input features, got shape {tuple(x.shape)}"
    )


class _ConvBlock(nn.Module):
    """One same-length temporal convolution with normalization and activation."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class BiLSTM(nn.Module):
    """Bidirectional LSTM returning one tic-presence logit per time step."""

    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.recurrent_layers = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim).transpose(1, 2)
        x, _ = self.recurrent_layers(x)
        return self.classifier(x).squeeze(-1)


class CNN(nn.Module):
    """Temporal CNN returning one tic-presence logit per time step."""

    def __init__(self, input_dim, hidden_dim=128, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.feature_layers = nn.Sequential(
            _ConvBlock(input_dim, hidden_dim, kernel_size=5),
            nn.Dropout(dropout),
            _ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=2),
            nn.Dropout(dropout),
            _ConvBlock(hidden_dim, hidden_dim, kernel_size=3, dilation=4),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Conv1d(hidden_dim, 1, kernel_size=1)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim)
        x = self.feature_layers(x)
        return self.classifier(x).squeeze(1)


class CNN_BiLSTM(nn.Module):
    """Temporal CNN and BiLSTM returning one tic logit per time step."""

    def __init__(
        self,
        input_dim,
        cnn_dim=128,
        hidden_dim=128,
        num_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.feature_layers = nn.Sequential(
            _ConvBlock(input_dim, cnn_dim, kernel_size=5),
            nn.Dropout(dropout),
            _ConvBlock(cnn_dim, cnn_dim, kernel_size=3, dilation=2),
            nn.Dropout(dropout),
        )
        self.recurrent_layers = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(2 * hidden_dim, 1)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim)
        x = self.feature_layers(x).transpose(1, 2)
        x, _ = self.recurrent_layers(x)
        return self.classifier(x).squeeze(-1)
