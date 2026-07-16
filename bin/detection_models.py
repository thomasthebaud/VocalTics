"""Clip-level tic detection and group classification models."""

import torch
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


def _statistics_pooling(x):
    """Concatenate the temporal mean and standard deviation."""
    return torch.cat((x.mean(dim=-1), x.std(dim=-1, unbiased=False)), dim=1)


class _ConvBlock(nn.Module):
    """One temporal convolution followed by normalization and activation."""

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


class TDNN(nn.Module):
    """X-vector-style network for tic presence and group classification."""

    def __init__(self, input_dim, num_groups):
        super().__init__()
        self.input_dim = input_dim
        self.frame_layers = nn.Sequential(
            _ConvBlock(input_dim, 512, kernel_size=5),
            _ConvBlock(512, 512, kernel_size=3, dilation=2),
            _ConvBlock(512, 512, kernel_size=3, dilation=3),
            _ConvBlock(512, 512, kernel_size=1),
            _ConvBlock(512, 1500, kernel_size=1),
        )
        self.segment_layers = nn.Sequential(
            nn.Linear(3000, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.presence_classifier = nn.Linear(512, 2)
        self.group_classifier = nn.Linear(512, num_groups)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim)
        x = self.frame_layers(x)
        x = _statistics_pooling(x)
        x = self.segment_layers(x)
        return self.presence_classifier(x), self.group_classifier(x)


class _ResidualBlock(nn.Module):
    """Basic one-dimensional residual block used by ResNet34."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(self.layers(x) + self.shortcut(x))


class ResNet34(nn.Module):
    """Temporal ResNet-34 for tic presence and group classification."""

    def __init__(self, input_dim, num_groups):
        super().__init__()
        self.input_dim = input_dim
        self.stem = nn.Sequential(
            nn.Conv1d(
                input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )
        self.in_channels = 64
        self.residual_layers = nn.Sequential(
            self._make_layer(64, blocks=3, stride=1),
            self._make_layer(128, blocks=4, stride=2),
            self._make_layer(256, blocks=6, stride=2),
            self._make_layer(512, blocks=3, stride=2),
        )
        self.segment_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.presence_classifier = nn.Linear(512, 2)
        self.group_classifier = nn.Linear(512, num_groups)

    def _make_layer(self, out_channels, blocks, stride):
        layers = [_ResidualBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        layers.extend(
            _ResidualBlock(out_channels, out_channels) for _ in range(blocks - 1)
        )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim)
        x = self.stem(x)
        x = self.residual_layers(x)
        x = _statistics_pooling(x)
        x = self.segment_layer(x)
        return self.presence_classifier(x), self.group_classifier(x)


class _TemporalResidualBlock(nn.Module):
    """Dilated residual block used by TCNN."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.layers = nn.Sequential(
            _ConvBlock(channels, channels, kernel_size=3, dilation=dilation),
            nn.Dropout(0.2),
            _ConvBlock(channels, channels, kernel_size=3, dilation=dilation),
            nn.Dropout(0.2),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.layers(x))


class TCNN(nn.Module):
    """Temporal CNN for tic presence and group classification."""

    def __init__(self, input_dim, num_groups):
        super().__init__()
        self.input_dim = input_dim
        self.input_layer = _ConvBlock(input_dim, 256, kernel_size=1)
        self.temporal_layers = nn.Sequential(
            *(_TemporalResidualBlock(256, dilation) for dilation in (1, 2, 4, 8))
        )
        self.segment_layer = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )
        self.presence_classifier = nn.Linear(256, 2)
        self.group_classifier = nn.Linear(256, num_groups)

    def forward(self, x):
        x = _prepare_input(x, self.input_dim)
        x = self.input_layer(x)
        x = self.temporal_layers(x)
        x = _statistics_pooling(x)
        x = self.segment_layer(x)
        return self.presence_classifier(x), self.group_classifier(x)
