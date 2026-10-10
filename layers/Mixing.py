# TSMixer-style mixing layers.

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class TimeBatchNorm2d(nn.BatchNorm1d):
    # BatchNorm over the joint (time, feature) dims of [B, S, C].

    def __init__(self, normalized_shape: tuple[int, int]):
        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected 3D input tensor, but got {x.ndim}D tensor instead.")

        x = x.reshape(x.shape[0], -1, 1)

        x = super().forward(x)

        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)

        return x


class FeatureMixing(nn.Module):
    # MLP across features: [B, S, C_in] -> [B, S, C_out].

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable[[torch.Tensor], torch.Tensor] = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        super().__init__()

        self.norm_before = (
            norm_type((sequence_length, input_channels))
            if normalize_before
            else nn.Identity()
        )
        self.norm_after = (
            norm_type((sequence_length, output_channels))
            if not normalize_before
            else nn.Identity()
        )

        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.projection(x)

        x = self.norm_before(x)

        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x = x_proj + x

        return self.norm_after(x)


class ConditionalFeatureMixing(nn.Module):
    # FeatureMixing with static features concatenated to the input.

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.fr_static = nn.Linear(static_channels, output_channels)
        self.fm = FeatureMixing(
            sequence_length,
            input_channels + output_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(
        self, x: torch.Tensor, x_static: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.fr_static(x_static)
        v = v.unsqueeze(1).repeat(
            1, x.shape[1], 1
        )

        return (
            self.fm(
                torch.cat([x, v], dim=-1)
            ),
            v.detach(),
        )


class TimeMixing(nn.Module):
    # MLP across time, applied per feature: [B, S, C] -> [B, S, C].

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        norm_type: type[nn.Module] = TimeBatchNorm2d,
    ):
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(sequence_length, sequence_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp = feature_to_time(
            x
        )
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_temp = self.dropout(x_temp)
        x_res = time_to_feature(x_temp)

        return self.norm(x + x_res)


class MixerLayer(nn.Module):
    # TimeMixing followed by FeatureMixing.

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = FeatureMixing(
            sequence_length,
            input_channels,
            output_channels,
            ff_dim,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
            normalize_before=normalize_before,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x = self.feature_mixing(x)

        return x


class ConditionalMixerLayer(nn.Module):
    # TimeMixing followed by ConditionalFeatureMixing.

    def __init__(
        self,
        sequence_length: int,
        input_channels: int,
        output_channels: int,
        static_channels: int,
        ff_dim: int,
        activation_fn: Callable = F.relu,
        dropout_rate: float = 0.1,
        normalize_before: bool = False,
        norm_type: type[nn.Module] = nn.LayerNorm,
    ):
        super().__init__()

        self.time_mixing = TimeMixing(
            sequence_length,
            input_channels,
            activation_fn,
            dropout_rate,
            norm_type=norm_type,
        )
        self.feature_mixing = ConditionalFeatureMixing(
            sequence_length,
            input_channels,
            output_channels=output_channels,
            static_channels=static_channels,
            ff_dim=ff_dim,
            activation_fn=activation_fn,
            dropout_rate=dropout_rate,
            normalize_before=normalize_before,
            norm_type=norm_type,
        )

    def forward(self, x: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        x = self.time_mixing(x)
        x, _ = self.feature_mixing(x, x_static)

        return x


def time_to_feature(x: torch.Tensor) -> torch.Tensor:
    return x.permute(0, 2, 1)


feature_to_time = time_to_feature