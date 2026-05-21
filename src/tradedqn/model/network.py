"""Dueling Conv1D Deep Q-Network.

Maps a 30×10 market+portfolio state to 3 Q-values [Sell, Hold, Buy]. Conv1D
convolves the *time* axis (features are channels, never convolved across), then
a Dueling head splits a shared trunk into a scalar state-value V(s) and an
action-advantage A(s,a), recombined as Q = V + A − mean(A). Setting
``dueling=False`` collapses this to a plain DQN (Q = A) — the ablation used to
show the dueling head earns its place (§9 research).
"""

from __future__ import annotations

import torch
from torch import nn


class DuelingDQN(nn.Module):
    def __init__(
        self,
        window: int,
        n_features: int,
        n_actions: int,
        conv_channels: list[int],
        kernel_size: int,
        dense_units: int,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        self.dueling = dueling
        padding = kernel_size // 2  # 'same' length → flatten dim is deterministic
        convs: list[nn.Module] = []
        in_channels = n_features
        for out_channels in conv_channels:
            convs.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding))
            convs.append(nn.ReLU())
            in_channels = out_channels
        self.conv = nn.Sequential(*convs)
        flat_dim = conv_channels[-1] * window
        self.trunk = nn.Sequential(nn.Linear(flat_dim, dense_units), nn.ReLU())
        self.value_head = nn.Linear(dense_units, 1)
        self.advantage_head = nn.Linear(dense_units, n_actions)

    @classmethod
    def from_config(cls, cfg) -> DuelingDQN:
        return cls(
            window=cfg.features.window_size,
            n_features=cfg.features.features_count,
            n_actions=len(vars(cfg.actions)),
            conv_channels=list(cfg.network.conv_channels),
            kernel_size=cfg.network.kernel_size,
            dense_units=cfg.network.dense_units,
            dueling=bool(getattr(cfg.network, "dueling", True)),
        )

    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        # (B, window, features) → (B, features, window) so Conv1D spans time
        features = self.conv(x.transpose(1, 2))
        return self.trunk(torch.flatten(features, start_dim=1))

    def value_advantage(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the (value, advantage) streams — exposed for inspection/tests."""
        hidden = self._trunk(x)
        return self.value_head(hidden), self.advantage_head(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, advantage = self.value_advantage(x)
        if not self.dueling:  # ablation: plain DQN — the advantage head is the Q head
            return advantage
        return value + advantage - advantage.mean(dim=1, keepdim=True)
