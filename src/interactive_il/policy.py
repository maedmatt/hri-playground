"""BC/DAgger policy network definition."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class BCPolicy(nn.Module):
    """
    Behavioral Cloning policy network.

    Simple feedforward network with ReLU activations and Tanh output,
    compatible with any continuous control environment.
    """

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: tuple[int, ...] = (256, 256)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = obs_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers += [nn.Linear(d, act_dim), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    @torch.no_grad()
    def predict(self, obs_np: np.ndarray) -> np.ndarray:
        """Deterministic inference from numpy observation."""
        x = torch.as_tensor(obs_np, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        a = self.net(x).cpu().numpy()
        return a[0]
