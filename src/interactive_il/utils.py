from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from interactive_il.policy import BCPolicy


class PredictablePolicy(Protocol):
    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, object]:
        """Return an action and optional state info."""


class TorchPolicyAdapter:
    """Wrap a BCPolicy checkpoint to mimic the SB3 predict interface."""

    def __init__(
        self,
        policy: BCPolicy,
        obs_mean: np.ndarray | None,
        obs_std: np.ndarray | None,
        use_norm: bool,
    ) -> None:
        self._policy = policy
        self._obs_mean = obs_mean
        self._obs_std = obs_std
        self._use_norm = use_norm

    def predict(
        self, observation: np.ndarray, deterministic: bool = False
    ) -> tuple[np.ndarray, None]:
        obs_array = np.asarray(observation, dtype=np.float32)
        if self._use_norm and self._obs_mean is not None and self._obs_std is not None:
            obs_array = (obs_array - self._obs_mean) / self._obs_std
        action = self._policy.predict(obs_array)
        return action, None


def is_torch_checkpoint(model_path: Path) -> bool:
    return model_path.suffix == ".pth"


def load_torch_policy(
    model_path: Path, obs_dim: int, act_dim: int, device: str = "auto"
) -> TorchPolicyAdapter:
    """Load a BC/DAgger checkpoint stored as a Torch .pth file."""
    if not model_path.exists():
        msg = f"Policy checkpoint not found at {model_path}"
        raise FileNotFoundError(msg)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    obs_mean = checkpoint.get("mean")
    obs_std = checkpoint.get("std")
    use_norm = bool(checkpoint.get("use_norm", False))

    policy = BCPolicy(obs_dim, act_dim, device=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    return TorchPolicyAdapter(policy, obs_mean, obs_std, use_norm)
