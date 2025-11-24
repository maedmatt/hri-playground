from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from interactive_il.policy import BCPolicy, resolve_device


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

    device = resolve_device(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"]
    obs_mean = checkpoint.get("mean")
    obs_std = checkpoint.get("std")
    use_norm = bool(checkpoint.get("use_norm", False))

    policy = BCPolicy(obs_dim, act_dim, device=device)
    policy.load_state_dict(state_dict)
    policy.eval()

    return TorchPolicyAdapter(policy, obs_mean, obs_std, use_norm)


def make_bc_wandb_name(n_demos: int) -> str:
    """Construct wandb run name for BC: bc_30demos"""
    return f"bc_{n_demos}demos"


def make_bc_wandb_tags(seed: int, n_epochs: int, n_demos: int) -> list[str]:
    """Construct wandb tags for BC."""
    return ["bc", f"seed{seed}", f"{n_epochs}epochs", f"{n_demos}demos"]


def make_dagger_wandb_name(
    n_iterations: int, use_replay: bool, n_critical_states: int
) -> str:
    """
    Construct wandb run name for DAgger/DAgger-Replay.

    - DAgger: dagger_20iters
    - DAgger-Replay: dagger-replay_20iters_k100
    """
    algo = "dagger-replay" if use_replay else "dagger"
    name = f"{algo}_{n_iterations}iters"
    if use_replay:
        name += f"_k{n_critical_states}"
    return name


def make_dagger_wandb_tags(
    seed: int,
    n_iterations: int,
    n_demos_str: str,
    use_replay: bool,
    n_critical_states: int,
) -> list[str]:
    """Construct wandb tags for DAgger/DAgger-Replay."""
    algo = "dagger-replay" if use_replay else "dagger"
    tags = [algo, f"seed{seed}", f"{n_iterations}iters", f"{n_demos_str}demos"]
    if use_replay:
        tags.append(f"k{n_critical_states}")
    return tags


def parse_policy_filename(
    filename: str,
) -> dict[str, int | str | bool] | None:
    """
    Parse policy filename to extract training parameters.

    Returns dict with keys: algo, n_demos, n_epochs/n_iterations, n_critical_states
    Returns None if pattern doesn't match.
    """
    import re

    # BC: bc_{n_demos}demos_{n_epochs}epochs.pth
    bc_match = re.match(r"bc_(\d+)demos_(\d+)epochs\.pth", filename)
    if bc_match:
        return {
            "algo": "bc",
            "n_demos": int(bc_match.group(1)),
            "n_epochs": int(bc_match.group(2)),
        }

    # DAgger: dagger_{n_demos}demos_{n_iterations}iters.pth
    dagger_match = re.match(r"dagger_((?:\d+)|unknown)demos_(\d+)iters\.pth", filename)
    if dagger_match:
        n_demos_str = dagger_match.group(1)
        return {
            "algo": "dagger",
            "n_demos": n_demos_str,
            "n_iterations": int(dagger_match.group(2)),
            "use_replay": False,
        }

    # DAgger-Replay: dagger-replay_{n_demos}demos_{n_iterations}iters_k{n_critical_states}.pth
    dagger_replay_match = re.match(
        r"dagger-replay_((?:\d+)|unknown)demos_(\d+)iters_k(\d+)\.pth", filename
    )
    if dagger_replay_match:
        n_demos_str = dagger_replay_match.group(1)
        return {
            "algo": "dagger-replay",
            "n_demos": n_demos_str,
            "n_iterations": int(dagger_replay_match.group(2)),
            "n_critical_states": int(dagger_replay_match.group(3)),
            "use_replay": True,
        }

    return None
