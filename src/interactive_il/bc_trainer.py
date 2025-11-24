"""Behavioral Cloning trainer."""

from __future__ import annotations

import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from tqdm import trange

from interactive_il.policy import BCPolicy, resolve_device
from interactive_il.utils import make_bc_wandb_name, make_bc_wandb_tags

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]


def load_demonstrations(demos_path: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load demonstrations from a pickle file.

    Expected format: list of dicts with keys 'observations' and 'actions'.

    Returns:
        observations, actions, number of demonstrations
    """
    with open(demos_path, "rb") as f:
        demos = pickle.load(f)

    n_demos = len(demos)
    obs = np.concatenate([d["observations"] for d in demos], axis=0).astype(np.float32)
    acts = np.concatenate([d["actions"] for d in demos], axis=0).astype(np.float32)

    return obs, acts, n_demos


def compute_normalization(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and std for observation normalization."""
    mean = obs.mean(0).astype(np.float32)
    std = obs.std(0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def train_bc(
    env_id: str,
    demos_path: Path,
    n_epochs: int = 100,
    batch_size: int = 1024,
    lr: float = 3e-4,
    use_norm: bool = True,
    seed: int = 42,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "hri-playground",
    wandb_name: str | None = None,
) -> dict[str, float | list[float]]:
    """
    Train a Behavioral Cloning policy.

    The trained policy is saved to:
    models/interactive_il/{env_id}/bc/bc_{n_demos}demos_{n_epochs}epochs.pth

    Args:
        env_id: Gymnasium environment ID
        demos_path: Path to demonstrations pickle file
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        use_norm: Whether to normalize observations
        seed: Random seed
        device: PyTorch device (cpu, cuda, auto)
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (defaults to "bc-{env_id}")

    Returns:
        Dictionary with training info (losses, final loss, etc.)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = resolve_device(device)

    # Load demonstrations
    obs, acts, n_demos = load_demonstrations(demos_path)
    print(f"Loaded {n_demos} demonstrations ({len(obs)} transitions) from {demos_path}")

    # Initialize wandb
    if use_wandb:
        if wandb is None:
            msg = "wandb is not installed but use_wandb=True"
            raise RuntimeError(msg)
        wandb.init(
            project=wandb_project,
            name=wandb_name or make_bc_wandb_name(n_demos),
            group=env_id,
            tags=make_bc_wandb_tags(seed, n_epochs, n_demos),
            config={
                "env_id": env_id,
                "algo": "bc",
                "n_epochs": n_epochs,
                "n_demos": n_demos,
                "n_transitions": len(obs),
                "batch_size": batch_size,
                "lr": lr,
                "use_norm": use_norm,
                "seed": seed,
            },
        )

    # Compute normalization
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    if use_norm:
        obs_mean, obs_std = compute_normalization(obs)
        obs = (obs - obs_mean) / obs_std

    # Get environment dimensions
    env = gym.make(env_id)
    if not isinstance(env.observation_space, Box) or not isinstance(
        env.action_space, Box
    ):
        msg = "BC requires continuous (Box) observation and action spaces"
        raise TypeError(msg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    env.close()

    # Create policy and optimizer
    policy = BCPolicy(obs_dim, act_dim, device=device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(obs), torch.from_numpy(acts)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )

    # Training loop
    losses: list[float] = []
    for epoch in trange(1, n_epochs + 1, desc="Training BC"):
        epoch_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            optimizer.zero_grad()
            pred = policy(obs_batch)
            loss = loss_fn(pred, act_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * obs_batch.size(0)

        avg_loss = epoch_loss / len(dataset)
        losses.append(avg_loss)

        if use_wandb and wandb is not None:
            wandb.log({"epoch": epoch, "train/loss": avg_loss})

    # Construct save path: models/interactive_il/{env_id}/bc/bc_{n_demos}demos_{n_epochs}epochs.pth
    save_dir = Path("models/interactive_il") / env_id / "bc"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"bc_{n_demos}demos_{n_epochs}epochs.pth"

    torch.save(
        {
            "state_dict": policy.state_dict(),
            "mean": obs_mean,
            "std": obs_std,
            "use_norm": use_norm,
        },
        save_path,
    )
    print(f"Saved policy to {save_path}")

    if use_wandb and wandb is not None:
        wandb.save(str(save_path), base_path=str(save_path.parent.parent), policy="now")
        wandb.finish()

    return {
        "losses": losses,
        "final_loss": losses[-1],
        "n_epochs": n_epochs,
    }
