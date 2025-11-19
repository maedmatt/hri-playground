"""DAgger (Dataset Aggregation) trainer."""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from tqdm import trange

from interactive_il.policy import BCPolicy

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]


def load_expert(expert_path: Path, device: str = "auto"):
    """Load expert policy from SB3 checkpoint."""
    from stable_baselines3 import A2C, PPO, SAC, TD3

    ALGOS = {"sac": SAC, "ppo": PPO, "td3": TD3, "a2c": A2C}

    for algo_name, algo_cls in ALGOS.items():
        try:
            model = algo_cls.load(str(expert_path), device=device)
            print(f"Loaded expert ({algo_name}) from {expert_path}")
            return model
        except Exception:
            continue

    msg = f"Could not load expert from {expert_path}"
    raise ValueError(msg)


def collect_trajectory(env, policy, obs_mean, obs_std, max_steps: int = 1000):
    """Collect a single trajectory from the policy."""
    obs, _ = env.reset()
    observations, actions, rewards = [], [], []

    for _ in range(max_steps):
        # Normalize observation
        if obs_mean is not None and obs_std is not None:
            obs_norm = (obs - obs_mean) / obs_std
        else:
            obs_norm = obs

        action = policy.predict(obs_norm)
        observations.append(obs)
        actions.append(action)

        obs, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)

        if terminated or truncated:
            break

    return {
        "observation": np.array(observations, dtype=np.float32),
        "action": np.array(actions, dtype=np.float32),
        "reward": np.array(rewards, dtype=np.float32),
    }


def train_dagger(
    env_id: str,
    expert_path: Path,
    save_path: Path,
    bc_init_path: Path | None = None,
    n_iterations: int = 20,
    n_traj_per_iter: int = 10,
    n_epochs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    use_norm: bool = True,
    use_replay: bool = False,
    buffer_size: int = 200000,
    beta_decay: float = 0.95,
    seed: int = 42,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "hri-playground",
    wandb_name: str | None = None,
):
    """
    Train a DAgger policy.

    Args:
        env_id: Gymnasium environment ID
        expert_path: Path to expert policy checkpoint (.zip)
        save_path: Where to save the trained policy (.pth)
        bc_init_path: Optional BC checkpoint to initialize from
        n_iterations: Number of DAgger iterations
        n_traj_per_iter: Trajectories to collect per iteration
        n_epochs: Training epochs per iteration
        batch_size: Batch size for training
        lr: Learning rate
        use_norm: Whether to normalize observations
        use_replay: Use replay buffer (DAgger+Replay) vs pure DAgger
        buffer_size: Replay buffer size (only if use_replay=True)
        beta_decay: Beta decay rate (beta = beta_decay^iteration)
        seed: Random seed
        device: PyTorch device (cpu, cuda, auto)
        use_wandb: Enable Weights & Biases logging
        wandb_project: W&B project name
        wandb_name: W&B run name (defaults to "dagger-{env_id}")

    Returns:
        Dictionary with training info
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize wandb
    if use_wandb:
        if wandb is None:
            msg = "wandb is not installed but use_wandb=True"
            raise RuntimeError(msg)
        algo_name = "dagger-replay" if use_replay else "dagger"
        wandb.init(
            project=wandb_project,
            name=wandb_name or f"{algo_name}-{env_id}",
            group=env_id,
            tags=[algo_name, f"seed{seed}", f"{n_iterations}iters"],
            config={
                "env_id": env_id,
                "algo": algo_name,
                "n_iterations": n_iterations,
                "n_traj_per_iter": n_traj_per_iter,
                "n_epochs": n_epochs,
                "batch_size": batch_size,
                "lr": lr,
                "use_norm": use_norm,
                "use_replay": use_replay,
                "buffer_size": buffer_size if use_replay else None,
                "beta_decay": beta_decay,
                "seed": seed,
            },
        )

    # Load expert
    expert = load_expert(expert_path, device=device)

    # Create environment and get dimensions
    env = gym.make(env_id)
    if not isinstance(env.observation_space, Box) or not isinstance(
        env.action_space, Box
    ):
        msg = "DAgger requires continuous (Box) observation and action spaces"
        raise TypeError(msg)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Initialize policy
    policy = BCPolicy(obs_dim, act_dim, device=device)

    # Load BC initialization if provided
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    if bc_init_path is not None and bc_init_path.exists():
        checkpoint = torch.load(bc_init_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["state_dict"])
        if use_norm:
            obs_mean = checkpoint.get("mean")
            obs_std = checkpoint.get("std")
        print(f"Initialized from BC checkpoint: {bc_init_path}")
    elif use_norm:
        print(
            "Warning: use_norm=True but no BC checkpoint provided for normalization stats"
        )

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Storage for aggregated dataset (pure DAgger) or replay buffer (DAgger+Replay)
    if use_replay:
        replay_obs: list[np.ndarray] = []
        replay_labels: list[np.ndarray] = []
    else:
        dataset_obs: list[np.ndarray] = []
        dataset_labels: list[np.ndarray] = []

    # Training loop
    iteration_rewards: list[float] = []

    for iteration in trange(n_iterations, desc="DAgger Iterations"):
        beta = beta_decay**iteration

        # Collect trajectories with current policy
        trajs = []
        for _ in range(n_traj_per_iter):
            traj = collect_trajectory(env, policy, obs_mean, obs_std, max_steps=1000)
            trajs.append(traj)

        # Process trajectories: query expert and apply beta-mixing
        new_obs: list[np.ndarray] = []
        new_labels: list[np.ndarray] = []

        for traj in trajs:
            for i in range(len(traj["action"])):
                obs_raw = traj["observation"][i]

                # Normalize observation for storage
                if obs_mean is not None and obs_std is not None:
                    obs_norm = (obs_raw - obs_mean) / obs_std
                else:
                    obs_norm = obs_raw

                # Beta-mixing: with probability beta, use expert action
                if np.random.random() < beta:
                    expert_action, _ = expert.predict(obs_raw, deterministic=True)
                    label = expert_action
                else:
                    label = traj["action"][i]

                new_obs.append(obs_norm)
                new_labels.append(label)

        # Add to dataset/buffer
        if use_replay:
            replay_obs.extend(new_obs)
            replay_labels.extend(new_labels)
            # Keep only last buffer_size samples
            if len(replay_obs) > buffer_size:
                replay_obs = replay_obs[-buffer_size:]
                replay_labels = replay_labels[-buffer_size:]
            train_obs = np.array(replay_obs, dtype=np.float32)
            train_labels = np.array(replay_labels, dtype=np.float32)
        else:
            dataset_obs.extend(new_obs)
            dataset_labels.extend(new_labels)
            train_obs = np.array(dataset_obs, dtype=np.float32)
            train_labels = np.array(dataset_labels, dtype=np.float32)

        # Training on aggregated data
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(train_obs), torch.from_numpy(train_labels)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True
        )

        epoch_losses: list[float] = []
        for _ in range(n_epochs):
            epoch_loss = 0.0
            for obs_batch, label_batch in loader:
                obs_batch = obs_batch.to(device)
                label_batch = label_batch.to(device)
                optimizer.zero_grad()
                pred = policy(obs_batch)
                loss = loss_fn(pred, label_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * obs_batch.size(0)
            epoch_losses.append(epoch_loss / len(dataset))

        avg_loss = np.mean(epoch_losses)

        # Evaluate policy
        eval_trajs = [
            collect_trajectory(env, policy, obs_mean, obs_std, max_steps=1000)
            for _ in range(5)
        ]
        eval_returns = [traj["reward"].sum() for traj in eval_trajs]
        mean_reward = float(np.mean(eval_returns))
        std_reward = float(np.std(eval_returns))
        iteration_rewards.append(mean_reward)

        # Log to wandb
        if use_wandb and wandb is not None:
            wandb.log(
                {
                    "iteration": iteration + 1,
                    "beta": beta,
                    "train/loss": avg_loss,
                    "eval/mean_reward": mean_reward,
                    "eval/std_reward": std_reward,
                    "dataset_size": len(train_obs),
                }
            )

    env.close()

    # Save checkpoint
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "mean": obs_mean,
            "std": obs_std,
            "use_norm": use_norm,
            "use_replay": use_replay,
        },
        save_path,
    )
    print(f"Saved policy to {save_path}")

    if use_wandb and wandb is not None:
        wandb.save(str(save_path), base_path=str(save_path.parent.parent), policy="now")
        wandb.finish()

    return {
        "iteration_rewards": iteration_rewards,
        "final_reward": iteration_rewards[-1] if iteration_rewards else 0.0,
        "n_iterations": n_iterations,
    }
