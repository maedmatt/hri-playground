"""DAgger (Dataset Aggregation) trainer."""

from __future__ import annotations

import re
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Box
from tqdm import trange

from interactive_il.policy import BCPolicy, resolve_device

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]


def parse_n_demos_from_bc_path(bc_path: Path) -> int | None:
    """
    Parse number of demonstrations from BC checkpoint filename.

    Expected format: bc_{n_demos}demos_{n_epochs}epochs.pth
    Returns None if pattern doesn't match.
    """
    match = re.search(r"bc_(\d+)demos_", bc_path.name)
    return int(match.group(1)) if match else None


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
    n_critical_states: int = 10,
    seed: int = 42,
    device: str = "auto",
    use_wandb: bool = False,
    wandb_project: str = "hri-playground",
    wandb_name: str | None = None,
):
    """
    Train a DAgger policy.

    The trained policy is saved to:
    - DAgger: models/interactive_il/{env_id}/dagger/dagger_{n_demos}demos_{n_iterations}iters.pth
    - DAgger-Replay: models/interactive_il/{env_id}/dagger-replay/dagger-replay_{n_demos}demos_{n_iterations}iters_k{n_critical_states}.pth

    Where n_demos is parsed from bc_init_path filename, or "unknown" if not provided.

    Args:
        env_id: Gymnasium environment ID
        expert_path: Path to expert policy checkpoint (.zip)
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
        n_critical_states: Number of highest-error states per trajectory to add to replay buffer
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
    device = resolve_device(device)

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
                "n_critical_states": n_critical_states if use_replay else None,
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

    # Load BC initialization if provided and parse n_demos
    obs_mean: np.ndarray | None = None
    obs_std: np.ndarray | None = None
    n_demos_str = "unknown"
    if bc_init_path is not None and bc_init_path.exists():
        checkpoint = torch.load(bc_init_path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["state_dict"])
        if use_norm:
            obs_mean = checkpoint.get("mean")
            obs_std = checkpoint.get("std")
        n_demos = parse_n_demos_from_bc_path(bc_init_path)
        if n_demos is not None:
            n_demos_str = f"{n_demos}"
        print(f"Initialized from BC checkpoint: {bc_init_path} ({n_demos_str} demos)")
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

        # Process trajectories
        if use_replay:
            # DAgger+Replay: collect D_new (beta-mixed) and D_crit (expert labels)
            d_new_obs: list[np.ndarray] = []
            d_new_labels: list[np.ndarray] = []
            d_crit_obs: list[np.ndarray] = []
            d_crit_labels: list[np.ndarray] = []

            for traj in trajs:
                traj_obs: list[np.ndarray] = []
                traj_labels: list[np.ndarray] = []
                traj_errors: list[float] = []

                for i in range(len(traj["action"])):
                    obs_raw = traj["observation"][i]
                    policy_action = traj["action"][i]

                    if obs_mean is not None and obs_std is not None:
                        obs_norm = (obs_raw - obs_mean) / obs_std
                    else:
                        obs_norm = obs_raw

                    # Query expert and compute error
                    expert_action, _ = expert.predict(obs_raw, deterministic=True)
                    error = float(np.sum((policy_action - expert_action) ** 2))

                    # Beta-mixing for D_new
                    if np.random.random() < beta:
                        label = expert_action
                    else:
                        label = policy_action

                    d_new_obs.append(obs_norm)
                    d_new_labels.append(label)

                    # Track for critical state selection
                    traj_obs.append(obs_norm)
                    traj_labels.append(expert_action)
                    traj_errors.append(error)

                # Select top-k highest-error states from trajectory
                k = min(n_critical_states, len(traj_errors))
                if k > 0:
                    top_k_indices = np.argsort(traj_errors)[-k:]
                    for idx in top_k_indices:
                        d_crit_obs.append(traj_obs[idx])
                        d_crit_labels.append(traj_labels[idx])

            # Combine D_new with replay buffer for training
            combined_obs = d_new_obs + replay_obs
            combined_labels = d_new_labels + replay_labels

            if len(combined_obs) > 0:
                train_obs = np.array(combined_obs, dtype=np.float32)
                train_labels = np.array(combined_labels, dtype=np.float32)
                dataset = torch.utils.data.TensorDataset(
                    torch.from_numpy(train_obs), torch.from_numpy(train_labels)
                )
                loader = torch.utils.data.DataLoader(
                    dataset, batch_size=batch_size, shuffle=True, drop_last=True
                )

                combined_epoch_losses: list[float] = []
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
                    combined_epoch_losses.append(epoch_loss / len(dataset))
                avg_loss = np.mean(combined_epoch_losses)
            else:
                avg_loss = 0.0

            # Add D_crit to replay buffer for next iteration
            replay_obs.extend(d_crit_obs)
            replay_labels.extend(d_crit_labels)
            if len(replay_obs) > buffer_size:
                replay_obs = replay_obs[-buffer_size:]
                replay_labels = replay_labels[-buffer_size:]
        else:
            # Pure DAgger: aggregate all data
            new_obs: list[np.ndarray] = []
            new_labels: list[np.ndarray] = []

            for traj in trajs:
                for i in range(len(traj["action"])):
                    obs_raw = traj["observation"][i]

                    if obs_mean is not None and obs_std is not None:
                        obs_norm = (obs_raw - obs_mean) / obs_std
                    else:
                        obs_norm = obs_raw

                    # Beta-mixing
                    if np.random.random() < beta:
                        expert_action, _ = expert.predict(obs_raw, deterministic=True)
                        label = expert_action
                    else:
                        label = traj["action"][i]

                    new_obs.append(obs_norm)
                    new_labels.append(label)

            dataset_obs.extend(new_obs)
            dataset_labels.extend(new_labels)
            train_obs = np.array(dataset_obs, dtype=np.float32)
            train_labels = np.array(dataset_labels, dtype=np.float32)

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
            log_dict = {
                "iteration": iteration + 1,
                "beta": beta,
                "train/loss": avg_loss,
                "eval/mean_reward": mean_reward,
                "eval/std_reward": std_reward,
            }
            if use_replay:
                log_dict["replay_buffer_size"] = len(replay_obs)
            else:
                log_dict["dataset_size"] = len(train_obs)
            wandb.log(log_dict)

    env.close()

    # Construct save path
    algo_name = "dagger-replay" if use_replay else "dagger"
    save_dir = Path("models/interactive_il") / env_id / algo_name
    save_dir.mkdir(parents=True, exist_ok=True)

    if use_replay:
        filename = f"{algo_name}_{n_demos_str}demos_{n_iterations}iters_k{n_critical_states}.pth"
    else:
        filename = f"{algo_name}_{n_demos_str}demos_{n_iterations}iters.pth"

    save_path = save_dir / filename

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
