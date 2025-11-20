"""
Collect expert demonstrations from a trained policy.

Example:
    Collect 30 demos from Walker2d expert:
        uv run src/collect_demonstrations.py --env-id Walker2d-v5 --expert-path models/SB3/Walker2d-v5/huggingface/walker2d-v5-SAC-medium.zip --n-episodes 30

    Custom save path:
        uv run src/collect_demonstrations.py --env-id Walker2d-v5 --expert-path expert.zip --n-episodes 50 --save-path datasets/Walker2d-v5/my_demos.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from tqdm import tqdm


def load_expert(expert_path: Path) -> BaseAlgorithm:
    """Load expert policy from checkpoint."""
    for algo_cls in [SAC, PPO, TD3, A2C]:
        try:
            return algo_cls.load(expert_path)
        except Exception:
            continue
    msg = f"Failed to load expert from {expert_path}"
    raise ValueError(msg)


def collect_single_trajectory(
    expert: BaseAlgorithm, env: gym.Env, max_steps: int = 1000
) -> dict:
    """Collect one trajectory from the expert policy."""
    obs, _ = env.reset()
    obs_buf, act_buf, rew_buf = [], [], []
    done, steps = False, 0

    while not done and steps < max_steps:
        action, _ = expert.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        obs_buf.append(obs)
        act_buf.append(action)
        rew_buf.append(reward)

        obs = next_obs
        steps += 1

    return {
        "observations": np.array(obs_buf, np.float32),
        "actions": np.array(act_buf, np.float32),
        "rewards": np.array(rew_buf, np.float32),
    }


def collect_demonstrations(
    env_id: str,
    expert_path: Path,
    n_episodes: int = 30,
    max_steps: int = 1000,
) -> list[dict]:
    """Collect multiple demonstrations from the expert policy."""
    expert = load_expert(expert_path)
    env = gym.make(env_id)

    demonstrations = []
    total_rewards = []

    for _ in tqdm(range(n_episodes), desc="Collecting demonstrations"):
        traj = collect_single_trajectory(expert, env, max_steps)
        demonstrations.append(traj)
        total_rewards.append(traj["rewards"].sum())

    env.close()

    rewards_array = np.array(total_rewards)
    print(f"\nCollected {n_episodes} episodes")
    print(
        f"Reward: {rewards_array.mean():.2f} ± {rewards_array.std():.2f} "
        f"(min: {rewards_array.min():.2f}, max: {rewards_array.max():.2f})"
    )

    return demonstrations


def save_demonstrations(demos: list[dict], save_path: Path) -> None:
    """Save demonstrations to disk."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("wb") as f:
        pickle.dump(demos, f)
    print(f"Saved to: {save_path}")


def load_demonstrations(filepath: Path) -> list[dict]:
    """Load demonstrations from disk."""
    with filepath.open("rb") as f:
        return pickle.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect expert demonstrations from a trained policy"
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Gymnasium environment ID (e.g., Walker2d-v5)",
    )
    parser.add_argument(
        "--expert-path",
        type=Path,
        required=True,
        help="Path to expert policy checkpoint",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=30,
        help="Number of demonstration episodes to collect",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Where to save demonstrations (defaults to datasets/<env-id>/<n>demos.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.save_path:
        save_path = args.save_path
    else:
        save_path = Path("datasets") / args.env_id / f"{args.n_episodes}demos.pkl"

    demos = collect_demonstrations(
        env_id=args.env_id,
        expert_path=args.expert_path,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
    )

    save_demonstrations(demos, save_path)


if __name__ == "__main__":
    main()
