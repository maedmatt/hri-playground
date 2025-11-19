"""
Play a trained policy on a Gymnasium environment with (PPO/SAC/TD3/A2C), Behavioral Cloning, and DAgger

Example:
    SB3 policy:
        uv run src/play.py --model-path models/SB3/Humanoid-v5/ppo/ppo_latest.zip --env-id Humanoid-v5 --algo ppo

    BC/DAgger policy:
        uv run src/play.py --model-path models/interactive_il/Walker2d-v5/seed42-100epochs/bc_policy.pth --env-id Walker2d-v5 --algo bc
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm

from interactive_il.utils import (
    PredictablePolicy,
    is_torch_checkpoint,
    load_torch_policy,
)

SB3_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}
ALGORITHM_CHOICES = tuple(sorted(tuple(SB3_ALGORITHMS) + ("bc",)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play a trained policy (.zip or .pth)")
    parser.add_argument(
        "--model-path", type=Path, required=True, help="Path to model (.zip or .pth)"
    )
    parser.add_argument(
        "--env-id", type=str, required=True, help="Gymnasium environment ID"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=ALGORITHM_CHOICES,
        help="Algorithm: ppo/a2c/sac/td3 for .zip files, bc for .pth files",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument(
        "--sleep", type=float, default=1 / 40, help="Delay between frames"
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=("human", "rgb_array"),
    )
    return parser.parse_args()


def load_policy(model_path: Path, algo: str, env: Env) -> PredictablePolicy:
    if not model_path.exists():
        msg = f"Policy checkpoint not found at {model_path}"
        raise FileNotFoundError(msg)

    if is_torch_checkpoint(model_path):
        if algo != "bc":
            msg = "Torch checkpoints (.pth) must be loaded with --algo bc"
            raise ValueError(msg)
        if not isinstance(env.observation_space, Box) or not isinstance(
            env.action_space, Box
        ):
            msg = "BC/DAgger policies require continuous (Box) observation and action spaces"
            raise TypeError(msg)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        return load_torch_policy(model_path, obs_dim, act_dim)

    if algo not in SB3_ALGORITHMS:
        msg = f"Unsupported algorithm '{algo}' for SB3 zip checkpoints."
        raise ValueError(msg)

    return SB3_ALGORITHMS[algo].load(str(model_path), device="auto")


def rollout(
    policy: PredictablePolicy,
    env: Env,
    episodes: int,
    max_steps: int,
    sleep: float,
    deterministic: bool,
    seed: int,
    render_mode: str,
) -> None:
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        for step in range(max_steps):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if render_mode != "human":
                env.render()
            time.sleep(sleep)
            if terminated or truncated:
                print(
                    f"Episode {episode + 1} finished after {step + 1} steps "
                    f"(reward {episode_return:.2f})."
                )
                break
        else:
            print(f"Episode {episode + 1} hit max steps ({max_steps}).")


def main() -> None:
    args = parse_args()
    env = gym.make(args.env_id, render_mode=args.render_mode)
    try:
        policy = load_policy(args.model_path, args.algo, env)
        rollout(
            policy=policy,
            env=env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            sleep=args.sleep,
            deterministic=args.deterministic,
            seed=args.seed,
            render_mode=args.render_mode,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
