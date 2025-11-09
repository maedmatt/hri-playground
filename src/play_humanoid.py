"""
Simple policy playback script for Gymnasium locomotion tasks

Usage (repo root):
    uv run python src/play_humanoid.py --model-path models/Humanoid-v5/ppo_latest.zip --env-id Humanoid-v5
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Final

import gymnasium as gym
from stable_baselines3 import PPO

DEFAULT_MODEL: Final = Path("models/Humanoid-v5/ppo_latest.zip")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a trained SB3 policy.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--env-id", type=str, default="Humanoid-v5")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument(
        "--sleep", type=float, default=1 / 40, help="Delay between frames."
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


def load_policy(model_path: Path) -> PPO:
    if not model_path.exists():
        msg = f"Policy checkpoint not found at {model_path}"
        raise FileNotFoundError(msg)
    return PPO.load(str(model_path), device="cpu")


def rollout(
    policy: PPO,
    env_id: str,
    episodes: int,
    max_steps: int,
    sleep: float,
    deterministic: bool,
    seed: int,
    render_mode: str,
) -> None:
    env = gym.make(env_id, render_mode=render_mode)
    try:
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
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    policy = load_policy(args.model_path)
    rollout(
        policy=policy,
        env_id=args.env_id,
        episodes=args.episodes,
        max_steps=args.max_steps,
        sleep=args.sleep,
        deterministic=args.deterministic,
        seed=args.seed,
        render_mode=args.render_mode,
    )


if __name__ == "__main__":
    main()
