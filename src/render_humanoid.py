"""
Minimal example that spins up the Gymnasium Humanoid environment and renders it

Usage (from repo root):
    uv run python src/render_humanoid.py --episodes 10 --max-steps 200
"""

from __future__ import annotations

import argparse
import time

import gymnasium as gym


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Gymnasium Humanoid environment."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="How many episodes to roll out. Each episode ends when the env terminates/truncates.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1_000,
        help="Safety cap on steps per episode to avoid infinite loops.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1 / 40,
        help="Seconds to sleep between frames for easier viewing.",
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=("human", "rgb_array"),
        help="Render mode to request from Gymnasium.",
    )
    return parser.parse_args()


def rollout(
    episodes: int, max_steps: int, frame_delay: float, render_mode: str
) -> None:
    env = gym.make("Humanoid-v5", render_mode=render_mode)
    _obs, _ = env.reset(seed=0)
    try:
        for episode in range(episodes):
            for step in range(max_steps):
                # Random policy is enough to demonstrate rendering.
                action = env.action_space.sample()
                _obs, reward, terminated, truncated, _ = env.step(action)
                if render_mode != "human":
                    env.render()
                time.sleep(frame_delay)
                if terminated or truncated:
                    print(
                        f"Episode {episode + 1} finished after {step + 1} steps (reward {reward:.2f})."
                    )
                    _obs, _ = env.reset()
                    break
            else:
                print(
                    f"Episode {episode + 1} hit max steps ({max_steps}). Resetting..."
                )
                _obs, _ = env.reset()
    finally:
        env.close()


def main() -> None:
    args = parse_args()
    rollout(args.episodes, args.max_steps, args.sleep, args.render_mode)


if __name__ == "__main__":
    main()
