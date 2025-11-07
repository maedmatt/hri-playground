"""
Play back a locally saved PPO Humanoid-v5 policy.

Usage (from repo root):
    uv run python src/play_humanoid.py --episodes 10 --deterministic
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

import gymnasium as gym
import numpy as np
from gymnasium.core import Env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

MODEL_PATH: Final = Path("models/ppo_humanoid/ppo_humanoid_standup_final.zip")
VECNORM_PATH: Final = Path("models/ppo_humanoid/vecnormalize_final.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Render Humanoid-v5 with a pretrained policy.")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--sleep", type=float, default=1 / 40, help="Delay between frames.")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=("human", "rgb_array"),
    )
    return parser.parse_args()


def load_policy() -> PPO:
    return PPO.load(str(MODEL_PATH), device="cpu")


def build_env(render_mode: str) -> VecEnv:
    def _make_env() -> Env[Any, Any]:
        return gym.make("Humanoid-v5", render_mode=render_mode)

    vec_env: VecEnv = DummyVecEnv([_make_env])
    if not VECNORM_PATH.exists():
        msg = f"Missing VecNormalize stats at {VECNORM_PATH}"
        raise FileNotFoundError(msg)
    vec_env = VecNormalize.load(str(VECNORM_PATH), vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def _extract_first_obs(raw_obs: VecEnvObs) -> np.ndarray:
    if isinstance(raw_obs, np.ndarray):
        return raw_obs
    if isinstance(raw_obs, Sequence):
        return raw_obs[0]
    raise TypeError(f"Unsupported observation type: {type(raw_obs)!r}")


def rollout(
    policy: PPO,
    env: VecEnv,
    episodes: int,
    max_steps: int,
    sleep: float,
    deterministic: bool,
    render_mode: str,
) -> None:
    raw_obs: VecEnvObs = env.reset()
    obs = _extract_first_obs(raw_obs)
    for episode in range(episodes):
        episode_return = 0.0
        for step in range(max_steps):
            action, _ = policy.predict(obs, deterministic=deterministic)
            raw_obs, rewards, dones, _ = env.step(action)
            obs = _extract_first_obs(raw_obs)
            reward_value = float(rewards[0])
            episode_return += reward_value
            if render_mode != "human":
                render_env = getattr(env, "envs", None)
                if render_env:
                    render_env[0].render()
            time.sleep(sleep)
            if dones[0]:
                print(
                    f"Episode {episode + 1} finished after {step + 1} steps "
                    f"(reward {episode_return:.2f})."
                )
                raw_obs = env.reset()
                obs = _extract_first_obs(raw_obs)
                break
        else:
            print(f"Episode {episode + 1} hit max steps ({max_steps}). Resetting...")
            raw_obs = env.reset()
            obs = _extract_first_obs(raw_obs)


def main() -> None:
    args = parse_args()
    policy = load_policy()
    env = build_env(args.render_mode)
    try:
        rollout(
            policy,
            env,
            args.episodes,
            args.max_steps,
            args.sleep,
            args.deterministic,
            args.render_mode,
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
