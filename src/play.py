"""
Play a trained policy on a Gymnasium environment with (PPO/SAC/TD3/A2C), Behavioral Cloning, and DAgger

Example:
    SB3 policy:
        uv run src/play.py --model-path models/SB3/Humanoid-v5/huggingface/humanoid-v5-sac-expert.zip --env-id Humanoid-v5 --algo ppo

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

try:
    import wandb
except ModuleNotFoundError:
    wandb = None  # type: ignore[assignment]

SB3_ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}
ALGORITHM_CHOICES = tuple(sorted(tuple(SB3_ALGORITHMS) + ("bc", "dagger")))


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
        help="Algorithm: ppo/a2c/sac/td3 for .zip files, bc/dagger for .pth files",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument(
        "--sleep", type=float, default=1 / 60, help="Delay between frames"
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=("human", "rgb_array", "none"),
        help="Render mode: 'human' for window, 'rgb_array' for offscreen, 'none' for no rendering",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project", type=str, default="hri-playground", help="Wandb project name"
    )
    parser.add_argument("--wandb-entity", type=str, help="Wandb entity/username")
    return parser.parse_args()


def load_policy(model_path: Path, algo: str, env: Env) -> PredictablePolicy:
    if not model_path.exists():
        msg = f"Policy checkpoint not found at {model_path}"
        raise FileNotFoundError(msg)

    if is_torch_checkpoint(model_path):
        if algo not in ("bc", "dagger"):
            msg = "Torch checkpoints (.pth) must be loaded with --algo bc or --algo dagger"
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


def init_wandb(
    project: str,
    entity: str | None,
    model_path: Path,
    env_id: str,
    algo: str,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    seed: int,
) -> None:
    if wandb is None:
        msg = "wandb not installed. Install with: uv add wandb"
        raise ImportError(msg)

    wandb.init(
        project=project,
        entity=entity,
        name=f"eval-{algo}-{env_id}",
        group=env_id,
        tags=["eval", algo, f"seed{seed}"],
        config={
            "model_path": str(model_path),
            "env_id": env_id,
            "algo": algo,
            "episodes": episodes,
            "max_steps": max_steps,
            "deterministic": deterministic,
            "seed": seed,
        },
        job_type="eval",
    )


def rollout(
    policy: PredictablePolicy,
    env: Env,
    episodes: int,
    max_steps: int,
    sleep: float,
    deterministic: bool,
    seed: int,
    render_mode: str,
    use_wandb: bool,
) -> dict[str, list[float | int | bool]]:
    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    episode_terminated: list[bool] = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        done_by_termination = False
        for step in range(max_steps):
            action, _ = policy.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            if render_mode == "human":
                time.sleep(sleep)
            if terminated or truncated:
                done_by_termination = terminated
                print(
                    f"Episode {episode + 1} finished after {step + 1} steps "
                    f"(reward {episode_return:.2f})."
                )
                break
        else:
            print(f"Episode {episode + 1} hit max steps ({max_steps}).")
            step = max_steps - 1

        episode_rewards.append(episode_return)
        episode_lengths.append(step + 1)
        episode_terminated.append(done_by_termination)

        if use_wandb:
            wandb.log(  # type: ignore[union-attr]
                {
                    "eval/episode_reward": episode_return,
                    "eval/episode_length": step + 1,
                    "eval/terminated": done_by_termination,
                    "episode": episode,
                }
            )

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "terminated": episode_terminated,
    }


def log_aggregate_stats(
    stats: dict[str, list[float | int | bool]], use_wandb: bool
) -> None:
    import numpy as np

    rewards = np.array(stats["rewards"])
    lengths = np.array(stats["lengths"])

    if use_wandb:
        wandb.log(  # type: ignore[union-attr]
            {
                "eval/mean_reward": float(np.mean(rewards)),
                "eval/std_reward": float(np.std(rewards)),
                "eval/min_reward": float(np.min(rewards)),
                "eval/max_reward": float(np.max(rewards)),
                "eval/mean_length": float(np.mean(lengths)),
                "eval/std_length": float(np.std(lengths)),
            }
        )

    print(f"\nAggregate statistics over {len(rewards)} episodes:")
    print(
        f"  Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f} "
        f"[{np.min(rewards):.2f}, {np.max(rewards):.2f}]"
    )
    print(
        f"  Length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f} "
        f"[{int(np.min(lengths))}, {int(np.max(lengths))}]"
    )


def main() -> None:
    args = parse_args()

    if args.wandb:
        init_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            model_path=args.model_path,
            env_id=args.env_id,
            algo=args.algo,
            episodes=args.episodes,
            max_steps=args.max_steps,
            deterministic=args.deterministic,
            seed=args.seed,
        )

    env = gym.make(
        args.env_id,
        render_mode=None if args.render_mode == "none" else args.render_mode,
        max_episode_steps=args.max_steps,
    )
    try:
        policy = load_policy(args.model_path, args.algo, env)
        stats = rollout(
            policy=policy,
            env=env,
            episodes=args.episodes,
            max_steps=args.max_steps,
            sleep=args.sleep,
            deterministic=args.deterministic,
            seed=args.seed,
            render_mode=args.render_mode,
            use_wandb=args.wandb,
        )

        log_aggregate_stats(stats, use_wandb=args.wandb)
    finally:
        env.close()
        if args.wandb:
            wandb.finish()  # type: ignore[union-attr]


if __name__ == "__main__":
    main()
