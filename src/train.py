"""
Small RL training helper around Stable-Baselines3

Example:
    uv run python src/train.py --env-id Humanoid-v5 --total-timesteps 500000 --n-envs 8 --wandb
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium.core import Env
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

try:  # pragma: no cover - wandb is optional at runtime
    from wandb.integration.sb3 import WandbCallback

    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None
    WandbCallback = None  # type: ignore[assignment]


EnvFactory = Callable[[], Env[Any, Any]]

ALGORITHMS: dict[str, type[BaseAlgorithm]] = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "td3": TD3,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Gymnasium agent with SB3.")
    parser.add_argument(
        "--env-id", type=str, default="Humanoid-v5", help="Gymnasium env id."
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="ppo",
        choices=sorted(ALGORITHMS),
        help="Which Stable-Baselines3 algorithm to use.",
    )
    parser.add_argument(
        "--policy", type=str, default="MlpPolicy", help="Policy class for SB3."
    )
    parser.add_argument("--total-timesteps", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42) # “Life, the universe, and everything”!
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Number of parallel environments for vectorized training.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="PyTorch device argument."
    )
    parser.add_argument(
        "--render-mode",
        type=str,
        default="none",
        choices=("none", "human", "rgb_array"),
        help="Pass-through render mode requested from Gymnasium.",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=Path("runs/tensorboard"),
        help="Directory for tensorboard summaries.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        help="Where to store the trained policy. Defaults to models/<env>/<algo>.zip.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("models"),
        help="Base directory for checkpoints when --save-path is not provided.",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging.",
    )
    return parser.parse_args()


def build_env(env_id: str, seed: int, render_mode: str, n_envs: int) -> VecEnv:
    def make_env(rank: int) -> EnvFactory:
        def _make() -> Env[Any, Any]:
            env = gym.make(
                env_id, render_mode=None if render_mode == "none" else render_mode
            )
            env.reset(seed=seed + rank)
            env.action_space.seed(seed + rank)
            return Monitor(env)

        return _make

    env_fns = [make_env(i) for i in range(n_envs)]

    if n_envs == 1:
        return DummyVecEnv(env_fns)
    else:
        return SubprocVecEnv(env_fns)


def build_algorithm(
    args: argparse.Namespace, env: VecEnv, log_dir: Path
) -> BaseAlgorithm:
    algo_cls = ALGORITHMS[args.algo]
    return algo_cls(  # pyright: ignore[reportCallIssue]
        args.policy,
        env,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
    )


def init_wandb(args: argparse.Namespace) -> tuple[Any, list[BaseCallback]]:
    if not args.wandb:
        return None, []
    if wandb is None or WandbCallback is None:  # pragma: no cover
        msg = "wandb is not installed but --wandb flag was provided."
        raise RuntimeError(msg)
    run = wandb.init(
        project="hri-playground",
        name=f"{args.algo}-{args.env_id}-seed{args.seed}-n_evs{args.n_envs}",
        config={
            "env_id": args.env_id,
            "algo": args.algo,
            "policy": args.policy,
            "total_timesteps": args.total_timesteps,
            "seed": args.seed,
            "n_envs": args.n_envs,
        },
        sync_tensorboard=True,
    )
    callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=str(args.log_dir / "wandb"),
        verbose=2,
    )
    return run, [callback]


def resolve_save_path(args: argparse.Namespace) -> Path:
    if args.save_path:
        return args.save_path
    default = args.log_dir / args.env_id / f"{args.algo}_latest.zip"
    default.parent.mkdir(parents=True, exist_ok=True)
    return default


def save_model(model: BaseAlgorithm, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def main() -> None:
    args = parse_args()
    tensorboard_log = args.tensorboard_log
    tensorboard_log.mkdir(parents=True, exist_ok=True)
    env = build_env(args.env_id, args.seed, args.render_mode, args.n_envs)
    model = build_algorithm(args, env, tensorboard_log)
    run, callbacks = init_wandb(args)
    save_path = resolve_save_path(args)
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            progress_bar=True,
            callback=callbacks or None,
        )
    finally:
        save_model(model, save_path)
        env.close()
        if run is not None:
            run.finish()


if __name__ == "__main__":
    main()
