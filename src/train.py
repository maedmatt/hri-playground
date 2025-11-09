"""
Small RL training helper around Stable-Baselines3

Example:
    uv run python src/train.py --env-id Humanoid-v5 --total-timesteps 500000 --wandb-project rl-play
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
from stable_baselines3.common.vec_env import DummyVecEnv
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
    parser.add_argument("--seed", type=int, default=0)
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
        "--wandb-project", type=str, help="Weights & Biases project name."
    )
    parser.add_argument(
        "--wandb-entity", type=str, help="Weights & Biases entity/workspace."
    )
    parser.add_argument("--wandb-run-name", type=str, help="Optional custom run name.")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=("online", "offline"),
        help="How wandb should operate.",
    )
    parser.add_argument(
        "--learning-rate",
        dest="learning_rate",
        type=float,
        help="Override SB3 learning rate (per algorithm).",
    )
    parser.add_argument(
        "--gamma", dest="gamma", type=float, help="Discount factor override."
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", type=int, help="Batch size override."
    )
    parser.add_argument(
        "--buffer-size",
        dest="buffer_size",
        type=int,
        help="Replay buffer size override (SAC/TD3).",
    )
    return parser.parse_args()


def build_env(env_id: str, seed: int, render_mode: str) -> VecEnv:
    def _make() -> Env[Any, Any]:
        env = gym.make(
            env_id, render_mode=None if render_mode == "none" else render_mode
        )
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return Monitor(env)

    # SB3 expects a VecEnv, so we wrap the single environment with DummyVecEnv.
    return DummyVecEnv([_make])


def build_algorithm(
    args: argparse.Namespace, env: VecEnv, log_dir: Path
) -> BaseAlgorithm:
    algo_cls = ALGORITHMS[args.algo]
    algo_kwargs: dict[str, Any] = {}
    for field in ("learning_rate", "gamma", "batch_size", "buffer_size"):
        value = getattr(args, field)
        if value is not None:
            algo_kwargs[field] = value
    return algo_cls(
        args.policy,
        env,
        tensorboard_log=str(log_dir),
        seed=args.seed,
        device=args.device,
        **algo_kwargs,
    )


def init_wandb(args: argparse.Namespace) -> tuple[Any, list[BaseCallback]]:
    if not args.wandb_project:
        return None, []
    if wandb is None or WandbCallback is None:  # pragma: no cover
        msg = "wandb is not installed but wandb flags were provided."
        raise RuntimeError(msg)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or f"{args.algo}-{args.env_id}",
        mode=args.wandb_mode,
        config=_args_to_config(args),
        sync_tensorboard=True,
    )
    callback = WandbCallback(
        gradient_save_freq=0,
        model_save_path=str(args.log_dir / "wandb"),
        verbose=2,
    )
    return run, [callback]


def _args_to_config(args: argparse.Namespace) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            config[key] = str(value)
        else:
            config[key] = value
    return config


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
    env = build_env(args.env_id, args.seed, args.render_mode)
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
