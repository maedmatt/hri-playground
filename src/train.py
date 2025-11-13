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
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
)
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


class VideoRecordingCallback(BaseCallback):
    """Records policy videos and uploads to wandb every N timesteps."""

    def __init__(
        self,
        env_id: str,
        record_freq: int,
        n_episodes: int = 1,
        video_length: int = 1000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.env_id = env_id
        self.record_freq = record_freq
        self.n_episodes = n_episodes
        self.video_length = video_length
        self.last_record_step = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_record_step >= self.record_freq:
            self._record_video()
            self.last_record_step = self.num_timesteps
        return True

    def _record_video(self) -> None:
        import tempfile

        from gymnasium.wrappers import RecordVideo

        with tempfile.TemporaryDirectory() as tmpdir:
            eval_env = RecordVideo(
                gym.make(self.env_id, render_mode="rgb_array"),
                video_folder=tmpdir,
                episode_trigger=lambda ep: ep < self.n_episodes,
                name_prefix=f"step_{self.num_timesteps}",
            )

            for _ in range(self.n_episodes):
                obs, _ = eval_env.reset()
                done = False
                steps = 0
                while not done and steps < self.video_length:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, _ = eval_env.step(action)
                    done = terminated or truncated
                    steps += 1

            eval_env.close()

            if wandb is not None and wandb.run is not None:
                video_files = list(Path(tmpdir).glob("*.mp4"))
                for video_path in video_files:
                    wandb.log(
                        {
                            "policy_video": wandb.Video(str(video_path), format="mp4"),
                        }
                    )


class WandbCheckpointCallback(CheckpointCallback):
    """Saves checkpoints locally and uploads them to wandb."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_uploaded_timestep = -1

    def _on_step(self) -> bool:
        result = super()._on_step()

        # If parent saved a new checkpoint, upload it
        if (
            self.num_timesteps != self.last_uploaded_timestep
            and self.n_calls % self.save_freq == 0
            and wandb is not None
            and wandb.run is not None
        ):
            checkpoint_path = (
                Path(self.save_path)
                / f"{self.name_prefix}_{self.num_timesteps}_steps.zip"
            )
            if checkpoint_path.exists():
                if self.verbose >= 1:
                    print(f"Uploading checkpoint to wandb: {checkpoint_path.name}")
                wandb.save(
                    str(checkpoint_path),
                    base_path=str(Path(self.save_path).parent),
                    policy="now",
                )
                self.last_uploaded_timestep = self.num_timesteps
            else:
                print(f"Warning: Expected checkpoint not found: {checkpoint_path}")
        return result


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
    parser.add_argument(
        "--seed", type=int, default=42
    )  # “Life, the universe, and everything”!
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
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=250000,
        help="Save checkpoints and videos every N environment timesteps.",
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
        name=f"{args.algo}-{args.env_id}-seed{args.seed}-n_envs{args.n_envs}-{args.total_timesteps}steps",
        group=args.env_id,
        tags=[args.algo, f"seed{args.seed}", f"n_envs{args.n_envs}"],
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
    # Use global_step (environment timesteps) as x-axis for all metrics
    wandb.define_metric("global_step")
    wandb.define_metric("*", step_metric="global_step")

    # Create unique run folder: models/{env_id}/{algo}-seed{seed}-n_envs{n_envs}-{timesteps}steps/
    run_name = (
        f"{args.algo}-seed{args.seed}-n_envs{args.n_envs}-{args.total_timesteps}steps"
    )
    run_dir = args.log_dir / args.env_id / run_name
    checkpoint_dir = run_dir / "checkpoints"

    callbacks = [
        WandbCallback(
            gradient_save_freq=0,
            model_save_freq=0,
            verbose=2,
        ),
        WandbCheckpointCallback(
            save_freq=max(args.checkpoint_freq // args.n_envs, 1),
            save_path=str(checkpoint_dir),
            name_prefix="model",
            verbose=2,
        ),
        VideoRecordingCallback(
            env_id=args.env_id,
            record_freq=args.checkpoint_freq,
            n_episodes=1,
            verbose=2,
        ),
    ]
    return run, callbacks


def resolve_save_path(args: argparse.Namespace) -> Path:
    if args.save_path:
        return args.save_path
    run_name = (
        f"{args.algo}-seed{args.seed}-n_envs{args.n_envs}-{args.total_timesteps}steps"
    )
    default = args.log_dir / args.env_id / run_name / f"{args.algo}_latest.zip"
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
            if wandb is not None:
                wandb.save(
                    str(save_path),
                    base_path=str(save_path.parent),
                    policy="now",
                )
            run.finish()


if __name__ == "__main__":
    main()
