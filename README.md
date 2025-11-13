# Human Robot Interaction Playground

Reinforcement learning experiments with Stable-Baselines3 and Gymnasium. Imitation learning support coming soon.

## Setup

```bash
uv sync
```

## Train

```bash
# Train with default settings
uv run src/train.py

# Train with wandb logging
uv run src/train.py --env-id HalfCheetah-v5 --algo ppo --wandb

# Customize training
uv run src/train.py --env-id Humanoid-v5 --algo sac --n-envs 8 --total-timesteps 2000000
```

## Play

```bash
uv run src/play.py --model-path models/HalfCheetah-v5/ppo-seed42-n_envs8-1000000steps/ppo_latest.zip --env-id HalfCheetah-v5 --algo ppo
```

## Key Arguments

**Training:**
- `--env-id`: Gymnasium environment (default: `Humanoid-v5`)
- `--algo`: `ppo`, `sac`, `td3`, `a2c` (default: `ppo`)
- `--total-timesteps`: Training steps (default: `1000000`)
- `--n-envs`: Parallel environments (default: `1`)
- `--checkpoint-freq`: Save checkpoints/videos every N steps (default: `250000`)
- `--wandb`: Enable wandb logging

**Playing:**
- `--model-path`: Path to trained model (required)
- `--env-id`: Gymnasium environment (required)
- `--algo`: Algorithm used (required)
- `--episodes`: Number of episodes (default: `5`)
