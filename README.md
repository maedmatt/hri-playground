# Human Robot Interaction Playground

Imitation Learning experiments for Social Robotics Final 2025.

**Authors**: Matteo Calabria, Giulia Benintendi, Stefano Abondio

<img width="1029" height="454" alt="wandb" src="https://github.com/user-attachments/assets/9cbb1b68-8e2a-4cdd-a0d1-c7a8b5605fd4" />

This project implements Behavioral Cloning and DAgger alongside standard RL baselines (PPO, SAC, TD3, A2C) using Gymnasium environments.

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
make install  # Install dependencies
make lint     # Run linting and type checking
make test     # Run tests
```

## Weights & Biases

This project is integrated with [Weights & Biases](https://wandb.ai/) for experiment tracking. Run `wandb login` before your first training session, then use the `--wandb` flag to log metrics, checkpoints, and videos.

## Usage

### 1. Download a pretrained expert

```bash
uv run src/download.py \
  --repo-id farama-minari/Walker2d-v5-SAC-medium \
  --filename walker2d-v5-SAC-medium.zip \
  --env-id Walker2d-v5
```

> **Note**: Additional pretrained experts are available on [HuggingFace](https://huggingface.co/maedmatt).

### 2. Collect demonstrations

```bash
uv run src/collect_demonstrations.py \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/hf/walker2d-v5-SAC-medium.zip \
  --n-episodes 30
# Saves to: datasets/Walker2d-v5/30demos.pkl
```

### 3. Train a policy

**Behavioral Cloning** (learn from demonstrations):
```bash
uv run src/train.py \
  --algo bc \
  --env-id Walker2d-v5 \
  --demos-path datasets/Walker2d-v5/30demos.pkl \
  --total-timesteps 100 \
  --wandb
```

**DAgger** (iterative imitation learning):
```bash
uv run src/train.py \
  --algo dagger \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/hf/walker2d-v5-SAC-medium.zip \
  --n-iterations 20 \
  --wandb
```

**DAgger+Replay** (with replay buffer):
```bash
uv run src/train.py \
  --algo dagger \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/hf/walker2d-v5-SAC-medium.zip \
  --n-iterations 20 \
  --use-replay \
  --wandb
```

**Reinforcement Learning** (PPO/SAC/TD3/A2C):
```bash
uv run src/train.py \
  --algo ppo \
  --env-id Walker2d-v5 \
  --total-timesteps 1000000 \
  --n-envs 8 \
  --wandb
```

### 4. Evaluate a policy

```bash
# SB3 policy (.zip)
uv run src/play.py \
  --env-id Walker2d-v5 \
  --model-path models/SB3/Walker2d-v5/ppo/ppo_latest.zip \
  --algo ppo

# BC/DAgger policy (.pth)
uv run src/play.py \
  --env-id Walker2d-v5 \
  --model-path models/interactive_il/Walker2d-v5/seed42-100epochs/bc_policy.pth \
  --algo bc
```

## Project Structure

```
hri-playground/
├── src/
│   ├── download.py                   # Download HuggingFace models
│   ├── collect_demonstrations.py     # Collect expert demonstrations
│   ├── train.py                      # Train BC/DAgger/SB3 policies
│   ├── play.py                       # Evaluate policies
│   └── interactive_il/               # BC/DAgger implementations
│       ├── bc_trainer.py
│       ├── dagger_trainer.py
│       ├── policy.py
│       └── utils.py
├── tests/                            # Unit tests
├── datasets/                         # Expert demonstrations
├── models/                           # Trained policies
└── docs/                             # Detailed documentation and report
```

## Documentation

For detailed algorithm explanations, experimental setup, and results, see the `docs/` folder.

## License

MIT License - see [LICENSE](LICENSE) for details.
