# Human Robot Interaction Playground

Imitation Learning experiments for Social Robotics Final 2025.

**Authors**: Matteo Calabria, Giulia Benintendi, Stefano Abondio

<table>
  <tr>
    <td align="center">
      <b>Expert (SAC)</b><br>
      <img src="assets/gifs/SAC_Humanoid-v5.gif" alt="Expert SAC" width="400">
    </td>
    <td align="center">
      <b>BC (30 demos, 100 epochs)</b><br>
      <img src="assets/gifs/bc_Humanoid-v5.gif" alt="BC" width="400">
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>DAgger (20 iterations)</b><br>
      <img src="assets/gifs/dagger_Humanoid-v5.gif" alt="DAgger" width="400">
    </td>
    <td align="center">
      <b>DAgger+Replay (20 iterations)</b><br>
      <img src="assets/gifs/dagger-replay_Humanoid-v5.gif" alt="DAgger+Replay" width="400">
    </td>
  </tr>
</table>

This project implements Behavioral Cloning (BC), DAgger, and DAgger+Replay for imitation learning, alongside standard RL baselines (PPO, SAC, TD3, A2C) using Gymnasium environments.

For detailed algorithm explanations, architecture details, and experimental results, see the `docs/` folder.

## Installation

Requires Python 3.11+ and [uv](https://github.com/astral-sh/uv).

```bash
make install  # Install dependencies
make lint     # Run linting and type checking
make test     # Run tests
```

> **Note**: For additional information about development tools and setup, see [development.md](development.md).

## Weights & Biases

This project integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking. Run `wandb login` before your first session, then add the `--wandb` flag to log metrics, checkpoints, and videos.

## Usage

### 1. Download a pretrained expert

```bash
uv run src/download.py \
  --repo-id maedmatt/Walker2d-v5-SAC-expert \
  --filename Walker2d-v5-SAC-expert.zip \
  --env-id Walker2d-v5
# Saves to: models/SB3/Walker2d-v5/huggingface/Walker2d-v5-SAC-expert.zip
```

> **Note**: Additional pretrained experts are available on [HuggingFace](https://huggingface.co/maedmatt).

### 2. Collect demonstrations

Collect expert demonstrations for BC training.

```bash
uv run src/collect_demonstrations.py \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/huggingface/Walker2d-v5-SAC-expert.zip \
  --n-episodes 30
# Saves to: datasets/Walker2d-v5/30demos.pkl
# Optional: --max-steps N (max steps per episode, default: 1000)
```

### 3. Train a policy

> **Note on GPU usage**: MuJoCo environments (Humanoid, Walker2d, etc.) run physics simulations on CPU only. Use `--device cuda` only for large networks or image-based policies.

**Behavioral Cloning**: Learn a policy directly from expert demonstrations. See `docs/` for algorithm details.

```bash
uv run src/train.py \
  --algo bc \
  --env-id Walker2d-v5 \
  --demos-path datasets/Walker2d-v5/30demos.pkl \
  --total-timesteps 30
# Note: --total-timesteps sets training epochs for BC (not env steps)
# Saves to: models/interactive_il/Walker2d-v5/bc/bc_30demos_30epochs.pth
# Optional: --wandb (enable W&B logging, default: off)
# Optional: --seed N (random seed, default: 42), --device DEVICE (pytorch device, default: cpu)
```

**DAgger**: Iterative imitation learning that aggregates on-policy data. Optionally initialize from BC checkpoint. See `docs/` for details.

```bash
uv run src/train.py \
  --algo dagger \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/huggingface/Walker2d-v5-SAC-expert.zip \
  --n-iterations 20
# Saves to: models/interactive_il/Walker2d-v5/dagger/dagger_unknowndemos_20iters.pth
# Optional: --bc-init-path PATH (initialize from BC checkpoint, default: random init)
# Optional: --wandb (enable W&B logging, default: off)
# Optional: --seed N (random seed, default: 42), --device DEVICE (pytorch device, default: cpu)
```

**DAgger+Replay**: DAgger with replay buffer of critical states. See `docs/` for details.

```bash
uv run src/train.py \
  --algo dagger-replay \
  --env-id Walker2d-v5 \
  --expert-path models/SB3/Walker2d-v5/huggingface/Walker2d-v5-SAC-expert.zip \
  --n-iterations 20
# Saves to: models/interactive_il/Walker2d-v5/dagger-replay/dagger-replay_unknowndemos_20iters_k100.pth
# Optional: --bc-init-path PATH (initialize from BC checkpoint, default: random init)
# Optional: --wandb (enable W&B logging, default: off)
# Optional: --seed N (random seed, default: 42), --device DEVICE (pytorch device, default: cpu)
```

**Reinforcement Learning**: Train from scratch with PPO, SAC, TD3, or A2C.

```bash
uv run src/train.py \
  --algo ppo \
  --env-id Walker2d-v5 \
  --total-timesteps 1000000 \
  --n-envs 8
# Optional: --wandb (enable W&B logging, default: off)
# Optional: --seed N (random seed, default: 42)
# Optional: --checkpoint-freq N (save checkpoint every N steps, default: 250000)
```

### 4. Evaluate a policy

```bash
# BC/DAgger policy (.pth)
uv run src/play.py \
  --env-id Walker2d-v5 \
  --model-path models/interactive_il/Walker2d-v5/bc/bc_30demos_30epochs.pth \
  --algo bc
# Optional: --episodes N (number of evaluation episodes, default: 5)
# Optional: --deterministic (use deterministic actions, default: stochastic)
# Optional: --wandb (log evaluation metrics to W&B, default: off)
# Optional: --render-mode MODE (human/rgb_array/none, default: human)
# Optional: --max-steps N (max steps per episode, default: 1000)

# SB3 policy (.zip)
uv run src/play.py \
  --env-id Walker2d-v5 \
  --model-path models/SB3/Walker2d-v5/ppo/ppo_latest.zip \
  --algo ppo
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

For detailed algorithm explanations, network architecture, experimental setup, and results, see the `docs/` folder.

## License

MIT License - see [LICENSE](LICENSE) for details.
