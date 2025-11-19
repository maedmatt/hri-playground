"""Test dagger_trainer functionality"""

from pathlib import Path

from huggingface_hub import hf_hub_download


def download_walker2d_expert() -> Path:
    """Download pre-trained Walker2d expert from HuggingFace for testing."""
    repo_id = "farama-minari/Walker2d-v5-SAC-medium"
    filename = "walker2d-v5-SAC-medium.zip"
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    return Path(path)


def test_dagger_trainer_pure():
    """Test pure DAgger (no replay buffer)."""
    from interactive_il.dagger_trainer import train_dagger

    expert_path = download_walker2d_expert()

    result = train_dagger(
        env_id="Walker2d-v5",
        expert_path=expert_path,
        save_path=Path("models/interactive_il/test_dagger_pure.pth"),
        bc_init_path=None,
        n_iterations=2,
        n_traj_per_iter=3,
        n_epochs=2,
        batch_size=256,
        lr=1e-3,
        use_norm=False,
        use_replay=False,
        seed=42,
        use_wandb=False,
    )

    assert "iteration_rewards" in result
    assert "final_reward" in result
    assert len(result["iteration_rewards"]) == 2
    assert Path("models/interactive_il/test_dagger_pure.pth").exists()


def test_dagger_trainer_replay():
    """Test DAgger with replay buffer."""
    from interactive_il.dagger_trainer import train_dagger

    expert_path = download_walker2d_expert()

    result = train_dagger(
        env_id="Walker2d-v5",
        expert_path=expert_path,
        save_path=Path("models/interactive_il/test_dagger_replay.pth"),
        bc_init_path=None,
        n_iterations=2,
        n_traj_per_iter=3,
        n_epochs=2,
        batch_size=256,
        lr=1e-3,
        use_norm=False,
        use_replay=True,
        buffer_size=1000,
        seed=42,
        use_wandb=False,
    )

    assert "iteration_rewards" in result
    assert "final_reward" in result
    assert len(result["iteration_rewards"]) == 2
    assert Path("models/interactive_il/test_dagger_replay.pth").exists()


if __name__ == "__main__":
    print("Testing pure DAgger...")
    test_dagger_trainer_pure()
    print("✓ Pure DAgger test passed!")

    print("\nTesting DAgger + Replay...")
    test_dagger_trainer_replay()
    print("✓ DAgger + Replay test passed!")
