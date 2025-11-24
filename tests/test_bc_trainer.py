"""Test bc_trainer functionality"""

import pickle
from pathlib import Path

import numpy as np

from interactive_il.bc_trainer import train_bc


def create_fake_demos(save_path: Path, n_demos: int = 5, traj_length: int = 100):
    """Create fake demonstrations for testing."""
    demos = []
    for _ in range(n_demos):
        obs = np.random.randn(traj_length, 17).astype(np.float32)
        acts = np.random.randn(traj_length, 6).astype(np.float32)
        demos.append({"observations": obs, "actions": acts})

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(demos, f)


def test_bc_trainer():
    """Test that BC trainer runs without errors."""
    # Create fake demos for testing
    demos_path = Path("models/interactive_il/test_demos.pkl")
    expected_policy_path = Path(
        "models/interactive_il/Walker2d-v5/bc/bc_5demos_2epochs.pth"
    )

    try:
        create_fake_demos(demos_path, n_demos=5, traj_length=100)

        result = train_bc(
            env_id="Walker2d-v5",
            demos_path=demos_path,
            n_epochs=2,
            batch_size=64,
            lr=3e-4,
            use_norm=True,
            seed=42,
            use_wandb=False,
        )

        assert "train_losses" in result
        assert "final_train_loss" in result
        train_losses = result["train_losses"]
        assert isinstance(train_losses, list)
        assert len(train_losses) == 2
        assert expected_policy_path.exists()

    finally:
        # Cleanup test artifacts
        if demos_path.exists():
            demos_path.unlink()
        if expected_policy_path.exists():
            expected_policy_path.unlink()
        # Clean up test directories
        test_dir = Path("models/interactive_il/Walker2d-v5")
        if test_dir.exists():
            import shutil

            shutil.rmtree(test_dir)


def test_bc_trainer_with_validation():
    """Test BC trainer with validation split."""
    demos_path = Path("models/interactive_il/test_demos_val.pkl")
    expected_policy_path = Path(
        "models/interactive_il/Walker2d-v5/bc/bc_5demos_2epochs.pth"
    )

    try:
        create_fake_demos(demos_path, n_demos=5, traj_length=100)

        result = train_bc(
            env_id="Walker2d-v5",
            demos_path=demos_path,
            n_epochs=2,
            batch_size=64,
            lr=3e-4,
            use_norm=True,
            val_split=0.2,
            seed=42,
            use_wandb=False,
        )

        assert "train_losses" in result
        assert "val_losses" in result
        assert "final_train_loss" in result
        assert "final_val_loss" in result
        train_losses = result["train_losses"]
        val_losses = result["val_losses"]
        assert isinstance(train_losses, list) and len(train_losses) == 2
        assert isinstance(val_losses, list) and len(val_losses) == 2
        assert expected_policy_path.exists()

    finally:
        if demos_path.exists():
            demos_path.unlink()
        if expected_policy_path.exists():
            expected_policy_path.unlink()
        test_dir = Path("models/interactive_il/Walker2d-v5")
        if test_dir.exists():
            import shutil

            shutil.rmtree(test_dir)


if __name__ == "__main__":
    test_bc_trainer()
    test_bc_trainer_with_validation()
    print("✓ All tests passed!")
