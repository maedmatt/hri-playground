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
    policy_path = Path("models/interactive_il/test_bc_policy.pth")

    try:
        create_fake_demos(demos_path, n_demos=5, traj_length=100)

        result = train_bc(
            env_id="Walker2d-v5",
            demos_path=demos_path,
            save_path=policy_path,
            n_epochs=2,
            batch_size=64,
            lr=3e-4,
            use_norm=True,
            seed=42,
            use_wandb=False,
        )

        assert "losses" in result
        assert "final_loss" in result
        losses = result["losses"]
        assert isinstance(losses, list)
        assert len(losses) == 2
        assert policy_path.exists()

    finally:
        # Cleanup test artifacts
        if demos_path.exists():
            demos_path.unlink()
        if policy_path.exists():
            policy_path.unlink()


if __name__ == "__main__":
    test_bc_trainer()
    print("✓ Test passed!")
