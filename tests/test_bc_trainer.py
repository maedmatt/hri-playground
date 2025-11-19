"""Test bc_trainer functionality"""

from pathlib import Path

from interactive_il.bc_trainer import train_bc


def test_bc_trainer():
    """Test that BC trainer runs without errors."""
    result = train_bc(
        env_id="Walker2d-v5",
        demos_path=Path("models/interactive_il/walker2d_demos.pkl"),
        save_path=Path("models/interactive_il/test_bc_policy.pth"),
        n_epochs=2,
        batch_size=1024,
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
    assert Path("models/interactive_il/test_bc_policy.pth").exists()


if __name__ == "__main__":
    test_bc_trainer()
    print("✓ Test passed!")
