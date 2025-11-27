"""
Download pretrained models from HuggingFace.

See README.md for usage examples and detailed documentation.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pretrained models from HuggingFace"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID (e.g., farama-minari/Walker2d-v5-SAC-medium)",
    )
    parser.add_argument(
        "--filename",
        type=str,
        required=True,
        help="Model filename in the repo (e.g., walker2d-v5-SAC-medium.zip)",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Environment ID (e.g., Walker2d-v5)",
    )
    parser.add_argument(
        "--save-name",
        type=str,
        help="Custom name for saved file (defaults to original filename)",
    )
    return parser.parse_args()


def download_model(
    repo_id: str,
    filename: str,
    env_id: str,
    save_name: str | None = None,
) -> Path:
    """Download model from HuggingFace and save to standard location."""
    print(f"Downloading {filename} from {repo_id}...")

    # Download from HuggingFace
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename)
    print(f"Downloaded to temp location: {downloaded_path}")

    # Save to standard location: models/SB3/<env-id>/hf/<name>
    save_name = save_name or filename
    target_path = Path("models") / "SB3" / env_id / "huggingface" / save_name
    target_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy to standard location
    shutil.copy(downloaded_path, target_path)
    print(f"Saved to: {target_path}")

    return target_path


def main() -> None:
    args = parse_args()

    target_path = download_model(
        repo_id=args.repo_id,
        filename=args.filename,
        env_id=args.env_id,
        save_name=args.save_name,
    )

    print("\nYou can now use it with:")
    print(
        f"  uv run src/train.py --algo dagger --env-id {args.env_id} --expert-path {target_path} --wandb"
    )
    print(
        f"  uv run src/play.py --env-id {args.env_id} --model-path {target_path} --algo <algo>"
    )


if __name__ == "__main__":
    main()
