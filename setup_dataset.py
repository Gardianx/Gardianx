"""Create the training and validation directory structure for image datasets."""

from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parent
DATASET_DIR = WORKSPACE_DIR / "dataset"
SPLITS = ("train", "validation")
CLASSES = ("buy", "sell", "hold")


def create_dataset_directories(dataset_dir: str | Path = DATASET_DIR) -> list[Path]:
    """Create dataset split/class folders and return their paths."""
    root = Path(dataset_dir)
    directories = [
        root / split / class_name
        for split in SPLITS
        for class_name in CLASSES
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Dataset folders created successfully in: {root}")
    return directories


def main() -> int:
    create_dataset_directories()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
