"""Train a compact PyTorch CNN on the labeled chart-image dataset."""

import argparse
from collections import Counter
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
VALIDATION_DIR = PROJECT_DIR / "dataset" / "validation"
MODEL_PATH = PROJECT_DIR / "trading_vision_model.pth"
CLASS_NAMES = ("buy", "sell", "hold")
IMAGE_SIZE = 224


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive number") from error
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a three-class CNN using dataset/train and dataset/validation."
    )
    parser.add_argument("--epochs", type=_positive_int, default=15)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=_positive_float, default=0.001)
    parser.add_argument("--train-dir", type=Path, default=TRAIN_DIR)
    parser.add_argument("--validation-dir", type=Path, default=VALIDATION_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device (default: use CUDA when available, otherwise CPU).",
    )
    return parser


def _load_training_dependencies():
    try:
        import torch
        from torchvision import datasets, transforms
    except ImportError as error:
        raise RuntimeError(
            "Training requires PyTorch and torchvision. Install the appropriate "
            "packages for your system using `python -m pip install -r "
            "requirements-training.txt`."
        ) from error
    return torch, datasets, transforms


def _validate_dataset_directories(train_dir: Path, validation_dir: Path) -> None:
    for split_name, split_dir in (
        ("training", train_dir),
        ("validation", validation_dir),
    ):
        if not split_dir.is_dir():
            raise FileNotFoundError(f"{split_name.capitalize()} folder not found: {split_dir}")
        missing = [name for name in CLASS_NAMES if not (split_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"{split_name.capitalize()} folder {split_dir} is missing class folders: "
                + ", ".join(missing)
            )


def build_model(nn_module: Any):
    """Build a small CNN whose three logits follow ImageFolder class indices."""
    return nn_module.Sequential(
        nn_module.Conv2d(3, 16, kernel_size=3, padding=1),
        nn_module.ReLU(),
        nn_module.MaxPool2d(2),
        nn_module.Conv2d(16, 32, kernel_size=3, padding=1),
        nn_module.ReLU(),
        nn_module.MaxPool2d(2),
        nn_module.Conv2d(32, 64, kernel_size=3, padding=1),
        nn_module.ReLU(),
        nn_module.AdaptiveAvgPool2d((1, 1)),
        nn_module.Flatten(),
        nn_module.Linear(64, len(CLASS_NAMES)),
    )


def _run_epoch(model, loader, criterion, device, torch, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            if is_training:
                optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    if total_samples == 0:
        raise ValueError("Cannot run an epoch on an empty image dataset.")
    return total_loss / total_samples, total_correct / total_samples


def train_model(
    *,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    train_dir: str | Path = TRAIN_DIR,
    validation_dir: str | Path = VALIDATION_DIR,
    output_path: str | Path = MODEL_PATH,
    device_name: str = "auto",
) -> Path:
    """Train and save model weights with class mapping and input-size metadata."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda.")

    training_path = Path(train_dir)
    validation_path = Path(validation_dir)
    _validate_dataset_directories(training_path, validation_path)
    torch, datasets, transforms = _load_training_dependencies()

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA-enabled PyTorch device is available.")
    device = torch.device(
        "cuda" if device_name == "cuda" or (
            device_name == "auto" and torch.cuda.is_available()
        ) else "cpu"
    )
    image_transforms = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    training_data = datasets.ImageFolder(str(training_path), transform=image_transforms)
    validation_data = datasets.ImageFolder(str(validation_path), transform=image_transforms)

    expected_classes = set(CLASS_NAMES)
    if set(training_data.classes) != expected_classes:
        raise ValueError(
            f"Training classes must be {sorted(expected_classes)}; "
            f"found {training_data.classes}."
        )
    if training_data.class_to_idx != validation_data.class_to_idx:
        raise ValueError(
            "Training and validation folders must contain the same classes "
            "with matching ImageFolder indices."
        )
    for split_name, dataset in (
        ("training", training_data),
        ("validation", validation_data),
    ):
        counts = Counter(label for _, label in dataset.samples)
        missing = [
            class_name
            for class_name, class_index in dataset.class_to_idx.items()
            if counts[class_index] == 0
        ]
        if missing:
            raise ValueError(
                f"{split_name.capitalize()} data has no images for: "
                + ", ".join(missing)
            )

    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model = build_model(torch.nn).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Training on {device}; classes: {training_data.class_to_idx}")
    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = _run_epoch(
            model, training_loader, criterion, device, torch, optimizer
        )
        validation_loss, validation_accuracy = _run_epoch(
            model, validation_loader, criterion, device, torch
        )
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss: {train_loss:.4f}, train accuracy: {train_accuracy:.2%}, "
            f"validation loss: {validation_loss:.4f}, "
            f"validation accuracy: {validation_accuracy:.2%}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": training_data.class_to_idx,
            "image_size": (IMAGE_SIZE, IMAGE_SIZE),
            "architecture": "simple_cnn",
        },
        output,
    )
    print(f"Saved trained model weights to: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        train_dir=args.train_dir,
        validation_dir=args.validation_dir,
        output_path=args.output,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
