"""Train a binary ResNet-18 chart/non-chart image classifier."""

import argparse
import math
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_DIR / "dataset" / "chart_recognition"
MODEL_PATH = PROJECT_DIR / "chart_recognition_resnet18.pth"
CLASS_NAMES = ("charts", "non-charts")
IMAGE_SIZE = 224


def _positive_int(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=_positive_int, default=15)
    parser.add_argument("--batch-size", type=_positive_int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def _validate_dataset(dataset_dir: Path) -> None:
    for split in ("train", "validation"):
        split_dir = dataset_dir / split
        for class_name in CLASS_NAMES:
            class_dir = split_dir / class_name
            if not class_dir.is_dir() or not any(class_dir.iterdir()):
                raise FileNotFoundError(
                    f"Expected at least one image in: {class_dir}"
                )


def _build_model(models_module: Any, nn_module: Any):
    model = models_module.resnet18(weights=None)
    model.fc = nn_module.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model


def train_model(
    *,
    dataset_dir: str | Path = DATASET_DIR,
    output_path: str | Path = MODEL_PATH,
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device_name: str = "auto",
) -> Path:
    dataset_path = Path(dataset_dir)
    _validate_dataset(dataset_path)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive.")
    if device_name not in {"auto", "cpu", "cuda"}:
        raise ValueError("device_name must be auto, cpu, or cuda.")

    try:
        import torch
        from torchvision import datasets, models, transforms
    except ImportError as error:
        raise RuntimeError(
            "Chart-classifier training requires PyTorch and torchvision; "
            "install requirements-training.txt."
        ) from error

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device = torch.device(
        "cuda"
        if device_name == "cuda" or (device_name == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )
    train_data = datasets.ImageFolder(str(dataset_path / "train"), transform=transform)
    validation_data = datasets.ImageFolder(
        str(dataset_path / "validation"),
        transform=transform,
    )
    expected_classes = set(CLASS_NAMES)
    if set(train_data.classes) != expected_classes:
        raise ValueError(f"Training folders must be named {CLASS_NAMES}.")
    if train_data.class_to_idx != validation_data.class_to_idx:
        raise ValueError("Training and validation class mappings must match.")

    loaders = (
        torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        ),
        torch.utils.data.DataLoader(
            validation_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        ),
    )
    model = _build_model(models, torch.nn).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        metrics = []
        for loader, training in ((loaders[0], True), (loaders[1], False)):
            model.train(training)
            total_loss = 0.0
            total_correct = 0
            total_examples = 0
            with torch.set_grad_enabled(training):
                for images, labels in loader:
                    images, labels = images.to(device), labels.to(device)
                    if training:
                        optimizer.zero_grad()
                    logits = model(images)
                    loss = criterion(logits, labels)
                    if training:
                        loss.backward()
                        optimizer.step()
                    count = labels.size(0)
                    total_loss += loss.item() * count
                    total_correct += (logits.argmax(dim=1) == labels).sum().item()
                    total_examples += count
            if total_examples == 0:
                raise ValueError("Cannot train or validate on an empty image split.")
            metrics.append((total_loss / total_examples, total_correct / total_examples))
        print(
            f"Epoch {epoch}/{epochs} - "
            f"train loss={metrics[0][0]:.4f} accuracy={metrics[0][1]:.2%}; "
            f"validation loss={metrics[1][0]:.4f} accuracy={metrics[1][1]:.2%}"
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "class_to_idx": train_data.class_to_idx,
            "image_size": (IMAGE_SIZE, IMAGE_SIZE),
            "architecture": "resnet18",
        },
        output,
    )
    print(f"Saved chart-recognition model to: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_model(
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device_name=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
