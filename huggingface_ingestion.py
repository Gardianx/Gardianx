"""Download labeled stock-chart images into a separate binary dataset."""

import argparse
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Mapping


PROJECT_DIR = Path(__file__).resolve().parent
DATASET_ID = "StephanAkkerman/stock-charts"
DATASET_SPLIT = "train"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "chart_recognition"
MAX_IMAGES = 1500
VALIDATION_EVERY = 5
CLASS_NAMES = ("charts", "non-charts")


def _positive_int(value: str) -> int:
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("limit must be a positive integer") from error
    if count <= 0:
        raise argparse.ArgumentTypeError("limit must be a positive integer")
    if count > MAX_IMAGES:
        raise argparse.ArgumentTypeError(f"limit cannot exceed {MAX_IMAGES}")
    return count


def _load_dataset(dataset_id: str, split: str):
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Hugging Face ingestion requires the `datasets` package. "
            "Install it with `python -m pip install -r requirements-training.txt`."
        ) from error
    return load_dataset(dataset_id, split=split, streaming=True)


def _class_mapping(dataset) -> dict[int, str]:
    features = getattr(dataset, "features", None)
    label_feature = features.get("label") if isinstance(features, Mapping) else None
    label_names = getattr(label_feature, "names", None)
    if not isinstance(label_names, (tuple, list)):
        raise ValueError("Hugging Face dataset must expose named ClassLabel values.")

    normalized_names = {str(name).strip().lower(): index for index, name in enumerate(label_names)}
    if set(normalized_names) != set(CLASS_NAMES):
        raise ValueError(
            f"Expected dataset labels {CLASS_NAMES}; found {tuple(label_names)}."
        )
    return {
        index: str(name).strip().lower()
        for index, name in enumerate(label_names)
    }


def _save_image(image, output_path: Path) -> None:
    if not callable(getattr(image, "convert", None)):
        raise TypeError("Dataset image values must be decoded image objects.")
    rgb_image = image.convert("RGB")
    rgb_image.save(output_path, format="JPEG", quality=95)


def ingest_images(
    dataset_id: str = DATASET_ID,
    output_dir: str | Path = OUTPUT_DIR,
    *,
    limit: int = MAX_IMAGES,
    split: str = DATASET_SPLIT,
    dataset_loader: Callable[[str, str], Iterable] = _load_dataset,
) -> dict[str, int]:
    """Save at most `limit` source images with labels and stratified train/val splits."""
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_IMAGES:
        raise ValueError(f"limit must be an integer between 1 and {MAX_IMAGES}.")

    dataset = dataset_loader(dataset_id, split)
    label_mapping = _class_mapping(dataset)
    destination_root = Path(output_dir)
    counts: Counter[str] = Counter()
    saved_count = 0

    for row in islice(dataset, limit):
        if not isinstance(row, Mapping) or "image" not in row or "label" not in row:
            raise ValueError("Each dataset row must contain image and label fields.")
        label_value = row["label"]
        if isinstance(label_value, bool) or not isinstance(label_value, int):
            raise ValueError("Dataset label values must be integer ClassLabel indices.")
        if label_value not in label_mapping:
            raise ValueError(f"Dataset contains an unknown label index: {label_value}.")

        class_name = label_mapping[label_value]
        class_index = counts[class_name]
        dataset_split = (
            "validation"
            if (class_index + 1) % VALIDATION_EVERY == 0
            else "train"
        )
        class_output_dir = destination_root / dataset_split / class_name
        class_output_dir.mkdir(parents=True, exist_ok=True)
        output_path = class_output_dir / f"{class_name}_{class_index:05d}.jpg"
        _save_image(row["image"], output_path)
        counts[class_name] += 1
        saved_count += 1

    if not saved_count:
        raise ValueError(f"No images were available in {dataset_id!r} split {split!r}.")

    summary = {name: counts[name] for name in CLASS_NAMES}
    print(f"Saved {saved_count} labeled images from {dataset_id}: {summary}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download up to 1,500 chart/non-chart images from Hugging Face."
    )
    parser.add_argument("--dataset-id", default=DATASET_ID)
    parser.add_argument("--split", default=DATASET_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=_positive_int, default=MAX_IMAGES)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ingest_images(
        args.dataset_id,
        args.output_dir,
        limit=args.limit,
        split=args.split,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
