"""Convert broker UI box annotations into image/prompt/response VLM manifests."""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PROJECT_DIR = Path(__file__).resolve().parent
DATA_CONFIG = PROJECT_DIR / "dataset" / "ui" / "data.yaml"
OUTPUT_DIR = PROJECT_DIR / "dataset" / "vlm"
IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
PROMPT_TEMPLATE = (
    "Inspect this broker interface screenshot. Detect every visible UI element "
    "from this label set: {labels}. Return only a JSON object with a "
    '"detections" array. Each item must contain "label" and "bbox_2d"; '
    "bbox_2d is [left, top, right, bottom] normalized from 0 to 1000."
)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "VLM dataset preparation requires PyYAML. Install "
            "`requirements-vlm.txt`."
        ) from error

    with path.open(encoding="utf-8") as data_file:
        config = yaml.safe_load(data_file)
    if not isinstance(config, dict):
        raise ValueError("Dataset configuration must contain a YAML mapping.")
    return config


def _class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("names")
    if isinstance(names, list):
        class_names = [str(name) for name in names]
    elif isinstance(names, dict):
        try:
            indexed_names = sorted(
                ((int(index), str(name)) for index, name in names.items()),
                key=lambda item: item[0],
            )
        except (TypeError, ValueError) as error:
            raise ValueError("Class-name map keys must be integer indices.") from error
        if [index for index, _ in indexed_names] != list(range(len(indexed_names))):
            raise ValueError("Class-name map indices must be contiguous starting at 0.")
        class_names = [name for _, name in indexed_names]
    else:
        raise ValueError("Dataset configuration must define class names as a list or map.")

    if not class_names or any(not name.strip() for name in class_names):
        raise ValueError("Dataset class names must be non-empty.")
    if len(set(class_names)) != len(class_names):
        raise ValueError("Dataset class names must be unique.")
    return class_names


def _resolve_image_dir(config_path: Path, config: dict[str, Any], split: str) -> Path:
    split_value = config.get(split)
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError(f"Dataset configuration must define a non-empty {split!r} split.")

    configured_root = config.get("path", ".")
    if not isinstance(configured_root, str) or not configured_root.strip():
        raise ValueError("Dataset configuration 'path' must be a non-empty string.")
    root = Path(configured_root)
    if not root.is_absolute():
        root = config_path.parent / root

    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"{split.capitalize()} image directory not found: {image_dir}")
    return image_dir


def _resolve_label_dir(image_dir: Path) -> Path:
    parts = list(image_dir.parts)
    image_component = next(
        (index for index in range(len(parts) - 1, -1, -1) if parts[index] == "images"),
        None,
    )
    if image_component is None:
        raise ValueError(
            f"Cannot infer annotation directory for {image_dir}; image paths must "
            "contain an 'images' directory component."
        )
    parts[image_component] = "labels"
    label_dir = Path(*parts)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Annotation directory not found: {label_dir}")
    return label_dir


def _parse_annotation(
    annotation_path: Path,
    class_names: list[str],
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    with annotation_path.open(encoding="utf-8") as annotation_file:
        for line_number, line in enumerate(annotation_file, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5:
                raise ValueError(
                    f"{annotation_path}:{line_number}: expected class, center_x, "
                    "center_y, width, height."
                )
            try:
                class_index = int(fields[0])
                center_x, center_y, width, height = (
                    float(value) for value in fields[1:]
                )
            except ValueError as error:
                raise ValueError(
                    f"{annotation_path}:{line_number}: annotation values must be numeric."
                ) from error
            if not 0 <= class_index < len(class_names):
                raise ValueError(
                    f"{annotation_path}:{line_number}: class index {class_index} "
                    f"is outside [0, {len(class_names) - 1}]."
                )
            if not all(
                math.isfinite(value)
                for value in (center_x, center_y, width, height)
            ):
                raise ValueError(
                    f"{annotation_path}:{line_number}: coordinates must be finite."
                )
            if not (
                0.0 <= center_x <= 1.0
                and 0.0 <= center_y <= 1.0
                and 0.0 < width <= 1.0
                and 0.0 < height <= 1.0
            ):
                raise ValueError(
                    f"{annotation_path}:{line_number}: YOLO coordinates must be "
                    "normalized to [0, 1] and box sizes must be positive."
                )

            left = max(0.0, center_x - width / 2)
            top = max(0.0, center_y - height / 2)
            right = min(1.0, center_x + width / 2)
            bottom = min(1.0, center_y + height / 2)
            if right <= left or bottom <= top:
                raise ValueError(
                    f"{annotation_path}:{line_number}: box is empty after clipping."
                )
            detections.append(
                {
                    "label": class_names[class_index],
                    "bbox_2d": [
                        round(left * 1000, 2),
                        round(top * 1000, 2),
                        round(right * 1000, 2),
                        round(bottom * 1000, 2),
                    ],
                }
            )
    return detections


def _convert_split(
    config_path: Path,
    config: dict[str, Any],
    split: str,
    class_names: list[str],
    output_dir: Path,
    *,
    allow_missing_labels: bool,
) -> list[dict[str, Any]]:
    image_dir = _resolve_image_dir(config_path, config, split)
    label_dir = _resolve_label_dir(image_dir)
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise ValueError(f"No supported images found in {image_dir}.")

    prompt = PROMPT_TEMPLATE.format(labels=", ".join(class_names))
    records: list[dict[str, Any]] = []
    for image_path in images:
        annotation_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if not annotation_path.is_file():
            if not allow_missing_labels:
                raise FileNotFoundError(
                    f"Annotation file not found for {image_path}: {annotation_path}. "
                    "Use --allow-missing-labels only for confirmed background images."
                )
            detections: list[dict[str, Any]] = []
        else:
            detections = _parse_annotation(annotation_path, class_names)
        records.append(
            {
                "image": Path(
                    os.path.relpath(image_path.resolve(), output_dir.resolve())
                ).as_posix(),
                "prompt": prompt,
                "completion": json.dumps(
                    {"detections": detections},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        )
    return records


def prepare_dataset(
    *,
    data_path: str | Path = DATA_CONFIG,
    output_dir: str | Path = OUTPUT_DIR,
    allow_missing_labels: bool = False,
) -> dict[str, int]:
    """Write train and validation JSONL manifests from YOLO-format box labels."""
    config_path = Path(data_path).resolve()
    output = Path(output_dir).resolve()
    config = _load_yaml(config_path)
    class_names = _class_names(config)
    records_by_split = {
        split: _convert_split(
            config_path,
            config,
            split,
            class_names,
            output,
            allow_missing_labels=allow_missing_labels,
        )
        for split in ("train", "val")
    }

    output.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, records in records_by_split.items():
        manifest_path = output / f"{'validation' if split == 'val' else split}.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest:
            for record in records:
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
        counts[split] = len(records)
    return counts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--allow-missing-labels",
        action="store_true",
        help="Treat missing annotation files as empty only for confirmed background images.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    counts = prepare_dataset(
        data_path=args.data,
        output_dir=args.output_dir,
        allow_missing_labels=args.allow_missing_labels,
    )
    print(
        f"VLM manifests saved to {args.output_dir}: "
        f"{counts['train']} training and {counts['val']} validation images."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
