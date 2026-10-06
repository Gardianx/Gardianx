"""Interactively crop, label, resize, and organize raw chart screenshots."""

from pathlib import Path
from typing import Callable

from setup_dataset import CLASSES, DATASET_DIR, SPLITS, create_dataset_directories


WORKSPACE_DIR = Path(__file__).resolve().parent
RAW_CHARTS_DIR = WORKSPACE_DIR / "raw_charts"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_SIZE = (224, 224)


def _load_cv2(cv2_module=None):
    if cv2_module is None:
        import cv2 as cv2_module
    return cv2_module


def _validate_crop(
    crop_region: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    if len(crop_region) != 4:
        raise ValueError("Crop region must contain x, y, width, and height.")
    x, y, width, height = crop_region
    if (
        any(not isinstance(value, int) or isinstance(value, bool) for value in crop_region)
        or x < 0
        or y < 0
        or width <= 0
        or height <= 0
        or x + width > image_width
        or y + height > image_height
    ):
        raise ValueError("Crop region must be a positive rectangle inside the source image.")
    return x, y, width, height


def _next_output_path(source_path: Path, destination: Path) -> Path:
    candidate = destination / f"{source_path.stem}.png"
    suffix = 1
    while candidate.exists():
        candidate = destination / f"{source_path.stem}_{suffix}.png"
        suffix += 1
    return candidate


def preprocess_image(
    image_path: str | Path,
    label: str,
    split: str,
    *,
    crop_region: tuple[int, int, int, int],
    dataset_dir: str | Path = DATASET_DIR,
    cv2_module=None,
) -> Path:
    """Crop a raw image, resize to 224x224, and save it in its labeled split."""
    normalized_label = label.lower()
    normalized_split = split.lower()
    if normalized_label not in CLASSES:
        raise ValueError(f"Label must be one of: {', '.join(CLASSES)}.")
    if normalized_split not in SPLITS:
        raise ValueError(f"Split must be one of: {', '.join(SPLITS)}.")

    cv2 = _load_cv2(cv2_module)
    source_path = Path(image_path)
    image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image file: {source_path}")
    image_height, image_width = image.shape[:2]
    x, y, width, height = _validate_crop(crop_region, image_width, image_height)
    crop = image[y : y + height, x : x + width]
    resized = cv2.resize(crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    destination = Path(dataset_dir) / normalized_split / normalized_label
    destination.mkdir(parents=True, exist_ok=True)
    output_path = _next_output_path(source_path, destination)
    if not cv2.imwrite(str(output_path), resized):
        raise OSError(f"Could not write preprocessed image: {output_path}")
    return output_path


def _prompt_choice(
    prompt: str,
    choices: tuple[str, ...],
    *,
    input_fn: Callable[[str], str],
    report: Callable[[str], None],
) -> str | None:
    choices_text = "/".join(choices)
    while True:
        answer = input_fn(f"{prompt} ({choices_text}; q to quit): ").strip().lower()
        if answer == "q":
            return None
        if answer in choices:
            return answer
        report(f"Invalid choice. Enter one of: {choices_text}, or q to quit.")


def process_raw_charts(
    raw_dir: str | Path = RAW_CHARTS_DIR,
    dataset_dir: str | Path = DATASET_DIR,
    *,
    input_fn: Callable[[str], str] = input,
    report: Callable[[str], None] = print,
    cv2_module=None,
) -> list[Path]:
    """Interactively crop and label supported images from a raw image folder.

    In each OpenCV selection window, drag a box around the chart candles/price
    action, then press ENTER or SPACE to accept, or C to skip that image.
    """
    cv2 = _load_cv2(cv2_module)
    source_dir = Path(raw_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    create_dataset_directories(dataset_dir)
    image_paths = sorted(
        (
            path
            for path in source_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: path.name.lower(),
    )
    if not image_paths:
        report(f"No supported images found in: {source_dir}")
        return []

    saved_paths: list[Path] = []
    try:
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                report(f"Skipped unreadable image: {image_path}")
                continue

            window_name = f"Crop chart candles: {image_path.name}"
            crop_region = cv2.selectROI(
                window_name,
                image,
                showCrosshair=True,
                fromCenter=False,
            )
            x, y, width, height = crop_region
            if width <= 0 or height <= 0:
                report(f"Skipped crop for: {image_path.name}")
                continue

            label = _prompt_choice(
                f"Label {image_path.name}",
                CLASSES,
                input_fn=input_fn,
                report=report,
            )
            if label is None:
                report("Labeling stopped.")
                break
            split = _prompt_choice(
                f"Dataset split for {image_path.name}",
                SPLITS,
                input_fn=input_fn,
                report=report,
            )
            if split is None:
                report("Labeling stopped.")
                break

            output_path = preprocess_image(
                image_path,
                label,
                split,
                crop_region=(x, y, width, height),
                dataset_dir=dataset_dir,
                cv2_module=cv2,
            )
            saved_paths.append(output_path)
            report(f"Saved {image_path.name} as {output_path}")
    finally:
        cv2.destroyAllWindows()

    return saved_paths


def main() -> int:
    process_raw_charts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
