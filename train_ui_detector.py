"""Train and validate the YOLO detector used to locate broker UI elements."""

import argparse
from pathlib import Path
import shutil
from typing import Any, Callable


PROJECT_DIR = Path(__file__).resolve().parent
DATA_CONFIG = PROJECT_DIR / "dataset" / "ui" / "data.yaml"
BASE_MODEL = "yolo11n.pt"
OUTPUT_MODEL = PROJECT_DIR / "models" / "best.pt"
RUNS_DIR = PROJECT_DIR / "runs" / "ui_detection"
REQUIRED_CLASSES = frozenset(
    {
        "exness",
        "pocket_option",
        "active_asset_label",
        "buy_button",
        "sell_button",
        "call_button",
        "put_button",
        "lot_size_input",
        "investment_input",
        "expiration_input",
        "insufficient_balance",
        "market_closed",
        "otc_dropdown_menu",
        "pocket_buy",
        "pocket_sell",
        "pocket_time",
        "pocket_amount",
        "pocket_payout",
        "pocket_trades",
        "pocket_wallet",
        "pocket_candle_timer",
        "bearish_otc_candle",
        "bullish_otc_candle",
        "active_exness_tab",
        "active_pocket_option_tab",
    }
)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return number


def _validate_data_config(data_path: Path) -> None:
    if not data_path.is_file():
        raise FileNotFoundError(f"YOLO dataset config not found: {data_path}")
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YOLO training requires PyYAML; install requirements-training.txt."
        ) from error

    with data_path.open(encoding="utf-8") as data_file:
        config = yaml.safe_load(data_file)
    if not isinstance(config, dict):
        raise ValueError("YOLO dataset config must contain a YAML mapping.")

    names = config.get("names")
    if isinstance(names, dict):
        class_names = {str(name) for name in names.values()}
    elif isinstance(names, list):
        class_names = {str(name) for name in names}
    else:
        raise ValueError("YOLO dataset config must define class names as a list or map.")

    missing = sorted(REQUIRED_CLASSES - class_names)
    unexpected = sorted(class_names - REQUIRED_CLASSES)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing classes: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected classes: {', '.join(unexpected)}")
        raise ValueError("Dataset class names do not match the UI detector: " + "; ".join(details))

    for split in ("train", "val"):
        if split not in config:
            raise ValueError(f"YOLO dataset config is missing the {split!r} split.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA_CONFIG)
    parser.add_argument(
        "--base-model",
        default=BASE_MODEL,
        help="Ultralytics detection checkpoint used as the starting point.",
    )
    parser.add_argument("--epochs", type=_positive_int, default=100)
    parser.add_argument("--imgsz", type=_positive_int, default=1280)
    parser.add_argument("--batch", type=_positive_int, default=8)
    parser.add_argument(
        "--device",
        default=None,
        help="GPU index or cpu; omitted lets Ultralytics select automatically.",
    )
    parser.add_argument("--project", type=Path, default=RUNS_DIR)
    parser.add_argument("--name", default="broker-ui")
    parser.add_argument("--output", type=Path, default=OUTPUT_MODEL)
    return parser


def train_detector(
    *,
    data_path: str | Path = DATA_CONFIG,
    base_model: str = BASE_MODEL,
    epochs: int = 100,
    image_size: int = 1280,
    batch_size: int = 8,
    device: str | None = None,
    project: str | Path = RUNS_DIR,
    run_name: str = "broker-ui",
    output_path: str | Path = OUTPUT_MODEL,
    yolo_factory: Callable[[str], Any] | None = None,
) -> Path:
    """Train, validate, and copy the best checkpoint to the application path."""
    data = Path(data_path)
    run_dir = Path(project) / run_name
    output = Path(output_path)
    for name, value in (
        ("epochs", epochs),
        ("image_size", image_size),
        ("batch_size", batch_size),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if not run_name.strip():
        raise ValueError("run_name must not be empty.")
    _validate_data_config(data)

    if yolo_factory is None:
        try:
            from ultralytics import YOLO
        except ImportError as error:
            raise RuntimeError(
                "YOLO training requires ultralytics; install requirements-training.txt."
            ) from error
        yolo_factory = YOLO

    model = yolo_factory(base_model)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=image_size,
        batch=batch_size,
        device=device,
        project=str(project),
        name=run_name,
        exist_ok=True,
    )
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(
            f"Training completed without producing the expected checkpoint: {best_weights}"
        )

    best_model = yolo_factory(str(best_weights))
    best_model.val(data=str(data), imgsz=image_size, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    print(f"Validated best weights saved for the application: {output}")
    return output


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    train_detector(
        data_path=args.data,
        base_model=args.base_model,
        epochs=args.epochs,
        image_size=args.imgsz,
        batch_size=args.batch,
        device=args.device,
        project=args.project,
        run_name=args.name,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
