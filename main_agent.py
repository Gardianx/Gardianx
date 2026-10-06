"""Run the stateful GUI trading controller until ESC is pressed."""

import argparse
import math
from pathlib import Path
import sys
from typing import Callable

from model_brain import SignalPrediction, TradingBrain
from performance_logger import log_trade
from safe_executor import EmergencyStop, KillSwitch, SafeExecutor
from screen_detector import Detection, ImageDetector, Region
from trading_logic import TradingAction, TradingLogic


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_DIR / "templates"
BUY_TEMPLATE = TEMPLATE_DIR / "buy_button.png"
SELL_TEMPLATE = TEMPLATE_DIR / "sell_button.png"
CHART_REGION = Region(left=2, top=0, width=1360, height=728)
BUY_CLICK_TARGET = (1188, 23)
SELL_CLICK_TARGET = (1308, 21)


class DryRunExecutor(SafeExecutor):
    """Keep the kill switch and logging, but never move or click the mouse."""

    def move_and_click(self, x: int, y: int, duration: float = 0.25) -> None:
        self.check_kill_switch()
        self.log_action("DRY_RUN", x, y)


def _interval(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("interval must be a number from 1 to 2 seconds") from error
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 2.0:
        raise argparse.ArgumentTypeError("interval must be between 1 and 2 seconds")
    return seconds


def _cooldown(value: str) -> float:
    try:
        seconds = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("cooldown must be a non-negative number") from error
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("cooldown must be a finite, non-negative number")
    return seconds


def _threshold(value: str) -> float:
    try:
        threshold = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return threshold


def _capture_desktop_screenshot():
    import pyautogui

    return pyautogui.screenshot()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Monitor a screen region for Buy/Sell templates until ESC is pressed."
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=int,
        metavar=("LEFT", "TOP", "WIDTH", "HEIGHT"),
        default=CHART_REGION.as_tuple(),
        help="Screen region to monitor (default: calibrated chart area).",
    )
    parser.add_argument(
        "--interval",
        type=_interval,
        default=1.5,
        help="Delay between scans, from 1 to 2 seconds (default: 1.5).",
    )
    parser.add_argument(
        "--cooldown",
        type=_cooldown,
        default=3.0,
        help="Minimum seconds between clicks (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=_threshold,
        default=0.9,
        help="Template-match confidence threshold from 0 to 1 (default: 0.9).",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=PROJECT_DIR / "gui_actions.log",
        help="File to append click or dry-run records to.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable real mouse clicks; without this flag, run in dry-run mode.",
    )
    return parser


def run_agent(
    brain: TradingBrain,
    logic: TradingLogic,
    buy_detector: ImageDetector,
    sell_detector: ImageDetector,
    executor: SafeExecutor,
    interval_seconds: float,
    *,
    report: Callable[[str], None] = print,
    trade_logger: Callable[[str, float | str, float], object] = log_trade,
    chart_region: Region = CHART_REGION,
    screenshot_provider: Callable | None = None,
) -> None:
    """Capture the chart, infer a signal, apply safeguards, and log each cycle."""
    if screenshot_provider is None:
        screenshot_provider = _capture_desktop_screenshot

    while True:
        executor.check_kill_switch()

        desktop_screenshot = screenshot_provider()
        chart_image = desktop_screenshot.crop(
            (
                chart_region.left,
                chart_region.top,
                chart_region.left + chart_region.width,
                chart_region.top + chart_region.height,
            )
        )
        buy_detection = buy_detector.find(chart_image)
        executor.check_kill_switch()
        sell_detection = sell_detector.find(chart_image)
        executor.check_kill_switch()

        current_features = _build_features(
            buy_detection,
            sell_detection,
            is_holding=logic.IS_HOLDING,
        )
        prediction = brain.predict_signal(chart_image)
        signal, confidence = _validate_prediction(prediction)
        executor.check_kill_switch()

        action = logic.execute_prediction(signal, buy_detection, sell_detection)
        logged_action = (
            action.upper()
            if action in {TradingAction.BUY, TradingAction.SELL}
            else "HOLD"
        )
        asset_state = (
            f"price unavailable; model_signal={signal}; holding={logic.IS_HOLDING}; "
            f"buy_match={current_features['buy_confidence']:.3f}; "
            f"sell_match={current_features['sell_confidence']:.3f}; "
            f"model_placeholder={prediction.is_placeholder}; "
            f"mode={'live' if not isinstance(executor, DryRunExecutor) else 'dry_run'}"
        )
        trade_logger(logged_action, asset_state, confidence)
        report(
            f"Model={signal} confidence={confidence:.3f}; action={logged_action}; "
            f"IS_HOLDING={logic.IS_HOLDING}"
        )

        if executor.wait_for_stop(interval_seconds):
            return


def _build_features(
    buy_detection: Detection | None,
    sell_detection: Detection | None,
    *,
    is_holding: bool,
) -> dict[str, float]:
    """Convert template detections and position state into logged measurements."""
    return {
        "buy_detected": float(buy_detection is not None),
        "buy_confidence": buy_detection.confidence if buy_detection else 0.0,
        "sell_detected": float(sell_detection is not None),
        "sell_confidence": sell_detection.confidence if sell_detection else 0.0,
        "is_holding": float(is_holding),
    }


def _validate_prediction(prediction: SignalPrediction) -> tuple[str, float]:
    if not isinstance(prediction, SignalPrediction):
        raise TypeError("TradingBrain.predict_signal() must return a SignalPrediction.")
    if not isinstance(prediction.signal, str):
        raise TypeError("Model signal must be a string.")
    signal = prediction.signal.upper()
    if signal not in {"BUY", "SELL", "HOLD"}:
        raise ValueError("Model signal must be BUY, SELL, or HOLD.")
    confidence = prediction.confidence_score
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.0 <= confidence <= 1.0
    ):
        raise ValueError("Model confidence must be a finite number between 0 and 1.")
    return signal, float(confidence)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    missing_templates = [
        str(path) for path in (BUY_TEMPLATE, SELL_TEMPLATE) if not path.is_file()
    ]
    if missing_templates:
        parser.error(
            "template files not found; run template_cropper.py first: "
            + ", ".join(missing_templates)
        )

    try:
        region = Region(*args.region)
    except ValueError as error:
        parser.error(str(error))

    buy_detector = ImageDetector(BUY_TEMPLATE, region, args.threshold)
    sell_detector = ImageDetector(SELL_TEMPLATE, region, args.threshold)
    brain = TradingBrain()
    brain.load_model()
    kill_switch = KillSwitch()
    executor_type = SafeExecutor if args.execute else DryRunExecutor
    executor = executor_type(args.log_file, kill_switch=kill_switch)
    logic = TradingLogic(
        buy_detector,
        sell_detector,
        executor,
        cooldown_seconds=args.cooldown,
        buy_click_target=BUY_CLICK_TARGET,
        sell_click_target=SELL_CLICK_TARGET,
    )

    mode = "LIVE EXECUTION" if args.execute else "DRY RUN (no mouse clicks)"
    print(f"Starting {mode}. Monitoring screen region {region.as_tuple()}; press ESC to stop.")

    try:
        with executor:
            run_agent(
                brain,
                logic,
                buy_detector,
                sell_detector,
                executor,
                args.interval,
            )
    except EmergencyStop:
        print("ESC kill switch activated; automation stopped.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
