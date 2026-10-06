"""Continuous, state-aware UI monitoring with explicit one-shot trade intent."""

import argparse
import logging
import math
from pathlib import Path
import threading
from typing import Callable, Sequence

from brokers import ExnessAdapter, PocketOptionAdapter
from config import AppConfig, DEFAULT_CONFIG
from core.assets import Asset
from core.detection import DEFAULT_MODEL_PATH, YOLODetection, YOLODetector
from core.safety import SafetyInterlock, SafetyInterlockError
from core.screen_capture import capture_screen
from core.state import Broker, StateConflict, SystemState
from core.vision import AssetOCR, UIVision
from utils.mouse import MouseController


LOGGER = logging.getLogger("trading_agent")
BROKER_ADAPTERS = {
    Broker.EXNESS: ExnessAdapter,
    Broker.POCKET_OPTION: PocketOptionAdapter,
}
BROKER_ACTIONS = {
    Broker.EXNESS: {"buy", "sell"},
    Broker.POCKET_OPTION: {"call", "put"},
}


def _positive_float(value: str) -> float:
    try:
        amount = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("amount must be a positive number") from error
    if not math.isfinite(amount) or amount <= 0:
        raise argparse.ArgumentTypeError("amount must be a finite, positive number")
    return amount


def _positive_int(value: str) -> int:
    try:
        seconds = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expiration must be a positive integer") from error
    if seconds <= 0:
        raise argparse.ArgumentTypeError("expiration must be a positive integer")
    return seconds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--interval", type=_positive_float, default=DEFAULT_CONFIG.scan_interval_seconds)
    parser.add_argument(
        "--action",
        choices=("buy", "sell", "call", "put"),
        help="One-shot external trade intent; never inferred from a visible button.",
    )
    parser.add_argument(
        "--asset",
        choices=tuple(asset.value for asset in Asset),
        help="Asset expected from active-label OCR; required with --action.",
    )
    parser.add_argument(
        "--amount",
        type=_positive_float,
        help="Exness lot size or Pocket Option investment amount.",
    )
    parser.add_argument(
        "--expiration-seconds",
        type=_positive_int,
        help="Required for Pocket Option call/put actions.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Enable actual PyAutoGUI input; omitted means dry-run only.",
    )
    return parser


def _make_state(
    screenshot,
    vision: UIVision,
    config: AppConfig,
) -> tuple[list[YOLODetection], SystemState]:
    detections = vision.detect_ui_elements(screenshot)
    try:
        active_asset_text = vision.read_active_asset(screenshot, detections)
    except Exception as error:
        raise SafetyInterlockError(
            f"Could not verify the active asset with OCR: {error}"
        ) from error
    try:
        state = SystemState.from_detections(
            detections,
            active_asset_text,
            config=config,
        )
    except StateConflict as error:
        raise SafetyInterlockError(str(error)) from error
    return detections, state


def _broker_from_state(state: SystemState):
    try:
        return BROKER_ADAPTERS[state.broker]
    except KeyError as error:
        raise SafetyInterlockError("No supported broker is active on screen.") from error


def _validate_request(
    action: str | None,
    asset: Asset | None,
    amount: float | None,
    expiration_seconds: int | None,
) -> None:
    if action is None:
        return
    if asset is None:
        raise ValueError("--asset is required when --action is specified.")
    if amount is None:
        raise ValueError("--amount is required when --action is specified.")
    if action in {"call", "put"} and expiration_seconds is None:
        raise ValueError("--expiration-seconds is required for call/put actions.")
    if action in {"buy", "sell"} and expiration_seconds is not None:
        raise ValueError("--expiration-seconds only applies to call/put actions.")


def run(
    vision: UIVision,
    *,
    action: str | None = None,
    expected_asset: Asset | None = None,
    amount: float | None = None,
    expiration_seconds: int | None = None,
    execute: bool = False,
    config: AppConfig = DEFAULT_CONFIG,
    screenshot_provider: Callable | None = None,
    stop_event: threading.Event | None = None,
    interlock: SafetyInterlock | None = None,
    mouse_factory: Callable[..., MouseController] = MouseController,
    report: Callable[[str], None] = print,
    max_iterations: int | None = None,
) -> int:
    """Continuously verify the screen; consume one explicit action at most once."""
    _validate_request(action, expected_asset, amount, expiration_seconds)
    if action is not None and amount is not None:
        if isinstance(amount, bool) or not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount must be finite and greater than zero.")
    if max_iterations is not None and max_iterations <= 0:
        raise ValueError("max_iterations must be positive when provided.")

    stop = stop_event or threading.Event()
    safety = interlock or SafetyInterlock(config=config)
    pending_action = action
    iteration = 0

    def current_state():
        screenshot = capture_screen(screenshot_provider)
        current_detections, current_system_state = _make_state(
            screenshot,
            vision,
            config,
        )
        safety.validate(
            current_system_state,
            expected_asset=expected_asset if pending_action else None,
        )
        return screenshot, current_detections, current_system_state

    def verify_before_each_input() -> None:
        if stop.is_set():
            raise SafetyInterlockError("Emergency stop requested; input is disabled.")
        _, _, fresh_state = current_state()
        if fresh_state.broker is not state.broker:
            raise SafetyInterlockError("Broker changed during the execution sequence.")
        if fresh_state.asset is not state.asset:
            raise SafetyInterlockError("Asset changed during the execution sequence.")

    while not stop.is_set():
        iteration += 1
        try:
            _, detections, state = current_state()
            report(
                f"Broker={state.broker.value}; asset={state.active_asset_text}; "
                f"detections={len(detections)}"
            )

            if pending_action is not None:
                adapter_type = _broker_from_state(state)
                if pending_action not in BROKER_ACTIONS[state.broker]:
                    raise SafetyInterlockError(
                        f"Action {pending_action!r} is not valid for "
                        f"{state.broker.value}."
                    )
                mouse = mouse_factory(
                    execute=execute,
                    check_safety=verify_before_each_input,
                )
                adapter = adapter_type(mouse=mouse)
                result = adapter.execute(
                    state.asset,
                    pending_action,
                    amount,
                    detections,
                    expiration_seconds=expiration_seconds,
                )
                report(result.message)
                for step in result.steps:
                    report(f"Execution step: {step}")
                pending_action = None

        except SafetyInterlockError:
            LOGGER.exception("Critical safety interlock; agent is paused.")
            report("CRITICAL SAFETY INTERLOCK: paused. Press ESC to stop.")
            while not stop.wait(0.5):
                pass
            return 2
        except Exception:
            LOGGER.critical(
                "Agent operation failed; input remains disabled.",
                exc_info=True,
            )
            report("Agent error: paused for safety. Press ESC to stop.")
            while not stop.wait(0.5):
                pass
            return 1

        if max_iterations is not None and iteration >= max_iterations:
            break
        stop.wait(config.scan_interval_seconds)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    try:
        expected_asset = Asset.parse(args.asset) if args.asset else None
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.action and expected_asset is None:
        raise SystemExit("--asset is required when --action is specified.")
    if args.action and args.amount is None:
        raise SystemExit("--amount is required when --action is specified.")
    if args.action in {"call", "put"} and args.expiration_seconds is None:
        raise SystemExit("--expiration-seconds is required for call/put actions.")
    if args.action in {"buy", "sell"} and args.expiration_seconds is not None:
        raise SystemExit("--expiration-seconds only applies to call/put actions.")

    detector = YOLODetector(args.model, confidence=DEFAULT_CONFIG.model_confidence)
    vision = UIVision(detector, AssetOCR(language=DEFAULT_CONFIG.ocr_language))
    stop_event = threading.Event()
    from pynput import keyboard

    def on_press(key):
        if key == keyboard.Key.esc:
            LOGGER.warning("ESC emergency stop requested.")
            stop_event.set()
            return False
        return None

    LOGGER.info(
        "Starting state-aware detection in %s mode; press ESC to stop.",
        "LIVE INPUT" if args.execute else "DRY RUN",
    )
    with keyboard.Listener(on_press=on_press) as listener:
        return run(
            vision,
            action=args.action,
            expected_asset=expected_asset,
            amount=args.amount,
            expiration_seconds=args.expiration_seconds,
            execute=args.execute,
            stop_event=stop_event,
        )


if __name__ == "__main__":
    raise SystemExit(main())
