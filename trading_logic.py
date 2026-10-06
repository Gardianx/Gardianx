"""Stateful, image-gated Buy/Sell click controller."""

import math
import time
from typing import Callable

from safe_executor import SafeExecutor
from screen_detector import Detection, ImageDetector


class TradingAction:
    NONE = "none"
    BUY = "buy"
    SELL = "sell"
    COOLDOWN = "cooldown"


class TradingLogic:
    """Click detected buttons according to an in-memory position state."""

    def __init__(
        self,
        buy_detector: ImageDetector,
        sell_detector: ImageDetector,
        executor: SafeExecutor,
        *,
        cooldown_seconds: float = 3.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0:
            raise ValueError("Cooldown must be a finite, non-negative number.")

        self._buy_detector = buy_detector
        self._sell_detector = sell_detector
        self._executor = executor
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self.IS_HOLDING = False
        self._last_click_at: float | None = None

    def step(self) -> str:
        """Check only the relevant template and perform at most one click."""
        if self._cooldown_active():
            return TradingAction.COOLDOWN
        if self.IS_HOLDING:
            return self.execute_prediction("SELL", None, self._sell_detector.find())
        return self.execute_prediction("BUY", self._buy_detector.find(), None)

    def execute_prediction(
        self,
        signal: str,
        buy_detection: Detection | None,
        sell_detection: Detection | None,
    ) -> str:
        """Apply a model signal only when state and the matching image agree."""
        normalized_signal = signal.upper()
        if normalized_signal not in {"BUY", "SELL", "HOLD"}:
            raise ValueError("Prediction signal must be BUY, SELL, or HOLD.")
        if normalized_signal == "HOLD":
            return TradingAction.NONE

        if self._cooldown_active():
            return TradingAction.COOLDOWN

        if normalized_signal == "BUY":
            if self.IS_HOLDING or buy_detection is None:
                return TradingAction.NONE
            detection = buy_detection
        else:
            if not self.IS_HOLDING or sell_detection is None:
                return TradingAction.NONE
            detection = sell_detection

        self._executor.move_and_click(*detection.center)
        self.IS_HOLDING = normalized_signal == "BUY"
        self._last_click_at = self._clock()
        return normalized_signal.lower()

    def _cooldown_active(self) -> bool:
        return (
            self._last_click_at is not None
            and self._clock() - self._last_click_at < self._cooldown_seconds
        )
