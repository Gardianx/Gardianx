"""Safe shared interface for broker integrations."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Iterable

from core.assets import Asset
from core.detection import YOLODetection
from utils.mouse import MouseController


@dataclass(frozen=True)
class ExecutionResult:
    broker: str
    asset: Asset
    action: str
    executed: bool
    message: str
    steps: tuple[str, ...] = ()


class BrokerAdapter(ABC):
    name: str

    def __init__(self, mouse: MouseController | None = None) -> None:
        self.mouse = mouse or MouseController()

    @staticmethod
    def validate_amount(amount: float) -> float:
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise TypeError("Amount must be numeric.")
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Amount must be finite and greater than zero.")
        return float(amount)

    @staticmethod
    def validate_expiration(expiration_seconds: int | None) -> int:
        if (
            isinstance(expiration_seconds, bool)
            or not isinstance(expiration_seconds, int)
            or expiration_seconds <= 0
        ):
            raise ValueError("Expiration must be a positive integer number of seconds.")
        return expiration_seconds

    @staticmethod
    def find_control(
        detections: Iterable[YOLODetection],
        label: str,
    ) -> YOLODetection:
        matches = [
            detection for detection in detections if detection.label.lower() == label
        ]
        if not matches:
            raise ValueError(f"Required UI control was not detected: {label}")
        return max(matches, key=lambda detection: detection.confidence)

    @staticmethod
    def find_control_any(
        detections: Iterable[YOLODetection],
        labels: tuple[str, ...],
    ) -> YOLODetection:
        accepted_labels = {label.lower() for label in labels}
        matches = [
            detection
            for detection in detections
            if detection.label.lower() in accepted_labels
        ]
        if not matches:
            raise ValueError(
                "Required UI control was not detected: "
                + " or ".join(labels)
            )
        return max(matches, key=lambda detection: detection.confidence)

    @abstractmethod
    def execute(
        self,
        asset: Asset,
        action: str,
        amount: float,
        detections: Iterable[YOLODetection],
        *,
        expiration_seconds: int | None = None,
    ) -> ExecutionResult:
        """Execute one explicitly requested action after caller-side safety checks."""
