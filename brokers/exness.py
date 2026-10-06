"""Exness UI adapter for validated, explicit buy/sell requests."""

from brokers.base import BrokerAdapter, ExecutionResult
from core.assets import Asset
from core.detection import YOLODetection


class ExnessAdapter(BrokerAdapter):
    name = "exness"

    def execute(
        self,
        asset: Asset,
        action: str,
        amount: float,
        detections: list[YOLODetection],
        *,
        expiration_seconds: int | None = None,
    ) -> ExecutionResult:
        normalized_action = action.lower()
        if normalized_action not in {"buy", "sell"}:
            raise ValueError("Exness actions must be buy or sell.")
        lot_size = self.validate_amount(amount)
        amount_field = self.find_control(detections, "lot_size_input")
        action_button = self.find_control(detections, f"{normalized_action}_button")
        steps = (f"set lot size to {lot_size:g}", f"click {normalized_action}")
        self.mouse.replace_text(*map(round, amount_field.center), f"{lot_size:g}")
        self.mouse.click(*map(round, action_button.center))
        return ExecutionResult(
            broker=self.name,
            asset=asset,
            action=normalized_action.upper(),
            executed=self.mouse.execute_enabled,
            message=(
                "Exness UI action sequence completed."
                if self.mouse.execute_enabled
                else "Dry run only; no broker action was sent."
            ),
            steps=steps,
        )
