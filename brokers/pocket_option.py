"""Pocket Option UI adapter for validated, explicit call/put requests."""

from brokers.base import BrokerAdapter, ExecutionResult
from core.assets import Asset
from core.detection import YOLODetection


class PocketOptionAdapter(BrokerAdapter):
    name = "pocket_option"

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
        if normalized_action not in {"call", "put"}:
            raise ValueError("Pocket Option actions must be call or put.")
        investment = self.validate_amount(amount)
        expiration = self.validate_expiration(expiration_seconds)
        investment_field = self.find_control_any(
            detections,
            ("pocket_amount", "investment_input"),
        )
        expiration_field = self.find_control_any(
            detections,
            ("pocket_time", "expiration_input"),
        )
        action_button = self.find_control_any(
            detections,
            (
                ("pocket_buy", "call_button")
                if normalized_action == "call"
                else ("pocket_sell", "put_button")
            ),
        )
        steps = (
            f"set investment to {investment:g}",
            f"set expiration to {expiration} seconds",
            f"click {normalized_action}",
        )
        self.mouse.replace_text(*map(round, investment_field.center), f"{investment:g}")
        self.mouse.replace_text(*map(round, expiration_field.center), str(expiration))
        self.mouse.click(*map(round, action_button.center))
        return ExecutionResult(
            broker=self.name,
            asset=asset,
            action=normalized_action.upper(),
            executed=self.mouse.execute_enabled,
            message=(
                "Pocket Option UI action sequence completed."
                if self.mouse.execute_enabled
                else "Dry run only; no broker action was sent."
            ),
            steps=steps,
        )
