"""Central fail-closed safety checks and critical event reporting."""

import logging

from config import AppConfig, DEFAULT_CONFIG
from core.assets import Asset
from core.state import AssetType, Broker, SystemState


LOGGER = logging.getLogger("trading_safety")


class SafetyInterlockError(RuntimeError):
    """Raised when the UI state is unsafe for further interaction."""


class SafetyInterlock:
    def __init__(
        self,
        *,
        config: AppConfig = DEFAULT_CONFIG,
        logger: logging.Logger = LOGGER,
    ) -> None:
        self.config = config
        self.logger = logger

    def validate(
        self,
        state: SystemState,
        *,
        expected_broker: Broker | None = None,
        expected_asset: Asset | None = None,
    ) -> None:
        reasons = []
        if state.safety_alerts:
            reasons.append(
                "safety popup detected: " + ", ".join(state.safety_alerts)
            )
        if state.broker is Broker.UNKNOWN:
            reasons.append("active broker is unknown")
        if expected_broker is not None and state.broker is not expected_broker:
            reasons.append(
                f"broker mismatch: expected {expected_broker.value}, "
                f"detected {state.broker.value}"
            )
        if state.asset_type is AssetType.UNKNOWN:
            reasons.append(
                f"active asset label could not be verified with OCR: "
                f"{state.active_asset_text!r}"
            )
        if expected_asset is not None and state.asset is not expected_asset:
            reasons.append(
                f"asset mismatch: expected {expected_asset.value}, "
                f"detected {state.asset.value if state.asset else 'UNKNOWN'}"
            )
        if (
            self.config.exness_rejects_otc
            and state.broker is Broker.EXNESS
            and state.asset_type is AssetType.OTC
        ):
            reasons.append("Exness/OTC broker-asset mismatch")
        if reasons:
            message = "; ".join(reasons)
            self.logger.critical("SAFETY INTERLOCK: %s", message)
            raise SafetyInterlockError(message)
