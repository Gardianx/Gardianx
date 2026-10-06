"""Parse YOLO labels and OCR text into a fail-closed broker/asset state."""

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from config import (
    ACTIVE_BROKER_TAB_ALIASES,
    BROKER_LABEL_ALIASES,
    SAFETY_POPUP_ALIASES,
    AppConfig,
    DEFAULT_CONFIG,
)
from core.assets import Asset
from core.detection import YOLODetection


class Broker(str, Enum):
    EXNESS = "EXNESS"
    POCKET_OPTION = "POCKET_OPTION"
    UNKNOWN = "UNKNOWN"


class AssetType(str, Enum):
    STANDARD = "STANDARD"
    OTC = "OTC"
    UNKNOWN = "UNKNOWN"


class StateConflict(ValueError):
    """Raised when detections disagree about the active broker or asset."""


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class SystemState:
    broker: Broker
    asset_type: AssetType
    asset: Asset | None
    active_asset_text: str
    safety_alerts: tuple[str, ...] = ()

    @classmethod
    def from_detections(
        cls,
        detections: Iterable[YOLODetection],
        ocr_text: str = "",
        *,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> "SystemState":
        accepted = [
            detection
            for detection in detections
            if detection.confidence
            >= config.label_confidence_thresholds.get(
                _normalize_label(detection.label),
                config.min_detection_confidence,
            )
        ]
        active_tab_brokers = {
            broker
            for detection in accepted
            for broker, aliases in ACTIVE_BROKER_TAB_ALIASES.items()
            if _normalize_label(detection.label) in aliases
        }
        detected_brokers = active_tab_brokers or {
            broker
            for detection in accepted
            for broker, aliases in BROKER_LABEL_ALIASES.items()
            if _normalize_label(detection.label) in aliases
        }
        if len(detected_brokers) > 1:
            raise StateConflict(
                "Conflicting broker detections: "
                + ", ".join(sorted(detected_brokers))
            )
        if detected_brokers == {"exness"}:
            broker = Broker.EXNESS
        elif detected_brokers == {"pocket_option"}:
            broker = Broker.POCKET_OPTION
        else:
            broker = Broker.UNKNOWN

        normalized_ocr = " ".join(ocr_text.strip().upper().split())
        asset = None
        asset_type = AssetType.UNKNOWN
        if normalized_ocr:
            try:
                asset = Asset.parse(normalized_ocr)
            except ValueError:
                asset_type = AssetType.UNKNOWN
            else:
                asset_type = (
                    AssetType.OTC
                    if asset.value.endswith(" OTC")
                    else AssetType.STANDARD
                )

        alerts = tuple(
            sorted(
                {
                    detection.label.strip().lower()
                    for detection in accepted
                    if any(
                        _normalize_label(detection.label) in aliases
                        and canonical in config.safety_popup_labels
                        for canonical, aliases in SAFETY_POPUP_ALIASES.items()
                    )
                }
            )
        )
        return cls(
            broker=broker,
            asset_type=asset_type,
            asset=asset,
            active_asset_text=normalized_ocr,
            safety_alerts=alerts,
        )
