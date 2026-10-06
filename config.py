"""Central configuration for screen recognition and safe automation defaults."""

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BROKER_LABEL_ALIASES = {
    "exness": frozenset({"exness", "exness_logo", "broker_exness"}),
    "pocket_option": frozenset(
        {
            "pocket_option",
            "pocket_option_logo",
            "broker_pocket_option",
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
        }
    ),
}
SAFETY_POPUP_ALIASES = {
    "insufficient_balance": frozenset(
        {"insufficient_balance", "insufficient_balance_popup"}
    ),
    "market_closed": frozenset({"market_closed", "market_closed_popup"}),
}


@dataclass(frozen=True)
class AppConfig:
    model_path: Path = PROJECT_DIR / "models" / "best.pt"
    scan_interval_seconds: float = 0.5
    model_confidence: float = 0.25
    min_detection_confidence: float = 0.55
    asset_label_min_confidence: float = 0.65
    ocr_language: str = "eng"
    exness_rejects_otc: bool = True
    safety_popup_labels: frozenset[str] = frozenset(
        {
            "insufficient_balance",
            "insufficient_balance_popup",
            "market_closed",
            "market_closed_popup",
        }
    )
    label_confidence_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "buy_button": 0.65,
            "sell_button": 0.65,
            "call_button": 0.65,
            "put_button": 0.65,
            "active_asset_label": 0.60,
            "exness": 0.70,
            "exness_logo": 0.70,
            "broker_exness": 0.70,
            "pocket_option": 0.70,
            "pocket_option_logo": 0.70,
            "broker_pocket_option": 0.70,
            "otc_dropdown_menu": 0.65,
            "pocket_buy": 0.65,
            "pocket_sell": 0.65,
            "pocket_time": 0.60,
            "pocket_amount": 0.60,
            "pocket_payout": 0.60,
            "pocket_trades": 0.60,
            "pocket_wallet": 0.60,
            "pocket_candle_timer": 0.60,
            "bearish_otc_candle": 0.60,
            "bullish_otc_candle": 0.60,
            "insufficient_balance": 0.60,
            "insufficient_balance_popup": 0.60,
            "market_closed": 0.60,
            "market_closed_popup": 0.60,
            "lot_size_input": 0.60,
            "investment_input": 0.60,
            "expiration_input": 0.60,
        }
    )


DEFAULT_CONFIG = AppConfig()
