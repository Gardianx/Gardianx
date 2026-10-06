"""Supported symbols, including OTC variants as distinct asset identities."""

from enum import Enum
import re


class Asset(str, Enum):
    EUR_USD = "EUR/USD"
    EUR_USD_OTC = "EUR/USD OTC"
    AED_CNY_OTC = "AED/CNY OTC"
    AUD_NZD_OTC = "AUD/NZD OTC"
    EUR_NZD_OTC = "EUR/NZD OTC"
    CAD_CHF_OTC = "CAD/CHF OTC"
    USD_JPY_OTC = "USD/JPY OTC"
    EUR_RUB_OTC = "EUR/RUB OTC"
    GBP_JPY_OTC = "GBP/JPY OTC"
    GBP_CAD_OTC = "GBP/CAD OTC"

    @classmethod
    def parse(cls, value: str) -> "Asset":
        normalized = " ".join(value.strip().upper().replace("_", " ").split())
        normalized = re.sub(r"\s*/\s*", "/", normalized)
        aliases = {}
        for asset in cls:
            aliases[asset.value] = asset
            aliases[asset.value.replace("/", " / ")] = asset
            aliases[asset.value.replace("/", "")] = asset
        try:
            return aliases[normalized]
        except KeyError as error:
            raise ValueError(
                f"Unsupported asset {value!r}; choose a configured asset: "
                + ", ".join(asset.value for asset in cls)
            ) from error
