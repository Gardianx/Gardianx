"""Append trade and signal records to a local CSV history."""

import csv
from datetime import datetime
import math
from pathlib import Path


TRADE_HISTORY_FILE = Path(__file__).resolve().parent / "trade_history.csv"
FIELDNAMES = (
    "Timestamp",
    "Action",
    "Approximate Price or Asset State",
    "Model Confidence Score",
)
VALID_ACTIONS = {"BUY", "SELL", "HOLD"}


def log_trade(action: str, price: float | int | str, confidence_score: float) -> Path:
    """Append a validated trade/signal row and create the CSV with a header."""
    if not isinstance(action, str) or action.upper() not in VALID_ACTIONS:
        raise ValueError("Action must be BUY, SELL, or HOLD.")
    normalized_action = action.upper()

    if isinstance(price, bool) or not isinstance(price, (float, int, str)):
        raise TypeError("Price must be numeric or a descriptive asset-state string.")
    if isinstance(price, str):
        if not price.strip():
            raise ValueError("Asset-state description cannot be empty.")
        recorded_price: float | int | str = price.strip()
    else:
        if not math.isfinite(price):
            raise ValueError("Numeric price must be finite.")
        recorded_price = price

    if isinstance(confidence_score, bool) or not isinstance(confidence_score, (float, int)):
        raise TypeError("Confidence score must be a number between 0 and 1.")
    if not math.isfinite(confidence_score) or not 0 <= confidence_score <= 1:
        raise ValueError("Confidence score must be finite and between 0 and 1.")

    timestamp = datetime.now().astimezone().isoformat(timespec="microseconds")
    with TRADE_HISTORY_FILE.open("a+", newline="", encoding="utf-8") as csv_file:
        csv_file.seek(0, 2)
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        if csv_file.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "Timestamp": timestamp,
                "Action": normalized_action,
                "Approximate Price or Asset State": recorded_price,
                "Model Confidence Score": confidence_score,
            }
        )

    return TRADE_HISTORY_FILE
