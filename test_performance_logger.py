import csv
from datetime import datetime
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from performance_logger import FIELDNAMES, log_trade


class PerformanceLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.csv_path = Path(self.temp_dir.name) / "trade_history.csv"
        self.path_patch = patch("performance_logger.TRADE_HISTORY_FILE", self.csv_path)
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_creates_csv_with_header_and_trade_row(self):
        returned_path = log_trade("buy", 123.45, 0.91)

        self.assertEqual(returned_path, self.csv_path)
        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            self.assertEqual(reader.fieldnames, list(FIELDNAMES))
            rows = list(reader)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Action"], "BUY")
        self.assertEqual(rows[0]["Approximate Price or Asset State"], "123.45")
        self.assertEqual(rows[0]["Model Confidence Score"], "0.91")
        timestamp = datetime.fromisoformat(rows[0]["Timestamp"])
        self.assertIsNotNone(timestamp.tzinfo)
        self.assertRegex(rows[0]["Timestamp"], r"\.\d{6}[+-]\d{2}:\d{2}$")

    def test_appends_rows_without_duplicate_headers_and_accepts_asset_state(self):
        log_trade("SELL", 125.0, 0.88)
        log_trade("HOLD", "price unavailable; flat", 0.5)

        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

        self.assertEqual(len(rows), 2)
        self.assertEqual([row["Action"] for row in rows], ["SELL", "HOLD"])
        self.assertEqual(rows[1]["Approximate Price or Asset State"], "price unavailable; flat")

    def test_writes_header_when_file_exists_but_is_empty(self):
        self.csv_path.touch()

        log_trade("HOLD", "no signal", 0)

        with self.csv_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Action"], "HOLD")

    def test_rejects_invalid_action_price_and_confidence(self):
        invalid_calls = (
            ("WAIT", 10, 0.5, ValueError),
            ("BUY", "", 0.5, ValueError),
            ("BUY", float("inf"), 0.5, ValueError),
            ("BUY", 10, -0.1, ValueError),
            ("BUY", 10, float("nan"), ValueError),
            ("BUY", 10, "high", TypeError),
        )
        for action, price, confidence, error in invalid_calls:
            with self.subTest(action=action, price=price, confidence=confidence):
                with self.assertRaises(error):
                    log_trade(action, price, confidence)
        self.assertFalse(self.csv_path.exists())


if __name__ == "__main__":
    unittest.main()
