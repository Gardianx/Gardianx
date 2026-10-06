import argparse
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from brokers import ExnessAdapter, PocketOptionAdapter
from config import AppConfig
from core.assets import Asset
from core.detection import YOLODetection
from core.safety import SafetyInterlock, SafetyInterlockError
from core.state import AssetType, Broker, StateConflict, SystemState
from core.vision import AssetOCR, UIVision
from main import _positive_float, build_parser, run
from utils.mouse import MouseController


def detection(label, confidence=0.95, box=(10, 20, 50, 40)):
    return YOLODetection(
        label=label,
        confidence=confidence,
        bounding_box=box,
        center=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
    )


class FakeImage:
    size = (1920, 1080)

    def __init__(self, crops=None):
        self.crops = crops if crops is not None else []

    def crop(self, bounds):
        self.crops.append(bounds)
        return self


class ModularArchitectureTests(unittest.TestCase):
    def test_pocket_option_specific_detections_identify_the_broker(self):
        state = SystemState.from_detections(
            [detection("pocket_buy"), detection("pocket_amount")],
            "AED/CNY OTC",
        )

        self.assertEqual(state.broker, Broker.POCKET_OPTION)
        self.assertEqual(state.asset, Asset.AED_CNY_OTC)
        self.assertEqual(state.asset_type, AssetType.OTC)

    def test_active_tab_identifies_broker_even_when_other_broker_ui_is_visible(self):
        state = SystemState.from_detections(
            [
                detection("active_pocket_option_tab"),
                detection("exness"),
                detection("pocket_amount"),
            ],
            "AED/CNY OTC",
        )

        self.assertEqual(state.broker, Broker.POCKET_OPTION)

    def test_conflicting_active_tabs_are_rejected(self):
        with self.assertRaises(StateConflict):
            SystemState.from_detections(
                [
                    detection("active_exness_tab"),
                    detection("active_pocket_option_tab"),
                ],
                "EUR/USD",
            )

    def test_configured_pocket_option_otc_symbols_parse(self):
        symbols = {
            "EUR/USD_OTC": Asset.EUR_USD_OTC,
            "AED/CNY_OTC": Asset.AED_CNY_OTC,
            "AUD/NZD_OTC": Asset.AUD_NZD_OTC,
            "EUR/NZD_OTC": Asset.EUR_NZD_OTC,
            "CAD/CHF_OTC": Asset.CAD_CHF_OTC,
            "USD/JPY_OTC": Asset.USD_JPY_OTC,
            "EUR/RUB_OTC": Asset.EUR_RUB_OTC,
            "GBP/JPY_OTC": Asset.GBP_JPY_OTC,
            "GBP/CAD_OTC": Asset.GBP_CAD_OTC,
        }
        for text, expected_asset in symbols.items():
            with self.subTest(text=text):
                self.assertEqual(Asset.parse(text), expected_asset)

    def test_broker_asset_and_popup_state_are_parsed_from_detections(self):
        state = SystemState.from_detections(
            [
                detection("broker_pocket_option"),
                detection("active_asset_label"),
                detection("market_closed_popup"),
            ],
            "EUR/USD OTC",
        )

        self.assertEqual(state.broker, Broker.POCKET_OPTION)
        self.assertEqual(state.asset, Asset.EUR_USD_OTC)
        self.assertEqual(state.asset_type, AssetType.OTC)
        self.assertEqual(state.safety_alerts, ("market_closed_popup",))

    def test_conflicting_broker_detection_is_rejected(self):
        with self.assertRaises(StateConflict):
            SystemState.from_detections(
                [detection("exness"), detection("pocket_option")],
                "EUR/USD",
            )

    def test_safety_interlock_fails_closed_on_unknown_and_exness_otc(self):
        interlock = SafetyInterlock(logger=Mock())
        for state in (
            SystemState(Broker.UNKNOWN, AssetType.STANDARD, Asset.EUR_USD, "EUR/USD"),
            SystemState(
                Broker.EXNESS,
                AssetType.OTC,
                Asset.EUR_USD_OTC,
                "EUR/USD OTC",
            ),
            SystemState(
                Broker.POCKET_OPTION,
                AssetType.STANDARD,
                Asset.EUR_USD,
                "EUR/USD",
                ("insufficient_balance",),
            ),
        ):
            with self.subTest(state=state), self.assertRaises(SafetyInterlockError):
                interlock.validate(state)

    def test_asset_ocr_crops_active_asset_box_and_reads_text(self):
        screenshot = FakeImage()
        detector = Mock()
        detector.detect.return_value = [
            detection("active_asset_label", box=(-5, 8, 60, 28))
        ]
        reader = Mock()
        reader.image_to_string.return_value = "EUR/USD OTC"
        vision = UIVision(detector, AssetOCR(reader=reader))

        found = vision.detect_ui_elements(screenshot)
        text = vision.read_active_asset(screenshot, found)

        self.assertEqual(text, "EUR/USD OTC")
        self.assertEqual(screenshot.crops, [(0, 8, 60, 28)])
        reader.image_to_string.assert_called_once_with(screenshot, lang="eng")

    def test_exness_requires_buy_sell_and_lot_size_controls(self):
        mouse = Mock()
        adapter = ExnessAdapter(mouse=mouse)
        detections = [
            detection("lot_size_input"),
            detection("buy_button"),
            detection("sell_button"),
        ]

        result = adapter.execute(Asset.EUR_USD, "buy", 0.02, detections)

        self.assertEqual(result.steps, ("set lot size to 0.02", "click buy"))
        mouse.replace_text.assert_called_once_with(30, 30, "0.02")
        mouse.click.assert_called_once_with(30, 30)
        with self.assertRaises(ValueError):
            adapter.execute(Asset.EUR_USD, "call", 0.02, detections)

    def test_pocket_option_sets_investment_and_expiration_then_call_or_put(self):
        mouse = Mock()
        adapter = PocketOptionAdapter(mouse=mouse)
        detections = [
            detection("investment_input", box=(0, 0, 20, 20)),
            detection("expiration_input", box=(20, 20, 40, 40)),
            detection("call_button", box=(40, 40, 60, 60)),
        ]

        result = adapter.execute(
            Asset.EUR_USD_OTC,
            "call",
            5,
            detections,
            expiration_seconds=60,
        )

        self.assertEqual(result.action, "CALL")
        self.assertEqual(
            mouse.replace_text.call_args_list,
            [
                unittest.mock.call(10, 10, "5"),
                unittest.mock.call(30, 30, "60"),
            ],
        )
        mouse.click.assert_called_once_with(50, 50)

    def test_pocket_option_uses_user_defined_control_labels(self):
        mouse = Mock()
        adapter = PocketOptionAdapter(mouse=mouse)
        detections = [
            detection("pocket_amount", box=(0, 0, 20, 20)),
            detection("pocket_time", box=(20, 20, 40, 40)),
            detection("pocket_buy", box=(40, 40, 60, 60)),
        ]

        result = adapter.execute(
            Asset.AED_CNY_OTC,
            "call",
            5,
            detections,
            expiration_seconds=60,
        )

        self.assertEqual(result.action, "CALL")
        self.assertEqual(
            mouse.replace_text.call_args_list,
            [
                unittest.mock.call(10, 10, "5"),
                unittest.mock.call(30, 30, "60"),
            ],
        )
        mouse.click.assert_called_once_with(50, 50)

    def test_mouse_is_dry_run_by_default_and_safety_checked_before_input(self):
        mouse_module = Mock()
        guard = Mock()
        controller = MouseController(
            pyautogui_module=mouse_module,
            check_safety=guard,
        )

        controller.click(1, 2)
        controller.replace_text(3, 4, "0.1")

        self.assertEqual(guard.call_count, 2)
        mouse_module.click.assert_not_called()
        mouse_module.write.assert_not_called()

    def test_orchestrator_runs_continuously_and_consumes_action_once(self):
        detector = Mock()
        detector.detect.return_value = [
            detection("exness"),
            detection("active_asset_label"),
            detection("lot_size_input"),
            detection("buy_button"),
        ]
        reader = Mock()
        reader.image_to_string.return_value = "EUR/USD"
        vision = UIVision(detector, AssetOCR(reader=reader))
        screenshot_provider = Mock(return_value=FakeImage())
        pyautogui = Mock()
        reports = []
        stop = threading.Event()

        def report(message):
            reports.append(message)
            if len([line for line in reports if line.startswith("Broker=")]) == 2:
                stop.set()

        run(
            vision,
            action="buy",
            expected_asset=Asset.EUR_USD,
            amount=0.01,
            screenshot_provider=screenshot_provider,
            stop_event=stop,
            mouse_factory=lambda *, execute, check_safety: MouseController(
                pyautogui_module=pyautogui,
                execute=execute,
                check_safety=check_safety,
            ),
            report=report,
            max_iterations=2,
            config=AppConfig(scan_interval_seconds=0),
        )

        self.assertEqual(screenshot_provider.call_count, 4)
        pyautogui.click.assert_not_called()
        pyautogui.write.assert_not_called()
        self.assertTrue(any("Dry run only" in message for message in reports))

    def test_orchestrator_pauses_on_otc_mismatch_without_creating_mouse(self):
        detector = Mock()
        detector.detect.return_value = [
            detection("exness"),
            detection("active_asset_label"),
        ]
        reader = Mock()
        reader.image_to_string.return_value = "EUR/USD OTC"
        vision = UIVision(detector, AssetOCR(reader=reader))
        stop = threading.Event()
        reports = []

        def report(message):
            reports.append(message)
            if "CRITICAL SAFETY INTERLOCK" in message:
                stop.set()

        mouse_factory = Mock()
        with patch("main.LOGGER.exception"):
            result = run(
                vision,
                action="buy",
                expected_asset=Asset.EUR_USD,
                amount=0.01,
                screenshot_provider=Mock(return_value=FakeImage()),
                stop_event=stop,
                mouse_factory=mouse_factory,
                report=report,
                max_iterations=1,
            )

        self.assertEqual(result, 2)
        mouse_factory.assert_not_called()
        self.assertTrue(any("CRITICAL SAFETY INTERLOCK" in item for item in reports))

    def test_parser_requires_explicit_asset_and_amount_with_action(self):
        args = build_parser().parse_args(["--action", "buy", "--asset", "EUR/USD", "--amount", "0.1"])
        self.assertEqual(args.action, "buy")
        self.assertEqual(args.amount, 0.1)
        self.assertFalse(args.execute)
        for invalid in ("0", "nan", "inf", "-2"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _positive_float(invalid)


if __name__ == "__main__":
    unittest.main()
