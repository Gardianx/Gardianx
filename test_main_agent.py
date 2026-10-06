import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from main_agent import (
    DryRunExecutor,
    _build_features,
    _cooldown,
    _interval,
    _threshold,
    _validate_prediction,
    run_agent,
)
from model_brain import SignalPrediction
from safe_executor import EmergencyStop, KillSwitch
from screen_detector import Detection
from trading_logic import TradingAction, TradingLogic


class MainAgentTests(unittest.TestCase):
    def test_run_agent_checks_logic_and_waits_until_stop(self):
        brain = Mock()
        brain.predict_signal.return_value = SignalPrediction("HOLD", 0.0)
        logic = Mock()
        logic.IS_HOLDING = False
        logic.execute_prediction.return_value = TradingAction.NONE
        buy_detector = Mock()
        buy_detector.find.return_value = None
        sell_detector = Mock()
        sell_detector.find.return_value = None
        executor = Mock()
        executor.wait_for_stop.side_effect = [False, True]
        report = Mock()
        trade_logger = Mock()

        run_agent(
            brain,
            logic,
            buy_detector,
            sell_detector,
            executor,
            1.5,
            report=report,
            trade_logger=trade_logger,
        )

        self.assertEqual(brain.predict_signal.call_count, 2)
        self.assertEqual(logic.execute_prediction.call_count, 2)
        self.assertEqual(executor.check_kill_switch.call_count, 8)
        executor.wait_for_stop.assert_any_call(1.5)
        self.assertEqual(executor.wait_for_stop.call_count, 2)
        self.assertEqual(trade_logger.call_count, 2)
        trade_logger.assert_called_with(
            "HOLD",
            "price unavailable; model_signal=HOLD; holding=False; "
            "buy_match=0.000; sell_match=0.000; model_placeholder=True; mode=live",
            0.0,
        )
        self.assertEqual(report.call_count, 2)

    def test_run_agent_passes_vision_and_position_features_to_model(self):
        buy = Detection(confidence=0.95, center=(10, 20))
        sell = Detection(confidence=0.8, center=(30, 40))
        brain = Mock()
        brain.predict_signal.return_value = SignalPrediction("HOLD", 0.0)
        logic = Mock()
        logic.IS_HOLDING = True
        logic.execute_prediction.return_value = TradingAction.NONE
        buy_detector = Mock()
        buy_detector.find.return_value = buy
        sell_detector = Mock()
        sell_detector.find.return_value = sell
        executor = Mock()
        executor.wait_for_stop.return_value = True

        run_agent(
            brain,
            logic,
            buy_detector,
            sell_detector,
            executor,
            1.0,
            report=Mock(),
            trade_logger=Mock(),
        )

        brain.predict_signal.assert_called_once_with(
            {
                "buy_detected": 1.0,
                "buy_confidence": 0.95,
                "sell_detected": 1.0,
                "sell_confidence": 0.8,
                "is_holding": 1.0,
            }
        )
        logic.execute_prediction.assert_called_once_with("HOLD", buy, sell)

    def test_kill_switch_stops_before_model_or_mouse_action(self):
        brain = Mock()
        logic = Mock()
        buy_detector = Mock()
        sell_detector = Mock()
        executor = Mock()
        executor.check_kill_switch.side_effect = EmergencyStop("ESC")

        with self.assertRaises(EmergencyStop):
            run_agent(
                brain,
                logic,
                buy_detector,
                sell_detector,
                executor,
                1.0,
                trade_logger=Mock(),
            )

        brain.predict_signal.assert_not_called()
        logic.execute_prediction.assert_not_called()

    def test_model_buy_runs_through_visual_and_position_safeguards(self):
        buy = Detection(confidence=0.95, center=(10, 20))
        sell = Detection(confidence=0.8, center=(30, 40))
        brain = Mock()
        brain.predict_signal.return_value = SignalPrediction("BUY", 0.87)
        executor = Mock()
        executor.wait_for_stop.return_value = True
        logic = TradingLogic(
            Mock(),
            Mock(),
            executor,
        )
        buy_detector = Mock()
        buy_detector.find.return_value = buy
        sell_detector = Mock()
        sell_detector.find.return_value = sell
        trade_logger = Mock()

        run_agent(
            brain,
            logic,
            buy_detector,
            sell_detector,
            executor,
            1.0,
            report=Mock(),
            trade_logger=trade_logger,
        )

        self.assertTrue(logic.IS_HOLDING)
        executor.move_and_click.assert_called_once_with(10, 20)
        self.assertEqual(trade_logger.call_args.args[0], "BUY")
        self.assertEqual(trade_logger.call_args.args[2], 0.87)

    def test_kill_switch_listener_sets_stop_event_on_escape(self):
        listener = Mock()
        listener_factory = Mock(return_value=listener)
        kill_switch = KillSwitch(listener_factory=listener_factory, esc_key="ESC")

        kill_switch.start()
        callback = listener_factory.call_args.kwargs["on_press"]
        self.assertFalse(callback("OTHER"))
        self.assertFalse(kill_switch.wait(0))
        self.assertFalse(callback("ESC"))
        self.assertTrue(kill_switch.wait(0))
        with self.assertRaises(EmergencyStop):
            kill_switch.check()
        kill_switch.stop()
        listener.join.assert_called_once_with()

    def test_build_features_encodes_detected_templates_and_position(self):
        features = _build_features(
            Detection(confidence=0.7, center=(1, 2)),
            None,
            is_holding=False,
        )

        self.assertEqual(
            features,
            {
                "buy_detected": 1.0,
                "buy_confidence": 0.7,
                "sell_detected": 0.0,
                "sell_confidence": 0.0,
                "is_holding": 0.0,
            },
        )

    def test_prediction_validation_checks_signal_and_confidence(self):
        self.assertEqual(_validate_prediction(SignalPrediction("buy", 0.75)), ("BUY", 0.75))
        for prediction in (
            SimpleNamespace(signal="BUY", confidence_score=0.5),
            SignalPrediction("INVALID", 0.5),
            SignalPrediction("BUY", 1.5),
        ):
            with self.subTest(prediction=prediction):
                with self.assertRaises((TypeError, ValueError)):
                    _validate_prediction(prediction)

    def test_dry_run_executor_logs_without_mouse_actions(self):
        pyautogui = Mock()
        executor = DryRunExecutor(
            "unused.log",
            pyautogui_module=pyautogui,
            listener_factory=Mock(),
            esc_key="ESC",
        )
        executor.log_action = Mock()

        executor.move_and_click(12, 34)

        executor.log_action.assert_called_once_with("DRY_RUN", 12, 34)
        pyautogui.moveTo.assert_not_called()
        pyautogui.click.assert_not_called()

    def test_cli_numeric_options_enforce_safe_ranges(self):
        self.assertEqual(_interval("1"), 1.0)
        self.assertEqual(_interval("2"), 2.0)
        self.assertEqual(_cooldown("0"), 0.0)
        self.assertEqual(_threshold("0.9"), 0.9)
        for invalid in ("0.99", "2.01", "nan", "inf"):
            with self.subTest(interval=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _interval(invalid)
        for invalid in ("-1", "nan", "inf"):
            with self.subTest(cooldown=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _cooldown(invalid)
        for invalid in ("-0.1", "1.1", "nan"):
            with self.subTest(threshold=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _threshold(invalid)


if __name__ == "__main__":
    unittest.main()
