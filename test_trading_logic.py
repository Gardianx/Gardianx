import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from trading_logic import TradingAction, TradingLogic


class TradingLogicTests(unittest.TestCase):
    def setUp(self):
        self.buy_detection = SimpleNamespace(center=(120, 240))
        self.sell_detection = SimpleNamespace(center=(320, 440))
        self.buy_detector = Mock()
        self.sell_detector = Mock()
        self.executor = Mock()
        self.now = 0.0
        self.logic = TradingLogic(
            self.buy_detector,
            self.sell_detector,
            self.executor,
            clock=lambda: self.now,
        )

    def test_buy_signal_clicks_buy_and_sets_holding_state(self):
        self.buy_detector.find.return_value = self.buy_detection

        action = self.logic.step()

        self.assertEqual(action, TradingAction.BUY)
        self.assertTrue(self.logic.IS_HOLDING)
        self.executor.move_and_click.assert_called_once_with(120, 240)
        self.buy_detector.find.assert_called_once_with()
        self.sell_detector.find.assert_not_called()

    def test_while_holding_only_sell_signal_is_checked(self):
        self.logic.IS_HOLDING = True
        self.sell_detector.find.return_value = None

        action = self.logic.step()

        self.assertEqual(action, TradingAction.NONE)
        self.buy_detector.find.assert_not_called()
        self.sell_detector.find.assert_called_once_with()
        self.executor.move_and_click.assert_not_called()

    def test_sell_signal_clicks_sell_and_resets_holding_state(self):
        self.logic.IS_HOLDING = True
        self.sell_detector.find.return_value = self.sell_detection

        action = self.logic.step()

        self.assertEqual(action, TradingAction.SELL)
        self.assertFalse(self.logic.IS_HOLDING)
        self.executor.move_and_click.assert_called_once_with(320, 440)
        self.buy_detector.find.assert_not_called()

    def test_cooldown_blocks_detection_and_click_until_elapsed(self):
        self.buy_detector.find.return_value = self.buy_detection
        self.assertEqual(self.logic.step(), TradingAction.BUY)

        self.now = 2.99
        self.assertEqual(self.logic.step(), TradingAction.COOLDOWN)
        self.assertEqual(self.buy_detector.find.call_count, 1)
        self.assertEqual(self.executor.move_and_click.call_count, 1)

        self.now = 3.0
        self.sell_detector.find.return_value = None
        self.assertEqual(self.logic.step(), TradingAction.NONE)
        self.sell_detector.find.assert_called_once_with()

    def test_failed_click_does_not_change_state_or_start_cooldown(self):
        self.buy_detector.find.return_value = self.buy_detection
        self.executor.move_and_click.side_effect = RuntimeError("click failed")

        with self.assertRaisesRegex(RuntimeError, "click failed"):
            self.logic.step()

        self.assertFalse(self.logic.IS_HOLDING)
        self.now = 0.1
        self.executor.move_and_click.side_effect = None
        self.assertEqual(self.logic.step(), TradingAction.BUY)

    def test_rejects_invalid_cooldown(self):
        for cooldown in (-1, float("inf"), float("nan")):
            with self.subTest(cooldown=cooldown):
                with self.assertRaises(ValueError):
                    TradingLogic(
                        self.buy_detector,
                        self.sell_detector,
                        self.executor,
                        cooldown_seconds=cooldown,
                    )

    def test_predicted_buy_requires_buy_match_and_flat_state(self):
        self.assertEqual(
            self.logic.execute_prediction("BUY", None, self.sell_detection),
            TradingAction.NONE,
        )
        self.assertFalse(self.logic.IS_HOLDING)
        self.executor.move_and_click.assert_not_called()

        self.assertEqual(
            self.logic.execute_prediction("BUY", self.buy_detection, self.sell_detection),
            TradingAction.BUY,
        )
        self.assertTrue(self.logic.IS_HOLDING)
        self.executor.move_and_click.assert_called_once_with(120, 240)

    def test_predicted_sell_requires_holding_and_sell_match(self):
        self.assertEqual(
            self.logic.execute_prediction("SELL", self.buy_detection, self.sell_detection),
            TradingAction.NONE,
        )
        self.logic.IS_HOLDING = True
        self.assertEqual(
            self.logic.execute_prediction("SELL", self.buy_detection, None),
            TradingAction.NONE,
        )
        self.assertTrue(self.logic.IS_HOLDING)
        self.executor.move_and_click.assert_not_called()

    def test_configured_click_targets_override_detection_centers(self):
        logic = TradingLogic(
            self.buy_detector,
            self.sell_detector,
            self.executor,
            cooldown_seconds=0,
            buy_click_target=(1188, 23),
            sell_click_target=(1308, 21),
        )

        self.assertEqual(
            logic.execute_prediction("BUY", self.buy_detection, self.sell_detection),
            TradingAction.BUY,
        )
        self.assertTrue(logic.IS_HOLDING)
        self.assertEqual(
            logic.execute_prediction("SELL", self.buy_detection, self.sell_detection),
            TradingAction.SELL,
        )
        self.executor.move_and_click.assert_has_calls(
            [
                unittest.mock.call(1188, 23),
                unittest.mock.call(1308, 21),
            ]
        )

    def test_rejects_invalid_click_targets(self):
        for target in ((1,), (1, True), (1.5, 2)):
            with self.subTest(target=target):
                with self.assertRaises(ValueError):
                    TradingLogic(
                        self.buy_detector,
                        self.sell_detector,
                        self.executor,
                        buy_click_target=target,
                    )

    def test_invalid_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            self.logic.execute_prediction("MAYBE", None, None)


if __name__ == "__main__":
    unittest.main()
