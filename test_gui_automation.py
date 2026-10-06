import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from safe_executor import EmergencyStop, SafeExecutor
from screen_detector import ImageDetector, Region


class ScreenDetectorTests(unittest.TestCase):
    def test_finds_template_and_returns_absolute_center(self):
        screenshot = object()
        cv2 = SimpleNamespace(
            IMREAD_GRAYSCALE=0,
            COLOR_RGB2GRAY=1,
            COLOR_RGBA2GRAY=2,
            TM_CCOEFF_NORMED=3,
            imread=Mock(return_value=SimpleNamespace(shape=(6, 8))),
            cvtColor=Mock(return_value=SimpleNamespace(shape=(20, 30))),
            matchTemplate=Mock(return_value="match-result"),
            minMaxLoc=Mock(return_value=(0.1, 0.95, (0, 0), (4, 7))),
        )
        numpy = SimpleNamespace(
            asarray=Mock(return_value=SimpleNamespace(ndim=3, shape=(20, 30, 3)))
        )
        screenshot_provider = Mock(return_value=screenshot)
        detector = ImageDetector(
            "indicator.png",
            Region(100, 200, 30, 20),
            cv2_module=cv2,
            numpy_module=numpy,
            screenshot_provider=screenshot_provider,
        )

        detection = detector.find()

        self.assertEqual(detection.center, (108, 210))
        self.assertEqual(detection.confidence, 0.95)
        screenshot_provider.assert_called_once_with(region=(100, 200, 30, 20))

    def test_returns_none_when_match_is_below_threshold(self):
        cv2 = SimpleNamespace(
            IMREAD_GRAYSCALE=0,
            COLOR_RGB2GRAY=1,
            COLOR_RGBA2GRAY=2,
            TM_CCOEFF_NORMED=3,
            imread=Mock(return_value=SimpleNamespace(shape=(2, 2))),
            matchTemplate=Mock(return_value="match-result"),
            minMaxLoc=Mock(return_value=(0.1, 0.5, (0, 0), (0, 0))),
        )
        numpy = SimpleNamespace(
            asarray=Mock(return_value=SimpleNamespace(ndim=2, shape=(20, 20)))
        )
        detector = ImageDetector(
            "indicator.png",
            Region(0, 0, 20, 20),
            cv2_module=cv2,
            numpy_module=numpy,
            screenshot_provider=Mock(return_value=object()),
        )

        self.assertIsNone(detector.find())


class SafeExecutorTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = Path(self.temp_dir.name) / "actions.log"
        self.mouse = SimpleNamespace(x=0, y=0)
        self.pyautogui = SimpleNamespace(
            PAUSE=0.1,
            position=Mock(side_effect=lambda: (self.mouse.x, self.mouse.y)),
            moveTo=Mock(side_effect=self._move),
            click=Mock(),
        )
        self.listener = Mock()
        self.executor = SafeExecutor(
            self.log_path,
            pyautogui_module=self.pyautogui,
            listener_factory=Mock(return_value=self.listener),
            esc_key="ESC",
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _move(self, x, y, duration=0):
        self.mouse.x, self.mouse.y = x, y

    def test_moves_clicks_logs_and_restores_pause(self):
        with self.executor:
            self.executor.move_and_click(20, 30, duration=0)

        self.pyautogui.moveTo.assert_called_once_with(20, 30, duration=0)
        self.pyautogui.click.assert_called_once_with(20, 30)
        self.assertEqual(self.pyautogui.PAUSE, 0.1)
        log = self.log_path.read_text(encoding="utf-8")
        self.assertIn("MOVE_COMPLETE | x=20 y=30", log)
        self.assertIn("CLICK_COMPLETE | x=20 y=30", log)

    def test_escape_prevents_click(self):
        with self.executor:
            self.executor._on_press("ESC")
            with self.assertRaises(EmergencyStop):
                self.executor.move_and_click(20, 30, duration=0)

        self.pyautogui.moveTo.assert_not_called()
        self.pyautogui.click.assert_not_called()

    def test_escape_during_movement_interrupts_before_click(self):
        original_move = self.pyautogui.moveTo

        def move_then_press_escape(x, y, duration=0):
            self._move(x, y, duration)
            self.executor._on_press("ESC")

        self.pyautogui.moveTo.side_effect = move_then_press_escape
        with self.executor:
            with self.assertRaises(EmergencyStop):
                self.executor.move_and_click(20, 30, duration=0.1)

        self.assertEqual(original_move.call_count, 1)
        self.pyautogui.click.assert_not_called()
        self.assertIn("MOVE_INTERRUPTED | x=20 y=30", self.log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
