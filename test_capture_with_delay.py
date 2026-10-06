import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from capture_with_delay import capture_templates


class CaptureWithDelayTests(unittest.TestCase):
    def test_countdown_precedes_full_screen_capture_and_saves_calibrated_crops(self):
        events = []
        crops = []
        report = Mock(side_effect=lambda message, **kwargs: events.append(("print", message)))
        sleep = Mock(side_effect=lambda seconds: events.append(("sleep", seconds)))

        class FakeCrop:
            def __init__(self, bounds):
                self.bounds = bounds

            def save(self, path, format):
                crops.append((Path(path), self.bounds, format))

        class FakeScreenshot:
            size = (1920, 1080)

            def crop(self, bounds):
                events.append(("crop", bounds))
                return FakeCrop(bounds)

        def screenshot_provider():
            events.append(("screenshot",))
            return FakeScreenshot()

        with tempfile.TemporaryDirectory() as temp_dir:
            saved = capture_templates(
                temp_dir,
                screenshot_provider=screenshot_provider,
                sleep=sleep,
                report=report,
            )

        self.assertEqual(
            [event for event in events if event[0] == "print"][:6],
            [
                ("print", "Switch to Firefox and maximize it now..."),
                ("print", "5..."),
                ("print", "4..."),
                ("print", "3..."),
                ("print", "2..."),
                ("print", "1..."),
            ],
        )
        self.assertEqual(sleep.call_count, 5)
        self.assertEqual([call.args for call in sleep.call_args_list], [(1,)] * 5)
        self.assertGreater(
            next(index for index, event in enumerate(events) if event[0] == "screenshot"),
            max(index for index, event in enumerate(events) if event[0] == "sleep"),
        )
        self.assertEqual(
            [(crop[0].name, crop[1]) for crop in crops],
            [
                ("buy_button.png", (1158, 8, 1218, 38)),
                ("sell_button.png", (1278, 6, 1338, 36)),
            ],
        )
        self.assertTrue(all(crop[2] == "PNG" for crop in crops))
        self.assertEqual(
            saved,
            [
                Path(temp_dir) / "buy_button.png",
                Path(temp_dir) / "sell_button.png",
            ],
        )

    def test_rejects_crop_outside_screen_before_creating_output(self):
        screenshot = SimpleNamespace(size=(100, 100))

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "templates"
            with self.assertRaisesRegex(ValueError, "outside the captured screen"):
                capture_templates(
                    output_dir,
                    screenshot_provider=Mock(return_value=screenshot),
                    sleep=Mock(),
                    report=Mock(),
                )
            self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
