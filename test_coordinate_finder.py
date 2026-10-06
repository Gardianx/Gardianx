import unittest
from types import SimpleNamespace

from coordinate_finder import run_coordinate_finder


class _FakeListener:
    def __init__(self, on_press):
        self.on_press = on_press

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return None

    def join(self):
        self.on_press(SimpleNamespace(char="c"))
        self.on_press("f8")
        self.on_press("f8")
        self.on_press(SimpleNamespace(char=None))


class _FakeMouse:
    def __init__(self, positions):
        self._positions = iter(positions)

    @property
    def position(self):
        return next(self._positions)


class CoordinateFinderTests(unittest.TestCase):
    def test_reports_current_coordinates_each_time_f8_is_pressed(self):
        messages = []
        positions = iter(((1280, 720), (15, 42)))
        keyboard = SimpleNamespace(
            Key=SimpleNamespace(f8="f8", esc="esc"),
            Listener=_FakeListener,
        )

        run_coordinate_finder(
            keyboard_module=keyboard,
            mouse_controller_factory=lambda: _FakeMouse(positions),
            report=messages.append,
        )

        self.assertEqual(
            messages,
            [
                "Coordinate finder is running.",
                "Move your mouse anywhere on the screen and press F8 to print its coordinates.",
                "Press ESC to stop.",
                "Mouse position: X=1280, Y=720",
                "Mouse position: X=15, Y=42",
            ],
        )

    def test_escape_stops_the_listener(self):
        messages = []
        escape = object()

        class EscapeListener(_FakeListener):
            def join(self):
                self.assert_stopped = self.on_press("esc") is False
                if not self.assert_stopped:
                    self.on_press("f8")

        keyboard = SimpleNamespace(
            Key=SimpleNamespace(f8="f8", esc="esc"),
            Listener=EscapeListener,
        )

        run_coordinate_finder(
            keyboard_module=keyboard,
            mouse_controller_factory=lambda: SimpleNamespace(position=(1, 2)),
            report=messages.append,
        )

        self.assertIn("Stopping coordinate finder.", messages)
        self.assertNotIn("Mouse position: X=1, Y=2", messages)


if __name__ == "__main__":
    unittest.main()
