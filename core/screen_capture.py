"""Full-screen capture adapter."""

from typing import Any, Callable


def capture_screen(screenshot_provider: Callable[[], Any] | None = None) -> Any:
    """Capture and return the full screen as a Pillow image."""
    if screenshot_provider is None:
        import pyautogui

        screenshot_provider = pyautogui.screenshot
    return screenshot_provider()
