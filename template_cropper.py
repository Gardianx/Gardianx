"""Capture the screen and crop Buy/Sell reference images interactively."""

from pathlib import Path
from typing import Callable


TEMPLATE_NAMES = ("buy_button.png", "sell_button.png")


def crop_templates(
    output_dir: str | Path = Path(__file__).resolve().parent / "templates",
    *,
    screenshot_provider: Callable | None = None,
    cv2_module=None,
    numpy_module=None,
) -> list[Path]:
    if screenshot_provider is None:
        import pyautogui

        screenshot_provider = pyautogui.screenshot
    if cv2_module is None:
        import cv2 as cv2_module
    if numpy_module is None:
        import numpy as numpy_module

    screenshot = numpy_module.asarray(screenshot_provider())
    if screenshot.ndim != 3 or screenshot.shape[2] not in (3, 4):
        raise ValueError("Expected a full-color RGB or RGBA screen capture.")

    color_conversion = (
        cv2_module.COLOR_RGB2BGR
        if screenshot.shape[2] == 3
        else cv2_module.COLOR_RGBA2BGR
    )
    screen = cv2_module.cvtColor(screenshot, color_conversion)
    screen_height, screen_width = screen.shape[:2]
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    try:
        for name in TEMPLATE_NAMES:
            window_title = f"Select {name} - drag region, ENTER to save, C to skip"
            x, y, width, height = cv2_module.selectROI(
                window_title, screen, showCrosshair=True, fromCenter=False
            )
            if width <= 0 or height <= 0:
                print(f"Skipped {name}.")
                continue
            if x < 0 or y < 0 or x + width > screen_width or y + height > screen_height:
                raise ValueError(f"Selected crop for {name} is outside the screenshot.")

            crop = screen[y : y + height, x : x + width]
            path = destination / name
            if not cv2_module.imwrite(str(path), crop):
                raise OSError(f"Could not save template image: {path}")
            saved.append(path)
            print(f"Saved {path}")
    finally:
        cv2_module.destroyAllWindows()

    return saved


def main() -> int:
    crop_templates()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
