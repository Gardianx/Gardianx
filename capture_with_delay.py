"""Capture fixed Buy/Sell templates after a short desktop-switch countdown."""

from pathlib import Path
import time
from typing import Callable


PROJECT_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = PROJECT_DIR / "templates"
BUTTON_CROPS = {
    "buy_button.png": (1188, 23),
    "sell_button.png": (1308, 21),
}
CROP_WIDTH = 60
CROP_HEIGHT = 30


def capture_templates(
    output_dir: str | Path = TEMPLATE_DIR,
    *,
    screenshot_provider: Callable | None = None,
    sleep: Callable[[float], None] = time.sleep,
    report: Callable[..., None] = print,
) -> list[Path]:
    """Capture the full screen and save 60x30 crops around the calibrated buttons."""
    if screenshot_provider is None:
        import pyautogui

        screenshot_provider = pyautogui.screenshot

    report("Switch to Firefox and maximize it now...", flush=True)
    for count in (5, 4, 3, 2, 1):
        report(f"{count}...", flush=True)
        sleep(1)

    screenshot = screenshot_provider()
    screen_width, screen_height = screenshot.size
    crop_bounds = {}
    for filename, (center_x, center_y) in BUTTON_CROPS.items():
        left = center_x - CROP_WIDTH // 2
        top = center_y - CROP_HEIGHT // 2
        bounds = (left, top, left + CROP_WIDTH, top + CROP_HEIGHT)
        if bounds[0] < 0 or bounds[1] < 0 or bounds[2] > screen_width or bounds[3] > screen_height:
            raise ValueError(f"{filename} crop is outside the captured screen bounds.")
        crop_bounds[filename] = bounds

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for filename, bounds in crop_bounds.items():
        output_path = destination / filename
        screenshot.crop(bounds).save(output_path, format="PNG")
        report(f"Saved {output_path}")
        saved_paths.append(output_path)
    return saved_paths


def main() -> int:
    capture_templates()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
