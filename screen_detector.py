"""Detect a template image inside a selected screen region."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Region width and height must be positive.")

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.left, self.top, self.width, self.height)


@dataclass(frozen=True)
class Detection:
    confidence: float
    center: tuple[int, int]


class ImageDetector:
    def __init__(
        self,
        template_path: str | Path,
        region: Region,
        threshold: float = 0.9,
        *,
        cv2_module=None,
        numpy_module=None,
        screenshot_provider: Callable | None = None,
    ) -> None:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        if cv2_module is None:
            import cv2 as cv2_module
        if numpy_module is None:
            import numpy as numpy_module
        if screenshot_provider is None:
            import pyautogui

            screenshot_provider = pyautogui.screenshot

        self._cv2 = cv2_module
        self._numpy = numpy_module
        self._screenshot = screenshot_provider
        self._region = region
        self._threshold = threshold
        self._template = self._cv2.imread(str(template_path), self._cv2.IMREAD_GRAYSCALE)
        if self._template is None:
            raise FileNotFoundError(f"Could not load template image: {template_path}")

    def find(self, screenshot=None) -> Detection | None:
        if screenshot is None:
            screenshot = self._screenshot(region=self._region.as_tuple())
        image = self._numpy.asarray(screenshot)
        if image.ndim == 3:
            color_conversion = (
                self._cv2.COLOR_RGBA2GRAY if image.shape[2] == 4 else self._cv2.COLOR_RGB2GRAY
            )
            image = self._cv2.cvtColor(image, color_conversion)
        elif image.ndim != 2:
            raise ValueError("Screenshot must be a grayscale, RGB, or RGBA image.")

        template_height, template_width = self._template.shape[:2]
        image_height, image_width = image.shape[:2]
        if template_width > image_width or template_height > image_height:
            raise ValueError("Template image is larger than the selected screen region.")

        result = self._cv2.matchTemplate(
            image, self._template, self._cv2.TM_CCOEFF_NORMED
        )
        _, confidence, _, location = self._cv2.minMaxLoc(result)
        if not math.isfinite(confidence) or confidence < self._threshold:
            return None

        x, y = location
        center = (
            self._region.left + x + template_width // 2,
            self._region.top + y + template_height // 2,
        )
        return Detection(confidence=confidence, center=center)
