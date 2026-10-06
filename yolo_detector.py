"""Starter for detecting and reporting objects in a full-screen capture."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ultralytics import YOLO


DEFAULT_MODEL = "yolov8n.pt"


@dataclass(frozen=True)
class UIElement:
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    center: tuple[float, float]


_model = None


def _get_model(model_path: str | Path = DEFAULT_MODEL):
    global _model
    if _model is None:
        _model = YOLO(str(model_path))
    return _model


def _scalar(value: Any) -> float:
    if callable(getattr(value, "item", None)):
        value = value.item()
    return float(value)


def detect_ui_elements(image: Any, model=None) -> list[UIElement]:
    """Run YOLO on an image and print each label, box, and center coordinate."""
    active_model = model if model is not None else _get_model()
    detections: list[UIElement] = []
    for result in active_model(image, verbose=False):
        names = result.names
        for box in result.boxes:
            class_id = int(_scalar(box.cls[0]))
            if isinstance(names, dict):
                label = str(names[class_id])
            else:
                label = str(names[class_id])
            x1, y1, x2, y2 = (
                float(coordinate)
                for coordinate in box.xyxy[0].tolist()
            )
            confidence = _scalar(box.conf[0])
            center = ((x1 + x2) / 2, (y1 + y2) / 2)
            detection = UIElement(
                label=label,
                confidence=confidence,
                bounding_box=(x1, y1, x2, y2),
                center=center,
            )
            detections.append(detection)
            print(
                f"{label}: box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), "
                f"center=({center[0]:.1f}, {center[1]:.1f}), "
                f"confidence={confidence:.3f}"
            )
    if not detections:
        print("No objects detected.")
    return detections


def main() -> int:
    import pyautogui

    model = _get_model()
    screenshot = pyautogui.screenshot()
    detect_ui_elements(screenshot, model=model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
