"""YOLO model adapter returning typed, screen-coordinate detections."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from ultralytics import YOLO

from config import DEFAULT_CONFIG

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best.pt"


@dataclass(frozen=True)
class YOLODetection:
    label: str
    confidence: float
    bounding_box: tuple[float, float, float, float]
    center: tuple[float, float]


class YOLODetector:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        model: Any | None = None,
        model_factory=None,
        confidence: float = DEFAULT_CONFIG.model_confidence,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("YOLO confidence must be between 0 and 1.")
        self.model_path = Path(model_path)
        self.confidence = confidence
        if model is None and model_factory is None and not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_path}. "
                "Provide UI-trained weights with --model."
            )
        if model is not None:
            self._model = model
        else:
            self._model = (model_factory or YOLO)(str(self.model_path))

    def detect(self, image: Any) -> list[YOLODetection]:
        results = self._model(image, conf=self.confidence, verbose=False)
        detections: list[YOLODetection] = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                name_map = result.names
                label = name_map[class_id]
                x1, y1, x2, y2 = (
                    float(coordinate) for coordinate in box.xyxy[0].tolist()
                )
                confidence = float(box.conf[0].item())
                if (
                    not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence))
                    or not 0.0 <= confidence <= 1.0
                    or x2 <= x1
                    or y2 <= y1
                ):
                    raise ValueError("YOLO returned an invalid detection box or confidence.")
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                detections.append(
                    YOLODetection(
                        label=str(label),
                        confidence=confidence,
                        bounding_box=(x1, y1, x2, y2),
                        center=center,
                    )
                )
        return detections
