"""YOLO UI detections and OCR over the active-asset label region."""

from typing import Any, Callable, Iterable

from config import AppConfig, DEFAULT_CONFIG
from core.detection import YOLODetection, YOLODetector


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("-", "_").replace(" ", "_")


class AssetOCR:
    def __init__(
        self,
        *,
        language: str = DEFAULT_CONFIG.ocr_language,
        reader=None,
        reader_factory=None,
    ) -> None:
        self.language = language
        self._reader = reader
        self._reader_factory = reader_factory

    def read(self, image: Any) -> str:
        if self._reader is None:
            if self._reader_factory is None:
                try:
                    import pytesseract
                except ImportError as error:
                    raise RuntimeError(
                        "OCR requires pytesseract and the Tesseract OCR executable."
                    ) from error
                self._reader = pytesseract
            else:
                self._reader = self._reader_factory()

        if callable(getattr(self._reader, "image_to_string", None)):
            text = self._reader.image_to_string(image, lang=self.language)
        elif callable(getattr(self._reader, "readtext", None)):
            results = self._reader.readtext(image)
            text = " ".join(str(result[1]) for result in results)
        else:
            raise TypeError("OCR reader must provide image_to_string() or readtext().")
        return str(text).strip()


class UIVision:
    def __init__(
        self,
        detector: YOLODetector,
        ocr: AssetOCR,
        *,
        config: AppConfig = DEFAULT_CONFIG,
    ) -> None:
        self.detector = detector
        self.ocr = ocr
        self.config = config

    def detect_ui_elements(self, image: Any) -> list[YOLODetection]:
        return [
            detection
            for detection in self.detector.detect(image)
            if detection.confidence
            >= self.config.label_confidence_thresholds.get(
                _normalize_label(detection.label),
                self.config.min_detection_confidence,
            )
        ]

    def read_active_asset(self, image: Any, detections: Iterable[YOLODetection]) -> str:
        active_labels = [
            detection
            for detection in detections
            if _normalize_label(detection.label) == "active_asset_label"
            and detection.confidence >= self.config.asset_label_min_confidence
        ]
        if not active_labels:
            raise ValueError("YOLO did not find a confident active_asset_label box.")

        label_box = max(active_labels, key=lambda detection: detection.confidence)
        left, top, right, bottom = label_box.bounding_box
        image_width, image_height = image.size
        crop_bounds = (
            max(0, int(left)),
            max(0, int(top)),
            min(image_width, int(right)),
            min(image_height, int(bottom)),
        )
        if crop_bounds[2] <= crop_bounds[0] or crop_bounds[3] <= crop_bounds[1]:
            raise ValueError("Active asset label detection has an invalid crop box.")
        return self.ocr.read(image.crop(crop_bounds))
