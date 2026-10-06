import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from yolo_detector import UIElement, detect_ui_elements


class YoloDetectorTests(unittest.TestCase):
    def test_logs_labels_boxes_and_centers(self):
        box = SimpleNamespace(
            cls=[1],
            conf=[0.875],
            xyxy=[SimpleNamespace(tolist=lambda: [10, 20, 50, 80])],
        )
        model = Mock(return_value=[SimpleNamespace(names={1: "button"}, boxes=[box])])

        with patch("builtins.print") as report:
            detections = detect_ui_elements(object(), model=model)

        self.assertEqual(
            detections,
            [
                UIElement(
                    label="button",
                    confidence=0.875,
                    bounding_box=(10.0, 20.0, 50.0, 80.0),
                    center=(30.0, 50.0),
                )
            ],
        )
        model.assert_called_once()
        report.assert_called_once_with(
            "button: box=(10.0, 20.0, 50.0, 80.0), "
            "center=(30.0, 50.0), confidence=0.875"
        )

    def test_reports_when_no_objects_are_detected(self):
        with patch("builtins.print") as report:
            result = detect_ui_elements(
                object(),
                model=Mock(return_value=[SimpleNamespace(names={}, boxes=[])]),
            )

        self.assertEqual(result, [])
        report.assert_called_once_with("No objects detected.")


if __name__ == "__main__":
    unittest.main()
