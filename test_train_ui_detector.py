import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import yaml

from train_ui_detector import REQUIRED_CLASSES, _validate_data_config, train_detector


class TrainUIDetectorTests(unittest.TestCase):
    def test_required_classes_include_pocket_candle_time(self):
        self.assertIn("pocket_candle_time", REQUIRED_CLASSES)

    def _write_config(self, directory: Path, names: list[str]) -> Path:
        config_path = directory / "data.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "path": str(directory),
                    "train": "images/train",
                    "val": "images/val",
                    "names": names,
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_accepts_config_with_application_class_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(
                Path(temporary_directory), sorted(REQUIRED_CLASSES)
            )

            _validate_data_config(config_path)

    def test_rejects_missing_application_class_names(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            config_path = self._write_config(Path(temporary_directory), ["buy_button"])

            with self.assertRaisesRegex(ValueError, "missing classes"):
                _validate_data_config(config_path)

    def test_trains_validates_and_copies_best_weights(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._write_config(root, sorted(REQUIRED_CLASSES))
            best_weights = root / "runs" / "broker-ui" / "weights" / "best.pt"
            best_weights.parent.mkdir(parents=True)
            best_weights.write_bytes(b"trained weights")
            output_path = root / "models" / "best.pt"

            trainer = Mock()
            validator = Mock()

            def model_factory(path: str):
                return trainer if path == "yolo11n.pt" else validator

            result = train_detector(
                data_path=config_path,
                project=root / "runs",
                output_path=output_path,
                yolo_factory=model_factory,
            )

            self.assertEqual(result, output_path)
            self.assertEqual(output_path.read_bytes(), b"trained weights")
            trainer.train.assert_called_once()
            validator.val.assert_called_once()

    def test_rejects_invalid_training_settings(self):
        with self.assertRaisesRegex(ValueError, "epochs"):
            train_detector(epochs=0)


if __name__ == "__main__":
    unittest.main()
