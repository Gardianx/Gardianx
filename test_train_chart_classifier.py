import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from train_chart_classifier import (
    CLASS_NAMES,
    _build_model,
    _validate_dataset,
    build_parser,
)


class TrainChartClassifierTests(unittest.TestCase):
    def test_validation_requires_both_real_source_label_folders(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for split in ("train", "validation"):
                for class_name in CLASS_NAMES:
                    class_dir = root / split / class_name
                    class_dir.mkdir(parents=True)
                    (class_dir / "sample.jpg").touch()

            _validate_dataset(root)

            (root / "validation" / "charts" / "sample.jpg").unlink()
            with self.assertRaisesRegex(FileNotFoundError, "charts"):
                _validate_dataset(root)

    def test_resnet_classifier_has_two_outputs(self):
        model = SimpleNamespace(fc=SimpleNamespace(in_features=512))
        models = SimpleNamespace(resnet18=Mock(return_value=model))
        nn = SimpleNamespace(
            Linear=Mock(return_value=SimpleNamespace(in_features=512, out_features=2))
        )

        result = _build_model(models, nn)

        models.resnet18.assert_called_once_with(weights=None)
        nn.Linear.assert_called_once_with(512, 2)
        self.assertEqual(result.fc.out_features, 2)

    def test_cli_defaults_to_chart_recognition_dataset(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.epochs, 15)
        self.assertEqual(args.dataset_dir.name, "chart_recognition")


if __name__ == "__main__":
    unittest.main()
