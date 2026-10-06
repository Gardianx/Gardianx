import argparse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from train_vision_model import (
    CLASS_NAMES,
    _positive_float,
    _positive_int,
    _validate_dataset_directories,
    build_parser,
)


class TrainVisionModelTests(unittest.TestCase):
    def test_cli_defaults_to_15_epochs(self):
        self.assertEqual(build_parser().parse_args([]).epochs, 15)

    def test_dataset_folder_validation_accepts_expected_class_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            for split in (train_dir, validation_dir):
                for class_name in CLASS_NAMES:
                    (split / class_name).mkdir(parents=True)

            _validate_dataset_directories(train_dir, validation_dir)

    def test_dataset_folder_validation_reports_missing_split_or_class(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_dir = root / "train"
            validation_dir = root / "validation"
            train_dir.mkdir()
            for class_name in CLASS_NAMES:
                (train_dir / class_name).mkdir()

            with self.assertRaises(FileNotFoundError):
                _validate_dataset_directories(train_dir, validation_dir)
            validation_dir.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "buy"):
                _validate_dataset_directories(train_dir, validation_dir)

    def test_cli_accepts_epoch_and_training_options(self):
        args = build_parser().parse_args(
            ["--epochs", "4", "--batch-size", "8", "--learning-rate", "0.002"]
        )

        self.assertEqual(args.epochs, 4)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.learning_rate, 0.002)

    def test_cli_numeric_arguments_must_be_positive(self):
        for parser_type, invalid in (
            (_positive_int, "0"),
            (_positive_int, "-2"),
            (_positive_float, "0"),
            (_positive_float, "nan"),
            (_positive_float, "inf"),
        ):
            with self.subTest(parser_type=parser_type, invalid=invalid):
                with self.assertRaises(argparse.ArgumentTypeError):
                    parser_type(invalid)

    def test_train_fails_before_dependency_loading_if_dataset_is_missing(self):
        from train_vision_model import train_model

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("train_vision_model._load_training_dependencies") as load:
                with self.assertRaises(FileNotFoundError):
                    train_model(
                        train_dir=Path(temp_dir) / "train",
                        validation_dir=Path(temp_dir) / "validation",
                    )
                load.assert_not_called()


if __name__ == "__main__":
    unittest.main()
