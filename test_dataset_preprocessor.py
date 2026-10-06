import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from dataset_preprocessor import preprocess_image, process_raw_charts


class DatasetPreprocessorTests(unittest.TestCase):
    def test_preprocesses_selected_crop_to_224_and_saves_labeled_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "chart.png"
            source.touch()
            image = _FakeImage((300, 500, 3))
            resized = _FakeImage((224, 224, 3))
            cv2 = SimpleNamespace(
                IMREAD_COLOR=1,
                INTER_AREA=2,
                imread=Mock(return_value=image),
                resize=Mock(return_value=resized),
                imwrite=Mock(return_value=True),
            )
            output = preprocess_image(
                source,
                "BUY",
                "train",
                crop_region=(10, 20, 120, 80),
                dataset_dir=Path(temp_dir) / "dataset",
                cv2_module=cv2,
            )

            self.assertEqual(output, Path(temp_dir) / "dataset" / "train" / "buy" / "chart.png")
            self.assertEqual(image.last_slice, (slice(20, 100), slice(10, 130)))
            resized_crop, size = cv2.resize.call_args.args
            self.assertEqual(resized_crop.shape, (80, 120, 3))
            self.assertEqual(size, (224, 224))
            self.assertEqual(
                cv2.resize.call_args.kwargs,
                {"interpolation": cv2.INTER_AREA},
            )
            cv2.imwrite.assert_called_once_with(str(output), resized)
            self.assertTrue(output.parent.is_dir())

    def test_preprocessor_avoids_overwriting_existing_image(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "chart.jpg"
            source.touch()
            destination = Path(temp_dir) / "dataset" / "train" / "sell"
            destination.mkdir(parents=True)
            (destination / "chart.png").touch()
            cv2 = SimpleNamespace(
                IMREAD_COLOR=1,
                INTER_AREA=2,
                imread=Mock(return_value=_FakeImage((50, 60, 3))),
                resize=Mock(return_value=_FakeImage((224, 224, 3))),
                imwrite=Mock(return_value=True),
            )

            output = preprocess_image(
                source,
                "sell",
                "train",
                crop_region=(0, 0, 60, 50),
                dataset_dir=Path(temp_dir) / "dataset",
                cv2_module=cv2,
            )

            self.assertEqual(output, destination / "chart_1.png")

    def test_rejects_bad_labels_splits_and_crop_bounds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "chart.png"
            source.touch()
            cv2 = SimpleNamespace(
                IMREAD_COLOR=1,
                imread=Mock(return_value=_FakeImage((20, 30, 3))),
            )
            for label, split, region in (
                ("unknown", "train", (0, 0, 10, 10)),
                ("buy", "test", (0, 0, 10, 10)),
                ("buy", "train", (25, 0, 10, 10)),
                ("buy", "train", (0, 0, 0, 10)),
            ):
                with self.subTest(label=label, split=split, region=region):
                    with self.assertRaises(ValueError):
                        preprocess_image(
                            source,
                            label,
                            split,
                            crop_region=region,
                            dataset_dir=Path(temp_dir) / "dataset",
                            cv2_module=cv2,
                        )

    def test_interactive_loop_crops_labels_and_assigns_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw_charts"
            raw_dir.mkdir()
            first = raw_dir / "a.png"
            second = raw_dir / "b.JPG"
            ignored = raw_dir / "notes.txt"
            first.touch()
            second.touch()
            ignored.touch()
            image = _FakeImage((100, 100, 3))
            cv2 = SimpleNamespace(
                IMREAD_COLOR=1,
                INTER_AREA=2,
                imread=Mock(return_value=image),
                selectROI=Mock(side_effect=[(5, 6, 30, 40), (0, 0, 0, 0)]),
                resize=Mock(return_value=_FakeImage((224, 224, 3))),
                imwrite=Mock(return_value=True),
                destroyAllWindows=Mock(),
            )
            answers = iter(("buy", "validation"))

            saved = process_raw_charts(
                raw_dir,
                Path(temp_dir) / "dataset",
                input_fn=lambda _prompt: next(answers),
                report=Mock(),
                cv2_module=cv2,
            )

            self.assertEqual(
                saved,
                [Path(temp_dir) / "dataset" / "validation" / "buy" / "a.png"],
            )
            self.assertEqual(cv2.selectROI.call_count, 2)
            self.assertEqual(cv2.imwrite.call_count, 1)
            cv2.destroyAllWindows.assert_called_once_with()
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_empty_raw_directory_is_created_and_reported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            raw_dir = Path(temp_dir) / "raw_charts"
            report = Mock()
            cv2 = SimpleNamespace(destroyAllWindows=Mock())

            saved = process_raw_charts(
                raw_dir,
                Path(temp_dir) / "dataset",
                report=report,
                cv2_module=cv2,
            )

            self.assertEqual(saved, [])
            self.assertTrue(raw_dir.is_dir())
            report.assert_called_once()
            cv2.destroyAllWindows.assert_not_called()


class _FakeImage:
    def __init__(self, shape):
        self.shape = shape
        self.last_slice = None

    def __getitem__(self, image_slice):
        self.last_slice = image_slice
        height = image_slice[0].stop - image_slice[0].start
        width = image_slice[1].stop - image_slice[1].start
        return _FakeImage((height, width, self.shape[2]))


if __name__ == "__main__":
    unittest.main()
