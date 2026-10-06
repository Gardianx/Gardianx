import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from huggingface_ingestion import (
    CLASS_NAMES,
    MAX_IMAGES,
    _positive_int,
    ingest_images,
)


class _FakeImage:
    def __init__(self, saved):
        self.saved = saved

    def convert(self, mode):
        self.mode = mode
        return self

    def save(self, path, format, quality):
        output_path = Path(path)
        output_path.touch()
        self.saved.append((output_path, format, quality))


class _FakeDataset:
    features = {"label": SimpleNamespace(names=["non-charts", "charts"])}

    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)


class HuggingFaceIngestionTests(unittest.TestCase):
    def test_ingests_maximum_1500_and_preserves_source_labels(self):
        saved = []
        rows = [
            {"image": _FakeImage(saved), "label": index % 2}
            for index in range(MAX_IMAGES + 10)
        ]
        dataset = _FakeDataset(rows)
        loader_calls = []

        def loader(dataset_id, split):
            loader_calls.append((dataset_id, split))
            return dataset

        with tempfile.TemporaryDirectory() as temp_dir:
            counts = ingest_images(
                "test/stock-charts",
                temp_dir,
                dataset_loader=loader,
            )

            self.assertEqual(sum(counts.values()), MAX_IMAGES)
            self.assertEqual(counts, {"charts": 750, "non-charts": 750})
            self.assertEqual(len(saved), MAX_IMAGES)
            self.assertEqual(loader_calls, [("test/stock-charts", "train")])
            self.assertTrue(all(path.is_file() for path, _, _ in saved))
            for class_name in CLASS_NAMES:
                self.assertTrue(
                    (Path(temp_dir) / "train" / class_name).is_dir()
                )
                self.assertTrue(
                    (Path(temp_dir) / "validation" / class_name).is_dir()
                )
            self.assertEqual(
                sum("validation" in path.parts for path, _, _ in saved),
                300,
            )
            self.assertTrue(all(fmt == "JPEG" and quality == 95 for _, fmt, quality in saved))

    def test_maps_true_source_label_names_to_directories(self):
        saved = []
        dataset = _FakeDataset(
            [
                {"image": _FakeImage(saved), "label": 1},
                {"image": _FakeImage(saved), "label": 0},
            ]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            counts = ingest_images(
                "test/stock-charts",
                temp_dir,
                limit=2,
                dataset_loader=lambda dataset_id, split: dataset,
            )

        self.assertEqual(counts, {"charts": 1, "non-charts": 1})
        self.assertIn(Path(temp_dir) / "train" / "charts" / "charts_00000.jpg", [p for p, _, _ in saved])
        self.assertIn(Path(temp_dir) / "train" / "non-charts" / "non-charts_00000.jpg", [p for p, _, _ in saved])

    def test_rejects_non_source_labels_and_invalid_limits(self):
        dataset = _FakeDataset(
            [{"image": _FakeImage([]), "label": 4}]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "unknown label"):
                ingest_images(
                    output_dir=temp_dir,
                    limit=1,
                    dataset_loader=lambda dataset_id, split: dataset,
                )
        for value in ("0", str(MAX_IMAGES + 1), "bad"):
            with self.subTest(value=value):
                with self.assertRaises(argparse.ArgumentTypeError):
                    _positive_int(value)


if __name__ == "__main__":
    unittest.main()
