import json
import tempfile
import unittest
from pathlib import Path

import yaml

from prepare_vlm_dataset import prepare_dataset


class PrepareVLMDatasetTests(unittest.TestCase):
    def _dataset(self, root: Path) -> Path:
        dataset_root = root / "ui"
        for split in ("train", "val"):
            (dataset_root / "images" / split).mkdir(parents=True)
            (dataset_root / "labels" / split).mkdir(parents=True)
        (dataset_root / "images" / "train" / "capture.png").write_bytes(b"image")
        (dataset_root / "labels" / "train" / "capture.txt").write_text(
            "0 0.5 0.5 0.4 0.2\n", encoding="utf-8"
        )
        (dataset_root / "images" / "val" / "empty.png").write_bytes(b"image")
        (dataset_root / "labels" / "val" / "empty.txt").write_text(
            "", encoding="utf-8"
        )
        config_path = dataset_root / "data.yaml"
        config_path.write_text(
            yaml.safe_dump(
                {
                    "path": ".",
                    "train": "images/train",
                    "val": "images/val",
                    "names": ["buy_button"],
                }
            ),
            encoding="utf-8",
        )
        return config_path

    def test_converts_box_labels_to_normalized_jsonl_records(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._dataset(root)
            output = root / "manifests"

            counts = prepare_dataset(data_path=config_path, output_dir=output)

            self.assertEqual(counts, {"train": 1, "val": 1})
            train_record = json.loads((output / "train.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(
                json.loads(train_record["completion"]),
                {
                    "detections": [
                        {
                            "label": "buy_button",
                            "bbox_2d": [300.0, 400.0, 700.0, 600.0],
                        }
                    ]
                },
            )
            self.assertEqual(train_record["image"], "../ui/images/train/capture.png")
            validation_record = json.loads(
                (output / "validation.jsonl").read_text(encoding="utf-8")
            )
            self.assertEqual(
                json.loads(validation_record["completion"]),
                {"detections": []},
            )

    def test_missing_annotation_fails_unless_explicitly_allowed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._dataset(root)
            (root / "ui" / "labels" / "val" / "empty.txt").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "--allow-missing-labels"):
                prepare_dataset(data_path=config_path, output_dir=root / "manifests")

            counts = prepare_dataset(
                data_path=config_path,
                output_dir=root / "manifests",
                allow_missing_labels=True,
            )
            self.assertEqual(counts["val"], 1)

    def test_rejects_invalid_annotations_without_writing_manifests(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_path = self._dataset(root)
            (root / "ui" / "labels" / "train" / "capture.txt").write_text(
                "0 0.5 nan 0.4 0.2\n", encoding="utf-8"
            )
            output = root / "manifests"

            with self.assertRaisesRegex(ValueError, "finite"):
                prepare_dataset(data_path=config_path, output_dir=output)

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
