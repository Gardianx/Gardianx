import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from setup_dataset import create_dataset_directories


class DatasetSetupTests(unittest.TestCase):
    def test_creates_all_split_and_class_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            output = io.StringIO()

            with contextlib.redirect_stdout(output):
                created = create_dataset_directories(dataset_dir)

            expected = [
                dataset_dir / split / class_name
                for split in ("train", "validation")
                for class_name in ("buy", "sell", "hold")
            ]
            self.assertEqual(created, expected)
            self.assertTrue(all(path.is_dir() for path in expected))
            self.assertIn("created successfully", output.getvalue())

    def test_is_safe_to_run_again(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"

            create_dataset_directories(dataset_dir)
            create_dataset_directories(dataset_dir)

            self.assertTrue((dataset_dir / "train" / "buy").is_dir())
            self.assertTrue((dataset_dir / "validation" / "hold").is_dir())


if __name__ == "__main__":
    unittest.main()
