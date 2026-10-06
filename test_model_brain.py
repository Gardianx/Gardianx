import tempfile
import unittest
from pathlib import Path

from model_brain import DEFAULT_MODEL_PATH, SignalPrediction, TradingBrain


class TradingBrainTests(unittest.TestCase):
    def test_initializes_model_path_placeholder(self):
        brain = TradingBrain()

        self.assertEqual(brain.model_path, DEFAULT_MODEL_PATH)
        self.assertIsNone(brain.model)

    def test_accepts_custom_model_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.pkl"

            brain = TradingBrain(model_path)

            self.assertEqual(brain.model_path, model_path)

    def test_prediction_returns_safe_placeholder_for_features(self):
        brain = TradingBrain()

        prediction = brain.predict_signal({"last_price": 100.5, "volume": 12.0})

        self.assertEqual(
            prediction,
            SignalPrediction(signal="HOLD", confidence_score=0.0, is_placeholder=True),
        )

    def test_prediction_rejects_non_mapping_features(self):
        brain = TradingBrain()

        with self.assertRaises(TypeError):
            brain.predict_signal([100.5, 12.0])

    def test_training_reports_not_implemented(self):
        brain = TradingBrain()

        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            brain.train_model("training.csv")


if __name__ == "__main__":
    unittest.main()
