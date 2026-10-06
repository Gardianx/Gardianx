import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from model_brain import DEFAULT_MODEL_PATH, SignalPrediction, TradingBrain


class _FakeTensor:
    def __init__(self, value=None):
        self.value = value

    def unsqueeze(self, dimension):
        self.unsqueeze_dimension = dimension
        return self

    def to(self, device):
        self.device = device
        return self

    def item(self):
        return self.value


class _FakeProbabilities:
    def max(self, dim):
        return _FakeTensor(0.91), _FakeTensor(2)


class _FakeModel:
    def __init__(self):
        self.loaded_state_dict = None
        self.input = None
        self.fc = SimpleNamespace(in_features=512)

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state_dict, strict):
        self.loaded_state_dict = state_dict
        self.load_state_dict_strict = strict
        return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    def eval(self):
        self.evaluating = True

    def __call__(self, tensor):
        self.input = tensor
        return "logits"


class _Resize:
    def __init__(self, size):
        self.size = size


class _Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        self.image = image
        return _FakeTensor()


class TradingBrainTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.model_path = Path(self.temp_dir.name) / "model.pth"
        self.model_path.touch()
        self.checkpoint = {
            "conv1.weight": "test convolution weights",
            "fc.weight": "test classifier weights",
            "fc.bias": "test classifier bias",
        }
        self.model = _FakeModel()
        self.transforms = SimpleNamespace(
            Compose=_Compose,
            Resize=_Resize,
            ToTensor=lambda: object(),
        )
        self.torch = SimpleNamespace(
            device=lambda name: name,
            load=Mock(return_value=self.checkpoint),
            inference_mode=nullcontext,
            softmax=Mock(return_value=_FakeProbabilities()),
            nn=SimpleNamespace(
                Linear=Mock(
                    side_effect=lambda input_features, output_features: SimpleNamespace(
                        in_features=input_features,
                        out_features=output_features,
                    )
                )
            ),
        )
        self.brain = TradingBrain(
            self.model_path,
            torch_module=self.torch,
            transforms_module=self.transforms,
            model_factory=lambda: self.model,
        )

    def test_uses_project_root_checkpoint_by_default(self):
        brain = TradingBrain()

        self.assertEqual(brain.model_path, DEFAULT_MODEL_PATH)
        self.assertIsNone(brain.model)

    def test_accepts_custom_model_path(self):
        brain = TradingBrain(self.model_path)

        self.assertEqual(brain.model_path, self.model_path)

    def test_loads_checkpoint_and_predicts_from_chart_image(self):
        class ChartImage:
            def crop(self, bounds):
                return self

        chart_image = ChartImage()
        prediction = self.brain.predict_signal(chart_image)

        self.assertEqual(
            prediction,
            SignalPrediction(signal="SELL", confidence_score=0.91),
        )
        self.assertFalse(prediction.is_placeholder)
        self.assertEqual(self.model.loaded_state_dict, self.checkpoint)
        self.assertFalse(self.model.load_state_dict_strict)
        self.assertEqual(self.model.fc.out_features, 3)
        self.torch.nn.Linear.assert_called_once_with(512, 3)
        self.assertTrue(self.model.evaluating)
        self.assertIs(self.brain._preprocess.image, chart_image)
        self.torch.load.assert_called_once_with(
            self.model_path,
            map_location="cpu",
            weights_only=True,
        )

    def test_missing_checkpoint_is_reported(self):
        brain = TradingBrain(self.model_path.parent / "missing.pth")

        with self.assertRaisesRegex(FileNotFoundError, "checkpoint not found"):
            brain.load_model()

    def test_prediction_requires_image_input(self):
        with self.assertRaises(TypeError):
            self.brain.predict_signal({"last_price": 100.5})

    def test_unexpected_state_dict_keys_are_rejected(self):
        self.model.load_state_dict = Mock(
            return_value=SimpleNamespace(
                missing_keys=["layer4.weight"],
                unexpected_keys=["unknown.weight"],
            )
        )

        with self.assertRaisesRegex(ValueError, "keys do not match"):
            self.brain.load_model()

    def test_strips_data_parallel_prefix_before_loading(self):
        self.checkpoint = {
            "module.conv1.weight": "test convolution weights",
            "module.fc.weight": "test classifier weights",
            "module.fc.bias": "test classifier bias",
        }
        self.torch.load.return_value = self.checkpoint

        self.brain.load_model()

        self.assertEqual(
            set(self.model.loaded_state_dict),
            {"conv1.weight", "fc.weight", "fc.bias"},
        )
        self.assertFalse(self.model.load_state_dict_strict)

    def test_training_reports_not_implemented(self):
        with self.assertRaisesRegex(NotImplementedError, "not implemented"):
            self.brain.train_model("training.csv")


if __name__ == "__main__":
    unittest.main()
