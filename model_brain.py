"""Load the trained chart-image CNN and produce trading signal predictions."""

from contextlib import nullcontext
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "trading_vision_model.pth"
CLASS_NAMES = ("buy", "sell", "hold")
IMAGE_SIZE = 224
DEFAULT_CLASS_TO_IDX = {"buy": 0, "hold": 1, "sell": 2}


@dataclass(frozen=True)
class SignalPrediction:
    signal: str
    confidence_score: float
    is_placeholder: bool = False


class TradingBrain:
    """Run the trained CNN against a cropped chart screenshot."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        torch_module=None,
        transforms_module=None,
        model_factory=None,
        models_module=None,
    ) -> None:
        self.model_path = Path(model_path)
        self.model = None
        self._torch = torch_module
        self._transforms = transforms_module
        self._model_factory = model_factory
        self._models = models_module
        self._device = None
        self._preprocess = None
        self._class_to_idx: dict[str, int] | None = None

    def load_model(self) -> None:
        """Load and validate the checkpoint before starting the agent loop."""
        if self.model is not None:
            return
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Trained model checkpoint not found: {self.model_path}")

        if self._torch is None or self._transforms is None or (
            self._model_factory is None and self._models is None
        ):
            try:
                import torch
                from torchvision import models, transforms
            except ImportError as error:
                raise RuntimeError(
                    "Model inference requires PyTorch and torchvision. Install "
                    "the appropriate builds using `python -m pip install -r "
                    "requirements-training.txt`."
                ) from error
            if self._torch is None:
                self._torch = torch
            if self._transforms is None:
                self._transforms = transforms
            if self._models is None:
                self._models = models

        self._device = self._torch.device("cpu")
        checkpoint = self._torch.load(
            self.model_path,
            map_location=self._device,
            weights_only=True,
        )
        if not isinstance(checkpoint, dict):
            raise ValueError("Model checkpoint must contain a state dictionary.")

        state_dict = checkpoint.get("state_dict", checkpoint)
        class_to_idx = checkpoint.get("class_to_idx", DEFAULT_CLASS_TO_IDX)
        if (
            not isinstance(state_dict, dict)
            or not state_dict
            or any(not isinstance(key, str) for key in state_dict)
        ):
            raise ValueError("Model checkpoint is missing a valid state_dict.")
        if (
            not isinstance(class_to_idx, dict)
            or set(class_to_idx) != set(CLASS_NAMES)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in class_to_idx.values()
            )
            or set(class_to_idx.values()) != set(range(len(CLASS_NAMES)))
        ):
            raise ValueError("Model checkpoint has an invalid class_to_idx mapping.")

        if all(key.startswith("module.") for key in state_dict):
            state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}

        if self._model_factory is not None:
            model = self._model_factory()
        else:
            model = self._models.resnet18(weights=None)
        model.fc = self._torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
        incompatible_keys = model.load_state_dict(state_dict, strict=False)
        if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
            raise ValueError(
                "Checkpoint keys do not match the ResNet-18 model; "
                f"missing={incompatible_keys.missing_keys}, "
                f"unexpected={incompatible_keys.unexpected_keys}."
            )
        model = model.to(self._device)
        model.eval()
        self._preprocess = self._transforms.Compose(
            [
                self._transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                self._transforms.ToTensor(),
            ]
        )
        self._class_to_idx = class_to_idx
        self.model = model

    def predict_signal(self, chart_image: Any) -> SignalPrediction:
        """Classify a PIL chart crop with the three-class ResNet-18 model."""
        if chart_image is None or not callable(getattr(chart_image, "crop", None)):
            raise TypeError("chart_image must be a PIL image or compatible image object.")
        self.load_model()

        tensor = self._preprocess(chart_image).unsqueeze(0).to(self._device)
        inference_context = getattr(self._torch, "inference_mode", nullcontext)
        with inference_context():
            probabilities = self._torch.softmax(self.model(tensor), dim=1)
            confidence_tensor, index_tensor = probabilities.max(dim=1)
        confidence = float(confidence_tensor.item())
        class_index = int(index_tensor.item())
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Model returned an invalid confidence score.")

        class_names_by_index = {
            index: name for name, index in self._class_to_idx.items()
        }
        if class_index not in class_names_by_index:
            raise ValueError(f"Model returned an unknown class index: {class_index}.")
        class_name = class_names_by_index[class_index]
        return SignalPrediction(
            signal=class_name.upper(),
            confidence_score=confidence,
        )

    def train_model(self, training_data_path: str | Path) -> None:
        """Reserve the training API; fitting remains in train_vision_model.py."""
        raise NotImplementedError(
            f"Model training is not implemented here: {Path(training_data_path)}"
        )
