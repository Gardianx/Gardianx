"""Placeholder interface for future trading signal models."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "trading_model.pkl"


@dataclass(frozen=True)
class SignalPrediction:
    signal: str
    confidence_score: float
    is_placeholder: bool = True


class TradingBrain:
    """Stable interface for loading, predicting, and eventually training a model."""

    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path)
        self.model = None

    def predict_signal(self, current_features: Mapping[str, float]) -> SignalPrediction:
        """Return a safe placeholder prediction until an actual model is added."""
        if not isinstance(current_features, Mapping):
            raise TypeError("current_features must be a mapping of feature names to values.")
        return SignalPrediction(signal="HOLD", confidence_score=0.0)

    def train_model(self, training_data_path: str | Path) -> None:
        """Reserve the training API; fitting and persistence are not implemented yet."""
        raise NotImplementedError(
            f"Model training is not implemented; training data was not loaded: "
            f"{Path(training_data_path)}"
        )
