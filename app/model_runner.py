from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


class ModelLoadError(RuntimeError):
    """Raised when a model artifact cannot be loaded."""


@dataclass(slots=True)
class Prediction:
    label: int
    score: float


class ModelRunner:
    def __init__(self, model_path: Path, version: str) -> None:
        self.model_path = model_path
        self.version = version
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.threshold: float = 0.5
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.model_path}")

        data = np.load(self.model_path)
        try:
            self.weights = np.asarray(data["weights"], dtype=np.float64)
            self.bias = float(data["bias"])
            self.threshold = float(data["threshold"])
        except KeyError as exc:
            raise ModelLoadError(f"Missing key in model artifact: {exc}") from exc

        if self.weights.ndim != 1:
            raise ModelLoadError("weights must be a 1D array")

        self._loaded = True

    def predict(self, features: list[float]) -> Prediction:
        if not self._loaded:
            self.load()

        if self.weights is None:
            raise ModelLoadError("Model weights are unavailable after load")

        x = np.asarray(features, dtype=np.float64)
        if x.shape != self.weights.shape:
            raise ValueError(
                f"Invalid feature length {x.shape[0]}; expected {self.weights.shape[0]}"
            )

        linear_score = float(np.dot(x, self.weights) + self.bias)
        score = float(1.0 / (1.0 + np.exp(-linear_score)))
        label = int(score >= self.threshold)
        return Prediction(label=label, score=score)

    def predict_batch(self, items: list[list[float]]) -> list[Prediction]:
        return [self.predict(features) for features in items]


class ModelRegistry:
    def __init__(self, models_root: Path) -> None:
        self.models_root = models_root
        self._lock = Lock()
        self._runners: dict[str, ModelRunner] = {}

    def _artifact_path(self, version: str) -> Path:
        return self.models_root / version / "model.npz"

    def load(self, version: str) -> ModelRunner:
        if version in self._runners:
            return self._runners[version]

        with self._lock:
            if version in self._runners:
                return self._runners[version]

            runner = ModelRunner(self._artifact_path(version), version=version)
            runner.load()
            self._runners[version] = runner
            return runner

    def is_loaded(self, version: str) -> bool:
        runner = self._runners.get(version)
        return bool(runner and runner.loaded)

