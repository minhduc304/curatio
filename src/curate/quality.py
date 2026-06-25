"""C5 — distilled quality classifier inference over embeddings. See TDD §4.4.

prob = sigmoid(coef . embedding + intercept); reject if prob < threshold.
A plain dot product — identical in Python (numpy) and Rust.
"""
from __future__ import annotations

import numpy as np

from src.models import QualityModel


class QualityClassifier:
    def __init__(self, model: QualityModel) -> None:
        self.model = model
        self.coef = np.asarray(model.coef, dtype=np.float32)

    @classmethod
    def from_json(cls, path: str) -> "QualityClassifier":
        with open(path) as f:
            return cls(QualityModel.model_validate_json(f.read()))

    def predict_proba(self, embedding: np.ndarray) -> float:
        z = float(self.coef @ np.asarray(embedding, dtype=np.float32).ravel()) + self.model.intercept
        return 1.0 / (1.0 + np.exp(-z))
