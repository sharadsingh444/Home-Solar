"""Small serializable weighted-average regressor used by HomeSolar."""
from __future__ import annotations

from typing import Any

import numpy as np


class WeightedBlendRegressor:
    """Combine two already-fitted regressors with validation-selected weights.

    This intentionally has only a ``predict`` method: model fitting and blend
    weight selection happen in ``train.py`` using a chronological validation
    period. The contained estimators are persisted together with this object.
    """

    def __init__(self, estimators: dict[str, Any], weights: dict[str, float]):
        if not estimators or set(estimators) != set(weights):
            raise ValueError("estimators and weights must contain the same non-empty keys")
        if any(float(w) < 0 for w in weights.values()):
            raise ValueError("blend weights cannot be negative")
        total = sum(float(w) for w in weights.values())
        if total <= 0:
            raise ValueError("at least one blend weight must be positive")
        self.estimators = dict(estimators)
        self.weights = {key: float(value) / total for key, value in weights.items()}

    def predict(self, X):
        predictions = []
        weights = []
        for name, estimator in self.estimators.items():
            predictions.append(np.asarray(estimator.predict(X), dtype=np.float64))
            weights.append(self.weights[name])
        matrix = np.vstack(predictions)
        return np.average(matrix, axis=0, weights=np.asarray(weights, dtype=np.float64))
