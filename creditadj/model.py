"""Champion and benchmark models, the data split and the calibration decision."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import CURRENCY_KEYS, FEATURE_KEYS, MONOTONE

# Small on purpose: 200 trees of depth 3 export to about 120 kB of JSON and
# every tree can be read by eye. n_jobs=1 removes the thread count as a source
# of run-to-run differences. The fit still takes well under a second.
CHAMPION_PARAMS = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "tree_method": "hist",
    "n_jobs": 1,
}

# Platt scaling is applied only if the validation set shows the raw scores
# are off by more than this, on the logit scale.
CALIBRATION_TOLERANCE = 0.10


@dataclass(frozen=True)
class Split:
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray


def make_split(y: np.ndarray, seed: int) -> Split:
    """Stratified 60 / 20 / 20 split. The data has no dates, so it is random."""
    index = np.arange(len(y))
    rest, test = train_test_split(index, test_size=0.20, stratify=y, random_state=seed)
    train, valid = train_test_split(rest, test_size=0.25, stratify=y[rest], random_state=seed)
    return Split(np.sort(train), np.sort(valid), np.sort(test))


def fit_xgb(
    X: pd.DataFrame,
    y: np.ndarray,
    seed: int,
    monotone: tuple[int, ...] | None,
) -> xgb.XGBClassifier:
    """Gradient-boosted trees with a fixed, known starting score.

    base_score is set to the training default rate so the exported model can
    state its intercept exactly instead of relying on XGBoost's estimate.
    """
    model = xgb.XGBClassifier(
        **CHAMPION_PARAMS,
        random_state=seed,
        base_score=round(float(np.mean(y)), 4),
        monotone_constraints=monotone,
    )
    model.fit(X, y)
    return model


def fit_champion(X: pd.DataFrame, y: np.ndarray, seed: int) -> xgb.XGBClassifier:
    if list(X.columns) != FEATURE_KEYS:
        raise ValueError("The champion is trained on the engineered features only.")
    return fit_xgb(X, y, seed, MONOTONE)


def _benchmark_transform(X: np.ndarray, currency_idx: list[int]) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64).copy()
    # Amounts are heavily right-skewed; a log keeps a few very large limits
    # and payments from dominating the linear fit.
    X[:, currency_idx] = np.log1p(X[:, currency_idx])
    return X


def fit_benchmark(X: pd.DataFrame, y: np.ndarray) -> Pipeline:
    """Logistic regression on the same 12 features: log amounts, then z-scores."""
    currency_idx = [list(X.columns).index(k) for k in CURRENCY_KEYS]
    pipeline = Pipeline(
        [
            (
                "log_amounts",
                FunctionTransformer(_benchmark_transform, kw_args={"currency_idx": currency_idx}),
            ),
            ("scale", StandardScaler()),
            ("logit", LogisticRegression(C=1.0, max_iter=1000)),
        ]
    )
    pipeline.fit(X.to_numpy(), y)
    return pipeline


def margin(model: xgb.XGBClassifier, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Raw model output on the logit scale, before any calibration."""
    return model.predict(X, output_margin=True).astype(np.float64)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=np.float64)))


def fit_platt(margin_valid: np.ndarray, y_valid: np.ndarray) -> tuple[float, float]:
    """Slope a and intercept b of logit(P(default)) = a * margin + b."""
    fit = LogisticRegression(C=1e6, max_iter=1000)
    fit.fit(margin_valid.reshape(-1, 1), y_valid)
    return float(fit.coef_[0, 0]), float(fit.intercept_[0])


def decide_calibration(a: float, b: float) -> dict | None:
    """Keep the raw scores unless validation says they are measurably off."""
    if abs(a - 1.0) > CALIBRATION_TOLERANCE or abs(b) > CALIBRATION_TOLERANCE:
        return {"kind": "platt", "a": a, "b": b}
    return None


def apply_calibration(margin_values: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Probability of default from the raw margin."""
    if calibration is None:
        return sigmoid(margin_values)
    return sigmoid(calibration["a"] * margin_values + calibration["b"])
