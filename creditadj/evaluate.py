"""Discrimination, calibration and sampling uncertainty.

Scores throughout are predicted probabilities of default, so a higher score
means higher risk and decile 1 is the safest tenth of accounts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def auc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score))


def gini(y: np.ndarray, score: np.ndarray) -> float:
    return 2.0 * auc(y, score) - 1.0


def ks_statistic(y: np.ndarray, score: np.ndarray) -> float:
    """Largest gap between the cumulative score distributions of bads and goods."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    n_bad = int(y.sum())
    n_good = int(len(y) - n_bad)
    if n_bad == 0 or n_good == 0:
        raise ValueError("KS needs both goods and bads.")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    sorted_y = y[order]
    cum_bad = np.cumsum(sorted_y) / n_bad
    cum_good = np.cumsum(1 - sorted_y) / n_good
    # Compare the two distributions only where the score changes, so tied
    # scores are treated as one step.
    last_of_tie = np.r_[sorted_score[1:] != sorted_score[:-1], True]
    return float(np.max(np.abs(cum_bad[last_of_tie] - cum_good[last_of_tie])))


def discrimination(y: np.ndarray, score: np.ndarray) -> dict:
    a = auc(y, score)
    return {"auc": a, "gini": 2.0 * a - 1.0, "ks": ks_statistic(y, score)}


def decile_table(y: np.ndarray, score: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-count score bands, safest first, with observed default rate."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    rows = []
    for i, chunk in enumerate(np.array_split(order, bins), start=1):
        rows.append(
            {
                "decile": i,
                "n": int(len(chunk)),
                "defaults": int(y[chunk].sum()),
                "default_rate": float(y[chunk].mean()),
                "avg_score": float(score[chunk].mean()),
                "min_score": float(score[chunk].min()),
                "max_score": float(score[chunk].max()),
            }
        )
    return pd.DataFrame(rows)


def is_monotone(values) -> bool:
    values = np.asarray(values, dtype=np.float64)
    return bool(np.all(np.diff(values) >= 0))


def inversions(values) -> list[int]:
    """1-based positions where a value is lower than the one before it."""
    values = np.asarray(values, dtype=np.float64)
    return [int(i) + 2 for i in np.flatnonzero(np.diff(values) < 0)]


def calibration_summary(y: np.ndarray, score: np.ndarray, bins: int = 10) -> dict:
    """Mean predicted against observed, overall and by score band."""
    table = decile_table(y, score, bins)
    gap = (table["avg_score"] - table["default_rate"]).abs()
    weights = table["n"] / table["n"].sum()
    return {
        "mean_predicted": float(np.mean(score)),
        "observed_rate": float(np.mean(y)),
        "brier": float(np.mean((np.asarray(score) - np.asarray(y)) ** 2)),
        "expected_calibration_error": float((gap * weights).sum()),
        "max_band_gap": float(gap.max()),
    }


def bootstrap_auc_ci(
    y: np.ndarray,
    score: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile interval for the AUC from resampling rows with replacement."""
    low, high = bootstrap_auc_diff_ci(y, score, None, n_boot, seed, level)
    return low, high


def bootstrap_auc_diff_ci(
    y: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray | None,
    n_boot: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Interval for AUC(a), or for AUC(a) - AUC(b) on the same resampled rows."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        while y[idx].min() == y[idx].max():  # a resample needs both classes
            idx = rng.integers(0, n, n)
        value = roc_auc_score(y[idx], score_a[idx])
        if score_b is not None:
            value -= roc_auc_score(y[idx], score_b[idx])
        stats[i] = value
    alpha = (1.0 - level) / 2.0
    return float(np.quantile(stats, alpha)), float(np.quantile(stats, 1.0 - alpha))
