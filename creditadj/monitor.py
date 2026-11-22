"""Population stability index (PSI) for the inputs and the score.

PSI compares the share of accounts in each bin between a reference sample
and a new one: sum((new - ref) * ln(new / ref)). Usual reading:

    below 0.10      stable
    0.10 to 0.25    moderate shift, investigate
    0.25 and above  significant shift, review the model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_WATCH = 0.10
PSI_ACTION = 0.25

# Empty bins would make the log blow up; every share is floored at this.
_FLOOR = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI of ``actual`` against ``expected`` using quantile bins of ``expected``.

    Repeated quantiles are merged, so a feature with a few distinct values
    (a count from 0 to 6, say) is binned by value.
    """
    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)[1:-1]))
    expected_share = np.bincount(
        np.searchsorted(edges, expected, side="left"), minlength=len(edges) + 1
    ) / len(expected)
    actual_share = np.bincount(
        np.searchsorted(edges, actual, side="left"), minlength=len(edges) + 1
    ) / len(actual)
    expected_share = np.clip(expected_share, _FLOOR, None)
    actual_share = np.clip(actual_share, _FLOOR, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def psi_status(value: float) -> str:
    if value < PSI_WATCH:
        return "stable"
    if value < PSI_ACTION:
        return "investigate"
    return "review"


def psi_table(
    X_ref: pd.DataFrame,
    X_new: pd.DataFrame,
    score_ref: np.ndarray,
    score_new: np.ndarray,
) -> list[dict]:
    """PSI for the score and for each feature, score first."""
    rows = [{"variable": "score", "psi": psi(score_ref, score_new)}]
    for key in X_ref.columns:
        rows.append({"variable": key, "psi": psi(X_ref[key], X_new[key])})
    for row in rows:
        row["status"] = psi_status(row["psi"])
    return rows


def shifted_copy(X: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """A deliberately stressed copy of ``X``, to show what PSI looks like when it fires.

    The scenario: balances rise by 30 percent across the book, customers repay
    20 percent less of what they are billed, and one account in ten that was
    up to date misses its latest payment. It is made up for the demonstration
    and is not a forecast.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    out["util_current"] = np.clip(out["util_current"] * 1.3, 0.0, 1.5)
    out["util_avg_6m"] = np.clip(out["util_avg_6m"] * 1.3, 0.0, 1.5)
    out["pay_ratio"] = out["pay_ratio"] * 0.8

    current = (out["recent_status"] == 0).to_numpy()
    newly_late = current & (rng.random(len(out)) < 0.10)
    out.loc[newly_late, "recent_status"] = 1
    out.loc[newly_late, "worst_status_6m"] = np.maximum(out.loc[newly_late, "worst_status_6m"], 1)
    out.loc[newly_late, "months_late_6m"] = np.minimum(out.loc[newly_late, "months_late_6m"] + 1, 6)
    out.loc[newly_late, "months_since_late"] = 0
    return out.round(4).astype(np.float32)
