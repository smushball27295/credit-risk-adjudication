"""Reason codes for a decline, by one-at-a-time replacement.

Method, for one applicant:

1. Score the applicant as they are.
2. For each feature in turn, replace that one value with a fixed reference
   value (the median of approved, non-defaulting accounts in training) and
   score again.
3. The drop in predicted probability is that feature's contribution.
4. Rank by drop and report the plain-language text for the top four.

This needs nothing but the model itself, so a browser can reproduce it with
13 calls to its own scoring function. It is not a Shapley decomposition: the
drops do not sum to the gap between the applicant and the reference, and
changing one feature at a time can describe an account that could not exist
(for example, no late months but a worst status of 3). ``contrib_agreement``
measures how often it names the same features as XGBoost's own TreeSHAP.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import xgboost as xgb

from .features import FEATURE_KEYS

TOP_N = 4
# A feature has to move the probability by at least a tenth of a percentage
# point to be cited as a reason.
MIN_DROP = 0.001

# Every feature has a monotonic constraint, so a positive drop can only come
# from a value on the risky side of the reference. Each sentence can therefore
# state a direction and be right by construction (tests/test_reasons.py checks it).
REASON_TEXT: dict[str, str] = {
    "credit_limit": "Existing credit limit is low",
    "recent_status": "Most recent payment was late",
    "worst_status_6m": "Worst delinquency in the last six months was too severe",
    "months_late_6m": "Too many late payments in the last six months",
    "months_since_late": "Last late payment was too recent",
    "util_current": "Current balance is high relative to the credit limit",
    "util_avg_6m": "Average balance over six months is high relative to the credit limit",
    "util_trend": "Balances have been rising relative to the credit limit",
    "pay_ratio": "Payments cover too small a share of the amounts billed",
    "last_payment": "Most recent payment amount was low",
    "months_paid_full": "Balance is rarely paid in full",
    "months_zero_balance": "Little or no card use in recent months",
}

ScoreFn = Callable[[np.ndarray], np.ndarray]


def reference_values(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    pd_train: np.ndarray,
    cutoff: float,
) -> np.ndarray:
    """Median of each feature among approved, non-defaulting training accounts.

    I take the lower median so the reference is always a value that occurs in
    the data (a count of 2, never 2.5).
    """
    keep = (np.asarray(pd_train) < cutoff) & (np.asarray(y_train) == 0)
    if not keep.any():
        raise ValueError("No approved good accounts at this cutoff.")
    values = X_train.to_numpy(dtype=np.float32)[keep]
    return np.quantile(values, 0.5, axis=0, method="lower").astype(np.float32)


def drops(score_fn: ScoreFn, X: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Matrix of probability drops, one row per applicant and one column per feature."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    n, k = X.shape
    base = score_fn(X)
    # Stack k perturbed copies of every row and score them in one call.
    perturbed = np.repeat(X, k, axis=0)
    cols = np.tile(np.arange(k), n)
    perturbed[np.arange(n * k), cols] = np.tile(reference, n)
    rescored = score_fn(perturbed).reshape(n, k)
    return base[:, None] - rescored


def rank_reasons(drop_row: np.ndarray, top_n: int = TOP_N) -> list[int]:
    """Feature positions ordered by drop, largest first, ties by feature order."""
    order = sorted(range(len(drop_row)), key=lambda j: (-float(drop_row[j]), j))
    return [j for j in order if drop_row[j] >= MIN_DROP][:top_n]


def explain(
    score_fn: ScoreFn,
    x: np.ndarray,
    reference: np.ndarray,
    top_n: int = TOP_N,
) -> list[dict]:
    """Top reasons for one applicant, most important first."""
    x = np.asarray(x, dtype=np.float32).ravel()
    drop_row = drops(score_fn, x, reference)[0]
    return [
        {
            "feature": FEATURE_KEYS[j],
            "reason": REASON_TEXT[FEATURE_KEYS[j]],
            "value": float(x[j]),
            "reference": float(reference[j]),
            "probability_drop": float(drop_row[j]),
        }
        for j in rank_reasons(drop_row, top_n)
    ]


def contrib_agreement(
    booster: xgb.Booster,
    score_fn: ScoreFn,
    X: np.ndarray,
    reference: np.ndarray,
    top_n: int = TOP_N,
) -> dict:
    """Compare the top reasons with XGBoost's ``pred_contribs`` (TreeSHAP).

    The two methods answer slightly different questions. TreeSHAP attributes
    the gap between this score and the average score. Replacement attributes
    the gap to a typical approved good account. Full agreement is not expected.
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float32))
    drop = drops(score_fn, X, reference)
    dmatrix = xgb.DMatrix(X, feature_names=FEATURE_KEYS)
    contribs = booster.predict(dmatrix, pred_contribs=True)[:, :-1]

    top1_same = 0
    top1_in_other = 0
    overlap_total = 0.0
    compared = 0
    for i in range(len(X)):
        mine = rank_reasons(drop[i], top_n)
        if not mine:
            continue
        shap_order = sorted(range(X.shape[1]), key=lambda j: (-float(contribs[i, j]), j))
        theirs = [j for j in shap_order if contribs[i, j] > 0][:top_n]
        if not theirs:
            continue
        compared += 1
        top1_same += int(mine[0] == theirs[0])
        top1_in_other += int(mine[0] in theirs)
        overlap_total += len(set(mine) & set(theirs)) / min(len(mine), len(theirs))
    if compared == 0:
        raise ValueError("No applicants with reasons from both methods.")
    return {
        "n_compared": compared,
        "top1_identical": top1_same / compared,
        "top1_in_shap_top4": top1_in_other / compared,
        "mean_top4_overlap": overlap_total / compared,
    }
