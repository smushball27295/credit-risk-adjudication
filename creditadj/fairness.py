"""Outcome check on attributes the model never sees.

Excluding sex, age and marital status from the inputs does not guarantee
equal outcomes, because the permitted features can still correlate with them.
This module measures the outcomes: approval rate by group at the chosen
cutoff, and the adverse-impact ratio (lowest group approval rate divided by
the highest). A ratio below 0.80 is the usual "four-fifths" screening flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEX_LABELS = {1: "Male", 2: "Female"}
MARRIAGE_LABELS = {1: "Married", 2: "Single", 3: "Other"}
AGE_BANDS = [(21, 29, "21-29"), (30, 39, "30-39"), (40, 49, "40-49"), (50, 200, "50+")]

FOUR_FIFTHS = 0.80


def sex_groups(raw: pd.DataFrame) -> np.ndarray:
    return raw["SEX"].map(SEX_LABELS).to_numpy()


def age_groups(raw: pd.DataFrame) -> np.ndarray:
    age = raw["AGE"].to_numpy()
    labels = np.empty(len(age), dtype=object)
    for low, high, name in AGE_BANDS:
        labels[(age >= low) & (age <= high)] = name
    return labels


def marriage_groups(raw: pd.DataFrame) -> np.ndarray:
    # Undocumented code 0 is folded into "Other", as in features.protected_features.
    marriage = raw["MARRIAGE"].where(raw["MARRIAGE"].isin([1, 2]), 3)
    return marriage.map(MARRIAGE_LABELS).to_numpy()


def group_table(
    groups: np.ndarray,
    y: np.ndarray,
    pd_hat: np.ndarray,
    cutoff: float,
    order: list[str],
) -> list[dict]:
    """Approval and default rates for each group, in the given order."""
    y = np.asarray(y)
    approved = np.asarray(pd_hat) < cutoff
    rows = []
    for name in order:
        member = groups == name
        if not member.any():
            continue
        good = member & (y == 0)
        rows.append(
            {
                "group": name,
                "n": int(member.sum()),
                "approval_rate": float(approved[member].mean()),
                "default_rate": float(y[member].mean()),
                "default_rate_approved": (
                    float(y[member & approved].mean())
                    if (member & approved).any()
                    else float("nan")
                ),
                # Among customers who did not default, how many were approved.
                "approval_rate_goods": float(approved[good].mean()),
            }
        )
    return rows


def adverse_impact_ratio(rows: list[dict]) -> float:
    """Lowest group approval rate divided by the highest."""
    rates = [r["approval_rate"] for r in rows]
    if max(rates) == 0:
        return float("nan")
    return float(min(rates) / max(rates))


def fairness_report(raw: pd.DataFrame, y: np.ndarray, pd_hat: np.ndarray, cutoff: float) -> dict:
    by_sex = group_table(sex_groups(raw), y, pd_hat, cutoff, list(SEX_LABELS.values()))
    by_age = group_table(age_groups(raw), y, pd_hat, cutoff, [name for _, _, name in AGE_BANDS])
    by_marriage = group_table(
        marriage_groups(raw), y, pd_hat, cutoff, list(MARRIAGE_LABELS.values())
    )
    return {
        "cutoff": float(cutoff),
        "by_sex": by_sex,
        "by_age_band": by_age,
        "by_marital_status": by_marriage,
        "adverse_impact_ratio_sex": adverse_impact_ratio(by_sex),
        "adverse_impact_ratio_age": adverse_impact_ratio(by_age),
        "adverse_impact_ratio_marital_status": adverse_impact_ratio(by_marriage),
        "four_fifths_threshold": FOUR_FIFTHS,
    }


def age_gap_probe(
    score_fn,
    X: np.ndarray,
    reference: np.ndarray,
    feature_keys: list[str],
    raw: pd.DataFrame,
    y: np.ndarray,
    cutoff: float,
) -> list[dict]:
    """Which input carries the age gap? Neutralise one feature at a time.

    For each feature, every account gets the reference value for that feature,
    everything is re-scored, and the age adverse-impact ratio is recomputed at
    the same cutoff. A feature whose removal moves the ratio towards 1 is one
    through which age reaches the decision. This is a diagnostic, not a fix:
    it changes the overall approval rate as well.
    """
    X = np.asarray(X, dtype=np.float32)
    groups = age_groups(raw)
    order = [name for _, _, name in AGE_BANDS]
    rows = []
    for j, key in enumerate(feature_keys):
        neutral = X.copy()
        neutral[:, j] = reference[j]
        table = group_table(groups, y, score_fn(neutral), cutoff, order)
        rows.append({"feature": key, "adverse_impact_ratio_age": adverse_impact_ratio(table)})
    return sorted(rows, key=lambda r: -r["adverse_impact_ratio_age"])


def median_limit_by_age_band(raw: pd.DataFrame) -> dict:
    groups = age_groups(raw)
    limit = raw["LIMIT_BAL"].to_numpy()
    return {name: float(np.median(limit[groups == name])) for _, _, name in AGE_BANDS}
