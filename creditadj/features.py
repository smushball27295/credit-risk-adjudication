"""The 12 engineered features. The champion model sees these and nothing else.

Month index 1 is September 2005 (the most recent month), index 6 is April.

How the repayment status codes are read (PAY_1 .. PAY_6):

    -2   no balance to pay that month (card not used)    undocumented by UCI
    -1   statement paid in full                          documented as "pay duly"
     0   paid at least the minimum, carried a balance    undocumented by UCI
    1-8  months past due                                  documented

Only 1 and above count as late. I never treat -2 < -1 < 0 as an ordered risk
scale. Months coded -1 are counted in months_paid_full. Inactivity is measured
from the bills themselves (months_zero_balance), not from the undocumented -2.

Every feature has a signed monotonic constraint. That is what lets each reason
code carry a fixed, directional sentence.

Every feature is rounded (ratios to 4 decimals) and stored as float32. XGBoost
compares float32 values against float32 thresholds, so a browser that applies
Math.fround to its inputs gets bit-identical tree paths.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import BILL_COLS, PAID_COLS, PAY_COLS

UTIL_CAP = 1.5
RATIO_DECIMALS = 4


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    help: str
    unit: str  # "currency" | "ratio" | "count" | "status"
    monotone: int  # +1 risk never falls as the value rises, -1 it never rises
    min: float
    max: float
    step: float


FEATURES: list[FeatureSpec] = [
    FeatureSpec(
        "credit_limit",
        "Credit limit",
        "Credit line on the card in New Taiwan dollars.",
        "currency",
        -1,
        10_000,
        1_000_000,
        10_000,
    ),
    FeatureSpec(
        "recent_status",
        "Most recent repayment status",
        "Months past due on the latest statement. 0 means not late.",
        "status",
        1,
        0,
        8,
        1,
    ),
    FeatureSpec(
        "worst_status_6m",
        "Worst repayment status in 6 months",
        "Largest number of months past due in any of the last six statements.",
        "status",
        1,
        0,
        8,
        1,
    ),
    FeatureSpec(
        "months_late_6m",
        "Months delinquent in 6 months",
        "How many of the last six months the account was at least one month past due.",
        "count",
        1,
        0,
        6,
        1,
    ),
    FeatureSpec(
        "months_since_late",
        "Months since last delinquency",
        "0 means late this month. 6 means no late payment in the six-month window.",
        "count",
        -1,
        0,
        6,
        1,
    ),
    FeatureSpec(
        "util_current",
        "Current utilisation",
        "Latest statement balance divided by the credit limit. Capped at 1.5.",
        "ratio",
        1,
        0.0,
        UTIL_CAP,
        0.01,
    ),
    FeatureSpec(
        "util_avg_6m",
        "Average utilisation over 6 months",
        "Average statement balance over six months divided by the credit limit.",
        "ratio",
        1,
        0.0,
        UTIL_CAP,
        0.01,
    ),
    FeatureSpec(
        "util_trend",
        "Utilisation trend",
        "Average utilisation in the latest three months minus the three months "
        "before. Positive means balances are rising.",
        "ratio",
        1,
        -1.0,
        1.0,
        0.01,
    ),
    FeatureSpec(
        "pay_ratio",
        "Share of billed amount repaid",
        "Payments in the last five months divided by the statements they were "
        "due on. 1 means everything billed was repaid.",
        "ratio",
        -1,
        0.0,
        1.0,
        0.01,
    ),
    FeatureSpec(
        "last_payment",
        "Last payment amount",
        "Amount paid in the most recent month in New Taiwan dollars.",
        "currency",
        -1,
        0,
        100_000,
        500,
    ),
    FeatureSpec(
        "months_paid_full",
        "Months paid in full",
        "How many of the last six statements were paid in full.",
        "count",
        -1,
        0,
        6,
        1,
    ),
    FeatureSpec(
        "months_zero_balance",
        "Months with no balance",
        "How many of the last six statements showed no balance (card not in use).",
        "count",
        1,
        0,
        6,
        1,
    ),
]

FEATURE_KEYS: list[str] = [f.key for f in FEATURES]
MONOTONE: tuple[int, ...] = tuple(f.monotone for f in FEATURES)
CURRENCY_KEYS: list[str] = [f.key for f in FEATURES if f.unit == "currency"]


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn raw UCI columns into the 12 model inputs (float32, no missing values)."""
    limit = raw["LIMIT_BAL"].to_numpy(dtype=np.float64)
    status = raw[PAY_COLS].to_numpy(dtype=np.int64)
    bills = raw[BILL_COLS].to_numpy(dtype=np.float64)
    paid = raw[PAID_COLS].to_numpy(dtype=np.float64)

    # A negative bill is a credit balance (the customer overpaid). Nothing is
    # owed, so it counts as zero balance.
    balance = np.clip(bills, 0.0, None)

    late = status >= 1
    months_past_due = np.where(late, status, 0)
    months_late = late.sum(axis=1)
    # argmax returns the first True, which is the most recent late month.
    months_since_late = np.where(months_late > 0, late.argmax(axis=1), 6)

    util_current = np.clip(balance[:, 0] / limit, 0.0, UTIL_CAP)
    util_avg = np.clip(balance.mean(axis=1) / limit, 0.0, UTIL_CAP)
    util_trend = np.clip(
        (balance[:, :3].mean(axis=1) - balance[:, 3:].mean(axis=1)) / limit, -1.0, 1.0
    )

    # The payment made in month k settles the statement of month k+1 (one
    # month older). So payments 1..5 are matched to statements 2..6.
    billed = balance[:, 1:6].sum(axis=1)
    repaid = paid[:, 0:5].sum(axis=1)
    safe_billed = np.where(billed > 0, billed, 1.0)
    # Nothing billed means nothing was left unpaid, so the ratio is 1.
    pay_ratio = np.where(billed > 0, np.clip(repaid / safe_billed, 0.0, 1.0), 1.0)

    features = {
        "credit_limit": limit,
        "recent_status": months_past_due[:, 0],
        "worst_status_6m": months_past_due.max(axis=1),
        "months_late_6m": months_late,
        "months_since_late": months_since_late,
        "util_current": np.round(util_current, RATIO_DECIMALS),
        "util_avg_6m": np.round(util_avg, RATIO_DECIMALS),
        "util_trend": np.round(util_trend, RATIO_DECIMALS),
        "pay_ratio": np.round(pay_ratio, RATIO_DECIMALS),
        "last_payment": paid[:, 0],
        "months_paid_full": (status == -1).sum(axis=1),
        "months_zero_balance": (balance <= 0).sum(axis=1),
    }
    frame = pd.DataFrame(features, index=raw.index)[FEATURE_KEYS]
    return frame.astype(np.float32)


def exposure(raw: pd.DataFrame) -> np.ndarray:
    """Exposure at the decision date: the most recent statement balance, floored at 0."""
    return np.clip(raw["BILL_AMT1"].to_numpy(dtype=np.float64), 0.0, None)


def protected_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Columns used only by the comparison model and the fairness check.

    EDUCATION codes 0, 5 and 6 and MARRIAGE code 0 are not in the UCI
    documentation. I fold them into the documented "others" category.
    """
    education = raw["EDUCATION"].where(raw["EDUCATION"].isin([1, 2, 3]), 4)
    marriage = raw["MARRIAGE"].where(raw["MARRIAGE"].isin([1, 2]), 3)
    frame = pd.DataFrame(
        {
            "age": raw["AGE"],
            "sex_female": (raw["SEX"] == 2).astype(int),
            "married": (marriage == 1).astype(int),
            "single": (marriage == 2).astype(int),
            "edu_graduate": (education == 1).astype(int),
            "edu_university": (education == 2).astype(int),
            "edu_high_school": (education == 3).astype(int),
        },
        index=raw.index,
    )
    return frame.astype(np.float32)
