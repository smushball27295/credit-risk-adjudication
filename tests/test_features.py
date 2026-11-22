"""Feature engineering on rows small enough to check by hand."""

import numpy as np
import pandas as pd
import pytest

from creditadj.data import BILL_COLS, PAID_COLS, PAY_COLS
from creditadj.features import FEATURE_KEYS, build_features, exposure


def make_row(limit, status, bills, paid):
    row = {"LIMIT_BAL": limit}
    row.update(dict(zip(PAY_COLS, status)))
    row.update(dict(zip(BILL_COLS, bills)))
    row.update(dict(zip(PAID_COLS, paid)))
    return row


# Index 0 is the most recent month in every list.
PAYS_IN_FULL = make_row(
    100_000, [-1, -1, -1, -1, -2, -2], [5000, 4000, 3000, 2000, 0, 0], [4000, 3000, 2000, 0, 0, 0]
)
DELINQUENT = make_row(
    50_000,
    [2, 2, 0, 0, 1, -1],
    [60_000, 55_000, 50_000, 45_000, 40_000, -500],
    [0, 1000, 2000, 2000, 500, 0],
)
OLD_LATE_OVER_LIMIT = make_row(
    20_000, [0, 0, 0, 3, 0, 0], [40_000, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]
)


@pytest.fixture(scope="module")
def features():
    raw = pd.DataFrame([PAYS_IN_FULL, DELINQUENT, OLD_LATE_OVER_LIMIT])
    return build_features(raw), raw


def test_clean_payer(features):
    f = features[0].iloc[0]
    assert f["recent_status"] == 0 and f["worst_status_6m"] == 0
    assert f["months_late_6m"] == 0 and f["months_since_late"] == 6
    assert f["util_current"] == pytest.approx(0.05)
    assert f["util_avg_6m"] == pytest.approx(14_000 / 6 / 100_000, abs=1e-4)
    assert f["util_trend"] == pytest.approx((4000 - 2000 / 3) / 100_000, abs=1e-4)
    # Payments 4000 + 3000 + 2000 settle the statements 4000 + 3000 + 2000.
    assert f["pay_ratio"] == pytest.approx(1.0)
    assert f["last_payment"] == 4000
    assert f["months_paid_full"] == 4 and f["months_zero_balance"] == 2


def test_delinquent_account(features):
    f = features[0].iloc[1]
    assert f["recent_status"] == 2 and f["worst_status_6m"] == 2
    assert f["months_late_6m"] == 3 and f["months_since_late"] == 0
    assert f["util_current"] == pytest.approx(1.2)
    # The -500 bill is a credit balance and counts as zero.
    assert f["util_avg_6m"] == pytest.approx(250_000 / 6 / 50_000, abs=1e-4)
    assert f["pay_ratio"] == pytest.approx(5500 / 190_000, abs=1e-4)
    # ... and as a month with no balance.
    assert f["months_paid_full"] == 1 and f["months_zero_balance"] == 1


def test_undocumented_codes_and_caps(features):
    frame, raw = features
    f = frame.iloc[2]
    # Status 0 (revolving) is not late; the single late month was 3 months ago.
    assert f["recent_status"] == 0 and f["worst_status_6m"] == 3
    assert f["months_late_6m"] == 1 and f["months_since_late"] == 3
    assert f["util_current"] == pytest.approx(1.5)  # 2.0 capped at 1.5
    assert f["pay_ratio"] == pytest.approx(1.0)  # nothing was billed in months 2 to 6
    assert f["months_zero_balance"] == 5
    assert list(frame.columns) == FEATURE_KEYS
    assert (frame.dtypes == np.float32).all() and not frame.isna().any().any()
    assert exposure(raw).tolist() == [5000.0, 60_000.0, 40_000.0]
