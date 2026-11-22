"""Reason codes: known answer on a toy scorer, then behaviour on the real model."""

import numpy as np

from creditadj import reasons
from creditadj.features import FEATURE_KEYS, FEATURES


def test_ranking_on_a_linear_toy_model():
    """With a linear score the drops are weight * (value - reference), so the order is known."""
    weights = np.zeros(len(FEATURE_KEYS))
    weights[[1, 5, 8]] = [0.30, 0.10, 0.20]

    def score(X):
        return 0.2 + X @ weights

    reference = np.zeros(len(FEATURE_KEYS), dtype=np.float32)
    x = np.zeros(len(FEATURE_KEYS), dtype=np.float32)
    x[[1, 5, 8]] = 1.0
    result = reasons.explain(score, x, reference)
    assert [r["feature"] for r in result] == [FEATURE_KEYS[1], FEATURE_KEYS[8], FEATURE_KEYS[5]]
    assert np.allclose([r["probability_drop"] for r in result], [0.30, 0.20, 0.10], atol=1e-6)
    # An applicant identical to the reference has nothing to explain.
    assert reasons.explain(score, reference, reference) == []


def test_reasons_are_deterministic_and_point_the_right_way(art, metrics):
    test = art.split.test
    declined = test[art.pd[test] >= art.recommended_cutoff][:150]
    monotone = {spec.key: spec.monotone for spec in FEATURES}
    for i in declined:
        x = art.X.to_numpy()[i]
        first = reasons.explain(art.score, x, art.reference)
        assert first == reasons.explain(art.score, x, art.reference)
        assert len(first) <= reasons.TOP_N
        drops = [r["probability_drop"] for r in first]
        assert drops == sorted(drops, reverse=True)
        for r in first:
            assert r["probability_drop"] >= reasons.MIN_DROP
            assert r["reason"] == reasons.REASON_TEXT[r["feature"]]
            # The sentence states a direction, so the applicant has to be on
            # the risky side of the reference for that feature.
            assert (r["value"] - r["reference"]) * monotone[r["feature"]] > 0

    # Agreement with XGBoost's own pred_contribs is reported, whatever it is.
    # The bound is not a quality bar, only a guard against the two methods
    # disagreeing outright, which would mean one of them is wired up wrongly.
    agreement = metrics["reasons"]["agreement_with_pred_contribs"]
    assert agreement["n_compared"] >= 200
    assert 0.5 < agreement["top1_in_shap_top4"] <= 1.0
    assert 0.0 <= agreement["top1_identical"] <= 1.0
