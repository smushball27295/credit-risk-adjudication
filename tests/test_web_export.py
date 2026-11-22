"""The exported JSON must score exactly like XGBoost.

``walk`` below is a deliberately small pure-Python tree walker. It is the
reference for what the browser does: float32 inputs, "yes" when x < t.
"""

import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from creditadj import export
from creditadj.features import FEATURE_KEYS

COMMITTED = Path(__file__).resolve().parents[1] / "web_export" / "model.json"


def to_float32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def walk(payload: dict, values: list[float]) -> float:
    x = [to_float32(v) for v in values]
    total = payload["model"]["base_score_logit"]
    for tree in payload["model"]["trees"]:
        node = tree[0]
        while "v" not in node:
            value = x[node["f"]]
            if value != value:  # NaN
                node = tree[node["m"]]
            elif value < node["t"]:
                node = tree[node["y"]]
            else:
                node = tree[node["n"]]
        total += node["v"]
    calibration = payload["calibration"]
    if calibration is not None:
        total = calibration["a"] * total + calibration["b"]
    return 1.0 / (1.0 + math.exp(-total))


@pytest.fixture(scope="module")
def payload(art, metrics, tmp_path_factory):
    path = tmp_path_factory.mktemp("web") / "model.json"
    size = export.write_payload(export.build_payload(art, metrics), path)
    assert size < export.MAX_BYTES
    return json.loads(path.read_text())


def test_json_reproduces_xgboost_probabilities(art, payload):
    rows = art.split.test[:200]
    X = art.X.iloc[rows]
    expected = art.champion.predict_proba(X)[:, 1]
    if payload["calibration"] is not None:
        expected = art.pd[rows]
    walked = [walk(payload, row) for row in X.to_numpy().tolist()]
    assert np.max(np.abs(np.array(walked) - expected)) < 1e-6
    # The vectorised scorer used inside export.py agrees as well.
    vectorised = export.score_model_block(payload["model"], payload["calibration"], X.to_numpy())
    assert np.max(np.abs(vectorised - expected)) < 1e-6


def test_export_has_the_agreed_shape(payload):
    assert list(payload) == [
        "meta",
        "features",
        "model",
        "calibration",
        "cutoff",
        "metrics",
        "deciles",
        "fairness",
        "reasons",
        "presets",
    ]
    assert set(payload["meta"]) == {"dataset", "n_train", "n_test", "generated_at", "seed"}
    assert [f["key"] for f in payload["features"]] == FEATURE_KEYS
    for feature in payload["features"]:
        assert set(feature) == {
            "key",
            "label",
            "help",
            "unit",
            "min",
            "max",
            "step",
            "median",
            "reference",
            "monotone",
        }
        assert feature["unit"] in {"currency", "ratio", "count", "status"}
        assert feature["monotone"] in {-1, 0, 1}
        assert feature["min"] <= feature["reference"] <= feature["max"]
    for tree in payload["model"]["trees"]:
        for node in tree:
            assert set(node) in ({"v"}, {"f", "t", "y", "n", "m"})
    assert len(payload["model"]["trees"]) == 200
    assert set(payload["reasons"]) == set(FEATURE_KEYS)
    grid_keys = {
        "cutoff",
        "approval_rate",
        "default_rate_approved",
        "defaulters_declined",
        "defaulted_balance_declined",
        "goods_declined",
        "illustrative_profit_index",
    }
    assert all(set(row) == grid_keys for row in payload["cutoff"]["grid"])
    assert payload["cutoff"]["recommended"] in [row["cutoff"] for row in payload["cutoff"]["grid"]]
    assert len(payload["deciles"]) == 10 and len(payload["presets"]) == 5
    assert set(payload["fairness"]) == {
        "cutoff",
        "by_sex",
        "by_age_band",
        "adverse_impact_ratio_sex",
    }


def test_committed_outputs_are_consistent():
    """The committed export scores its own presets, and the README quotes metrics.json."""
    assert COMMITTED.exists(), "run `python -m creditadj all`"
    assert COMMITTED.stat().st_size < export.MAX_BYTES
    payload = json.loads(COMMITTED.read_text())
    keys = [f["key"] for f in payload["features"]]
    probabilities = []
    for preset in payload["presets"]:
        assert "id" not in {k.lower() for k in preset["values"]}
        walked = walk(payload, [preset["values"][k] for k in keys])
        assert walked == pytest.approx(preset["expected_probability"], abs=1e-6)
        probabilities.append(walked)
    assert probabilities == sorted(probabilities)

    root = COMMITTED.parents[1]
    metrics = json.loads((root / "reports" / "metrics.json").read_text())
    assert payload["metrics"]["champion"]["auc"] == pytest.approx(
        metrics["champion"]["auc"], abs=1e-6
    )
    readme = (root / "README.md").read_text()
    champion, benchmark = metrics["champion"], metrics["benchmark"]
    at_cutoff = metrics["cutoff"]["at_recommended_test"]
    agreement = metrics["reasons"]["agreement_with_pred_contribs"]
    quoted = [
        f"| {champion['auc']:.3f} | {champion['gini']:.3f} | {champion['ks']:.3f} |",
        f"| {benchmark['auc']:.3f} | {benchmark['gini']:.3f} | {benchmark['ks']:.3f} |",
        f"{champion['auc_ci_low']:.3f} to {champion['auc_ci_high']:.3f}",
        f"{metrics['with_protected']['auc']:.3f} with them",
        f"{metrics['unconstrained']['auc']:.3f} without them",
        f"Recommended cutoff: {metrics['cutoff']['recommended']:.2f}",
        f"approves {100 * at_cutoff['approval_rate']:.1f}% of accounts",
        f"declines {100 * at_cutoff['defaulters_declined']:.1f}% of defaulters",
        f"{metrics['fairness']['adverse_impact_ratio_sex']:.3f} by sex",
        f"{metrics['fairness']['adverse_impact_ratio_age']:.3f} by age band",
        f"| Same top reason | {100 * agreement['top1_identical']:.1f}% |",
    ]
    missing = [text for text in quoted if text not in readme]
    assert not missing, f"README is out of date with reports/metrics.json: {missing}"
