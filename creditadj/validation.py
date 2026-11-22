"""The tests a validator can run on a submission, and the evidence they return.

Each test returns a short list of evidence items. An item has an id and one
sentence of text with the measured numbers in it. The validation memo may
only state a number if it appears in an item the memo cites, so the wording
here is the single source of every figure in a memo.

Tests report what was measured and the usual conventions for reading it.
They never say whether a result is acceptable. That judgement belongs to the
validator, human or agent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import evaluate, fairness, model, monitor
from . import cutoff as cutoff_mod
from .candidates import DICTIONARY, Candidate, recommended_cutoff
from .features import FEATURE_KEYS, MONOTONE

FAIRNESS_ATTRIBUTES = ("sex", "age_band", "marital_status")

# name -> (description shown to the validator, argument name or None)
TESTS: dict[str, tuple[str, str | None]] = {
    "performance": ("Discrimination on the training, validation and test samples, and the model's size.", None),
    "benchmark": ("A plain logistic regression on the 12 standard inputs, scored on the same test sample.", None),
    "calibration": ("Predicted probability of default against the observed default rate.", None),
    "rank_ordering": ("Observed default rate by score decile.", None),
    "inputs": ("Every model input, its share of the model's total gain and its single-input AUC.", None),
    "feature_detail": ("The data dictionary entry and statistics for one input.", "feature"),
    "data_integrity": ("Row counts, default rates, missing values and overlap between the samples.", None),
    "stability": ("Population stability index between the development sample and the most recent population.", None),
    "fairness": ("Approval rates by group at the recommended cutoff.", "attribute"),
    "sensitivity": ("Whether the score moves in the expected direction when one input is changed.", None),
    "cutoff": ("The recommended cutoff and what it does on the test sample.", None),
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def _item(evidence_id: str, text: str) -> dict:
    return {"id": evidence_id, "text": text}


def _result(name: str, items: list[dict]) -> dict:
    return {"test": name, "items": items}


def _scores(c: Candidate) -> dict[str, np.ndarray]:
    return {name: c.score(c.X.iloc[idx]) for name, idx in (("train", c.train), ("valid", c.valid), ("test", c.test))}


def performance(c: Candidate) -> dict:
    s = _scores(c)
    auc = {k: evaluate.auc(c.y[idx], s[k]) for k, idx in (("train", c.train), ("valid", c.valid), ("test", c.test))}
    low, high = evaluate.bootstrap_auc_ci(c.y[c.test], s["test"], n_boot=500, seed=0)
    disc = evaluate.discrimination(c.y[c.test], s["test"])
    booster = c.booster.get_booster()
    trees = booster.num_boosted_rounds()
    depth = int(c.booster.get_params()["max_depth"])
    return _result(
        "performance",
        [
            _item("PERF-TRAIN", f"AUC on the training sample ({len(c.train):,} accounts): {auc['train']:.3f}."),
            _item("PERF-VALID", f"AUC on the validation sample ({len(c.valid):,} accounts): {auc['valid']:.3f}."),
            _item(
                "PERF-TEST",
                f"AUC on the test sample ({len(c.test):,} accounts): {auc['test']:.3f}, "
                f"95% bootstrap interval {low:.3f} to {high:.3f}.",
            ),
            _item("PERF-KS", f"On the test sample the KS statistic is {disc['ks']:.3f} and the Gini is {disc['gini']:.3f}."),
            _item("PERF-GAP", f"Training AUC minus test AUC: {auc['train'] - auc['test']:.3f}."),
            _item("PERF-SIZE", f"The model has {trees} trees with a maximum depth of {depth} and {len(c.columns)} inputs."),
            _item(
                "PERF-NOTE",
                "Convention: a training-minus-test gap above 0.05 AUC is usually read as overfitting, and consumer "
                "credit default models on behavioural data typically reach a test AUC between 0.70 and 0.85.",
            ),
        ],
    )


def benchmark(c: Candidate) -> dict:
    # The benchmark always uses the 12 standard inputs, whatever the submission uses.
    from .pipeline import build

    art = build()
    y_test = c.y[c.test]
    bench = art.pd_benchmark[c.test]
    cand = c.score(c.X.iloc[c.test])
    a_b, a_c = evaluate.auc(y_test, bench), evaluate.auc(y_test, cand)
    low, high = evaluate.bootstrap_auc_diff_ci(y_test, cand, bench, n_boot=500, seed=0)
    return _result(
        "benchmark",
        [
            _item("BENCH-AUC", f"Logistic regression benchmark on the 12 standard inputs: test AUC {a_b:.3f}, KS {evaluate.ks_statistic(y_test, bench):.3f}."),
            _item(
                "BENCH-DIFF",
                f"Submission AUC minus benchmark AUC on the same test accounts: {a_c - a_b:+.3f}, "
                f"95% bootstrap interval {low:+.3f} to {high:+.3f}.",
            ),
        ],
    )


def calibration(c: Candidate) -> dict:
    y, score = c.y[c.test], c.score(c.X.iloc[c.test])
    summary = evaluate.calibration_summary(y, score)
    a, b = model.fit_platt(model.margin(c.booster, c.X.iloc[c.valid][c.columns]), c.y[c.valid])
    table = evaluate.decile_table(y, score)
    worst = int((table["avg_score"] - table["default_rate"]).abs().idxmax())
    row = table.iloc[worst]
    return _result(
        "calibration",
        [
            _item(
                "CAL-MEAN",
                f"Mean predicted probability of default on the test sample: {_pct(summary['mean_predicted'])}. "
                f"Observed default rate: {_pct(summary['observed_rate'])}.",
            ),
            _item(
                "CAL-BANDS",
                f"Largest gap between predicted and observed in any score decile: {100 * summary['max_band_gap']:.1f} points "
                f"(decile {int(row['decile'])}: predicted {_pct(row['avg_score'])}, observed {_pct(row['default_rate'])}).",
            ),
            _item("CAL-SLOPE", f"Calibration fit on the validation sample: slope {a:.3f}, intercept {b:.3f}. A calibrated score has slope 1 and intercept 0."),
            _item("CAL-BRIER", f"Brier score on the test sample: {summary['brier']:.4f}."),
        ],
    )


def rank_ordering(c: Candidate) -> dict:
    table = evaluate.decile_table(c.y[c.test], c.score(c.X.iloc[c.test]))
    rates = table["default_rate"].tolist()
    inv = evaluate.inversions(rates)
    listing = ", ".join(_pct(r) for r in rates)
    return _result(
        "rank_ordering",
        [
            _item("RANK-DECILES", f"Observed default rate by score decile on the test sample, safest first: {listing}."),
            _item(
                "RANK-INV",
                "Default rates rise in every decile." if not inv
                else f"Default rates do not rise at decile {', '.join(str(i) for i in inv)}: {len(inv)} inversion(s).",
            ),
        ],
    )


def _gain_share(c: Candidate) -> dict[str, float]:
    gain = c.booster.get_booster().get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    return {key: gain.get(key, 0.0) / total for key in c.columns}


def _single_auc(c: Candidate, key: str) -> float:
    a = evaluate.auc(c.y[c.test], c.X.iloc[c.test][key].to_numpy(dtype=np.float64))
    return max(a, 1.0 - a)


def inputs(c: Candidate) -> dict:
    share = _gain_share(c)
    items = [_item("INP-LIST", f"The model has {len(c.columns)} inputs: {', '.join(c.columns)}.")]
    for key in sorted(c.columns, key=lambda k: -share[k]):
        items.append(
            _item(
                f"INP-{key}",
                f"{key}: {_pct(share[key])} of the model's total gain; AUC of this input on its own {_single_auc(c, key):.3f}.",
            )
        )
    return _result("inputs", items)


def feature_detail(c: Candidate, feature: str) -> dict:
    if feature not in c.columns:
        return _result("feature_detail", [_item(f"FEAT-{feature}-NONE", f"{feature} is not an input to this model.")])
    column = c.X[feature].to_numpy(dtype=np.float64)
    corr = float(np.corrcoef(column[c.test], c.y[c.test])[0, 1])
    return _result(
        "feature_detail",
        [
            _item(f"FEAT-{feature}-DICT", f"Data dictionary entry for {feature}: {DICTIONARY[feature]}"),
            _item(
                f"FEAT-{feature}-STATS",
                f"{feature}: {_pct(_gain_share(c)[feature])} of total gain, AUC on its own {_single_auc(c, feature):.3f}, "
                f"correlation with default {corr:+.3f}, mean {np.mean(column):,.2f}.",
            ),
        ],
    )


def data_integrity(c: Candidate) -> dict:
    overlap = int(np.intersect1d(c.train, c.test).size)
    rate = {name: float(c.y[idx].mean()) for name, idx in (("training", c.train), ("validation", c.valid), ("test", c.test))}
    missing = int(c.X[c.columns].isna().sum().sum())
    return _result(
        "data_integrity",
        [
            _item("DATA-ROWS", f"Sample sizes: training {len(c.train):,}, validation {len(c.valid):,}, test {len(c.test):,}."),
            _item("DATA-RATES", "Default rate by sample: " + ", ".join(f"{k} {_pct(v)}" for k, v in rate.items()) + "."),
            _item("DATA-OVERLAP", f"Accounts that appear in both the training sample and the test sample: {overlap:,} of {len(c.test):,} test accounts."),
            _item("DATA-MISSING", f"Missing values across all model inputs: {missing:,}."),
            _item("DATA-WINDOW", "All accounts are observed over the same six months, April to September 2005, with the outcome in the following month. The split is random. There is no out-of-time sample."),
        ],
    )


def stability(c: Candidate) -> dict:
    ref = c.X.iloc[c.train]
    recent = c.X_recent
    s_ref, s_new = c.score(ref), c.score(recent)
    rows = monitor.psi_table(ref[c.columns], recent[c.columns], s_ref, s_new)
    by_psi = sorted(rows[1:], key=lambda r: -r["psi"])[:3]
    items = [
        _item("STAB-SCORE", f"Population stability index of the score, development sample against the most recent population ({len(recent):,} accounts): {rows[0]['psi']:.3f}."),
        _item("STAB-MEAN", f"Mean predicted probability of default: {_pct(float(s_ref.mean()))} on the development sample, {_pct(float(s_new.mean()))} on the most recent population."),
    ]
    for row in by_psi:
        items.append(_item(f"STAB-{row['variable']}", f"Population stability index of {row['variable']}: {row['psi']:.3f}."))
    items.append(_item("STAB-NOTE", "Convention: a PSI below 0.10 is read as stable, 0.10 to 0.25 as worth investigating, and 0.25 or more as a material shift."))
    return _result("stability", items)


def fairness_test(c: Candidate, attribute: str) -> dict:
    if attribute not in FAIRNESS_ATTRIBUTES:
        return _result("fairness", [_item("FAIR-NONE", f"Unknown attribute. Choose one of: {', '.join(FAIRNESS_ATTRIBUTES)}.")])
    cutoff = recommended_cutoff(c)
    raw = c.raw.iloc[c.test]
    report = fairness.fairness_report(raw, c.y[c.test], c.score(c.X.iloc[c.test]), cutoff)
    rows = report[{"sex": "by_sex", "age_band": "by_age_band", "marital_status": "by_marital_status"}[attribute]]
    air = fairness.adverse_impact_ratio(rows)
    used = [k for k in c.columns if k in ("age", "sex_female", "married", "single")]
    return _result(
        "fairness",
        [
            _item(
                f"FAIR-{attribute}-RATES",
                f"Approval rate by {attribute.replace('_', ' ')} at a cutoff of {_pct(cutoff, 0)}: "
                + ", ".join(f"{r['group']} {_pct(r['approval_rate'])} ({r['n']:,} accounts)" for r in rows) + ".",
            ),
            _item(
                f"FAIR-{attribute}-AIR",
                f"Adverse-impact ratio by {attribute.replace('_', ' ')} (lowest approval rate divided by highest): {air:.3f}."
                if np.isfinite(air)
                else f"Adverse-impact ratio by {attribute.replace('_', ' ')}: not defined, because the cutoff approves no accounts.",
            ),
            _item(
                f"FAIR-{attribute}-GOODS",
                f"Approval rate among customers who did not default, by {attribute.replace('_', ' ')}: "
                + ", ".join(f"{r['group']} {_pct(r['approval_rate_goods'])}" for r in rows) + ".",
            ),
            _item(
                f"FAIR-{attribute}-INPUTS",
                "Demographic fields used as model inputs: " + (", ".join(used) if used else "none") + ".",
            ),
            _item("FAIR-NOTE", "Convention: a ratio under 0.80 is the four-fifths screen. It comes from US employment guidelines and is a first look, not a legal test for lending."),
        ],
    )


def sensitivity(c: Candidate) -> dict:
    X = c.X.iloc[c.test][c.columns]
    base = c.score(X)
    expected = dict(zip(FEATURE_KEYS, MONOTONE))
    checks, wrong = 0, 0
    worst_key, worst_share = "", 0.0
    for key in c.columns:
        sign = expected.get(key)
        if not sign:
            continue
        step = float(X[key].std()) or 1.0
        moved = X.copy()
        moved[key] = moved[key] + step
        delta = c.score(moved) - base
        bad = float(np.mean(delta * sign < -1e-9))
        checks += 1
        if bad > 0:
            wrong += 1
        if bad > worst_share:
            worst_key, worst_share = key, bad
    items = [
        _item(
            "SENS-DIR",
            f"Each of {checks} standard inputs was raised by one standard deviation for every test account. "
            f"For {wrong} of them the score moved against the expected direction for at least one account.",
        )
    ]
    if worst_key:
        items.append(_item("SENS-WORST", f"Largest share of accounts moving the wrong way: {worst_key}, {_pct(worst_share)} of test accounts."))
    if "util_current" in c.columns:
        moved = X.copy()
        moved["util_current"] = np.clip(moved["util_current"] * 1.1, 0, 1.5)
        items.append(_item("SENS-UTIL", f"Raising current utilisation by 10% changes the mean predicted probability of default by {100 * float(np.mean(c.score(moved) - base)):+.2f} points."))
    return _result("sensitivity", items)


def cutoff_test(c: Candidate) -> dict:
    cutoff = recommended_cutoff(c)
    table = cutoff_mod.cutoff_table(c.y[c.test], c.score(c.X.iloc[c.test]), c.exposure[c.test])
    row = cutoff_mod.row_at(table, cutoff)
    return _result(
        "cutoff",
        [
            _item("CUT-REC", f"Recommended cutoff, chosen on the validation sample: approve below a predicted probability of default of {_pct(cutoff, 0)}."),
            _item(
                "CUT-TEST",
                f"On the test sample that cutoff approves {_pct(row['approval_rate'])} of accounts, declines "
                f"{_pct(row['defaulters_declined'])} of defaulters and {_pct(row['goods_declined'])} of good accounts. "
                + (
                    f"The default rate among approved accounts is {_pct(row['default_rate_approved'])}."
                    if np.isfinite(row["default_rate_approved"])
                    else "No accounts are approved, so there is no default rate among approved accounts."
                ),
            ),
        ],
    )


def run_test(c: Candidate, name: str, argument: str | None = None) -> dict:
    """Run one test by name. Unknown names return an evidence item saying so."""
    if name == "performance":
        return performance(c)
    if name == "benchmark":
        return benchmark(c)
    if name == "calibration":
        return calibration(c)
    if name == "rank_ordering":
        return rank_ordering(c)
    if name == "inputs":
        return inputs(c)
    if name == "feature_detail":
        return feature_detail(c, argument or "")
    if name == "data_integrity":
        return data_integrity(c)
    if name == "stability":
        return stability(c)
    if name == "fairness":
        return fairness_test(c, argument or "")
    if name == "sensitivity":
        return sensitivity(c)
    if name == "cutoff":
        return cutoff_test(c)
    return _result(name, [_item("TEST-NONE", f"There is no test called {name}.")])


def evidence_pack(c: Candidate) -> dict:
    """Every test result for one submission, keyed the way a tool call asks for it.

    The web demo serves tool calls from this pack, so the agent there sees
    exactly what the Python tests computed.
    """
    pack: dict[str, dict] = {}
    for name, (_, arg) in TESTS.items():
        if arg is None:
            pack[name] = run_test(c, name)
        elif name == "feature_detail":
            for key in c.columns:
                pack[f"feature_detail:{key}"] = run_test(c, name, key)
        elif name == "fairness":
            for attribute in FAIRNESS_ATTRIBUTES:
                pack[f"fairness:{attribute}"] = run_test(c, name, attribute)
    return pack
