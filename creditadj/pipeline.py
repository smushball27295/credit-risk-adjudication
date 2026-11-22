"""Fit everything once, in a fixed order, and collect the numbers.

Fitting everything takes about a second, so every CLI command rebuilds from
the cached data with the fixed seed instead of loading a pickled model. There
is no stale artefact to worry about, and the result of a command never
depends on which command ran before it.

Which sample is used for what:

    train (60%)   fit the champion, the benchmark and the comparison models;
                  reference values for reason codes
    valid (20%)   calibration decision; choice of the recommended cutoff
    test  (20%)   every reported number
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from . import SEED, evaluate, fairness, model, monitor, reasons
from . import cutoff as cutoff_mod
from .data import DATASET_NAME, TARGET, data_summary, load_raw, pay_code_evidence
from .features import FEATURE_KEYS, build_features, exposure, protected_features


@dataclass(frozen=True, eq=False)
class Artifacts:
    seed: int
    raw: pd.DataFrame
    X: pd.DataFrame
    y: np.ndarray
    exposure: np.ndarray
    split: model.Split
    champion: xgb.XGBClassifier
    benchmark: object
    platt: tuple[float, float]
    calibration: dict | None
    margin: np.ndarray  # champion output on the logit scale, all rows
    pd: np.ndarray  # champion probability of default, all rows
    pd_benchmark: np.ndarray
    lgd: float
    margin_assumption: float
    cutoff_valid: pd.DataFrame
    cutoff_test: pd.DataFrame
    recommended_cutoff: float
    reference: np.ndarray

    def score(self, X: np.ndarray) -> np.ndarray:
        """Champion probability of default for a float32 feature matrix."""
        frame = pd.DataFrame(np.asarray(X, dtype=np.float32), columns=FEATURE_KEYS)
        return model.apply_calibration(model.margin(self.champion, frame), self.calibration)


@lru_cache(maxsize=4)
def build(
    seed: int = SEED,
    lgd: float = cutoff_mod.DEFAULT_LGD,
    margin: float = cutoff_mod.DEFAULT_MARGIN,
    data_dir: str | None = None,
) -> Artifacts:
    raw = load_raw(data_dir)
    X = build_features(raw)
    y = raw[TARGET].to_numpy()
    ead = exposure(raw)
    split = model.make_split(y, seed)

    champion = model.fit_champion(X.iloc[split.train], y[split.train], seed)
    raw_margin = model.margin(champion, X)
    platt = model.fit_platt(raw_margin[split.valid], y[split.valid])
    calibration = model.decide_calibration(*platt)
    pd_hat = model.apply_calibration(raw_margin, calibration)

    benchmark = model.fit_benchmark(X.iloc[split.train], y[split.train])
    pd_benchmark = benchmark.predict_proba(X.to_numpy())[:, 1]

    cutoff_valid = cutoff_mod.cutoff_table(
        y[split.valid], pd_hat[split.valid], ead[split.valid], lgd, margin
    )
    cutoff_test = cutoff_mod.cutoff_table(
        y[split.test], pd_hat[split.test], ead[split.test], lgd, margin
    )
    recommended = cutoff_mod.best_cutoff(cutoff_valid)

    reference = reasons.reference_values(
        X.iloc[split.train], y[split.train], pd_hat[split.train], recommended
    )

    return Artifacts(
        seed=seed,
        raw=raw,
        X=X,
        y=y,
        exposure=ead,
        split=split,
        champion=champion,
        benchmark=benchmark,
        platt=platt,
        calibration=calibration,
        margin=raw_margin,
        pd=pd_hat,
        pd_benchmark=pd_benchmark,
        lgd=lgd,
        margin_assumption=margin,
        cutoff_valid=cutoff_valid,
        cutoff_test=cutoff_test,
        recommended_cutoff=recommended,
        reference=reference,
    )


def _with_protected_scores(art: Artifacts) -> np.ndarray:
    """Test-set scores of the same model with sex, age, marriage and education added."""
    X_all = pd.concat([art.X, protected_features(art.raw)], axis=1)
    monotone = tuple(model.MONOTONE) + (0,) * (X_all.shape[1] - len(FEATURE_KEYS))
    fitted = model.fit_xgb(X_all.iloc[art.split.train], art.y[art.split.train], art.seed, monotone)
    return fitted.predict_proba(X_all.iloc[art.split.test])[:, 1]


def _unconstrained_scores(art: Artifacts) -> np.ndarray:
    fitted = model.fit_xgb(art.X.iloc[art.split.train], art.y[art.split.train], art.seed, None)
    return fitted.predict_proba(art.X.iloc[art.split.test])[:, 1]


def reason_agreement(art: Artifacts, max_rows: int = 500) -> dict:
    """Agreement with TreeSHAP on a seeded sample of declined test accounts."""
    test = art.split.test
    declined = test[art.pd[test] >= art.recommended_cutoff]
    rng = np.random.default_rng(art.seed)
    if len(declined) > max_rows:
        declined = np.sort(rng.choice(declined, max_rows, replace=False))
    result = reasons.contrib_agreement(
        art.champion.get_booster(),
        art.score,
        art.X.to_numpy()[declined],
        art.reference,
    )
    result["n_declined_in_test"] = int((art.pd[test] >= art.recommended_cutoff).sum())
    return result


def reason_coverage(art: Artifacts) -> dict:
    """How many declined test accounts get reasons, and which reason comes first."""
    test = art.split.test
    declined = test[art.pd[test] >= art.recommended_cutoff]
    drop = reasons.drops(art.score, art.X.to_numpy()[declined], art.reference)
    ranked = [reasons.rank_reasons(row) for row in drop]
    first = [FEATURE_KEYS[r[0]] for r in ranked if r]
    return {
        "n_declined": int(len(declined)),
        "n_without_any_reason": int(sum(len(r) == 0 for r in ranked)),
        "n_with_fewer_than_four": int(sum(len(r) < reasons.TOP_N for r in ranked)),
        "share_by_top_reason": {
            key: first.count(key) / len(declined)
            for key in sorted(set(first), key=lambda k: -first.count(k))
        },
    }


def evaluate_all(art: Artifacts, n_boot: int = 1000) -> dict:
    """Every number the README and the model card quote."""
    tr, va, te = art.split.train, art.split.valid, art.split.test
    y_te = art.y[te]

    champion = evaluate.discrimination(y_te, art.pd[te])
    ci_low, ci_high = evaluate.bootstrap_auc_ci(y_te, art.pd[te], n_boot, art.seed)
    champion["auc_ci_low"], champion["auc_ci_high"] = ci_low, ci_high
    benchmark = evaluate.discrimination(y_te, art.pd_benchmark[te])
    champ_vs_bench = evaluate.bootstrap_auc_diff_ci(
        y_te, art.pd[te], art.pd_benchmark[te], n_boot, art.seed
    )

    auc_train = evaluate.auc(art.y[tr], art.pd[tr])
    auc_valid = evaluate.auc(art.y[va], art.pd[va])

    pd_protected = _with_protected_scores(art)
    protected_diff_ci = evaluate.bootstrap_auc_diff_ci(
        y_te, pd_protected, art.pd[te], n_boot, art.seed
    )
    pd_free = _unconstrained_scores(art)

    deciles = evaluate.decile_table(y_te, art.pd[te])
    raw_pd_test = model.sigmoid(art.margin[te])
    platt_pd_test = model.sigmoid(art.platt[0] * art.margin[te] + art.platt[1])

    recommended = art.recommended_cutoff
    X_test = art.X.iloc[te]
    X_shift = monitor.shifted_copy(X_test, art.seed)

    return {
        "dataset": DATASET_NAME,
        "seed": art.seed,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
        "data": {
            **data_summary(art.raw),
            "pay_code_evidence": pay_code_evidence(art.raw),
        },
        "split": {
            "n_train": int(len(tr)),
            "n_valid": int(len(va)),
            "n_test": int(len(te)),
            "default_rate_train": float(art.y[tr].mean()),
            "default_rate_valid": float(art.y[va].mean()),
            "default_rate_test": float(y_te.mean()),
        },
        "features": FEATURE_KEYS,
        "champion_params": dict(model.CHAMPION_PARAMS),
        "champion": champion,
        "benchmark": benchmark,
        "champion_minus_benchmark_auc": {
            "difference": champion["auc"] - benchmark["auc"],
            "ci_low": champ_vs_bench[0],
            "ci_high": champ_vs_bench[1],
        },
        "auc_train": auc_train,
        "auc_valid": auc_valid,
        "train_test_gap": auc_train - champion["auc"],
        "with_protected": {
            "auc": evaluate.auc(y_te, pd_protected),
            "auc_gain_over_champion": evaluate.auc(y_te, pd_protected) - champion["auc"],
            "gain_ci_low": protected_diff_ci[0],
            "gain_ci_high": protected_diff_ci[1],
        },
        "unconstrained": {
            "auc": evaluate.auc(y_te, pd_free),
            "auc_cost_of_constraints": evaluate.auc(y_te, pd_free) - champion["auc"],
        },
        "deciles": deciles.to_dict(orient="records"),
        "deciles_monotone": evaluate.is_monotone(deciles["default_rate"]),
        # Deciles whose default rate is below the decile before them.
        "decile_inversions": evaluate.inversions(deciles["default_rate"]),
        "calibration": {
            "platt_slope_valid": art.platt[0],
            "platt_intercept_valid": art.platt[1],
            "tolerance": model.CALIBRATION_TOLERANCE,
            "applied": art.calibration,
            "test_as_used": evaluate.calibration_summary(y_te, art.pd[te]),
            "test_raw": evaluate.calibration_summary(y_te, raw_pd_test),
            "test_if_platt_applied": evaluate.calibration_summary(y_te, platt_pd_test),
        },
        "cutoff": {
            "recommended": recommended,
            "chosen_on": "validation",
            "assumptions": {
                "lgd": art.lgd,
                "annual_margin": art.margin_assumption,
                "exposure": "most recent statement balance, floored at zero",
                "break_even_pd": cutoff_mod.break_even_pd(art.lgd, art.margin_assumption),
            },
            "at_recommended_valid": cutoff_mod.row_at(art.cutoff_valid, recommended),
            "at_recommended_test": cutoff_mod.row_at(art.cutoff_test, recommended),
            "best_on_test": cutoff_mod.best_cutoff(art.cutoff_test),
            "sensitivity": cutoff_mod.sensitivity(
                art.y[va],
                art.pd[va],
                art.exposure[va],
                y_te,
                art.pd[te],
                art.exposure[te],
            ),
            "grid_test": art.cutoff_test.to_dict(orient="records"),
        },
        "fairness": {
            **fairness.fairness_report(art.raw.iloc[te], y_te, art.pd[te], recommended),
            "age_gap_probe": fairness.age_gap_probe(
                art.score,
                X_test.to_numpy(),
                art.reference,
                FEATURE_KEYS,
                art.raw.iloc[te],
                y_te,
                recommended,
            ),
            "median_credit_limit_by_age_band": fairness.median_limit_by_age_band(art.raw.iloc[te]),
        },
        "reasons": {
            "reference": dict(zip(FEATURE_KEYS, map(float, art.reference))),
            "agreement_with_pred_contribs": reason_agreement(art),
            "coverage_declined_test": reason_coverage(art),
        },
        "psi": {
            "train_vs_test": monitor.psi_table(art.X.iloc[tr], X_test, art.pd[tr], art.pd[te]),
            "test_vs_shifted": monitor.psi_table(
                X_test, X_shift, art.pd[te], art.score(X_shift.to_numpy())
            ),
        },
    }
