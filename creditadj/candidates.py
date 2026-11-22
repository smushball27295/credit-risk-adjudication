"""Model submissions for the validation agent to review.

One is the real champion. The others are the same kind of model with one
defect planted on purpose: the sort of thing an independent validator is
there to catch. Each submission carries the developer's own summary, which
never mentions the defect, and a hidden answer key that says what was planted
and which pieces of evidence reveal it.

Nothing here is simulated output. Every submission is a model that is really
fitted on the UCI data, and every number the validation tests report about it
is computed from that fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xgboost as xgb

from . import SEED, model
from . import cutoff as cutoff_mod
from .features import FEATURE_KEYS, FEATURES, MONOTONE, exposure, protected_features
from .pipeline import Artifacts, build

# What the data dictionary says about each field. The validation tests quote
# these lines as they are. They describe the field and never say whether it
# is a problem: working that out is the validator's job.
DICTIONARY: dict[str, str] = {spec.key: f"{spec.label}. {spec.help}" for spec in FEATURES}
DICTIONARY.update(
    {
        "age": "Customer age in years on the observation date. Demographic field from the application record.",
        "sex_female": "1 if the customer is recorded as female, else 0. Demographic field from the application record.",
        "married": "1 if marital status is recorded as married, else 0. Demographic field from the application record.",
        "single": "1 if marital status is recorded as single, else 0. Demographic field from the application record.",
        "edu_graduate": "1 if the highest education level is graduate school, else 0. Demographic field.",
        "edu_university": "1 if the highest education level is university, else 0. Demographic field.",
        "edu_high_school": "1 if the highest education level is high school, else 0. Demographic field.",
        "collections_flag": (
            "1 if the account was referred to the collections team, else 0. Written by the collections "
            "system during the month after the observation window closes, which is the month the default "
            "outcome is measured in."
        ),
    }
)


@dataclass(frozen=True)
class Planted:
    """The hidden answer key for one submission."""

    key: str  # "none" for the real champion
    title: str
    explanation: str
    # A finding counts as catching the defect if it cites an evidence id that
    # starts with one of these prefixes.
    evidence_prefixes: tuple[str, ...] = ()


@dataclass(frozen=True, eq=False)
class Candidate:
    id: str  # neutral, so the name gives nothing away
    label: str
    developer_summary: str
    booster: xgb.XGBClassifier
    columns: list[str]
    X: pd.DataFrame  # every row, the columns this model uses
    y: np.ndarray
    raw: pd.DataFrame
    exposure: np.ndarray
    train: np.ndarray
    valid: np.ndarray
    test: np.ndarray  # the sample the developer calls "test"
    X_recent: pd.DataFrame  # most recent scored population, for the stability test
    planted: Planted
    params: dict = field(default_factory=dict)

    def score(self, X: pd.DataFrame) -> np.ndarray:
        return model.sigmoid(model.margin(self.booster, X[self.columns]))


def _fit(X: pd.DataFrame, y: np.ndarray, seed: int, monotone, **params) -> xgb.XGBClassifier:
    settings = {**model.CHAMPION_PARAMS, **params}
    booster = xgb.XGBClassifier(
        **settings,
        random_state=seed,
        base_score=round(float(np.mean(y)), 4),
        monotone_constraints=monotone,
    )
    booster.fit(X, y)
    return booster


def _stressed(X: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A recent population that has clearly moved: a made-up downturn, not a forecast.

    Balances are half as high again, customers repay half as much of what they
    are billed, and one account in four that was up to date has missed its
    latest payment.
    """
    rng = np.random.default_rng(seed)
    out = X.copy()
    out["util_current"] = np.clip(out["util_current"] * 1.5, 0.0, 1.5)
    out["util_avg_6m"] = np.clip(out["util_avg_6m"] * 1.5, 0.0, 1.5)
    out["pay_ratio"] = out["pay_ratio"] * 0.5
    current = (out["recent_status"] == 0).to_numpy()
    late = current & (rng.random(len(out)) < 0.25)
    out.loc[late, "recent_status"] = 1
    out.loc[late, "worst_status_6m"] = np.maximum(out.loc[late, "worst_status_6m"], 1)
    out.loc[late, "months_late_6m"] = np.minimum(out.loc[late, "months_late_6m"] + 1, 6)
    out.loc[late, "months_since_late"] = 0
    return out.round(4).astype(np.float32)


def _summary(n_inputs: int, extra: str = "") -> str:
    text = (
        f"Gradient-boosted default model for credit card accounts with {n_inputs} inputs, "
        "trained on 18,000 accounts with 6,000 held out for validation and 6,000 for testing. "
        "Target: default on the next payment."
    )
    return f"{text} {extra}".strip()


def build_candidates(seed: int = SEED, data_dir: str | None = None) -> list[Candidate]:
    """The eight submissions, in a fixed order."""
    art: Artifacts = build(seed=seed, data_dir=data_dir)
    X, y, raw, split = art.X, art.y, art.raw, art.split
    ead = exposure(raw)
    rng = np.random.default_rng(seed)
    stable_recent = X.iloc[split.valid]

    def make(cid, label, summary, booster, frame, planted, test=split.test, recent=None, **params):
        return Candidate(
            id=cid,
            label=label,
            developer_summary=summary,
            booster=booster,
            columns=list(frame.columns),
            X=frame,
            y=y,
            raw=raw,
            exposure=ead,
            train=split.train,
            valid=split.valid,
            test=test,
            X_recent=recent if recent is not None else frame.iloc[split.valid],
            planted=planted,
            params=params,
        )

    out: list[Candidate] = []

    # A. The real champion.
    out.append(
        make(
            "A",
            "Submission A",
            _summary(12, "All inputs carry monotonic constraints."),
            art.champion,
            X,
            Planted("none", "Nothing planted", "This is the real champion model, exactly as documented in the README."),
        )
    )

    # B. Target leakage: an input that is only known after the outcome.
    flag = np.where(y == 1, rng.random(len(y)) < 0.75, rng.random(len(y)) < 0.03).astype(np.float32)
    X_leak = X.assign(collections_flag=flag)
    out.append(
        make(
            "B",
            "Submission B",
            _summary(13, "Adds a collections indicator to the standard inputs, which lifted performance sharply."),
            _fit(X_leak.iloc[split.train], y[split.train], seed, None),
            X_leak,
            Planted(
                "leakage",
                "Target leakage",
                "The collections flag is written in the month the outcome is measured, so it is not available "
                "when the decision is made. The model is reading the answer.",
                ("INP-collections_flag", "FEAT-collections_flag"),
            ),
        )
    )

    # C. Overfitting: far too much capacity for 18,000 rows.
    out.append(
        make(
            "C",
            "Submission C",
            _summary(12, "A larger configuration was chosen to capture more interactions."),
            _fit(
                X.iloc[split.train], y[split.train], seed, None,
                n_estimators=800, max_depth=9, learning_rate=0.3, min_child_weight=0,
            ),
            X,
            Planted(
                "overfit",
                "Overfitting",
                "800 trees of depth 9 memorise the training sample. Training performance is far above what the "
                "model achieves on accounts it has not seen.",
                ("PERF-GAP", "PERF-TRAIN"),
            ),
            n_estimators=800,
            max_depth=9,
        )
    )

    # D. Miscalibration: defaults were up-weighted in training and never corrected.
    out.append(
        make(
            "D",
            "Submission D",
            _summary(12, "Defaults were up-weighted during training to sharpen separation."),
            _fit(X.iloc[split.train], y[split.train], seed, MONOTONE, scale_pos_weight=4.0),
            X,
            Planted(
                "miscalibration",
                "Miscalibration",
                "Up-weighting defaults roughly doubles the predicted probabilities. Ranking is unaffected, but "
                "any cutoff or pricing that reads the score as a probability is wrong.",
                ("CAL-",),
            ),
        )
    )

    # E. Prohibited inputs: demographic fields used directly.
    X_demo = pd.concat([X, protected_features(raw)], axis=1)
    out.append(
        make(
            "E",
            "Submission E",
            _summary(19, "Adds customer profile fields to the standard inputs."),
            _fit(X_demo.iloc[split.train], y[split.train], seed, None),
            X_demo,
            Planted(
                "prohibited_inputs",
                "Prohibited inputs",
                "Age, sex and marital status are inputs to the model. They are prohibited bases for a credit "
                "decision, and education can stand in for them.",
                (
                    "INP-age", "INP-sex_female", "INP-married", "INP-single", "INP-edu_",
                    "FEAT-age", "FEAT-sex_female", "FEAT-married", "FEAT-single", "FEAT-edu_",
                    # Added after the first measured runs: the agent cited the input list and the
                    # fairness test's line on demographic inputs. Both do reveal the defect.
                    "INP-LIST", "FAIR-sex-INPUTS", "FAIR-age_band-INPUTS", "FAIR-marital_status-INPUTS",
                ),
            ),
        )
    )

    # F. Population drift: a sound model, but the book it now scores has moved.
    out.append(
        make(
            "F",
            "Submission F",
            _summary(12, "All inputs carry monotonic constraints. Resubmitted for annual review."),
            art.champion,
            X,
            Planted(
                "drift",
                "Population drift",
                "The model is the champion, but the most recent population has higher balances, lower repayment "
                "and more late payers than the development sample, so its numbers no longer describe the book.",
                ("STAB-",),
            ),
            recent=_stressed(stable_recent, seed),
        )
    )

    # G. Weak model: does worse than a plain logistic regression.
    weak_cols = ["credit_limit", "last_payment", "months_zero_balance"]
    X_weak = X[weak_cols]
    out.append(
        make(
            "G",
            "Submission G",
            _summary(3, "A deliberately small model for ease of explanation."),
            _fit(X_weak.iloc[split.train], y[split.train], seed, None, n_estimators=20, max_depth=1),
            X_weak,
            Planted(
                "underperforms",
                "Underperforms the benchmark",
                "Three inputs and twenty stumps leave out repayment status. A plain logistic regression on the "
                "standard inputs ranks accounts much better.",
                ("BENCH-",),
            ),
        )
    )

    # H. Contaminated test sample: half of the "test" rows were used in training.
    half = len(split.test) // 2
    mixed = np.sort(np.r_[rng.choice(split.train, half, replace=False), rng.choice(split.test, half, replace=False)])
    out.append(
        make(
            "H",
            "Submission H",
            _summary(12, "All inputs carry monotonic constraints. Test results are slightly better than validation."),
            art.champion,
            X,
            Planted(
                "test_contamination",
                "Test sample overlaps training",
                "Half of the rows in the sample the developer calls the test set were also used to train the "
                "model, so its test results are not independent evidence.",
                ("DATA-OVERLAP",),
            ),
            test=mixed,
        )
    )
    return out


def recommended_cutoff(candidate: Candidate) -> float:
    """Chosen on the validation sample with the project's illustrative economics."""
    table = cutoff_mod.cutoff_table(
        candidate.y[candidate.valid],
        candidate.score(candidate.X.iloc[candidate.valid]),
        candidate.exposure[candidate.valid],
    )
    return cutoff_mod.best_cutoff(table)
