"""Write the champion model and its supporting numbers to one JSON file.

The file is everything a browser needs to score an applicant, show the
cutoff trade-off and produce reason codes, with no server.

Scoring from the file:

    1. Round each input to float32 (Math.fround in JavaScript). XGBoost
       stores inputs and thresholds as float32, so this is what makes the
       tree paths identical.
    2. For each tree start at node 0. At a split node go to "y" if x < t,
       to "m" if x is missing, otherwise to "n". Stop at a leaf {"v": value}.
    3. margin = base_score_logit + sum of the leaf values.
    4. If "calibration" is not null: margin = a * margin + b.
    5. probability of default = 1 / (1 + exp(-margin)).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

from . import model
from .data import DATASET_NAME
from .features import FEATURE_KEYS, FEATURES
from .pipeline import Artifacts
from .reasons import REASON_TEXT

MAX_BYTES = 600_000


def booster_to_trees(booster: xgb.Booster) -> list[list[dict]]:
    """Flatten every tree into a list of nodes, node 0 being the root."""
    dump = json.loads(booster.save_raw("json"))
    trees = []
    for tree in dump["learner"]["gradient_booster"]["model"]["trees"]:
        left = tree["left_children"]
        right = tree["right_children"]
        nodes = []
        for i in range(len(left)):
            # The model file prints float32 values. Passing them through
            # float32 again gives the exact number XGBoost compares against.
            value = float(np.float32(tree["split_conditions"][i]))
            if left[i] == -1:
                nodes.append({"v": value})
            else:
                nodes.append(
                    {
                        "f": int(tree["split_indices"][i]),
                        "t": value,
                        "y": int(left[i]),
                        "n": int(right[i]),
                        "m": int(left[i] if tree["default_left"][i] else right[i]),
                    }
                )
        trees.append(nodes)
    return trees


def leaf_sum(trees: list[list[dict]], X: np.ndarray) -> np.ndarray:
    """Sum of leaf values for each row of a float32 matrix (vectorised walk)."""
    X = np.asarray(X, dtype=np.float32)
    total = np.zeros(len(X), dtype=np.float64)
    rows = np.arange(len(X))
    for nodes in trees:
        is_leaf = np.array(["v" in n for n in nodes])
        feat = np.array([n.get("f", 0) for n in nodes])
        thr = np.array([n.get("t", 0.0) for n in nodes], dtype=np.float32)
        yes = np.array([n.get("y", 0) for n in nodes])
        no = np.array([n.get("n", 0) for n in nodes])
        miss = np.array([n.get("m", 0) for n in nodes])
        value = np.array([n.get("v", 0.0) for n in nodes])
        at = np.zeros(len(X), dtype=np.int64)
        while not is_leaf[at].all():
            x = X[rows, feat[at]]
            step = np.where(np.isnan(x), miss[at], np.where(x < thr[at], yes[at], no[at]))
            at = np.where(is_leaf[at], at, step)
        total += value[at]
    return total


def score_model_block(model_block: dict, calibration: dict | None, X: np.ndarray) -> np.ndarray:
    """Probability of default computed from the exported JSON alone."""
    raw = model_block["base_score_logit"] + leaf_sum(model_block["trees"], X)
    return model.apply_calibration(raw, calibration)


def _base_score_logit(art: Artifacts, trees: list[list[dict]]) -> float:
    """Recover the intercept as (XGBoost margin - sum of leaves) and check it is constant."""
    sample = art.X.to_numpy()[art.split.test[:200]]
    gap = model.margin(art.champion, art.X.iloc[art.split.test[:200]]) - leaf_sum(trees, sample)
    base = float(np.median(gap))
    if np.max(np.abs(gap - base)) > 1e-5:
        raise RuntimeError("Exported trees do not reproduce the XGBoost margin.")
    return base


def _clean_value(spec, value: float) -> float:
    if spec.unit == "ratio":
        return round(float(value), 4)
    return int(round(float(value)))


def _describe(values: dict) -> str:
    """One sentence about a preset, written from its actual values."""
    late = int(values["months_late_6m"])
    if late == 0:
        history = "no late payment in six months"
    elif values["recent_status"] > 0:
        history = (
            f"{int(values['recent_status'])} month(s) past due now, "
            f"late in {late} of the last six months"
        )
    else:
        history = f"up to date now, late in {late} of the last six months"
    return (
        f"Limit NT${values['credit_limit']:,.0f}; {history}; "
        f"utilisation {values['util_current']:.0%}; "
        f"repaid {values['pay_ratio']:.0%} of the amounts billed."
    )


def pick_presets(art: Artifacts) -> list[dict]:
    """Five test-set accounts, from clearly safe to clearly risky.

    Each one is picked by a rule (a risk percentile, or the median-risk
    account of a simple profile), so the choice is reproducible and not
    hand-picked. Values are within the slider ranges. No identifiers are exported.
    """
    test = art.split.test
    X = art.X.iloc[test]
    pd_test = art.pd[test]
    in_range = np.ones(len(X), dtype=bool)
    for spec in FEATURES:
        in_range &= (X[spec.key] >= spec.min).to_numpy() & (X[spec.key] <= spec.max).to_numpy()
    approved = pd_test < art.recommended_cutoff

    def nearest(target: float) -> int:
        candidates = np.flatnonzero(in_range)
        return int(candidates[np.argmin(np.abs(pd_test[candidates] - target))])

    def median_of(mask: np.ndarray, name: str) -> int:
        candidates = np.flatnonzero(mask & in_range)
        if len(candidates) == 0:
            raise RuntimeError(f"No test account matches the profile '{name}'.")
        ranked = candidates[np.argsort(pd_test[candidates], kind="mergesort")]
        return int(ranked[len(ranked) // 2])

    two_behind = (X["recent_status"] == 2).to_numpy() & (X["months_late_6m"] <= 2).to_numpy()
    long_behind = (X["recent_status"] >= 2).to_numpy() & (X["months_late_6m"] >= 5).to_numpy()
    chosen = [
        ("Among the safest 5% of accounts", nearest(float(np.quantile(pd_test, 0.025)))),
        ("Typical approved account", median_of(approved, "approved")),
        ("Close to the cutoff", nearest(art.recommended_cutoff)),
        ("Two months past due", median_of(two_behind, "two months past due")),
        ("Behind for most of six months", median_of(long_behind, "behind for most of six months")),
    ]

    presets = []
    for name, pos in chosen:
        values = {spec.key: _clean_value(spec, X.iloc[pos][spec.key]) for spec in FEATURES}
        vector = np.array([[values[k] for k in FEATURE_KEYS]], dtype=np.float32)
        presets.append(
            {
                "name": name,
                "note": _describe(values),
                "values": values,
                "expected_probability": round(float(art.score(vector)[0]), 8),
            }
        )
    probabilities = [p["expected_probability"] for p in presets]
    if probabilities != sorted(probabilities):
        raise RuntimeError("Presets are not ordered from safe to risky.")
    return presets


def _r(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def build_payload(art: Artifacts, metrics: dict) -> dict:
    """Assemble the export in the shape the web demo expects."""
    trees = booster_to_trees(art.champion.get_booster())
    train = art.X.iloc[art.split.train]

    features = []
    for j, spec in enumerate(FEATURES):
        median = np.quantile(train[spec.key].to_numpy(), 0.5, method="lower")
        features.append(
            {
                "key": spec.key,
                "label": spec.label,
                "help": spec.help,
                "unit": spec.unit,
                "min": spec.min,
                "max": spec.max,
                "step": spec.step,
                "median": _clean_value(spec, median),
                "reference": _clean_value(spec, art.reference[j]),
                "monotone": spec.monotone,
            }
        )

    grid = [
        {
            "cutoff": _r(row["cutoff"], 2),
            "approval_rate": _r(row["approval_rate"]),
            "default_rate_approved": _r(row["default_rate_approved"]),
            "defaulters_declined": _r(row["defaulters_declined"]),
            "defaulted_balance_declined": _r(row["defaulted_balance_declined"]),
            "goods_declined": _r(row["goods_declined"]),
            "illustrative_profit_index": _r(row["illustrative_profit_index"], 3),
        }
        for row in metrics["cutoff"]["grid_test"]
        if row["approval_rate"] > 0
    ]

    fair = metrics["fairness"]

    def slim(rows: list[dict]) -> list[dict]:
        return [
            {
                "group": r["group"],
                "approval_rate": _r(r["approval_rate"]),
                "default_rate": _r(r["default_rate"]),
            }
            for r in rows
        ]

    assumptions = dict(metrics["cutoff"]["assumptions"])
    assumptions["note"] = (
        "Illustrative only. lgd and annual_margin are assumptions, not estimates "
        "from the data. Index 100 is the best cutoff on the test-set grid; the "
        "recommended cutoff was chosen on the validation set."
    )

    return {
        "meta": {
            "dataset": DATASET_NAME,
            "n_train": int(len(art.split.train)),
            "n_test": int(len(art.split.test)),
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "seed": art.seed,
        },
        "features": features,
        "model": {
            "base_score_logit": _base_score_logit(art, trees),
            "trees": trees,
        },
        "calibration": art.calibration,
        "cutoff": {
            "recommended": _r(art.recommended_cutoff, 2),
            "assumptions": assumptions,
            "grid": grid,
        },
        "metrics": {
            "champion": {
                k: _r(metrics["champion"][k])
                for k in ("auc", "gini", "ks", "auc_ci_low", "auc_ci_high")
            },
            "benchmark": {k: _r(metrics["benchmark"][k]) for k in ("auc", "gini", "ks")},
            "with_protected": {"auc": _r(metrics["with_protected"]["auc"])},
            "train_test_gap": _r(metrics["train_test_gap"]),
        },
        "deciles": [
            {
                "decile": int(d["decile"]),
                "n": int(d["n"]),
                "default_rate": _r(d["default_rate"]),
                "avg_score": _r(d["avg_score"]),
            }
            for d in metrics["deciles"]
        ],
        "fairness": {
            "cutoff": _r(fair["cutoff"], 2),
            "by_sex": slim(fair["by_sex"]),
            "by_age_band": slim(fair["by_age_band"]),
            "adverse_impact_ratio_sex": _r(fair["adverse_impact_ratio_sex"]),
        },
        "reasons": dict(REASON_TEXT),
        "presets": pick_presets(art),
    }


def write_payload(payload: dict, path: Path | str) -> int:
    """Write compact JSON and return its size in bytes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise RuntimeError(f"Export is {len(text)} bytes, over the {MAX_BYTES} limit.")
    path.write_text(text + "\n", encoding="utf-8")
    return path.stat().st_size


def build_validation_payload(seed: int | None = None) -> dict:
    """Everything the web demo needs to run the validation agent.

    For each submission: the developer's summary, the full evidence pack that
    tool calls are served from, the rules baseline's memo, and the answer key.
    The answer key is shown to the visitor after a run. It is never sent to
    the language model.
    """
    from . import SEED
    from .candidates import build_candidates
    from .validation import FAIRNESS_ATTRIBUTES, TESTS, evidence_pack
    from .validator import MAX_TOOL_CALLS, SYSTEM_PROMPT, TOOLS, check_memo, rules_memo, score_memo

    submissions = []
    for c in build_candidates(SEED if seed is None else seed):
        pack = evidence_pack(c)
        memo = rules_memo(pack)
        seen = {i["id"]: i["text"] for result in pack.values() for i in result["items"]}
        submissions.append(
            {
                "id": c.id,
                "label": c.label,
                "developer_summary": c.developer_summary,
                "inputs": list(c.columns),
                "pack": pack,
                "rules": {"memo": memo, "check": check_memo(memo, seen).to_dict(), "score": score_memo(memo, c.planted)},
                "planted": {
                    "key": c.planted.key,
                    "title": c.planted.title,
                    "explanation": c.planted.explanation,
                    "evidence_prefixes": list(c.planted.evidence_prefixes),
                },
            }
        )
    return {
        "system_prompt": SYSTEM_PROMPT,
        "max_tool_calls": MAX_TOOL_CALLS,
        "tools": TOOLS,
        "tests": [{"name": name, "description": desc, "argument": arg} for name, (desc, arg) in TESTS.items()],
        "fairness_attributes": list(FAIRNESS_ATTRIBUTES),
        "submissions": submissions,
    }


def write_validation_payload(payload: dict, path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return path.stat().st_size
