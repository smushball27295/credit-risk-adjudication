"""What the champion is allowed to see, and how it has to behave."""

import numpy as np

from creditadj.data import PROTECTED_COLS
from creditadj.features import FEATURE_KEYS, FEATURES


def test_protected_attributes_are_not_model_inputs(art):
    booster_inputs = art.champion.get_booster().feature_names
    assert booster_inputs == FEATURE_KEYS
    assert list(art.X.columns) == FEATURE_KEYS
    banned = {c.lower() for c in PROTECTED_COLS} | {"gender", "married", "education"}
    for key in booster_inputs:
        assert not any(word in key.lower() for word in banned), key
    # The benchmark is fitted on the same 12 columns.
    assert art.benchmark.named_steps["scale"].n_features_in_ == len(FEATURE_KEYS)
    assert (len(art.split.train), len(art.split.valid), len(art.split.test)) == (
        18_000,
        6_000,
        6_000,
    )


def test_monotonic_constraints_hold_on_a_sweep(art):
    """Move one feature across its range, hold the rest, and check the direction."""
    rng = np.random.default_rng(0)
    base = art.X.to_numpy()[rng.choice(art.split.test, 40, replace=False)]
    for j, spec in enumerate(FEATURES):
        assert spec.monotone in (-1, 1), f"{spec.key} has no constraint"
        grid = np.linspace(spec.min, spec.max, 41, dtype=np.float32)
        swept = np.repeat(base, len(grid), axis=0)
        swept[:, j] = np.tile(grid, len(base))
        scores = art.score(swept).reshape(len(base), len(grid))
        steps = np.diff(scores, axis=1) * spec.monotone
        assert steps.min() >= -1e-9, f"{spec.key} breaks its constraint"
        assert steps.max() > 0, f"{spec.key} never moves the score"
