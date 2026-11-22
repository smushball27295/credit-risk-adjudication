"""Shared fixtures. The fitted pipeline is built once per test session."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def art():
    from creditadj.pipeline import build

    try:
        return build()
    except RuntimeError as exc:  # no network and no cached copy
        pytest.skip(f"Dataset not available: {exc}")


@pytest.fixture(scope="session")
def metrics(art):
    from creditadj.pipeline import evaluate_all

    # 50 bootstrap resamples keep the suite fast; the tests do not check the interval.
    return evaluate_all(art, n_boot=50)
