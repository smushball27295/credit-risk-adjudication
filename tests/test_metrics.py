"""KS, PSI, cutoff table and adverse-impact ratio on inputs with known answers."""

import numpy as np
import pytest

from creditadj.cutoff import best_cutoff, break_even_pd, cutoff_table
from creditadj.evaluate import decile_table, is_monotone, ks_statistic
from creditadj.fairness import adverse_impact_ratio
from creditadj.model import make_split
from creditadj.monitor import psi, psi_status


def test_ks_known_inputs():
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(
        1.0
    )
    assert ks_statistic(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.5, 0.5])) == pytest.approx(
        0.0
    )
    # Alternating good/bad: the two cumulative curves are never more than 1/3 apart.
    assert ks_statistic(np.array([0, 1, 0, 1, 0, 1]), np.arange(6.0)) == pytest.approx(1 / 3)
    # Against scipy on random scores with ties.
    stats = pytest.importorskip("scipy.stats")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 2000)
    score = np.round(rng.normal(size=2000) + 0.8 * y, 2)
    expected = stats.ks_2samp(score[y == 1], score[y == 0]).statistic
    assert ks_statistic(y, score) == pytest.approx(expected, abs=1e-12)


def test_psi_known_inputs():
    expected = np.array([0] * 50 + [1] * 50)
    assert psi(expected, expected) == pytest.approx(0.0, abs=1e-12)
    actual = np.array([0] * 80 + [1] * 20)
    by_hand = (0.8 - 0.5) * np.log(0.8 / 0.5) + (0.2 - 0.5) * np.log(0.2 / 0.5)
    assert psi(expected, actual) == pytest.approx(by_hand, abs=1e-9)
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "investigate"
    assert psi_status(0.30) == "review"


def test_cutoff_table_by_hand():
    y = np.array([0, 0, 1, 1])
    pd_hat = np.array([0.1, 0.3, 0.2, 0.6])
    balance = np.array([100.0, 200.0, 300.0, 400.0])
    table = cutoff_table(y, pd_hat, balance, lgd=0.75, margin=0.12, grid=np.array([0.25, 0.5]))
    tight, loose = table.iloc[0], table.iloc[1]
    # Cutoff 0.25 approves the first good (0.1) and the first bad (0.2).
    assert tight["approval_rate"] == pytest.approx(0.5)
    assert tight["default_rate_approved"] == pytest.approx(0.5)
    assert tight["defaulters_declined"] == pytest.approx(0.5)
    assert tight["defaulted_balance_declined"] == pytest.approx(400 / 700)
    assert tight["goods_declined"] == pytest.approx(0.5)
    assert tight["illustrative_profit"] == pytest.approx(0.12 * 100 - 0.75 * 300)
    assert loose["goods_declined"] == pytest.approx(0.0)
    assert loose["illustrative_profit"] == pytest.approx(0.12 * 300 - 0.75 * 300)
    assert best_cutoff(table) == pytest.approx(0.5)
    assert break_even_pd(0.75, 0.12) == pytest.approx(0.12 / 0.87)


def test_deciles_split_and_adverse_impact():
    rng = np.random.default_rng(0)
    score = rng.random(1000)
    y = (rng.random(1000) < score).astype(int)
    table = decile_table(y, score)
    assert table["n"].tolist() == [100] * 10
    assert table["defaults"].sum() == y.sum()
    assert is_monotone(table["avg_score"])
    assert not is_monotone([0.1, 0.3, 0.2])

    split = make_split(y, seed=42)
    parts = [split.train, split.valid, split.test]
    assert [len(p) for p in parts] == [600, 200, 200]
    assert len(np.unique(np.concatenate(parts))) == 1000
    rates = [y[p].mean() for p in parts]
    assert max(rates) - min(rates) < 0.01  # stratified

    assert adverse_impact_ratio(
        [{"approval_rate": 0.40}, {"approval_rate": 0.50}]
    ) == pytest.approx(0.8)
