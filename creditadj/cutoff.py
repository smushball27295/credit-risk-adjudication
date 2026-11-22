"""Where to put the approval cutoff, and what each choice costs.

Decision rule: approve when the predicted probability of default is below
the cutoff. Exposure is the most recent statement balance.

The profit view is illustrative. It rests on two assumptions that are not in
the data and that anyone can change by argument:

    lgd     share of the balance lost on an approved account that defaults
    margin  annual margin earned on the balance of an approved account that
            does not default

    profit = margin * (balance of approved goods) - lgd * (balance of approved bads)

Only the ratio of the two matters for where the maximum falls. A lender that
is paid exactly for the risk it takes breaks even on an account whose default
probability is margin / (margin + lgd).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_LGD = 0.75
DEFAULT_MARGIN = 0.12

CUTOFF_GRID = np.round(np.arange(0.04, 0.801, 0.01), 2)

SENSITIVITY_CASES = [
    (0.75, 0.04),
    (0.75, 0.08),
    (0.75, 0.12),
    (0.75, 0.16),
    (0.75, 0.20),
    (0.50, 0.12),
]


def break_even_pd(lgd: float, margin: float) -> float:
    return margin / (margin + lgd)


def cutoff_table(
    y: np.ndarray,
    pd_hat: np.ndarray,
    exposure: np.ndarray,
    lgd: float = DEFAULT_LGD,
    margin: float = DEFAULT_MARGIN,
    grid: np.ndarray = CUTOFF_GRID,
) -> pd.DataFrame:
    """One row per cutoff: who gets approved, who gets declined, and the profit view."""
    y = np.asarray(y)
    pd_hat = np.asarray(pd_hat, dtype=np.float64)
    exposure = np.asarray(exposure, dtype=np.float64)
    bad = y == 1
    good = ~bad
    bad_balance = exposure[bad].sum()

    rows = []
    for cutoff in grid:
        approved = pd_hat < cutoff
        declined = ~approved
        n_approved = int(approved.sum())
        profit = margin * exposure[approved & good].sum() - lgd * exposure[approved & bad].sum()
        rows.append(
            {
                "cutoff": float(cutoff),
                "approval_rate": float(approved.mean()),
                "default_rate_approved": (
                    float(y[approved].mean()) if n_approved else float("nan")
                ),
                "defaulters_declined": float((declined & bad).sum() / bad.sum()),
                "defaulted_balance_declined": (
                    float(exposure[declined & bad].sum() / bad_balance)
                    if bad_balance > 0
                    else float("nan")
                ),
                "goods_declined": float((declined & good).sum() / good.sum()),
                "illustrative_profit": float(profit),
            }
        )
    table = pd.DataFrame(rows)
    best = table["illustrative_profit"].max()
    # Index 100 is the best cutoff on the grid. Negative values lose money
    # under the stated assumptions. If no cutoff makes money there is no index.
    table["illustrative_profit_index"] = (
        100.0 * table["illustrative_profit"] / best if best > 0 else float("nan")
    )
    table["illustrative_profit_per_account"] = table["illustrative_profit"] / len(y)
    return table


def best_cutoff(table: pd.DataFrame) -> float:
    """Cutoff with the highest illustrative profit (the lowest one on a tie)."""
    return float(table.loc[table["illustrative_profit"].idxmax(), "cutoff"])


def row_at(table: pd.DataFrame, cutoff: float) -> dict:
    """The table row for one cutoff, as a plain dict."""
    match = table.loc[np.isclose(table["cutoff"], cutoff)]
    if match.empty:
        raise KeyError(f"Cutoff {cutoff} is not on the grid.")
    return {k: float(v) for k, v in match.iloc[0].items()}


def sensitivity(
    y_select: np.ndarray,
    pd_select: np.ndarray,
    exposure_select: np.ndarray,
    y_report: np.ndarray,
    pd_report: np.ndarray,
    exposure_report: np.ndarray,
    cases: list[tuple[float, float]] = SENSITIVITY_CASES,
) -> list[dict]:
    """How the profit-maximising cutoff moves when the assumptions move.

    The cutoff is chosen on one sample (validation) and its approval rate and
    capture are read off another (test), the same way the recommended cutoff is.
    """
    rows = []
    for lgd, margin in cases:
        chosen = best_cutoff(cutoff_table(y_select, pd_select, exposure_select, lgd, margin))
        report = row_at(cutoff_table(y_report, pd_report, exposure_report, lgd, margin), chosen)
        rows.append(
            {
                "lgd": lgd,
                "margin": margin,
                "break_even_pd": break_even_pd(lgd, margin),
                "best_cutoff": chosen,
                "approval_rate": report["approval_rate"],
                "defaulters_declined": report["defaulters_declined"],
                "goods_declined": report["goods_declined"],
            }
        )
    return rows
