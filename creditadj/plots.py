"""The four charts written to reports/. Static PNGs, one message each."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_MUTED, labelsize=9, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)


def _titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.02, 0.965, title, fontsize=13, fontweight="bold", color=INK, va="top")
    fig.text(0.02, 0.915, subtitle, fontsize=9.5, color=INK_SECONDARY, va="top")


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def _legend(ax, **kwargs) -> None:
    legend = ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, **kwargs)
    for line in legend.get_lines():
        line.set_linewidth(2)


def plot_roc(y, pd_champion, pd_benchmark, auc_champion, auc_benchmark, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    for scores, colour, label in [
        (pd_benchmark, ORANGE, f"Logistic regression benchmark, AUC {auc_benchmark:.3f}"),
        (pd_champion, BLUE, f"XGBoost champion, AUC {auc_champion:.3f}"),
    ]:
        fpr, tpr, _ = roc_curve(y, scores)
        ax.plot(fpr, tpr, color=colour, linewidth=2, label=label, solid_capstyle="round")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Share of good accounts flagged (false positive rate)")
    ax.set_ylabel("Share of defaulters flagged (true positive rate)")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles[::-1],
        labels[::-1],
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        loc="lower right",
    )
    _titles(fig, "ROC curve on the test set", "Grey diagonal is a model with no information.")
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.11)
    _save(fig, path)


def _rounded_bar(ax, x: float, height: float, width: float, colour: str) -> None:
    """A bar that is square at the baseline and rounded at the data end."""
    fig = ax.figure
    fig.canvas.draw()
    to_px = ax.transData.transform
    px_per_x = abs(to_px((1, 0))[0] - to_px((0, 0))[0])
    px_per_y = abs(to_px((0, 1))[1] - to_px((0, 0))[1])
    radius_px = 4 * fig.dpi / 72 * 0.75
    rx = min(radius_px / px_per_x, width / 2)
    ry = min(radius_px / px_per_y, height)
    left, right = x - width / 2, x + width / 2
    verts = [
        (left, 0),
        (left, height - ry),
        (left, height),
        (left + rx, height),
        (right - rx, height),
        (right, height),
        (right, height - ry),
        (right, 0),
        (left, 0),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CURVE3,
        MplPath.CURVE3,
        MplPath.LINETO,
        MplPath.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=colour, edgecolor="none"))


def plot_deciles(deciles: list[dict], overall_rate: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(False, axis="x")
    rates = [d["default_rate"] for d in deciles]
    xs = [d["decile"] for d in deciles]
    ax.set_xlim(0.4, len(xs) + 0.6)
    ax.set_ylim(0, max(rates) * 1.15)
    ax.set_xticks(xs)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Score decile (1 = lowest predicted risk)")
    ax.set_ylabel("Observed default rate")
    fig.subplots_adjust(left=0.11, right=0.97, top=0.83, bottom=0.13)
    for x, rate in zip(xs, rates):
        _rounded_bar(ax, x, rate, 0.42, BLUE)
    for x, rate in ((xs[0], rates[0]), (xs[-1], rates[-1])):
        ax.text(
            x,
            rate + max(rates) * 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.axhline(overall_rate, color=INK_MUTED, linewidth=1)
    ax.text(
        0.5,
        overall_rate + max(rates) * 0.015,
        f"All test accounts {overall_rate:.1%}",
        fontsize=9,
        color=INK_SECONDARY,
        va="bottom",
    )
    _titles(
        fig,
        "Default rate by score decile, test set",
        "Each bar is one tenth of test accounts, ordered by predicted risk.",
    )
    _save(fig, path)


def plot_calibration(deciles: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    predicted = [d["avg_score"] for d in deciles]
    observed = [d["default_rate"] for d in deciles]
    top = max(max(predicted), max(observed)) * 1.08
    ax.plot([0, top], [0, top], color=AXIS, linewidth=1)
    ax.plot(predicted, observed, color=BLUE, linewidth=2, solid_capstyle="round")
    ax.scatter(predicted, observed, s=52, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xlim(0, top)
    ax.set_ylim(0, top)
    ax.set_xlabel("Mean predicted probability of default in the decile")
    ax.set_ylabel("Observed default rate in the decile")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.text(
        top * 0.62,
        top * 0.66,
        "predicted = observed",
        fontsize=9,
        color=INK_MUTED,
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        transform_rotates_text=True,
    )
    _titles(
        fig,
        "Calibration on the test set",
        "One dot per score decile. Dots on the grey line are perfectly calibrated.",
    )
    fig.subplots_adjust(left=0.13, right=0.96, top=0.86, bottom=0.11)
    _save(fig, path)


def plot_cutoff(grid: list[dict], recommended: float, path: Path) -> None:
    """Two panels on one x-axis: the rates, then the illustrative profit index."""
    grid = [g for g in grid if g["approval_rate"] > 0]
    cut = np.array([g["cutoff"] for g in grid])
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(7.6, 7.4), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    fig.patch.set_facecolor(SURFACE)
    series = [
        ("defaulters_declined", "Defaulters declined", ORANGE),
        ("approval_rate", "Approval rate", BLUE),
        ("goods_declined", "Good accounts declined", AQUA),
        ("default_rate_approved", "Default rate among approved", YELLOW),
    ]
    _style(top)
    at = min(range(len(cut)), key=lambda i: abs(cut[i] - recommended))
    for key, label, colour in series:
        values = np.array([g[key] for g in grid])
        # The value at the recommended cutoff goes in the legend: two of the
        # four lines cross near it, so labels on the dots would collide.
        top.plot(
            cut,
            values,
            color=colour,
            linewidth=2,
            solid_capstyle="round",
            label=f"{label}: {values[at]:.0%}",
        )
        top.scatter(
            [cut[at]], [values[at]], s=52, color=colour, edgecolor=SURFACE, linewidth=2, zorder=3
        )
    top.axvline(recommended, color=INK_MUTED, linewidth=1)
    top.set_ylim(0, 1.0)
    top.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    top.set_ylabel("Share")
    _legend(
        top,
        loc="center right",
        title=f"At the recommended cutoff of {recommended:.2f}",
        title_fontsize=9,
        alignment="left",
    )

    _style(bottom)
    index = np.array([g["illustrative_profit_index"] for g in grid])
    bottom.axhline(0, color=AXIS, linewidth=1)
    bottom.plot(cut, index, color=BLUE, linewidth=2, solid_capstyle="round")
    bottom.scatter(
        [cut[at]], [index[at]], s=52, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3
    )
    bottom.axvline(recommended, color=INK_MUTED, linewidth=1)
    bottom.text(
        cut[at] + 0.012,
        index[at],
        f"Recommended cutoff {recommended:.2f}",
        fontsize=9,
        color=INK,
        va="bottom",
        ha="left",
    )
    bottom.set_ylabel("Illustrative profit index\n(100 = best cutoff)")
    bottom.set_xlabel("Cutoff: approve when predicted probability of default is below this")
    bottom.set_xlim(cut.min(), cut.max())

    _titles(
        fig,
        "What each cutoff costs, test set",
        "Top: who is approved and declined. Bottom: profit view under stated assumptions.",
    )
    fig.subplots_adjust(left=0.12, right=0.96, top=0.87, bottom=0.08, hspace=0.12)
    _save(fig, path)
