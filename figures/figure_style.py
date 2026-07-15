"""Shared publication style and output helpers for Figures 3-7."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from . import config


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data"
FINAL_SOURCE_DIR = SOURCE_DIR / "generated"
FINAL_OUTPUT_DIR = ROOT / "outputs" / "main_figures"

INK = "#172033"
TEXT_MUTED = "#687385"
GRID = "#E7EAF0"
BORDER = "#D6DCE5"
TRACK_BG = "#E9EDF2"
NA_COLOR = "#F1F3F5"

FAMILY_COLORS = {
    "factor": "#3B6FB6",
    "deep": "#E28A3A",
    "graph": "#259C8B",
    "relational": "#A65AA5",
}

DATASET_COLORS = {
    "pbmc3k": "#4C78A8",
    "paul15": "#59A14F",
    "heart_cell_atlas_subsampled": "#A779A7",
}

PASS_COLOR = "#238B68"
FAIL_COLOR = "#D1495B"
BOTTLENECK_GOLD = "#D9A441"
THRESHOLD_COLOR = "#66707C"

METHOD_ORDER = config.ANCHOR_METHODS
FAMILY_ORDER = config.FAMILY_ORDER
FAMILY_LABELS = config.FAMILY_LABELS

DATASET_LABELS = {
    "pbmc3k": "PBMC3k",
    "paul15": "Paul15",
    "heart_cell_atlas_subsampled": "Heart atlas",
}

METRIC_LABELS = {
    "local_retention": "local",
    "trustworthiness": "trust",
    "global_rank_corr": "global",
    "label_neighbor_recall": "label",
    "latent_distance_corr": "latent",
    "pseudotime_rank_corr": "pseudo rank",
    "pseudotime_neighborhood_retention": "pseudo local",
    "cell_type_label_recall": "cell label",
    "donor_entropy_norm": "donor entropy",
}

GATE_LABELS = {
    "label_support_gate": "label",
    "local_neighbourhood_gate": "local",
    "global_geometry_gate": "global",
    "continuum_gate": "continuum",
    "donor_aware_gate": "donor",
}

GATE_ORDER = [
    "label_support_gate",
    "local_neighbourhood_gate",
    "global_geometry_gate",
    "continuum_gate",
    "donor_aware_gate",
]


def ensure_final_dirs() -> None:
    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_SOURCE_DIR.mkdir(parents=True, exist_ok=True)


def apply_final_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.labelsize": 7.1,
            "axes.titlesize": 8.2,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 6.2,
            "ytick.labelsize": 6.2,
            "legend.fontsize": 6.1,
            "axes.linewidth": 0.65,
            "lines.linewidth": 1.0,
            "patch.linewidth": 0.65,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "axes.unicode_minus": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
        }
    )


def dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(str(dataset_id), str(dataset_id))


def family_label(family: str) -> str:
    return FAMILY_LABELS.get(str(family), str(family))


def method_family(method: str) -> str:
    return config.METHOD_FAMILY.get(str(method), "factor")


def family_color_for_method(method: str) -> str:
    return FAMILY_COLORS[method_family(method)]


def short_label(name: str) -> str:
    replacements = {
        "Atrial_Cardiomyocyte": "Atrial CM",
        "Ventricular_Cardiomyocyte": "Ventricular CM",
        "Smooth_muscle_cells": "Smooth muscle",
        "Dendritic cells": "DC",
        "Megakaryocytes": "Mega",
        "CD14+ Monocytes": "CD14 Mono",
        "FCGR3A+ Monocytes": "FCGR3A Mono",
        "T cells (CD4/CD8)": "T cell",
        "myeloid_monocyte": "myeloid/mono",
        "basophil_mast": "basophil/mast",
        "megakaryocyte": "mega",
        "smooth_muscle": "smooth muscle",
        "cardiomyocyte": "cardio",
    }
    return replacements.get(str(name), str(name).replace("_", " "))


def clean_axis(ax, *, grid: bool = False) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=INK, labelcolor=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    if grid:
        ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    else:
        ax.grid(False)


def panel_label(ax, letter: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        fontweight="bold",
        color="black",
        clip_on=False,
    )


def add_family_header(fig, axes: Sequence, family: str, label: str | None = None) -> None:
    fig.canvas.draw_idle()
    boxes = [ax.get_position() for ax in axes]
    x0 = min(box.x0 for box in boxes)
    x1 = max(box.x1 for box in boxes)
    y1 = max(box.y1 for box in boxes)
    color = FAMILY_COLORS[family]
    fig.add_artist(
        Line2D([x0, x1], [y1 + 0.013, y1 + 0.013], transform=fig.transFigure, color=color, lw=2.5)
    )
    fig.text(
        (x0 + x1) / 2,
        y1 + 0.020,
        (label or family_label(family)).upper(),
        ha="center",
        va="bottom",
        fontsize=6.0,
        color=color,
        fontweight="bold",
    )


def add_family_row_strip(ax, method_order: Sequence[str] = METHOD_ORDER) -> None:
    for idx, method in enumerate(method_order):
        ax.add_patch(
            Rectangle(
                (-0.70, idx - 0.42),
                0.08,
                0.84,
                facecolor=family_color_for_method(method),
                edgecolor="none",
                clip_on=False,
                zorder=3,
            )
        )
    for y in [1.5, 3.5, 5.5]:
        ax.axhline(y, color=BORDER, lw=0.8, zorder=2)


def direct_label_points(ax, x, y, labels, offsets=None, **kwargs) -> None:
    offsets = offsets or {}
    for xi, yi, label in zip(x, y, labels):
        dx, dy = offsets.get(label, (0.015, 0.015))
        ax.annotate(
            str(label),
            xy=(xi, yi),
            xytext=(xi + dx, yi + dy),
            textcoords="data",
            fontsize=kwargs.get("fontsize", 6.0),
            color=kwargs.get("color", INK),
            ha=kwargs.get("ha", "left"),
            va=kwargs.get("va", "center"),
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": TEXT_MUTED, "shrinkA": 1, "shrinkB": 2},
        )


def draw_horizontal_track(
    ax,
    x0: float,
    x1: float,
    y: float,
    value: float,
    color: str,
    threshold: float | None = None,
    value_label: str | None = None,
    lw: float = 4.0,
) -> None:
    ax.plot([x0, x1], [y, y], color=TRACK_BG, lw=lw, solid_capstyle="round", zorder=1)
    ax.plot([x0, value], [y, y], color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter([value], [y], s=18, color=color, edgecolor="white", linewidth=0.45, zorder=3)
    if threshold is not None:
        ax.plot([threshold, threshold], [y - 0.22, y + 0.22], color=THRESHOLD_COLOR, lw=0.8, ls="--", zorder=2)
    if value_label is not None:
        ax.text(x1 + 0.035 * (x1 - x0), y, value_label, ha="left", va="center", fontsize=5.7, color=TEXT_MUTED)


def draw_interval_track(
    ax,
    xmin: float,
    xmax: float,
    xmedian: float,
    y: float,
    color: str,
    zero: float = 0,
    xlim: tuple[float, float] = (-1.25, 1.25),
    status: str = "mixed",
) -> None:
    ax.plot([xlim[0], xlim[1]], [y, y], color=TRACK_BG, lw=3.2, solid_capstyle="round", zorder=1)
    ax.plot([zero, zero], [y - 0.23, y + 0.23], color=INK, lw=0.7, zorder=2)
    ax.plot([xmin, xmax], [y, y], color=color, lw=1.2, zorder=3)
    if status == "all_pass":
        ax.scatter([xmedian], [y], s=24, color=color, edgecolor="white", linewidth=0.5, zorder=4)
    elif status == "all_fail":
        ax.scatter([xmedian], [y], s=28, marker="x", color=FAIL_COLOR, linewidth=1.2, zorder=5)
    else:
        ax.scatter([xmedian], [y], s=28, facecolor="white", edgecolor=color, linewidth=1.2, zorder=4)


def draw_bullet_cell(ax, x: float, y: float, width: float, value: float, color: str, label=None, highlight=False) -> None:
    ax.add_patch(Rectangle((x, y - 0.11), width, 0.22, facecolor=TRACK_BG, edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((x, y - 0.11), width * value, 0.22, facecolor=color, edgecolor="none", zorder=2))
    ax.scatter([x + width * value], [y], s=14, facecolor=color, edgecolor=BOTTLENECK_GOLD if highlight else "white", linewidth=1.2 if highlight else 0.4, zorder=3)
    if label is not None:
        ax.text(x + width + 0.02, y, str(label), ha="left", va="center", fontsize=5.6, color=TEXT_MUTED)


def draw_failure_tile(ax, x: float, y: float, failure_fraction: float | None, width: float, height: float, na: bool = False) -> None:
    if na or pd.isna(failure_fraction):
        ax.add_patch(Rectangle((x, y), width, height, facecolor=NA_COLOR, edgecolor=BORDER, lw=0.35, hatch="///", zorder=1))
        return
    frac = float(np.clip(failure_fraction, 0, 1))
    ax.add_patch(Rectangle((x, y), width, height, facecolor="#FAFBFC", edgecolor=BORDER, lw=0.35, zorder=1))
    if frac > 0:
        ax.add_patch(Rectangle((x, y), width, height * frac, facecolor=FAIL_COLOR, edgecolor="none", alpha=0.88, zorder=2))
    if np.isclose(frac, 1.0):
        ax.text(x + width / 2, y + height / 2, "1.00", ha="center", va="center", fontsize=4.8, color="white", zorder=3)
    elif np.isclose(frac, 0.0):
        ax.text(x + width / 2, y + height / 2, "0", ha="center", va="center", fontsize=4.8, color=TEXT_MUTED, zorder=3)


def draw_claim_glyph(ax, x: float, y: float, pass_count: int, tested_count: int, applicable: bool = True, color: str = PASS_COLOR) -> None:
    if not applicable or tested_count == 0:
        ax.text(x, y, "n/a", ha="center", va="center", fontsize=5.0, color=TEXT_MUTED)
        return
    frac = pass_count / tested_count
    if np.isclose(frac, 1.0):
        ax.add_patch(Circle((x, y), 0.12, facecolor=color, edgecolor=INK, lw=0.45))
    elif np.isclose(frac, 0.0):
        ax.text(x, y, "x", ha="center", va="center", fontsize=8.5, fontweight="bold", color=FAIL_COLOR)
    else:
        ax.add_patch(Circle((x, y), 0.12, facecolor="white", edgecolor=color, lw=0.9))
        ax.add_patch(Rectangle((x - 0.12, y - 0.12), 0.12, 0.24, facecolor=color, edgecolor="none"))
        ax.add_patch(Circle((x, y), 0.12, facecolor="none", edgecolor=color, lw=0.9))
    ax.text(x, y - 0.24, f"{pass_count}/{tested_count}", ha="center", va="top", fontsize=4.8, color=TEXT_MUTED)


def draw_stat_badge(ax, value_text: str, label: str, sublabel: str, x: float, y: float, w: float = 0.28, h: float = 0.70) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor="white",
            edgecolor=BORDER,
            lw=0.9,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(x + 0.04, y + h * 0.62, value_text, transform=ax.transAxes, ha="left", va="center", fontsize=22, fontweight="bold", color=FAIL_COLOR)
    ax.text(x + 0.04, y + h * 0.34, label, transform=ax.transAxes, ha="left", va="center", fontsize=6.2, color=INK, fontweight="bold")
    ax.text(x + 0.04, y + h * 0.16, sublabel, transform=ax.transAxes, ha="left", va="center", fontsize=5.4, color=TEXT_MUTED)


def draw_workflow_ribbon(ax, steps: Sequence[str]) -> None:
    ax.set_axis_off()
    n = len(steps)
    gap = 0.014
    w = (1 - gap * (n - 1)) / n
    y = 0.32
    h = 0.34
    for i, step in enumerate(steps):
        x = i * (w + gap)
        edge = BOTTLENECK_GOLD if i == n - 1 else BORDER
        lw = 1.5 if i == n - 1 else 0.8
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.035",
                facecolor="white",
                edgecolor=edge,
                lw=lw,
                transform=ax.transAxes,
            )
        )
        ax.text(x + w / 2, y + h / 2, step, transform=ax.transAxes, ha="center", va="center", fontsize=6.0, color=INK, fontweight="bold")
        if i < n - 1:
            ax.annotate("", xy=(x + w + gap * 0.78, y + h / 2), xytext=(x + w + gap * 0.18, y + h / 2), xycoords=ax.transAxes, arrowprops={"arrowstyle": "->", "lw": 0.7, "color": TEXT_MUTED})


def save_final_figure(fig, basename: str) -> None:
    ensure_final_dirs()
    for ext in ["pdf", "svg", "png"]:
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(FINAL_OUTPUT_DIR / f"{basename}.{ext}", **kwargs)
    plt.close(fig)


def export_summary(df: pd.DataFrame, filename: str) -> Path:
    ensure_final_dirs()
    path = FINAL_SOURCE_DIR / filename
    df.to_csv(path, index=False, float_format="%.10g")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family_handles():
    return [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAMILY_COLORS[f], markeredgecolor="white", markersize=6, label=family_label(f))
        for f in FAMILY_ORDER
    ]


def assert_expected_methods(df: pd.DataFrame, where: str, methods: Iterable[str] = METHOD_ORDER) -> None:
    present = set(df["method"].dropna().astype(str))
    expected = set(methods)
    missing = expected - present
    unexpected = present - expected
    if missing or unexpected:
        raise AssertionError(f"{where}: method mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")


def assert_fraction_columns(df: pd.DataFrame, columns: Sequence[str], where: str) -> None:
    for col in columns:
        vals = df[col].dropna().astype(float)
        if not vals.between(0, 1).all():
            raise AssertionError(f"{where}: {col} outside [0, 1]")
