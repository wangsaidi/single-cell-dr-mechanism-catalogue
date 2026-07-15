"""Shared Nature-class plotting style, colours, source-data export, and saving."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Rectangle

from . import config

FAMILY_COLORS = {
    "factor": "#0072B2",
    "deep": "#D55E00",
    "graph": "#009E73",
    "relational": "#CC79A7",
}

METHOD_COLORS = {
    "PCA": "#2F6F9F",
    "GLM-PCA": "#78A9CF",
    "scScope": "#B54A00",
    "SAUCIE": "#E18A4A",
    "UMAP": "#008C67",
    "PHATE": "#62B89A",
    "t-SNE": "#B65A96",
    "PaCMAP": "#D98CB7",
}

OKABE_ITO = [
    "#000000",
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#A6761D",
    "#666666",
]

CELLTYPE_COLORS: dict[str, str] = {}


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 8,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "savefig.dpi": config.DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def init_celltype_colors(labels: Iterable[str]) -> dict[str, str]:
    global CELLTYPE_COLORS
    cats = [str(x) for x in pd.Categorical(labels).categories]
    CELLTYPE_COLORS = {cat: OKABE_ITO[i % len(OKABE_ITO)] for i, cat in enumerate(cats)}
    return CELLTYPE_COLORS


def panel_label(ax, letter: str) -> None:
    ax.text(
        -0.12,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=9,
        fontweight="bold",
        va="top",
        ha="left",
    )


def family_legend_handles():
    return [
        Patch(
            facecolor=FAMILY_COLORS[family],
            edgecolor="none",
            alpha=0.85,
            label=config.FAMILY_LABELS.get(family, family),
        )
        for family in config.FAMILY_ORDER
    ]


def add_family_legend(fig, *, loc: str = "upper center", bbox_to_anchor=(0.5, 0.995), ncol: int = 4) -> None:
    fig.legend(
        handles=family_legend_handles(),
        title="mechanism family",
        loc=loc,
        bbox_to_anchor=bbox_to_anchor,
        ncol=ncol,
        fontsize=5.2,
        title_fontsize=5.5,
        frameon=False,
    )


def add_family_pair_headers(fig, axes, groups) -> None:
    """Add compact colored headers above paired method panels."""
    fig.canvas.draw_idle()
    for ax_group, family in groups:
        bboxes = [ax.get_position() for ax in ax_group]
        x0 = min(b.x0 for b in bboxes)
        x1 = max(b.x1 for b in bboxes)
        y1 = max(b.y1 for b in bboxes)
        color = FAMILY_COLORS[family]
        header = Rectangle(
            (x0, y1 + 0.018),
            x1 - x0,
            0.018,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="none",
            alpha=0.14,
            clip_on=False,
            zorder=0,
        )
        fig.add_artist(header)
        fig.text(
            (x0 + x1) / 2,
            y1 + 0.027,
            config.FAMILY_LABELS.get(family, family),
            ha="center",
            va="center",
            fontsize=5.7,
            fontweight="bold",
            color=color,
        )


def add_method_group_dividers(ax, n_methods: int = 8) -> None:
    for y in [1.5, 3.5, 5.5]:
        if y < n_methods - 0.5:
            ax.axhline(y, color="white", lw=1.0, zorder=3)


def add_method_group_vdividers(ax, n_methods: int = 8) -> None:
    for x in [1.5, 3.5, 5.5]:
        if x < n_methods - 0.5:
            ax.axvline(x, color="#D9D9D9", lw=0.7, zorder=0)


def color_method_ticklabels(ax, axis: str = "x") -> None:
    ticklabels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    for label in ticklabels:
        method = label.get_text()
        if method in METHOD_COLORS:
            label.set_color(family_color(method))


def color_family_ticklabels(ax, axis: str = "y") -> None:
    ticklabels = ax.get_xticklabels() if axis == "x" else ax.get_yticklabels()
    reverse = {config.FAMILY_LABELS.get(k, k): k for k in config.FAMILY_ORDER}
    for label in ticklabels:
        family = reverse.get(label.get_text())
        if family in FAMILY_COLORS:
            label.set_color(FAMILY_COLORS[family])


def save(fig, name: str) -> None:
    config.ensure_dirs()
    for ext in config.FORMATS:
        fig.savefig(config.OUTPUT_DIR / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


def write_source_data(fig: str, panel: str, desc: str, df: pd.DataFrame) -> Path:
    config.ensure_dirs()
    safe_desc = (
        desc.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("+", "plus")
    )
    path = config.SOURCE_DIR / f"{fig}_panel{panel}_{safe_desc}.csv"
    out = df.copy()
    out.to_csv(path, index=False, float_format="%.10g")
    return path


def scatter_embedding(ax, coords: np.ndarray, labels, title: str) -> None:
    labels = np.asarray(labels).astype(str)
    for lab in sorted(np.unique(labels)):
        mask = labels == lab
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=3,
            lw=0,
            alpha=0.78,
            color=CELLTYPE_COLORS.get(lab, "#999999"),
            label=lab,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("dim 1", fontsize=5.5, labelpad=1)
    ax.set_ylabel("dim 2", fontsize=5.5, labelpad=1)


def heatmap(ax, matrix, title: str, cmap: str = "viridis", vmin=None, vmax=None):
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def family_color(method: str) -> str:
    return METHOD_COLORS.get(method, FAMILY_COLORS[config.METHOD_FAMILY.get(method, "factor")])
