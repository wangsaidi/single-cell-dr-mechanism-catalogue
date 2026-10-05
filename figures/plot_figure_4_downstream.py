"""Build revised Figure 4 from publication source-data tables.

The former threshold-gate Figure 4 is retained as Supplementary Fig. S14.
Only this main figure was replaced during revision because it directly reports
downstream clustering, population, trajectory and donor-related outcomes.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "source_data" / "figure4_downstream"
OUT = ROOT / "outputs" / "main_figures"
SUPP_OUT = ROOT / "outputs" / "supplementary_figures"

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "scVI", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
COLORS = {
    "PCA": "#2F6F9F",
    "GLM-PCA": "#78A9CF",
    "scScope": "#B54A00",
    "SAUCIE": "#E18A4A",
    "scVI": "#F2B47B",
    "UMAP": "#008C67",
    "PHATE": "#62B89A",
    "t-SNE": "#B65A96",
    "PaCMAP": "#D98CB7",
}
MARKERS = {
    "PCA": "o",
    "GLM-PCA": "s",
    "scScope": "^",
    "SAUCIE": "v",
    "scVI": "D",
    "UMAP": "P",
    "PHATE": "X",
    "t-SNE": "<",
    "PaCMAP": ">",
}
LINESTYLES = {
    "PCA": "-",
    "GLM-PCA": "--",
    "scScope": "-",
    "SAUCIE": "--",
    "scVI": ":",
    "UMAP": "-",
    "PHATE": "--",
    "t-SNE": "-",
    "PaCMAP": "--",
}


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
            "legend.fontsize": 5.7,
            "legend.frameon": False,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    sns.set_style("white")


def panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top", ha="left")


def clean(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", width=0.6, length=2.5)


def preserve_previous_figure_as_s14() -> None:
    """Keep the previous main Figure 4 as the threshold-sensitivity supplement."""
    SUPP_OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png", "jpg"):
        target = SUPP_OUT / f"Supplementary_Figure_S14_threshold_gate_sensitivity.{ext}"
        source = OUT / f"Figure_4.{ext}"
        if not target.exists() and source.exists():
            shutil.copy2(source, target)


def plot_resolution_curve(ax: plt.Axes, panel: str, title: str, ylabel: str) -> None:
    frame = pd.read_csv(SOURCE / f"Figure4_panel_{panel}.csv")
    for method in METHODS:
        part = frame[frame["method"].eq(method)].sort_values("resolution")
        if part.empty:
            continue
        x = part["resolution"].to_numpy(float)
        y = part["median"].to_numpy(float)
        ax.fill_between(x, part["low"], part["high"], color=COLORS[method], alpha=0.07, linewidth=0)
        ax.plot(
            x,
            y,
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            markersize=2.8,
            linewidth=0.9,
        )
    ax.set_title(title, pad=4)
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel(ylabel)
    clean(ax)
    panel_label(ax, panel)


def build() -> Path:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    preserve_previous_figure_as_s14()

    fig = plt.figure(figsize=(7.15, 8.7))
    outer = fig.add_gridspec(
        4,
        2,
        height_ratios=[1.0, 1.15, 1.05, 0.95],
        hspace=0.78,
        wspace=0.38,
        left=0.10,
        right=0.96,
        top=0.98,
        bottom=0.105,
    )

    plot_resolution_curve(fig.add_subplot(outer[0, 0]), "a", "Partition agreement", "Adjusted Rand index")
    plot_resolution_curve(fig.add_subplot(outer[0, 1]), "b", "Population recovery", "Macro-F1")

    ax_c = fig.add_subplot(outer[1, :])
    population = pd.read_csv(SOURCE / "Figure4_panel_c.csv")
    label_order = (
        population[["label", "prevalence"]]
        .drop_duplicates()
        .sort_values("prevalence", ascending=False)["label"]
        .tolist()
    )
    matrix = (
        population.groupby(["method", "label"], as_index=False)["f1_resolution_auc"]
        .median()
        .pivot(index="method", columns="label", values="f1_resolution_auc")
        .reindex(index=METHODS, columns=label_order)
    )
    prevalence = population.groupby("label")["prevalence"].first().reindex(label_order)
    sns.heatmap(
        matrix,
        ax=ax_c,
        cmap="mako",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 4.8},
        linewidths=0.25,
        linecolor="white",
        cbar_kws={"label": "Resolution-integrated F1", "shrink": 0.78, "pad": 0.015},
    )
    short = {
        "CD4 T cells": "CD4 T",
        "CD8 T cells": "CD8 T",
        "CD14+ Monocytes": "CD14 Mono",
        "FCGR3A+ Monocytes": "FCGR3A Mono",
        "Dendritic cells": "Dendritic",
        "Megakaryocytes": "Megakaryocyte",
    }
    ax_c.set_xticklabels(
        [f"{short.get(label, label)}\n{100 * prevalence[label]:.1f}%" for label in label_order],
        rotation=18,
        ha="right",
        rotation_mode="anchor",
    )
    ax_c.set_yticklabels(METHODS)
    for label in ax_c.get_yticklabels():
        label.set_color(COLORS.get(label.get_text(), "black"))
    ax_c.set_title("Population-specific recovery", pad=4)
    ax_c.set_xlabel("PBMC3k annotation (prevalence)")
    ax_c.set_ylabel("Method")
    panel_label(ax_c, "c", x=-0.055)

    ax_d = fig.add_subplot(outer[2, 0])
    trajectory = pd.read_csv(SOURCE / "Figure4_panel_d.csv").set_index("method").reindex(METHODS).reset_index()
    for row in trajectory.itertuples():
        ax_d.errorbar(
            row.lineage_median,
            row.dpt_median,
            xerr=[[row.lineage_median - row.lineage_min], [row.lineage_max - row.lineage_median]],
            yerr=[[row.dpt_median - row.dpt_min], [row.dpt_max - row.dpt_median]],
            fmt=MARKERS[row.method],
            color=COLORS[row.method],
            markersize=4.5,
            capsize=1.8,
            linewidth=0.7,
        )
    ax_d.axhline(0, color="#888888", lw=0.6, ls="--")
    ax_d.set_title("Lineage coherence and developmental order", pad=4)
    ax_d.set_xlabel("Same-lineage neighbour fraction")
    ax_d.set_ylabel("DPT order agreement, Spearman rho")
    clean(ax_d)
    panel_label(ax_d, "d", x=-0.10, y=1.12)

    ax_e = fig.add_subplot(outer[2, 1])
    heart = pd.read_csv(SOURCE / "Figure4_panel_e.csv")
    for method in METHODS:
        part = heart[heart["method"].eq(method)]
        if part.empty:
            continue
        ax_e.scatter(
            part["cell_type_accuracy"],
            part["donor_accuracy"],
            color=COLORS[method],
            marker=MARKERS[method],
            s=11,
            alpha=0.28,
            linewidth=0,
        )
        ax_e.scatter(
            part["cell_type_accuracy"].median(),
            part["donor_accuracy"].median(),
            color=COLORS[method],
            marker=MARKERS[method],
            s=32,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    ax_e.axhline(1 / 14, color="#777777", lw=0.7, ls="--", label="Donor chance, 1/14")
    ax_e.set_title("Cell type and donor are separate endpoints", pad=4)
    ax_e.set_xlabel("Held-donor-out cell-type balanced accuracy")
    ax_e.set_ylabel("Donor balanced accuracy")
    ax_e.legend(loc="lower right", fontsize=5.5)
    clean(ax_e)
    panel_label(ax_e, "e")

    ax_f = fig.add_subplot(outer[3, :])
    associations = pd.read_csv(SOURCE / "Figure4_panel_f.csv")
    associations = associations.sort_values(["comparison", "dataset_id"], kind="stable").reset_index(drop=True)
    positions = np.arange(len(associations))[::-1]
    significant = associations["q_bh_7_planned_tests"].lt(0.05)
    ax_f.axvline(0, color="#777777", lw=0.7, ls="--")
    ax_f.scatter(
        associations["spearman_rho"],
        positions,
        s=34,
        facecolors=np.where(significant, "#333333", "white"),
        edgecolors="#333333",
        linewidths=0.8,
        zorder=3,
    )
    for x, y, q_value in zip(
        associations["spearman_rho"], positions, associations["q_bh_7_planned_tests"]
    ):
        ax_f.text(x + 0.035, y, f"q={q_value:.3g}", va="center", ha="left", fontsize=5.4)
    ax_f.set_yticks(positions, associations["label"])
    ax_f.set_xlim(-1.0, 1.16)
    ax_f.set_xlabel("Spearman correlation across nine evaluated implementations")
    ax_f.set_title("Diagnostic associations depend on the downstream endpoint", pad=4)
    clean(ax_f)
    panel_label(ax_f, "f", x=-0.055, y=1.10)

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS[method],
            marker=MARKERS[method],
            linestyle=LINESTYLES[method],
            linewidth=1.0,
            markersize=4,
            label=method,
        )
        for method in METHODS
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.012), ncol=9,
               columnspacing=0.8, handletextpad=0.35)

    stem = OUT / "Figure_4"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".jpg"), dpi=300, bbox_inches="tight", pil_kwargs={"quality": 95})
    plt.close(fig)
    return stem.with_suffix(".pdf")


if __name__ == "__main__":
    print(build())
