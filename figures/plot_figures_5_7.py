"""Plot publication main figures 5-7 from locked source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source_data"
OUTPUT_DIR = ROOT / "outputs" / "main_figures"

METHOD_ORDER = config.ANCHOR_METHODS
FAMILY_ORDER = config.FAMILY_ORDER
FAMILY_LABELS = config.FAMILY_LABELS

METRIC_LABELS = {
    "local_retention": "local",
    "trustworthiness": "trust",
    "global_rank_corr": "global",
    "label_neighbor_recall": "label",
    "latent_distance_corr": "latent",
    "pseudotime_rank_corr": "pseudo rank",
    "pseudotime_neighborhood_retention": "pseudo local",
    "cell_type_label_recall": "cell-label",
    "donor_entropy_norm": "donor entropy",
}


def _apply_pub_style() -> None:
    style.apply_style()
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.7,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png", "jpg"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "jpg"}:
            kwargs["dpi"] = 450
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _method_colors(methods) -> list[str]:
    return [style.family_color(str(method)) for method in methods]


def _short_program(name: str) -> str:
    replacements = {
        "T cells (CD4/CD8)": "T cell",
        "CD14+ Monocytes": "CD14 Mono",
        "FCGR3A+ Monocytes": "FCGR3A Mono",
        "Dendritic cells": "DC",
        "Megakaryocytes": "Mega",
        "myeloid_monocyte": "myeloid/mono",
        "basophil_mast": "basophil/mast",
        "megakaryocyte": "mega",
        "smooth_muscle": "smooth muscle",
        "cardiomyocyte": "cardio",
    }
    return replacements.get(name, name.replace("_", " "))


def _short_label(name: str) -> str:
    replacements = {
        "Atrial_Cardiomyocyte": "Atrial CM",
        "Ventricular_Cardiomyocyte": "Ventricular CM",
        "Smooth_muscle_cells": "Smooth muscle",
        "Dendritic cells": "DC",
        "Megakaryocytes": "Mega",
        "CD14+ Monocytes": "CD14 Mono",
        "FCGR3A+ Monocytes": "FCGR3A Mono",
    }
    return replacements.get(name, name.replace("_", " "))


def _heatmap_with_values(ax, matrix: pd.DataFrame, *, cmap, vmin=None, vmax=None, norm=None, fmt: str = ".2f", fontsize: float = 4.8):
    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, norm=norm)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels([_short_label(str(x)) for x in matrix.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels([_short_program(str(x)) for x in matrix.index])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.notna(val):
                color = "white" if (norm(val) if norm is not None else (val - (vmin or 0)) / ((vmax or 1) - (vmin or 0))) > 0.72 else "#222222"
                if matrix.shape[0] * matrix.shape[1] <= 75:
                    ax.text(j, i, format(float(val), fmt), ha="center", va="center", fontsize=fontsize, color=color)
    return im


def build_fig5() -> None:
    """Figure 5: independent biological evidence anchors embedding claims."""
    _apply_pub_style()
    pbmc_marker = pd.read_csv(SOURCE_DIR / "fig5_pbmc_marker_program_table.csv")
    pbmc_support = pd.read_csv(SOURCE_DIR / "fig5_pbmc_marker_support.csv")
    paul_lineage = pd.read_csv(SOURCE_DIR / "fig5_paul15_lineage_marker_support.csv")
    heart_marker = pd.read_csv(SOURCE_DIR / "fig5_heart_marker_program_table.csv")
    rare = pd.read_csv(SOURCE_DIR / "fig5_rare_state_support.csv")
    audit = pd.read_csv(SOURCE_DIR / "fig5_independent_evidence_audit.csv")
    marker_vs_neighbour = pd.read_csv(SOURCE_DIR / "fig5_marker_vs_neighbour_support.csv")
    family_marker = pd.read_csv(SOURCE_DIR / "fig5_family_marker_support_summary.csv")
    rare_family = pd.read_csv(SOURCE_DIR / "fig5_rare_support_by_family.csv")
    heart_donor = pd.read_csv(SOURCE_DIR / "fig5_heart_identity_donor_support.csv")

    fig = plt.figure(figsize=(7.6, 11.2))
    gs = fig.add_gridspec(
        5,
        3,
        width_ratios=[1.18, 1.18, 1.0],
        height_ratios=[1.02, 1.02, 1.05, 0.92, 0.88],
        hspace=0.78,
        wspace=0.58,
    )

    ax = fig.add_subplot(gs[0, 0:2])
    pbmc_program_order = [
        "T cells (CD4/CD8)",
        "B cells",
        "CD14+ Monocytes",
        "FCGR3A+ Monocytes",
        "NK cells",
        "Dendritic cells",
        "Megakaryocytes",
    ]
    pbmc_group_order = [
        "CD4 T cells",
        "CD8 T cells",
        "B cells",
        "CD14+ Monocytes",
        "FCGR3A+ Monocytes",
        "NK cells",
        "Dendritic cells",
        "Megakaryocytes",
    ]
    mat = (
        pbmc_marker.pivot(index="program", columns="group", values="z_score")
        .reindex(index=pbmc_program_order, columns=pbmc_group_order)
    )
    norm = TwoSlopeNorm(vcenter=0, vmin=-1.2, vmax=2.8)
    im = _heatmap_with_values(ax, mat, cmap="RdBu_r", norm=norm, fmt=".1f", fontsize=4.5)
    ax.set_title("PBMC marker support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.set_ylabel("marker z-score", rotation=270, labelpad=8, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0:2])
    support_mat = (
        pbmc_support.pivot(index="method", columns="label", values="label_knn_recall")
        .reindex(index=METHOD_ORDER, columns=pbmc_group_order)
    )
    im = ax.imshow(support_mat.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(support_mat.shape[1]))
    ax.set_xticklabels([_short_label(x) for x in support_mat.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(support_mat.shape[0]))
    ax.set_yticklabels(support_mat.index)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(METHOD_ORDER))
    for i in range(support_mat.shape[0]):
        for j in range(support_mat.shape[1]):
            val = support_mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val < 0.25 else "#132A13")
    ax.set_title("PBMC label recall", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.set_ylabel("label recall", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    lineage = paul_lineage.copy()
    lineage["program_label"] = lineage["program"].map(_short_program)
    lineage = lineage.sort_values("value", ascending=False)
    colors = ["#4C78A8" if v >= 0 else "#C44E52" for v in lineage["value"]]
    ax.bar(np.arange(lineage.shape[0]), lineage["value"], color=colors, alpha=0.85)
    ax.axhline(0.25, color="#777777", lw=0.8, ls="--")
    ax.axhline(-0.25, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(np.arange(lineage.shape[0]))
    ax.set_xticklabels(lineage["program_label"], rotation=35, ha="right")
    ax.set_ylabel("Spearman rho")
    ax.set_ylim(-0.55, 0.55)
    ax.set_title("Paul15 lineage markers", loc="left")
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    audit_order = [
        "discrete_cell_identity",
        "trajectory_continuum",
        "donor_variable_tissue_identity",
        "rare_state_support",
    ]
    audit_labels = ["cell identity", "continuum", "donor tissue", "rare state"]
    audit = audit.set_index("claim_type").reindex(audit_order).reset_index()
    y = np.arange(audit.shape[0])
    ax.barh(y, audit["support_summary"], color=["#4C78A8", "#59A14F", "#B07AA1", "#D98CB7"], alpha=0.86)
    for yi, row in enumerate(audit.itertuples()):
        ax.text(row.support_summary + 0.02, yi, f"{row.support_summary:.2f}\nn={int(row.n_method_label_rows)}", va="center", fontsize=4.9)
    ax.set_yticks(y)
    ax.set_yticklabels(audit_labels)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("support fraction")
    ax.set_title("Evidence audit", loc="left")
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[2, 0:3])
    heart_program_order = [
        "cardiomyocyte",
        "endothelial",
        "fibroblast",
        "pericyte",
        "myeloid",
        "lymphoid",
        "smooth_muscle",
        "adipocyte",
        "neuronal",
        "mesothelial",
    ]
    heart_label_order = [
        "Atrial_Cardiomyocyte",
        "Ventricular_Cardiomyocyte",
        "Endothelial",
        "Fibroblast",
        "Pericytes",
        "Myeloid",
        "Lymphoid",
        "Smooth_muscle_cells",
        "Adipocytes",
        "Neuronal",
        "Mesothelial",
    ]
    hmat = (
        heart_marker.pivot(index="program", columns="label", values="z_score")
        .reindex(index=heart_program_order, columns=heart_label_order)
    )
    norm = TwoSlopeNorm(vcenter=0, vmin=-1.4, vmax=2.4)
    im = _heatmap_with_values(ax, hmat, cmap="RdBu_r", norm=norm, fmt=".1f", fontsize=4.2)
    ax.set_title("Heart marker support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.010)
    cbar.ax.set_ylabel("marker z-score", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "e")

    ax = fig.add_subplot(gs[3, 0:2])
    rare_order = ["Dendritic cells", "FCGR3A+ Monocytes", "Megakaryocytes", "NK cells", "Adipocytes", "Mesothelial", "Neuronal"]
    rare_mat = rare.pivot(index="method", columns="label", values="label_knn_recall").reindex(index=METHOD_ORDER, columns=rare_order)
    im = ax.imshow(rare_mat.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(rare_mat.shape[1]))
    ax.set_xticklabels([_short_label(x) for x in rare_mat.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(rare_mat.shape[0]))
    ax.set_yticklabels(rare_mat.index)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(METHOD_ORDER))
    for i in range(rare_mat.shape[0]):
        for j in range(rare_mat.shape[1]):
            val = rare_mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val < 0.25 else "#132A13")
    ax.set_title("Rare-state support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.set_ylabel("rare-label recall", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[3, 2])
    for family, part in marker_vs_neighbour.groupby("family", sort=False):
        ax.scatter(
            part["max_marker_z"],
            part["label_knn_recall"],
            s=np.clip(part["n_cells_label"].to_numpy(dtype=float) / 16.0, 14, 62),
            color=style.FAMILY_COLORS[family],
            alpha=0.66,
            edgecolor="white",
            linewidth=0.35,
            label=FAMILY_LABELS[family],
        )
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.set_xlabel("max marker z-score")
    ax.set_ylabel("label recall")
    ax.set_ylim(0, 1.02)
    ax.set_title("Marker vs recall", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=4.4, frameon=False)
    _panel_label(ax, "g", x=-0.20, y=1.14)

    ax = fig.add_subplot(gs[4, 0])
    dual = (
        family_marker.pivot(index="family", columns="label", values="dual_support_fraction")
        .reindex(index=FAMILY_ORDER, columns=pbmc_group_order)
    )
    im = ax.imshow(dual.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(dual.shape[1]))
    ax.set_xticklabels([_short_label(x) for x in dual.columns], rotation=35, ha="right", fontsize=4.8)
    ax.set_yticks(np.arange(dual.shape[0]))
    ax.set_yticklabels([FAMILY_LABELS[f] for f in dual.index], fontsize=5.0)
    style.color_family_ticklabels(ax, "y")
    for i in range(dual.shape[0]):
        for j in range(dual.shape[1]):
            val = dual.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Dual support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.018)
    cbar.ax.set_ylabel("dual support", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[4, 1])
    rare_family["dataset_label"] = rare_family["dataset_id"].map({"pbmc3k": "PBMC3k", "heart_cell_atlas_subsampled": "Heart atlas"})
    rare_family_mat = (
        rare_family.pivot(index="dataset_label", columns="family", values="rare_pass_fraction")
        .reindex(index=["PBMC3k", "Heart atlas"], columns=FAMILY_ORDER)
    )
    im = ax.imshow(rare_family_mat.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(FAMILY_ORDER)))
    ax.set_xticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER], rotation=35, ha="right", fontsize=4.8)
    style.color_family_ticklabels(ax, "x")
    ax.set_yticks(np.arange(rare_family_mat.shape[0]))
    ax.set_yticklabels(rare_family_mat.index, fontsize=5.0)
    for i in range(rare_family_mat.shape[0]):
        for j in range(rare_family_mat.shape[1]):
            val = rare_family_mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.6, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Rare-state pass", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.018)
    cbar.ax.set_ylabel("pass fraction", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[4, 2])
    label_positions = {
        "PCA": (0.635, 0.700),
        "GLM-PCA": (0.620, 0.790),
        "scScope": (0.735, 0.235),
        "SAUCIE": (0.625, 0.865),
        "UMAP": (0.525, 0.980),
        "PHATE": (0.545, 0.925),
        "t-SNE": (0.438, 1.005),
        "PaCMAP": (0.408, 0.955),
    }
    for method, row in heart_donor.set_index("method").reindex(METHOD_ORDER).iterrows():
        ax.scatter(
            row["donor_entropy_norm"],
            row["cell_type_label_recall"],
            s=42,
            color=style.family_color(method),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.86,
        )
        tx, ty = label_positions.get(method, (row["donor_entropy_norm"] + 0.010, row["cell_type_label_recall"]))
        ax.annotate(
            method,
            xy=(row["donor_entropy_norm"], row["cell_type_label_recall"]),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=4.3,
            color=style.family_color(method),
            va="center",
            ha="right" if tx < row["donor_entropy_norm"] else "left",
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#777777", "shrinkA": 1.5, "shrinkB": 2.0},
        )
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.axvline(0.50, color="#777777", lw=0.7, ls="--")
    ax.set_xlim(0.38, 0.78)
    ax.set_ylim(0.15, 1.02)
    ax.set_xlabel("donor entropy")
    ax.set_ylabel("cell-type recall")
    ax.set_title("Donor-aware identity", loc="left")
    _panel_label(ax, "j", x=-0.20, y=1.14)

    fig.legend(
        handles=style.family_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.50, 0.018),
        ncol=4,
        title="mechanism family",
        title_fontsize=5.6,
        fontsize=5.4,
        frameon=False,
    )
    _save(fig, "Figure_5")


def _line_panel(ax, df: pd.DataFrame, x_col: str, metric: str, title: str, xlabel: str, ylabel: str, methods: list[str]) -> None:
    sub = df[df["metric"].eq(metric)].copy()
    for method in methods:
        part = sub[sub["method"].eq(method)].sort_values(x_col)
        if part.empty:
            continue
        grouped = part.groupby(x_col)["value"].agg(["mean", "min", "max"]).reset_index()
        color = style.family_color(method)
        ax.plot(grouped[x_col], grouped["mean"], marker="o", ms=3.0, lw=1.1, color=color, label=method)
        if (grouped["max"] - grouped["min"]).abs().sum() > 0:
            ax.fill_between(grouped[x_col], grouped["min"], grouped["max"], color=color, alpha=0.12, lw=0)
    threshold = sub["threshold"].dropna().median()
    if pd.notna(threshold):
        ax.axhline(threshold, color="#777777", lw=0.8, ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(loc="lower right", fontsize=4.8, ncol=2, frameon=True, framealpha=0.9, edgecolor="#DDDDDD")


def build_fig6() -> None:
    """Figure 6: robustness axes define when a mechanism claim is stable."""
    _apply_pub_style()
    dim = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
    upstream = pd.read_csv(SOURCE_DIR / "fig6_upstream_pca_response.csv")
    perturb = pd.read_csv(SOURCE_DIR / "fig6_dropout_noise_response.csv")
    worst = pd.read_csv(SOURCE_DIR / "fig6_worst_case_support.csv")
    dim_slope = pd.read_csv(SOURCE_DIR / "fig6_dimension_sensitivity_slopes.csv")

    fig = plt.figure(figsize=(7.7, 10.0))
    gs = fig.add_gridspec(4, 3, height_ratios=[1.0, 1.0, 0.94, 0.90], hspace=0.76, wspace=0.72)

    metric_order_robust = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]

    def _failure_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        work = df.copy()
        work["failure"] = work["support"].eq("below_threshold").astype(float)
        return work.groupby(group_cols, as_index=False).agg(failure_fraction=("failure", "mean"), n_rows=("failure", "size"))

    def _draw_failure_heatmap(
        ax,
        matrix: pd.DataFrame,
        title: str,
        *,
        cbar_label: str = "fraction below threshold",
        xrot: float = 35,
    ) -> None:
        im = ax.imshow(matrix.values, vmin=0, vmax=1, cmap="Reds", aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=xrot, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.8, color="white" if val > 0.55 else "#4A1F1F")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.4, color="#777777")
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.020)
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=8, fontsize=5)
        cbar.ax.tick_params(labelsize=5)

    ax = fig.add_subplot(gs[0, 0])
    _line_panel(
        ax,
        dim[dim["output_dimension"].isin([2, 5, 10, 20])],
        "output_dimension",
        "label_neighbor_recall",
        "Dimension response",
        "output dimension",
        "label recall",
        ["PCA", "UMAP", "PHATE", "PaCMAP"],
    )
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    _line_panel(
        ax,
        upstream,
        "upstream_pca_dimension",
        "global_rank_corr",
        "Upstream PCA response",
        "upstream PCs",
        "global rank rho",
        ["UMAP", "PHATE", "t-SNE", "PaCMAP"],
    )
    _panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    _line_panel(
        ax,
        perturb[perturb["perturbation"].eq("dropout")],
        "level",
        "label_neighbor_recall",
        "Dropout response",
        "extra dropout level",
        "label recall",
        ["PCA", "UMAP", "PHATE", "PaCMAP"],
    )
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    _line_panel(
        ax,
        perturb[perturb["perturbation"].eq("noise")],
        "level",
        "trustworthiness",
        "Noise response",
        "noise level",
        "trustworthiness",
        ["PCA", "UMAP", "PHATE", "PaCMAP"],
    )
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1])
    dim_method_failure = (
        _failure_summary(dim, ["method", "metric"])
        .pivot(index="method", columns="metric", values="failure_fraction")
        .reindex(index=["PCA", "UMAP", "PHATE", "PaCMAP"], columns=metric_order_robust)
    )
    dim_method_failure.columns = [METRIC_LABELS[m] for m in metric_order_robust]
    _failure_summary(dim, ["method", "metric"]).to_csv(SOURCE_DIR / "fig6_dimension_failure_by_method_metric.csv", index=False)
    _draw_failure_heatmap(ax, dim_method_failure, "Dimension failure")
    style.color_method_ticklabels(ax, "y")
    _panel_label(ax, "e")

    ax = fig.add_subplot(gs[1, 2])
    sim_order = ["PCA", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
    metric_order = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall", "latent_distance_corr"]
    worst_mat = worst.pivot(index="method", columns="metric", values="worst_value").reindex(index=sim_order, columns=metric_order)
    support_mat = worst.pivot(index="method", columns="metric", values="support").reindex(index=sim_order, columns=metric_order)
    im = ax.imshow(worst_mat.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(sim_order)))
    ax.set_yticklabels(sim_order)
    style.color_method_ticklabels(ax, "y")
    for i in range(worst_mat.shape[0]):
        for j in range(worst_mat.shape[1]):
            val = worst_mat.iloc[i, j]
            if pd.notna(val):
                mark = "*" if support_mat.iloc[i, j] == "below_threshold" else ""
                ax.text(j, i, f"{val:.2f}{mark}", ha="center", va="center", fontsize=4.6)
    ax.set_title("Worst-case support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.020)
    cbar.ax.set_ylabel("worst score", rotation=270, labelpad=8, fontsize=5)
    ax.text(1.0, -0.28, "* below predeclared threshold", transform=ax.transAxes, ha="right", fontsize=4.8, color="#555555")
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[2, 0])
    upstream_method_failure = (
        _failure_summary(upstream, ["method", "metric"])
        .pivot(index="method", columns="metric", values="failure_fraction")
        .reindex(index=["UMAP", "PHATE", "t-SNE", "PaCMAP"], columns=metric_order_robust)
    )
    upstream_method_failure.columns = [METRIC_LABELS[m] for m in metric_order_robust]
    _failure_summary(upstream, ["method", "metric"]).to_csv(SOURCE_DIR / "fig6_upstream_failure_by_method_metric.csv", index=False)
    _draw_failure_heatmap(ax, upstream_method_failure, "Upstream-PCA failure")
    style.color_method_ticklabels(ax, "y")
    _panel_label(ax, "g")

    ax = fig.add_subplot(gs[2, 1:3])
    dim_failure = dim.copy()
    dim_failure["failure"] = dim_failure["support"].eq("below_threshold").astype(float)
    dataset_labels = {
        "pbmc3k": "PBMC3k",
        "paul15": "Paul15",
        "heart_cell_atlas_subsampled": "Heart atlas",
    }
    dim_failure_summary = (
        dim_failure.groupby(["dataset_id", "metric"], as_index=False)
        .agg(failure_fraction=("failure", "mean"), n_rows=("failure", "size"))
    )
    dim_failure_summary.to_csv(SOURCE_DIR / "fig6_dimension_failure_by_dataset_metric.csv", index=False)
    metric_order_dim = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
    dim_failure_mat = (
        dim_failure_summary.pivot(index="dataset_id", columns="metric", values="failure_fraction")
        .reindex(index=dataset_order, columns=metric_order_dim)
    )
    im = ax.imshow(dim_failure_mat.values, vmin=0, vmax=1, cmap="Reds", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order_dim)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order_dim], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels([dataset_labels[x] for x in dataset_order])
    for i in range(dim_failure_mat.shape[0]):
        for j in range(dim_failure_mat.shape[1]):
            val = dim_failure_mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.0, color="white" if val > 0.55 else "#4A1F1F")
    ax.set_xlabel("diagnostic metric")
    ax.set_title("Dataset failure", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.018)
    cbar.ax.set_ylabel("fraction below threshold", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[3, 0:1])
    dim_delta = (
        dim_slope.groupby(["method", "metric"], as_index=False)["dimension_delta"]
        .mean()
        .pivot(index="method", columns="metric", values="dimension_delta")
        .reindex(index=["PCA", "UMAP", "PHATE", "PaCMAP"], columns=["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"])
    )
    max_abs = max(0.05, float(np.nanmax(np.abs(dim_delta.values))))
    im = ax.imshow(dim_delta.values, vmin=-max_abs, vmax=max_abs, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(np.arange(dim_delta.shape[1]))
    ax.set_xticklabels([METRIC_LABELS.get(x, x) for x in dim_delta.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(dim_delta.shape[0]))
    ax.set_yticklabels(dim_delta.index)
    style.color_method_ticklabels(ax, "y")
    for i in range(dim_delta.shape[0]):
        for j in range(dim_delta.shape[1]):
            val = dim_delta.iloc[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=4.6, color="#222222")
    ax.set_title("Dimension shifts", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.020)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[3, 1:3])
    perturb_metric_failure = (
        _failure_summary(perturb, ["metric", "perturbation"])
        .pivot(index="metric", columns="perturbation", values="failure_fraction")
        .reindex(index=metric_order_robust, columns=["dropout", "noise"])
    )
    perturb_metric_failure.index = [METRIC_LABELS[m] for m in metric_order_robust]
    _failure_summary(perturb, ["metric", "perturbation"]).to_csv(SOURCE_DIR / "fig6_perturbation_failure_by_metric.csv", index=False)
    _draw_failure_heatmap(ax, perturb_metric_failure, "Metric stress failure", xrot=20)
    _panel_label(ax, "j")

    _save(fig, "Figure_6")


def build_fig7() -> None:
    """Figure 7: direct failure-mode analysis across diagnostic and stress-test axes."""
    _apply_pub_style()
    gate_matrix = pd.read_csv(SOURCE_DIR / "fig4_gate_pass_matrix.csv")
    local_gate = pd.read_csv(SOURCE_DIR / "fig4_local_gate_metrics.csv")
    global_gate = pd.read_csv(SOURCE_DIR / "fig4_global_gate_metrics.csv")
    continuum_gate = pd.read_csv(SOURCE_DIR / "fig4_paul15_continuum_metrics.csv")
    donor_gate = pd.read_csv(SOURCE_DIR / "fig4_heart_donor_gate_metrics.csv")
    dim = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
    perturb = pd.read_csv(SOURCE_DIR / "fig6_dropout_noise_response.csv")
    worst = pd.read_csv(SOURCE_DIR / "fig6_worst_case_support.csv")
    mechanism_sim = pd.read_csv(SOURCE_DIR / "fig6_mechanism_simulation_suite.csv")

    gate_order = [
        "local_neighbourhood_gate",
        "label_support_gate",
        "global_geometry_gate",
        "continuum_gate",
        "donor_aware_gate",
    ]
    gate_labels = ["local", "label", "global", "continuum", "donor"]
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
    dataset_labels = {"pbmc3k": "PBMC3k", "paul15": "Paul15", "heart_cell_atlas_subsampled": "Heart atlas"}
    metric_order = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
    metric_labels = [METRIC_LABELS[m] for m in metric_order]
    cmap_failure = mpl.colormaps["Reds"].copy()
    cmap_failure.set_bad("#F2F2F2")

    def _failure_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
        work = df.copy()
        work["failure"] = work["support"].eq("below_threshold").astype(float)
        return (
            work.groupby(group_cols, as_index=False)
            .agg(failure_fraction=("failure", "mean"), n_rows=("failure", "size"))
        )

    def _draw_heatmap(ax, matrix: pd.DataFrame, title: str, *, cbar_label: str = "failure fraction", fmt: str = ".2f"):
        masked = np.ma.masked_invalid(matrix.to_numpy(dtype=float))
        im = ax.imshow(masked, vmin=0, vmax=1, cmap=cmap_failure, aspect="auto")
        ax.set_xticks(np.arange(matrix.shape[1]))
        ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(matrix.shape[0]))
        ax.set_yticklabels(matrix.index)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix.iloc[i, j]
                if pd.notna(val):
                    ax.text(j, i, format(float(val), fmt), ha="center", va="center", fontsize=4.8, color="white" if val > 0.55 else "#4A1F1F")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.5, color="#777777")
        ax.set_title(title, loc="left")
        cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.018)
        cbar.ax.set_ylabel(cbar_label, rotation=270, labelpad=8, fontsize=5)
        cbar.ax.tick_params(labelsize=5)
        return im

    gate_failures = gate_matrix.copy()
    gate_failures["failure"] = gate_failures["support"].eq("below_threshold").astype(float)

    gate_by_gate = _failure_summary(gate_failures, ["claim_gate"]).set_index("claim_gate").reindex(gate_order).reset_index()
    gate_by_gate.to_csv(SOURCE_DIR / "fig7_gate_failure_by_gate.csv", index=False)
    gate_by_method = _failure_summary(gate_failures, ["method", "claim_gate"])
    gate_by_method.to_csv(SOURCE_DIR / "fig7_gate_failure_by_method.csv", index=False)
    gate_by_dataset = _failure_summary(gate_failures, ["dataset_id", "claim_gate"])
    gate_by_dataset.to_csv(SOURCE_DIR / "fig7_gate_failure_by_dataset.csv", index=False)
    gate_by_family = _failure_summary(gate_failures, ["family", "claim_gate"])
    gate_by_family.to_csv(SOURCE_DIR / "fig7_gate_failure_by_family.csv", index=False)

    component_metrics = pd.concat(
        [
            local_gate[["dataset_id", "method", "family", "metric", "value", "threshold", "gate_component_pass"]],
            global_gate[["dataset_id", "method", "family", "metric", "value", "threshold", "gate_component_pass"]],
            continuum_gate[["dataset_id", "method", "family", "metric", "value", "threshold", "gate_component_pass"]],
            donor_gate[donor_gate["threshold"].notna()][["dataset_id", "method", "family", "metric", "value", "threshold", "gate_component_pass"]],
        ],
        ignore_index=True,
    )
    component_metrics["failure"] = (~component_metrics["gate_component_pass"].astype(bool)).astype(float)
    component_metrics["threshold_margin"] = component_metrics["value"] - component_metrics["threshold"]
    component_failure = (
        component_metrics.groupby("metric", as_index=False)
        .agg(failure_fraction=("failure", "mean"), n_rows=("failure", "size"))
        .sort_values("failure_fraction", ascending=True)
    )
    component_failure.to_csv(SOURCE_DIR / "fig7_component_failure_by_metric.csv", index=False)
    component_margin = (
        component_metrics.groupby(["family", "metric"], as_index=False)
        .agg(mean_threshold_margin=("threshold_margin", "mean"), n_rows=("threshold_margin", "size"))
    )
    component_margin.to_csv(SOURCE_DIR / "fig7_component_margin_by_family_metric.csv", index=False)

    worst_failure = worst.copy()
    worst_failure["failure_fraction"] = worst_failure["support"].eq("below_threshold").astype(float)
    worst_failure.to_csv(SOURCE_DIR / "fig7_worst_failure_by_method_metric.csv", index=False)

    dim_failure = _failure_summary(dim, ["method", "metric"])
    dim_failure.to_csv(SOURCE_DIR / "fig7_dimension_failure_by_method_metric.csv", index=False)
    perturb_failure = _failure_summary(perturb, ["method", "perturbation"])
    perturb_failure.to_csv(SOURCE_DIR / "fig7_perturbation_failure_by_method.csv", index=False)
    sim_valid = mechanism_sim[mechanism_sim["support"].isin(["pass", "below_threshold"])].copy()
    sim_valid["failure"] = sim_valid["support"].eq("below_threshold").astype(float)
    sim_failure = (
        sim_valid.groupby(["method", "claim_axis"], as_index=False)
        .agg(failure_fraction=("failure", "mean"), n_rows=("failure", "size"))
    )
    sim_failure.to_csv(SOURCE_DIR / "fig7_simulation_failure_by_method_claim_axis.csv", index=False)

    fig = plt.figure(figsize=(7.8, 12.2))
    gs = fig.add_gridspec(5, 2, height_ratios=[0.92, 1.0, 1.0, 1.0, 1.04], width_ratios=[1.05, 1.15], hspace=0.78, wspace=0.62)

    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(gate_by_gate.shape[0])
    ax.barh(y, gate_by_gate["failure_fraction"], color="#4C78A8", alpha=0.88)
    for yi, row in enumerate(gate_by_gate.itertuples()):
        ax.text(row.failure_fraction + 0.025, yi, f"{row.failure_fraction:.2f}\nn={int(row.n_rows)}", va="center", fontsize=5.0)
    ax.set_yticks(y)
    ax.set_yticklabels(gate_labels)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("fraction below threshold")
    ax.set_title("Gate failure rates", loc="left")
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    method_gate_mat = (
        gate_by_method.pivot(index="method", columns="claim_gate", values="failure_fraction")
        .reindex(index=METHOD_ORDER, columns=gate_order)
    )
    method_gate_mat.columns = gate_labels
    _draw_heatmap(ax, method_gate_mat, "Method gate failure")
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(METHOD_ORDER))
    _panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 0])
    dataset_gate_mat = (
        gate_by_dataset.pivot(index="dataset_id", columns="claim_gate", values="failure_fraction")
        .reindex(index=dataset_order, columns=gate_order)
    )
    dataset_gate_mat.index = [dataset_labels[x] for x in dataset_gate_mat.index]
    dataset_gate_mat.columns = gate_labels
    _draw_heatmap(ax, dataset_gate_mat, "Dataset gate failure")
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1])
    family_gate_mat = (
        gate_by_family.pivot(index="family", columns="claim_gate", values="failure_fraction")
        .reindex(index=FAMILY_ORDER, columns=gate_order)
    )
    family_gate_mat.index = [FAMILY_LABELS[f] for f in family_gate_mat.index]
    family_gate_mat.columns = gate_labels
    _draw_heatmap(ax, family_gate_mat, "Family gate failure")
    for label, family in zip(ax.get_yticklabels(), FAMILY_ORDER):
        label.set_color(style.FAMILY_COLORS[family])
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[2, 0])
    component_failure["metric_label"] = component_failure["metric"].map(lambda x: METRIC_LABELS.get(x, x.replace("_", " ")))
    y = np.arange(component_failure.shape[0])
    ax.barh(y, component_failure["failure_fraction"], color="#C65D67", alpha=0.86)
    for yi, row in enumerate(component_failure.itertuples()):
        ax.text(row.failure_fraction + 0.025, yi, f"{row.failure_fraction:.2f}\nn={int(row.n_rows)}", va="center", fontsize=4.8)
    ax.set_yticks(y)
    ax.set_yticklabels(component_failure["metric_label"], fontsize=5.0)
    ax.set_xlim(0, 1.10)
    ax.set_xlabel("fraction below threshold")
    ax.set_title("Component failures", loc="left")
    _panel_label(ax, "e")

    ax = fig.add_subplot(gs[2, 1])
    component_order = [
        "local_retention",
        "trustworthiness",
        "label_neighbor_recall",
        "global_rank_corr",
        "pseudotime_rank_corr",
        "pseudotime_neighborhood_retention",
        "cell_type_label_recall",
        "donor_entropy_norm",
    ]
    margin_mat = (
        component_margin.pivot(index="family", columns="metric", values="mean_threshold_margin")
        .reindex(index=FAMILY_ORDER, columns=component_order)
    )
    max_abs = max(0.05, float(np.nanmax(np.abs(margin_mat.values))))
    im = ax.imshow(margin_mat.values, vmin=-max_abs, vmax=max_abs, cmap="RdBu", aspect="auto")
    ax.set_xticks(np.arange(len(component_order)))
    ax.set_xticklabels([METRIC_LABELS.get(x, x.replace("_", " ")) for x in component_order], rotation=35, ha="right", fontsize=4.7)
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels([FAMILY_LABELS[f] for f in FAMILY_ORDER])
    for label, family in zip(ax.get_yticklabels(), FAMILY_ORDER):
        label.set_color(style.FAMILY_COLORS[family])
    for i in range(margin_mat.shape[0]):
        for j in range(margin_mat.shape[1]):
            val = margin_mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=4.4, color="#222222")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=4.2, color="#777777")
    ax.set_title("Threshold margins", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.018)
    cbar.ax.set_ylabel("value - threshold", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[3, 0])
    worst_methods = ["PCA", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
    worst_metrics = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall", "latent_distance_corr"]
    worst_mat = (
        worst_failure.pivot(index="method", columns="metric", values="failure_fraction")
        .reindex(index=worst_methods, columns=worst_metrics)
    )
    worst_mat.columns = [METRIC_LABELS.get(x, x) for x in worst_mat.columns]
    _draw_heatmap(ax, worst_mat, "Worst-case failure")
    style.color_method_ticklabels(ax, "y")
    _panel_label(ax, "g")

    ax = fig.add_subplot(gs[3, 1])
    dim_mat = (
        dim_failure.pivot(index="method", columns="metric", values="failure_fraction")
        .reindex(index=["PCA", "UMAP", "PHATE", "PaCMAP"], columns=metric_order)
    )
    dim_mat.columns = metric_labels
    _draw_heatmap(ax, dim_mat, "Dimension failure")
    style.color_method_ticklabels(ax, "y")
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[4, 0])
    perturb_mat = (
        perturb_failure.pivot(index="method", columns="perturbation", values="failure_fraction")
        .reindex(index=["PCA", "UMAP", "PHATE", "PaCMAP"], columns=["dropout", "noise"])
    )
    _draw_heatmap(ax, perturb_mat, "Perturbation failure")
    style.color_method_ticklabels(ax, "y")
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[4, 1])
    claim_order = ["truth_geometry", "cell_state_support", "continuum_support", "rare_state_support", "batch_biology_tradeoff"]
    claim_labels = ["truth\ngeometry", "cell-state\nsupport", "continuum", "rare state", "batch-\nbiology"]
    sim_mat = (
        sim_failure.pivot(index="method", columns="claim_axis", values="failure_fraction")
        .reindex(index=METHOD_ORDER, columns=claim_order)
    )
    sim_mat.columns = claim_labels
    _draw_heatmap(ax, sim_mat, "Simulation failure")
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(METHOD_ORDER))
    _panel_label(ax, "j")

    _save(fig, "Figure_7")


def main() -> None:
    build_fig5()
    build_fig6()
    build_fig7()
    files = sorted(path for path in OUTPUT_DIR.glob("Figure_*.*") if path.stem in {"Figure_5", "Figure_6", "Figure_7"})
    manifest = pd.DataFrame(
        {
            "relative_path": [str(path.relative_to(ROOT)) for path in files],
            "size_bytes": [path.stat().st_size for path in files],
        }
    )
    manifest.to_csv(OUTPUT_DIR / "Figure_5_7_output_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()



