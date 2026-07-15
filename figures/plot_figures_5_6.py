"""Build source-data-backed main Figures 5 and 6."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle
from sklearn.neighbors import NearestNeighbors

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data"
REV_SOURCE_DIR = SOURCE_DIR / "generated"
OUTPUT_DIR = ROOT / "outputs" / "main_figures"

METHOD_ORDER = config.ANCHOR_METHODS
FAMILY_ORDER = config.FAMILY_ORDER
FAMILY_LABELS = config.FAMILY_LABELS

METHOD_COLORS = {
    method: mpl.colors.to_hex(mpl.colormaps["tab10"](i))
    for i, method in enumerate(METHOD_ORDER)
}


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#3B6F95")


def _color_method_labels(ax, axis: str) -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label in labels:
        label.set_color(_method_color(label.get_text()))

METRIC_LABELS = {
    "local_retention": "local",
    "trustworthiness": "trust",
    "global_rank_corr": "global",
    "label_neighbor_recall": "same-label",
    "latent_distance_corr": "latent",
    "pseudotime_rank_corr": "pseudo rank",
    "pseudotime_neighborhood_retention": "pseudo local",
    "cell_type_label_recall": "cell identity",
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


def _save(fig: plt.Figure, name: str, aliases: tuple[str, ...] = ()) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_name in (name, *aliases):
        for ext in ["pdf", "svg", "png", "jpg"]:
            kwargs = {"bbox_inches": "tight"}
            if ext in {"png", "jpg"}:
                kwargs["dpi"] = 450
            fig.savefig(OUTPUT_DIR / f"{target_name}.{ext}", **kwargs)
    plt.close(fig)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _method_colors(methods) -> list[str]:
    return [_method_color(str(method)) for method in methods]


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
    REV_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    pbmc_marker = pd.read_csv(SOURCE_DIR / "fig5_pbmc_marker_program_table.csv")
    pbmc_support = pd.read_csv(SOURCE_DIR / "fig5_pbmc_marker_support.csv")
    paul_lineage = pd.read_csv(SOURCE_DIR / "fig5_paul15_lineage_marker_support.csv")
    heart_marker = pd.read_csv(SOURCE_DIR / "fig5_heart_marker_program_table.csv")
    rare = pd.read_csv(SOURCE_DIR / "fig5_rare_state_support.csv")
    audit = pd.read_csv(SOURCE_DIR / "fig5_independent_evidence_audit.csv")
    marker_vs_neighbour = pd.read_csv(SOURCE_DIR / "fig5_marker_vs_neighbour_support.csv")
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
    _color_method_labels(ax, "y")
    for i in range(support_mat.shape[0]):
        for j in range(support_mat.shape[1]):
            val = support_mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val < 0.25 else "#132A13")
    ax.set_title("PBMC same-label neighbourhoods", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.set_ylabel("same-label neighbour fraction", rotation=270, labelpad=8, fontsize=5)
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
    evidence_rows = []
    for row in pbmc_support.itertuples(index=False):
        marker_margin = (float(row.max_marker_z) - 0.50) / 0.50
        neighbour_margin = (float(row.label_knn_recall) - 0.55) / 0.55
        evidence_rows.append(
            {
                "claim_type": "cell identity",
                "evidence_unit": f"{row.method} | {row.label}",
                "method": row.method,
                "family": row.family,
                "relative_evidence_margin": min(marker_margin, neighbour_margin),
                "component_definition": "minimum of marker-program and same-label-neighbour relative margins",
            }
        )
    for row in paul_lineage.itertuples(index=False):
        evidence_rows.append(
            {
                "claim_type": "continuum",
                "evidence_unit": row.program,
                "method": "",
                "family": "",
                "relative_evidence_margin": abs(float(row.value)) / 0.25 - 1.0,
                "component_definition": "absolute marker-pseudotime correlation relative to 0.25",
            }
        )
    for row in heart_donor.itertuples(index=False):
        identity_margin = (float(row.cell_type_label_recall) - 0.55) / 0.55
        donor_margin = (float(row.donor_entropy_norm) - 0.50) / 0.50
        evidence_rows.append(
            {
                "claim_type": "donor tissue",
                "evidence_unit": row.method,
                "method": row.method,
                "family": row.family,
                "relative_evidence_margin": min(identity_margin, donor_margin),
                "component_definition": "minimum of same-cell-type-neighbour and donor-entropy relative margins",
            }
        )
    for row in rare.itertuples(index=False):
        evidence_rows.append(
            {
                "claim_type": "rare state",
                "evidence_unit": f"{row.dataset_id} | {row.method} | {row.label}",
                "method": row.method,
                "family": row.family,
                "relative_evidence_margin": (float(row.label_knn_recall) - 0.55) / 0.55,
                "component_definition": "rare-label same-neighbour fraction relative to 0.55",
            }
        )
    evidence_units = pd.DataFrame(evidence_rows)
    evidence_units["supported"] = evidence_units["relative_evidence_margin"].ge(0)
    evidence_units.to_csv(REV_SOURCE_DIR / "Fig5d_evidence_unit_margins.csv", index=False)

    audit_order = ["cell identity", "continuum", "donor tissue", "rare state"]
    audit_colours = {
        "cell identity": "#4C78A8",
        "continuum": "#59A14F",
        "donor tissue": "#B07AA1",
        "rare state": "#D98CB7",
    }
    expected_support = {
        "cell identity": 0.859375,
        "continuum": 0.4,
        "donor tissue": 0.625,
        "rare state": 0.7321428571428571,
    }
    rng = np.random.default_rng(config.SEED)
    for x, claim in enumerate(audit_order):
        values = evidence_units.loc[evidence_units["claim_type"].eq(claim), "relative_evidence_margin"].to_numpy()
        observed_support = float(np.mean(values >= 0))
        if not np.isclose(observed_support, expected_support[claim]):
            raise ValueError(f"Fig. 5d support fraction changed for {claim}: {observed_support}")
        jitter = rng.uniform(-0.16, 0.16, len(values))
        ax.scatter(
            x + jitter,
            values,
            s=10,
            color=audit_colours[claim],
            alpha=0.42,
            edgecolor="white",
            linewidth=0.25,
        )
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        ax.vlines(x, q1, q3, color="#1F2937", linewidth=1.8)
        ax.scatter(x, median, s=25, facecolor="white", edgecolor="#1F2937", linewidth=0.8, zorder=4)
    counts = evidence_units.groupby("claim_type").size()
    ax.axhline(0, color="#777777", lw=0.8, ls="--")
    ax.set_xticks(
        np.arange(len(audit_order)),
        [f"{claim}\nn={int(counts[claim])}" for claim in audit_order],
        rotation=28,
        ha="right",
    )
    ax.set_ylim(-1.05, 1.02)
    ax.set_ylabel("Relative evidence margin")
    ax.set_title("Evidence-unit margins", loc="left")
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
    _color_method_labels(ax, "y")
    for i in range(rare_mat.shape[0]):
        for j in range(rare_mat.shape[1]):
            val = rare_mat.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val < 0.25 else "#132A13")
    ax.set_title("Rare-state support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.012)
    cbar.ax.set_ylabel("rare-label neighbour fraction", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[3, 2])
    for method, part in marker_vs_neighbour.groupby("method", sort=False):
        ax.scatter(
            part["max_marker_z"],
            part["label_knn_recall"],
            s=np.clip(part["n_cells_label"].to_numpy(dtype=float) / 16.0, 14, 62),
            color=_method_color(method),
            alpha=0.66,
            edgecolor="white",
            linewidth=0.35,
            label=method,
        )
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.set_xlabel("max marker z-score")
    ax.set_ylabel("same-label neighbour fraction")
    ax.set_ylim(0, 1.02)
    ax.set_title("Marker strength vs neighbourhood agreement", loc="left")
    ax.text(0.02, 0.05, "point area: cells per label", transform=ax.transAxes, fontsize=4.5, color="#555555")
    _panel_label(ax, "g", x=-0.20, y=1.14)

    ax = fig.add_subplot(gs[4, 0])
    pbmc_dual = marker_vs_neighbour[marker_vs_neighbour["dataset_id"].eq("pbmc3k")].copy()
    pbmc_dual["dual_supported_numeric"] = pbmc_dual["dual_supported"].astype(str).str.lower().eq("true").astype(float)
    dual = pbmc_dual.pivot(index="method", columns="label", values="dual_supported_numeric").reindex(index=METHOD_ORDER, columns=pbmc_group_order)
    pbmc_dual.to_csv(REV_SOURCE_DIR / "Fig5h_method_label_dual_support.csv", index=False)
    im = ax.imshow(dual.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(dual.shape[1]))
    ax.set_xticklabels([_short_label(x) for x in dual.columns], rotation=35, ha="right", fontsize=4.8)
    ax.set_yticks(np.arange(dual.shape[0]))
    ax.set_yticklabels(dual.index, fontsize=5.0)
    _color_method_labels(ax, "y")
    for i in range(dual.shape[0]):
        for j in range(dual.shape[1]):
            val = dual.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.2, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Method-label dual support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.018)
    cbar.ax.set_ylabel("dual support", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[4, 1])
    rare_method = rare.copy()
    rare_method["rare_pass"] = rare_method["label_knn_recall"].astype(float).ge(0.55).astype(float)
    rare_summary = rare_method.groupby(["method", "dataset_id"], as_index=False).agg(
        rare_pass_fraction=("rare_pass", "mean"),
        pass_count=("rare_pass", "sum"),
        n_labels=("rare_pass", "size"),
    )
    rare_summary.to_csv(REV_SOURCE_DIR / "Fig5i_method_context_rare_support.csv", index=False)
    rare_method_mat = rare_summary.pivot(index="method", columns="dataset_id", values="rare_pass_fraction").reindex(index=METHOD_ORDER, columns=["pbmc3k", "heart_cell_atlas_subsampled"])
    rare_count_mat = rare_summary.pivot(index="method", columns="dataset_id", values="pass_count").reindex(index=METHOD_ORDER, columns=["pbmc3k", "heart_cell_atlas_subsampled"])
    rare_n_mat = rare_summary.pivot(index="method", columns="dataset_id", values="n_labels").reindex(index=METHOD_ORDER, columns=["pbmc3k", "heart_cell_atlas_subsampled"])
    im = ax.imshow(rare_method_mat.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(["PBMC3k", "Heart atlas"], rotation=30, ha="right", fontsize=4.8)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER, fontsize=5.0)
    _color_method_labels(ax, "y")
    for i in range(rare_method_mat.shape[0]):
        for j in range(rare_method_mat.shape[1]):
            val = float(rare_method_mat.iloc[i, j])
            ax.text(j, i, f"{int(rare_count_mat.iloc[i, j])}/{int(rare_n_mat.iloc[i, j])}", ha="center", va="center", fontsize=4.6, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Rare-state support by method", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.018)
    cbar.ax.set_ylabel("pass fraction", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[4, 2])
    heart_coordinates = pd.read_csv(
        SOURCE_DIR / "fig3_heart_embedding_coordinates.csv",
        usecols=["method", "x", "y", "label"],
    )
    heart_label_marker = (
        heart_marker.groupby("label", as_index=False)
        .agg(max_marker_z=("z_score", "max"), n_cells_label=("n_cells", "max"))
    )
    heart_neighbour_rows = []
    for method in METHOD_ORDER:
        part = heart_coordinates[heart_coordinates["method"].eq(method)].reset_index(drop=True)
        coordinates = part[["x", "y"]].to_numpy(dtype=float)
        labels = part["label"].astype(str).to_numpy()
        neighbour_indices = NearestNeighbors(n_neighbors=16).fit(coordinates).kneighbors(return_distance=False)[:, 1:]
        same_label_fraction = (labels[neighbour_indices] == labels[:, None]).mean(axis=1)
        method_rows = pd.DataFrame({"label": labels, "same_label_fraction": same_label_fraction})
        method_summary = method_rows.groupby("label", as_index=False).agg(
            label_knn_recall=("same_label_fraction", "mean"),
            n_cells_coordinate=("same_label_fraction", "size"),
        )
        method_summary["method"] = method
        heart_neighbour_rows.append(method_summary)
    heart_marker_neighbour = pd.concat(heart_neighbour_rows, ignore_index=True).merge(
        heart_label_marker,
        on="label",
        how="left",
        validate="many_to_one",
    )
    if heart_marker_neighbour[["max_marker_z", "n_cells_label"]].isna().any().any():
        raise ValueError("Heart marker-neighbour analysis has unmatched labels.")
    heart_marker_neighbour["mean_label_knn_recall"] = heart_marker_neighbour.groupby("label")["label_knn_recall"].transform("mean")
    label_means = heart_marker_neighbour.drop_duplicates("label")[["label", "max_marker_z", "mean_label_knn_recall", "n_cells_label"]]
    marker_neighbour_rho = float(label_means["max_marker_z"].corr(label_means["mean_label_knn_recall"], method="spearman"))
    heart_marker_neighbour["descriptive_label_level_spearman_rho"] = marker_neighbour_rho
    heart_marker_neighbour.to_csv(REV_SOURCE_DIR / "Fig5j_heart_marker_neighbour_concordance.csv", index=False)
    for method, part in heart_marker_neighbour.groupby("method", sort=False):
        ax.scatter(
            part["max_marker_z"],
            part["label_knn_recall"],
            s=np.clip(part["n_cells_label"].to_numpy(dtype=float) / 28.0, 10, 55),
            color=_method_color(method),
            edgecolor="white",
            linewidth=0.3,
            alpha=0.55,
        )
    ax.scatter(
        label_means["max_marker_z"],
        label_means["mean_label_knn_recall"],
        s=23,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.65,
        zorder=4,
    )
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("max marker z-score")
    ax.set_ylabel("same-label neighbour fraction")
    ax.set_title("Heart marker-neighbour concordance", loc="left")
    ax.text(
        0.02,
        0.05,
        f"descriptive rho={marker_neighbour_rho:.2f}; n=11 labels",
        transform=ax.transAxes,
        fontsize=4.2,
        color="#555555",
    )
    _panel_label(ax, "j", x=-0.20, y=1.14)

    fig.legend(
        handles=[
            mpl.lines.Line2D([0], [0], marker="o", color="none", markerfacecolor=_method_color(method), markeredgecolor="white", markersize=4.2, label=method)
            for method in METHOD_ORDER
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.018),
        ncol=8,
        title="method",
        title_fontsize=5.4,
        fontsize=5.4,
        frameon=False,
    )
    _save(fig, "Figure_5", aliases=("Figure_5",))


def _line_panel(ax, df: pd.DataFrame, x_col: str, metric: str, title: str, xlabel: str, ylabel: str, methods: list[str]) -> None:
    sub = df[df["metric"].eq(metric)].copy()
    for method in methods:
        part = sub[sub["method"].eq(method)].sort_values(x_col)
        if part.empty:
            continue
        grouped = part.groupby(x_col)["value"].agg(["mean", "min", "max"]).reset_index()
        color = _method_color(method)
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
    REV_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
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
        cbar_label: str = "fraction below boundary",
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
        "same-label neighbour fraction",
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
        "same-label neighbour fraction",
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
    _failure_summary(dim, ["method", "metric"]).to_csv(REV_SOURCE_DIR / "Fig6e_dimension_failure_by_method_metric.csv", index=False)
    _draw_failure_heatmap(ax, dim_method_failure, "Dimension failure")
    _color_method_labels(ax, "y")
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
    _color_method_labels(ax, "y")
    for i in range(worst_mat.shape[0]):
        for j in range(worst_mat.shape[1]):
            val = worst_mat.iloc[i, j]
            if pd.notna(val):
                mark = "*" if support_mat.iloc[i, j] == "below_threshold" else ""
                ax.text(j, i, f"{val:.2f}{mark}", ha="center", va="center", fontsize=4.6)
    ax.set_title("Worst-case support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.020)
    cbar.ax.set_ylabel("worst score", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[2, 0])
    upstream_method_failure = (
        _failure_summary(upstream, ["method", "metric"])
        .pivot(index="method", columns="metric", values="failure_fraction")
        .reindex(index=["UMAP", "PHATE", "t-SNE", "PaCMAP"], columns=metric_order_robust)
    )
    upstream_method_failure.columns = [METRIC_LABELS[m] for m in metric_order_robust]
    _failure_summary(upstream, ["method", "metric"]).to_csv(REV_SOURCE_DIR / "Fig6g_upstream_failure_by_method_metric.csv", index=False)
    _draw_failure_heatmap(ax, upstream_method_failure, "Upstream-PCA failure")
    _color_method_labels(ax, "y")
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
    dim_failure_summary.to_csv(REV_SOURCE_DIR / "Fig6h_dimension_failure_by_dataset_metric.csv", index=False)
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
    cbar.ax.set_ylabel("fraction below boundary", rotation=270, labelpad=8, fontsize=5)
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
    _color_method_labels(ax, "y")
    for i in range(dim_delta.shape[0]):
        for j in range(dim_delta.shape[1]):
            val = dim_delta.iloc[i, j]
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=4.6, color="#222222")
    ax.set_title("Dimension shifts", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.020)
    cbar.ax.set_ylabel("mean score at d=20 minus d=2", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[3, 1:3])
    perturb_nonbaseline = perturb.loc[perturb["level"].astype(float).gt(0)].copy()
    perturb_nonbaseline["relative_threshold_margin"] = (
        perturb_nonbaseline["value"].astype(float) - perturb_nonbaseline["threshold"].astype(float)
    ) / perturb_nonbaseline["threshold"].astype(float)
    perturb_summary = (
        perturb_nonbaseline.groupby(["method", "perturbation", "level", "metric"], as_index=False)
        .agg(
            mean_relative_margin=("relative_threshold_margin", "mean"),
            minimum_relative_margin=("relative_threshold_margin", "min"),
            maximum_relative_margin=("relative_threshold_margin", "max"),
            n_replicates=("relative_threshold_margin", "size"),
        )
    )
    perturb_summary.to_csv(REV_SOURCE_DIR / "Fig6j_perturbation_condition_margins.csv", index=False)
    condition_order = [
        ("dropout", 0.2),
        ("dropout", 0.4),
        ("noise", 0.2),
        ("noise", 0.4),
    ]
    column_order = [
        (perturbation_name, level, metric)
        for perturbation_name, level in condition_order
        for metric in metric_order_robust
    ]
    perturb_summary["column_key"] = list(
        zip(perturb_summary["perturbation"], perturb_summary["level"].astype(float), perturb_summary["metric"])
    )
    margin_matrix = (
        perturb_summary.pivot(index="method", columns="column_key", values="mean_relative_margin")
        .reindex(index=["PCA", "UMAP", "PHATE", "PaCMAP"], columns=column_order)
    )
    if margin_matrix.isna().any().any():
        raise ValueError("Fig. 6j perturbation-margin matrix is incomplete.")
    display_matrix = np.clip(margin_matrix.values.astype(float), -1, 1)
    im = ax.imshow(
        display_matrix,
        cmap="RdBu_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(column_order)))
    ax.set_xticklabels(
        [METRIC_LABELS[metric] for _perturbation, _level, metric in column_order],
        rotation=90,
        ha="center",
        va="top",
        fontsize=4.8,
    )
    ax.set_yticks(np.arange(margin_matrix.shape[0]))
    ax.set_yticklabels(margin_matrix.index)
    _color_method_labels(ax, "y")
    for condition_index, (perturbation_name, level) in enumerate(condition_order):
        start = condition_index * len(metric_order_robust)
        centre = start + (len(metric_order_robust) - 1) / 2
        ax.text(
            centre,
            -0.68,
            f"{perturbation_name.title()} {level:.1f}",
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            clip_on=False,
        )
        if condition_index:
            ax.axvline(start - 0.5, color="white", linewidth=1.2)
    ax.set_title("Perturbation-specific margins", loc="left", pad=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.015)
    cbar.ax.set_ylabel("relative threshold margin", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "j", y=1.24)

    _save(fig, "Figure_6", aliases=("Figure_6",))



def main() -> None:
    build_fig5()
    build_fig6()

if __name__ == "__main__":
    main()
