"""Plot publication main figures 2-4 from locked source data."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import sparse

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source_data"
OUTPUT_DIR = ROOT / "outputs" / "main_figures"

DATASET_LABELS = {
    "pbmc3k": "PBMC3k",
    "paul15": "Paul15",
    "heart_cell_atlas_subsampled": "Heart atlas",
    "known_truth_count_simulation": "Known-truth simulation",
}

DATASET_COLORS = {
    "pbmc3k": "#4C78A8",
    "paul15": "#59A14F",
    "heart_cell_atlas_subsampled": "#B07AA1",
    "known_truth_count_simulation": "#9C755F",
}

CLAIM_ROWS = [
    "Discrete identity",
    "Continuum",
    "Donor-aware tissue",
    "Rare states",
    "Robustness",
]

DIAGNOSTIC_COLS = [
    "Local",
    "Label",
    "Global",
    "Pseudotime",
    "Donor",
    "Marker",
    "Stress",
]

CLAIM_DIAGNOSTIC_VALUES = np.array(
    [
        [2, 2, 1, 0, 0, 2, 1],
        [1, 1, 2, 2, 0, 2, 2],
        [1, 2, 1, 0, 2, 2, 2],
        [2, 2, 0, 0, 0, 2, 2],
        [2, 1, 1, 1, 1, 0, 2],
    ],
    dtype=float,
)

SUPPORT_CMAP = ListedColormap(["#F0F0F0", "#A8C7E6", "#2F6F9F"])


def _apply_pub_style() -> None:
    style.apply_style()
    mpl.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.6,
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


def _short_dataset(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def _write_fig2_claim_map_source() -> Path:
    rows = []
    for i, claim in enumerate(CLAIM_ROWS):
        for j, diag in enumerate(DIAGNOSTIC_COLS):
            rows.append(
                {
                    "claim_type": claim,
                    "diagnostic": diag,
                    "requirement_level": CLAIM_DIAGNOSTIC_VALUES[i, j],
                    "requirement_definition": "0 not primary; 1 supporting; 2 required before a main-text visual claim",
                    "source": "predeclared manuscript v3 claim-to-evidence design matrix",
                }
            )
    out = SOURCE_DIR / "fig2_claim_diagnostic_map.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _write_fig2_label_burden_source(composition: pd.DataFrame) -> pd.DataFrame:
    rows = []
    label_comp = composition[composition["field_role"].eq("label")].copy()
    for dataset_id, sub in label_comp.groupby("dataset_id", sort=False):
        fractions = sub["fraction"].astype(float).to_numpy()
        counts = sub["n_cells"].astype(float).to_numpy()
        entropy = -float(np.sum(fractions * np.log(np.clip(fractions, 1e-12, 1.0))))
        norm_entropy = entropy / np.log(len(fractions)) if len(fractions) > 1 else 0.0
        rare_mask = counts <= 200
        rows.append(
            {
                "dataset_id": dataset_id,
                "n_label_levels": int(sub.shape[0]),
                "label_imbalance_index": float(1.0 - norm_entropy),
                "largest_label_fraction": float(np.max(fractions)),
                "rare_cell_fraction": float(np.sum(fractions[rare_mask])),
                "rare_label_fraction": float(np.mean(rare_mask)),
                "minimum_label_fraction": float(np.min(fractions)),
                "source_definition": "computed from locked label-level cell counts in dataset_composition.csv",
            }
        )
    out = SOURCE_DIR / "fig2_label_burden_metrics.csv"
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    return result


def _write_fig2_expression_sparsity_source(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in manifest.itertuples(index=False):
        counts_path = ROOT / str(row.analysis_counts_path)
        adata = sc.read_h5ad(counts_path)
        X = adata.X
        if sparse.issparse(X):
            nnz = int(X.nnz)
            zero_fraction = 1.0 - (nnz / float(X.shape[0] * X.shape[1]))
            detected = np.asarray((X > 0).sum(axis=1)).ravel()
            totals = np.asarray(X.sum(axis=1)).ravel()
        else:
            arr = np.asarray(X)
            zero_fraction = float(np.mean(arr == 0))
            detected = np.sum(arr > 0, axis=1)
            totals = np.sum(arr, axis=1)
        rows.append(
            {
                "dataset_id": row.dataset_id,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "zero_fraction": float(zero_fraction),
                "mean_detected_genes_per_cell": float(np.mean(detected)),
                "median_detected_genes_per_cell": float(np.median(detected)),
                "mean_total_counts_per_cell": float(np.mean(totals)),
                "counts_path": str(row.analysis_counts_path),
                "source_definition": "computed directly from locked analysis count matrices",
            }
        )
    out = SOURCE_DIR / "fig2_expression_sparsity_metrics.csv"
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    return result


def _write_fig2_detection_quantiles_source(cell_complexity: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_id",
        "q10_detected_genes_per_cell",
        "median_detected_genes_per_cell",
        "q90_detected_genes_per_cell",
        "source_definition",
    ]
    result = cell_complexity[cols].copy()
    out = SOURCE_DIR / "fig2_detected_gene_quantiles.csv"
    result.to_csv(out, index=False)
    return result


def _write_fig2_count_depth_source(cell_complexity: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset_id",
        "mean_total_counts_per_cell",
        "median_total_counts_per_cell",
        "source_definition",
    ]
    result = cell_complexity[cols].copy()
    out = SOURCE_DIR / "fig2_total_count_depth.csv"
    result.to_csv(out, index=False)
    return result


def build_fig2() -> None:
    """Figure 2: data contexts and diagnostic burden."""
    _apply_pub_style()
    lock = pd.read_csv(SOURCE_DIR / "dataset_summary.csv")
    composition = pd.read_csv(SOURCE_DIR / "dataset_composition.csv")
    manifest = pd.read_csv(SOURCE_DIR / "analysis_object_manifest.csv")
    label_burden = _write_fig2_label_burden_source(composition)
    expression_sparsity = _write_fig2_expression_sparsity_source(manifest)
    rare_burden = pd.read_csv(SOURCE_DIR / "fig2_rare_label_burden.csv")
    pca_structure = pd.read_csv(SOURCE_DIR / "fig2_pca_variance_structure.csv")
    pc_separability = pd.read_csv(SOURCE_DIR / "fig2_pc_label_separability.csv")
    cell_complexity = pd.read_csv(SOURCE_DIR / "fig2_cell_complexity_metrics.csv")
    detection_quantiles = _write_fig2_detection_quantiles_source(cell_complexity)
    count_depth = _write_fig2_count_depth_source(cell_complexity)

    fig = plt.figure(figsize=(7.6, 9.6))
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[1.05, 1.05, 1.0, 1.0],
        width_ratios=[1.02, 1.05, 1.05],
        hspace=0.72,
        wspace=0.64,
    )

    order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]

    ax = fig.add_subplot(gs[0, 0])
    label_offsets = {
        "pbmc3k": (1.05, 1.25, "left"),
        "paul15": (1.05, 1.02, "left"),
        "heart_cell_atlas_subsampled": (0.94, 1.06, "right"),
    }
    for row in lock.itertuples(index=False):
        ax.scatter(
            row.n_obs,
            row.n_vars,
            s=36 + 0.010 * float(row.n_label_levels) ** 2,
            color=DATASET_COLORS[row.dataset_id],
            alpha=0.84,
            edgecolor="white",
            linewidth=0.6,
        )
        ox, oy, ha = label_offsets[row.dataset_id]
        ax.text(row.n_obs * ox, row.n_vars * oy, _short_dataset(row.dataset_id), color=DATASET_COLORS[row.dataset_id], fontsize=5.3, ha=ha)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(2300, 25000)
    ax.set_ylim(1100, 35000)
    ax.set_xticks([3000, 5000, 10000, 20000])
    ax.set_xticklabels(["3k", "5k", "10k", "20k"])
    ax.set_yticks([2000, 10000, 30000])
    ax.set_yticklabels(["2k", "10k", "30k"])
    ax.set_xlabel("locked cells")
    ax.set_ylabel("locked genes")
    ax.set_title("Dataset scale", loc="left", pad=4)
    ax.text(0.56, 0.07, "size: label levels", transform=ax.transAxes, fontsize=4.7, color="#555555")
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1:3])
    label_comp = composition[composition["field_role"].eq("label")].copy()
    jitter = {"pbmc3k": -0.18, "paul15": 0.0, "heart_cell_atlas_subsampled": 0.18}
    for i, dataset_id in enumerate(order):
        sub = label_comp[label_comp["dataset_id"].eq(dataset_id)].copy()
        x = np.full(sub.shape[0], i, dtype=float) + jitter[dataset_id] * np.linspace(-0.25, 0.25, sub.shape[0])
        ax.scatter(
            x,
            sub["n_cells"],
            s=28,
            color=DATASET_COLORS[dataset_id],
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            label=_short_dataset(dataset_id),
        )
        for _, rare in sub.nsmallest(2, "n_cells").iterrows():
            ax.text(i + 0.12, rare["n_cells"], str(rare["level"])[:16], fontsize=4.6, va="center", color="#555555")
    ax.axhline(200, color="#8A8A8A", ls="--", lw=0.8)
    ax.text(2.05, 215, "rare-state threshold", fontsize=5.2, color="#666666", va="bottom")
    ax.set_yscale("log")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([_short_dataset(x) for x in order])
    ax.set_ylabel("cells per label level (log)")
    ax.set_title("Label burden", loc="left", pad=7)
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.50, 0.99), fontsize=5.0, frameon=False)
    _panel_label(ax, "b", x=-0.055)

    ax = fig.add_subplot(gs[1, 0])
    donor = composition[
        composition["dataset_id"].eq("heart_cell_atlas_subsampled")
        & composition["field_role"].eq("batch_or_donor")
        & composition["field"].eq("donor")
    ].sort_values("n_cells")
    y = np.arange(donor.shape[0])
    ax.barh(y, donor["n_cells"], color=DATASET_COLORS["heart_cell_atlas_subsampled"], alpha=0.78)
    ax.set_yticks(y)
    ax.set_yticklabels(donor["level"], fontsize=4.8)
    ax.set_xlabel("cells")
    ax.set_title("Donor structure", loc="left")
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 1:3])
    burden_cols = [
        "label_imbalance_index",
        "largest_label_fraction",
        "rare_cell_fraction",
        "rare_label_fraction",
        "minimum_label_fraction",
    ]
    burden_labels = ["imbalance", "largest\nlabel", "rare\ncells", "rare\nlabels", "smallest\nlabel"]
    burden = label_burden.set_index("dataset_id").reindex(order)[burden_cols]
    im = ax.imshow(burden.values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(burden_cols)))
    ax.set_xticklabels(burden_labels, rotation=0)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([_short_dataset(x) for x in order])
    for i in range(burden.shape[0]):
        for j in range(burden.shape[1]):
            val = burden.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.3, color="white" if val > 0.55 else "#4A1F1F")
    ax.set_title("Label-burden metrics", loc="left", pad=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.016)
    cbar.ax.set_ylabel("fraction or index", rotation=270, labelpad=9, fontsize=5)
    _panel_label(ax, "d", x=-0.055)

    ax = fig.add_subplot(gs[2, 0])
    expr = expression_sparsity.set_index("dataset_id").reindex(order).reset_index()
    sizes = 30 + 0.006 * expr["n_cells"].astype(float)
    for row, size in zip(expr.itertuples(index=False), sizes):
        ax.scatter(
            row.mean_detected_genes_per_cell,
            row.zero_fraction,
            s=float(size),
            color=DATASET_COLORS[row.dataset_id],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.55,
        )
        ax.text(
            row.mean_detected_genes_per_cell * 1.01,
            row.zero_fraction + 0.004,
            _short_dataset(row.dataset_id),
            fontsize=5.0,
            color=DATASET_COLORS[row.dataset_id],
        )
    ax.set_xlabel("mean detected HVGs per cell")
    ax.set_ylabel("zero fraction")
    ax.set_ylim(max(0, expr["zero_fraction"].min() - 0.06), min(1, expr["zero_fraction"].max() + 0.08))
    ax.set_title("Expression sparsity", loc="left", pad=6)
    ax.text(0.02, 0.05, "size: cells", transform=ax.transAxes, fontsize=4.7, color="#555555")
    _panel_label(ax, "e", x=-0.18, y=1.12)

    ax = fig.add_subplot(gs[2, 1])
    pca = pca_structure.set_index("dataset_id").reindex(order)
    x = np.arange(len(order))
    ax.plot(x, pca["pc5_cumulative_variance"], marker="o", color="#4C78A8", lw=1.2, label="PC5")
    ax.plot(x, pca["pc10_cumulative_variance"], marker="o", color="#59A14F", lw=1.2, label="PC10")
    ax.plot(x, pca["pc20_cumulative_variance"], marker="o", color="#B07AA1", lw=1.2, label="PC20")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset(i) for i in order], rotation=25, ha="right")
    ax.set_ylabel("cumulative variance")
    ax.set_ylim(0, min(1.0, max(0.25, float(pca[["pc5_cumulative_variance", "pc10_cumulative_variance", "pc20_cumulative_variance"]].max().max()) + 0.08)))
    ax.set_title("PCA variance", loc="left", pad=4)
    ax.legend(loc="upper left", ncol=3, bbox_to_anchor=(0.0, 1.02), fontsize=4.7, frameon=False, handlelength=1.2, columnspacing=0.7)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[2, 2])
    sep = pc_separability.set_index("dataset_id").reindex(order)
    width = 0.36
    ax.bar(x - width / 2, sep["pc_label_neighbor_recall"], width, color="#4C78A8", alpha=0.82, label="PC-kNN recall")
    ax.bar(x + width / 2, sep["pc20_label_silhouette"], width, color="#B07AA1", alpha=0.82, label="PC20 silhouette")
    ax.axhline(0, color="#555555", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset(i) for i in order], rotation=25, ha="right")
    ax.set_ylim(min(-0.1, float(sep["pc20_label_silhouette"].min()) - 0.04), 1.0)
    ax.set_ylabel("score")
    ax.set_title("PC separability", loc="left", pad=4)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        ncol=1,
        fontsize=4.8,
        frameon=False,
        handlelength=1.2,
    )
    _panel_label(ax, "g", x=-0.18, y=1.12)

    ax = fig.add_subplot(gs[3, 0])
    dq = detection_quantiles.set_index("dataset_id").reindex(order)
    ax.errorbar(
        np.arange(len(order)),
        dq["median_detected_genes_per_cell"],
        yerr=[
            dq["median_detected_genes_per_cell"] - dq["q10_detected_genes_per_cell"],
            dq["q90_detected_genes_per_cell"] - dq["median_detected_genes_per_cell"],
        ],
        fmt="o",
        color="#4C78A8",
        ecolor="#9BB7D4",
        elinewidth=1.2,
        capsize=3,
    )
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([_short_dataset(i) for i in order], rotation=25, ha="right")
    ax.set_ylabel("detected HVGs per cell")
    ax.set_title("Detection-depth spread", loc="left", pad=5)
    ax.text(
        0.00,
        -0.34,
        "point: median; whiskers: 10-90%",
        transform=ax.transAxes,
        fontsize=4.6,
        color="#555555",
        va="top",
        clip_on=False,
    )
    _panel_label(ax, "h", x=-0.18, y=1.12)

    ax = fig.add_subplot(gs[3, 1])
    cd = count_depth.set_index("dataset_id").reindex(order)
    width = 0.34
    x = np.arange(len(order))
    ax.bar(x - width / 2, cd["median_total_counts_per_cell"], width, color="#76B7B2", alpha=0.82, label="median")
    ax.bar(x + width / 2, cd["mean_total_counts_per_cell"], width, color="#F28E2B", alpha=0.82, label="mean")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset(i) for i in order], rotation=25, ha="right")
    ax.set_ylabel("total counts per cell")
    ax.set_title("Library-size depth", loc="left", pad=5)
    ax.legend(loc="upper right", fontsize=4.7, frameon=False)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[3, 2])
    rare = rare_burden.set_index("dataset_id").reindex(order)
    comp = cell_complexity.set_index("dataset_id").reindex(order)
    x = np.arange(len(order))
    width = 0.24
    ax.bar(x - width, rare["rare_cell_fraction"], width, color="#E15759", alpha=0.82, label="rare-cell fraction")
    ax.bar(x, rare["rare_label_fraction"], width, color="#F28E2B", alpha=0.82, label="rare-label fraction")
    detected_scaled = comp["median_detected_genes_per_cell"] / comp["median_detected_genes_per_cell"].max()
    ax.bar(x + width, detected_scaled, width, color="#76B7B2", alpha=0.82, label="median detected genes (scaled)")
    for xi, row in enumerate(rare.itertuples(index=True)):
        ax.text(xi - width, row.rare_cell_fraction + 0.025, f"{row.n_rare_labels} rare", ha="center", fontsize=4.9, color="#555555")
    ax.set_xticks(x)
    ax.set_xticklabels([_short_dataset(i) for i in order], rotation=25, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("fraction or scaled value")
    ax.set_title("Rare-depth burden", loc="left", pad=5)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.00),
        borderaxespad=0.0,
        ncol=1,
        fontsize=4.4,
        frameon=False,
        handlelength=1.2,
    )
    _panel_label(ax, "j")

    _save(fig, "Figure_2")


def _label_palette(labels: pd.Series) -> dict[str, str]:
    cats = [str(x) for x in pd.Categorical(labels.astype(str)).categories]
    cmap = plt.get_cmap("tab20")
    return {cat: mpl.colors.to_hex(cmap(i % 20)) for i, cat in enumerate(cats)}


def _paul_group(label: str) -> str:
    if "Ery" in label:
        return "Erythroid"
    if "Mo" in label:
        return "Monocyte"
    if "Baso" in label:
        return "Basophil"
    if "MEP" in label or "Mk" in label:
        return "MEP/Mk"
    if "Neu" in label:
        return "Neutrophil"
    return "Progenitor"


def _embedding_source(dataset_id: str) -> pd.DataFrame:
    if dataset_id == "heart_cell_atlas_subsampled":
        return pd.read_csv(SOURCE_DIR / "fig3_heart_embedding_coordinates.csv")
    return pd.read_csv(SOURCE_DIR / f"fig3_{dataset_id}_embedding_coordinates.csv")


def _plot_embedding_row(fig: plt.Figure, gs, row: int, dataset_id: str, letter: str) -> None:
    df = _embedding_source(dataset_id)
    methods = config.ANCHOR_METHODS
    plot_label = df["label"].astype(str)
    if dataset_id == "paul15":
        plot_label = plot_label.map(_paul_group)
    palette = _label_palette(plot_label)
    df = df.assign(plot_label=plot_label)
    dataset_title = _short_dataset(dataset_id)
    for col, method in enumerate(methods):
        ax = fig.add_subplot(gs[row, col])
        sub = df[df["method"].eq(method)]
        for label, part in sub.groupby("plot_label", sort=True):
            ax.scatter(
                part["x"],
                part["y"],
                s=1.2 if dataset_id == "heart_cell_atlas_subsampled" else 2.0,
                color=palette[str(label)],
                alpha=0.62,
                lw=0,
                rasterized=True,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        if row == 0:
            ax.set_title(method, color=style.family_color(method), fontsize=6.7, pad=2)
        if col == 0:
            ax.set_ylabel(dataset_title, fontsize=7.2, fontweight="bold", rotation=90, labelpad=8)
            _panel_label(ax, letter, x=-0.38, y=1.14)
    ax_leg = fig.add_subplot(gs[row, 8])
    ax_leg.axis("off")
    handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for label, color in list(palette.items())[:14]
    ]
    ax_leg.legend(handles=handles, loc="center left", fontsize=4.6, title="labels", title_fontsize=5.0, frameon=False)


def build_fig3() -> None:
    """Figure 3: cross-context eight-method behaviour."""
    _apply_pub_style()
    fig = plt.figure(figsize=(9.4, 13.0))
    gs = fig.add_gridspec(
        6,
        9,
        height_ratios=[1, 1, 1, 1.05, 1.0, 1.0],
        width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 0.95],
        hspace=0.68,
        wspace=0.34,
    )

    _plot_embedding_row(fig, gs, 0, "pbmc3k", "a")
    _plot_embedding_row(fig, gs, 1, "paul15", "b")
    _plot_embedding_row(fig, gs, 2, "heart_cell_atlas_subsampled", "c")

    metrics = pd.read_csv(SOURCE_DIR / "fig3_family_local_label_metrics.csv")
    method_wide = pd.read_csv(SOURCE_DIR / "fig3_method_metric_wide.csv")
    family_metric_support = pd.read_csv(SOURCE_DIR / "fig3_family_metric_pass_fraction.csv")
    family_context = pd.read_csv(SOURCE_DIR / "fig3_family_context_pass_fraction.csv")
    metric_order = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
    metric_labels = {
        "local_retention": "local",
        "trustworthiness": "trust",
        "global_rank_corr": "global",
        "label_neighbor_recall": "label",
    }
    method_order = config.ANCHOR_METHODS
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]

    ax = fig.add_subplot(gs[3, 0:5])
    family_metric = (
        metrics.groupby(["family", "dataset_id", "metric"], as_index=False)["value"]
        .mean()
    )
    heat_rows = []
    row_labels = []
    col_labels = []
    for dataset_id in dataset_order:
        for metric in metric_order:
            col_labels.append(f"{_short_dataset(dataset_id)}\n{metric.replace('_', ' ').replace('neighbor ', '')[:8]}")
    for family in config.FAMILY_ORDER:
        vals = []
        for dataset_id in dataset_order:
            for metric in metric_order:
                value = family_metric[
                    family_metric["family"].eq(family)
                    & family_metric["dataset_id"].eq(dataset_id)
                    & family_metric["metric"].eq(metric)
                ]["value"].mean()
                vals.append(float(value))
        heat_rows.append(vals)
        row_labels.append(config.FAMILY_LABELS[family])
    mat = np.asarray(heat_rows)
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=4.5)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=5.6)
    style.color_family_ticklabels(ax, "y")
    for xline in [3.5, 7.5]:
        ax.axvline(xline, color="white", lw=0.9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] >= 0.75 or mat[i, j] <= 0.12:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=4.4, color="white" if mat[i, j] < 0.25 else "#132A13")
    ax.set_title("Family metric profiles", loc="left", pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.012, ticks=[0, 0.5, 1.0])
    cbar.ax.set_ylabel("score", rotation=270, labelpad=8, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[3, 6:9])
    support = pd.read_csv(SOURCE_DIR / "fig3_cross_context_support_matrix.csv")
    overall = support[support["metric"].eq("all_four_metrics")].copy()
    support_matrix = overall.pivot(index="method", columns="dataset_id", values="value").reindex(method_order)[dataset_order]
    im = ax.imshow(support_matrix.values, vmin=0, vmax=4, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels([_short_dataset(x) for x in dataset_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    for i in range(support_matrix.shape[0]):
        for j in range(support_matrix.shape[1]):
            ax.text(j, i, f"{support_matrix.iloc[i, j]:.0f}/4", ha="center", va="center", fontsize=5.4)
    ax.set_title("Diagnostic support")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.ax.set_ylabel("passed metrics", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "e")

    row4 = gs[4, 0:9].subgridspec(1, 3, wspace=0.72)
    ax_f = fig.add_subplot(row4[0, 0])
    ax = ax_f
    method_metric = (
        method_wide.groupby(["method", "family"], as_index=False)[metric_order]
        .mean()
        .set_index("method")
        .reindex(method_order)
    )
    mm = method_metric[metric_order]
    im = ax.imshow(mm.values, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(method_order))
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            val = mm.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.4, color="white" if val < 0.25 else "#132A13")
    ax.set_title("Method fingerprints", loc="left", pad=7)
    _panel_label(ax, "f")

    ax_g = fig.add_subplot(row4[0, 1])
    ax = ax_g
    pass_metric = (
        family_metric_support[family_metric_support["metric"].isin(metric_order)]
        .groupby(["family", "metric"], as_index=False)["pass_flag"]
        .mean()
        .pivot(index="family", columns="metric", values="pass_flag")
        .reindex(index=config.FAMILY_ORDER, columns=metric_order)
    )
    im = ax.imshow(pass_metric.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(pass_metric.index)))
    family_short = {"factor": "Factor", "deep": "Deep", "graph": "Graph", "relational": "Relational"}
    ax.set_yticklabels([family_short[f] for f in pass_metric.index], fontsize=5.2)
    for label, family in zip(ax.get_yticklabels(), pass_metric.index):
        label.set_color(style.FAMILY_COLORS[family])
    for i in range(pass_metric.shape[0]):
        for j in range(pass_metric.shape[1]):
            val = pass_metric.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.7, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Family support", loc="left", pad=7)
    _panel_label(ax, "g")

    ax_h = fig.add_subplot(row4[0, 2])
    ax = ax_h
    context_mat = (
        family_context.pivot(index="family", columns="dataset_id", values="pass_fraction")
        .reindex(index=config.FAMILY_ORDER, columns=dataset_order)
    )
    im = ax.imshow(context_mat.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels([_short_dataset(x) for x in dataset_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(context_mat.index)))
    ax.set_yticklabels([family_short[f] for f in context_mat.index], fontsize=5.2)
    for label, family in zip(ax.get_yticklabels(), context_mat.index):
        label.set_color(style.FAMILY_COLORS[family])
    for i in range(context_mat.shape[0]):
        for j in range(context_mat.shape[1]):
            val = context_mat.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.9, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Context support", loc="left", pad=7)
    cbar = fig.colorbar(im, ax=[ax_f, ax_g, ax_h], fraction=0.018, pad=0.018, ticks=[0, 0.5, 1.0])
    cbar.ax.set_ylabel("score / pass fraction", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[5, 0:4])
    context_range = (
        method_wide.groupby(["method", "family"], as_index=False)[metric_order]
        .agg(lambda col: float(col.max() - col.min()))
        .set_index("method")
        .reindex(method_order)
    )
    context_range_out = context_range.reset_index()
    context_range_out.to_csv(SOURCE_DIR / "fig3_method_context_metric_range.csv", index=False)
    cr = context_range[metric_order]
    context_range_vmax = max(0.25, float(np.nanmax(cr.values)))
    im = ax.imshow(cr.values, vmin=0, vmax=context_range_vmax, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(method_order))
    for i in range(cr.shape[0]):
        for j in range(cr.shape[1]):
            val = cr.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.4, color="white" if val > 0.34 else "#4A1F1F")
    ax.set_title("Context sensitivity", loc="left", pad=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.014, ticks=[0, round(context_range_vmax, 2)])
    cbar.ax.set_ylabel("max-min score", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[5, 5:9])
    tradeoff = method_wide[["dataset_id", "method", "family", "local_retention", "global_rank_corr", "label_neighbor_recall", "trustworthiness"]].copy()
    tradeoff.to_csv(SOURCE_DIR / "fig3_local_global_tradeoff_by_context.csv", index=False)
    markers = {"pbmc3k": "o", "paul15": "^", "heart_cell_atlas_subsampled": "s"}
    for dataset_id, part in tradeoff.groupby("dataset_id", sort=False):
        for row in part.itertuples(index=False):
            ax.scatter(
                row.global_rank_corr,
                row.local_retention,
                s=18 + 45 * float(row.label_neighbor_recall),
                marker=markers.get(dataset_id, "o"),
                color=style.family_color(row.method),
                alpha=0.84,
                edgecolor="white",
                linewidth=0.35,
            )
    ax.axvline(0.45, color="#777777", lw=0.7, ls="--")
    ax.axhline(0.30, color="#777777", lw=0.7, ls="--")
    ax.set_xlim(0, 0.82)
    ax.set_ylim(0, max(0.38, float(tradeoff["local_retention"].max()) + 0.06))
    ax.set_xlabel("global rank correlation")
    ax.set_ylabel("local retention", labelpad=1)
    ax.set_title("Local-global trade-off", loc="left", pad=7)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="PBMC3k"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="Paul15"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="Heart atlas"),
    ]
    ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.02, 0.62),
        borderaxespad=0.0,
        fontsize=4.7,
        frameon=False,
        title="dataset",
        title_fontsize=5.0,
    )
    ax.text(0.02, 0.05, "point size: label recall", transform=ax.transAxes, fontsize=4.8, color="#555555")
    _panel_label(ax, "j")

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
    _save(fig, "Figure_3")


def build_fig4() -> None:
    """Figure 4: claim-specific diagnostic gates."""
    _apply_pub_style()
    local = pd.read_csv(SOURCE_DIR / "fig4_local_gate_metrics.csv")
    global_df = pd.read_csv(SOURCE_DIR / "fig4_global_gate_metrics.csv")
    continuum = pd.read_csv(SOURCE_DIR / "fig4_paul15_continuum_metrics.csv")
    donor = pd.read_csv(SOURCE_DIR / "fig4_heart_donor_gate_metrics.csv")
    gate = pd.read_csv(SOURCE_DIR / "fig4_gate_pass_matrix.csv")
    gate_by_method = pd.read_csv(SOURCE_DIR / "fig4_gate_pass_by_method.csv")
    local_tradeoff = pd.read_csv(SOURCE_DIR / "fig4_local_metric_tradeoff.csv")

    fig = plt.figure(figsize=(7.6, 11.4))
    gs = fig.add_gridspec(
        4,
        3,
        width_ratios=[1.20, 1.0, 1.0],
        height_ratios=[1.05, 1.05, 1.0, 1.0],
        hspace=0.76,
        wspace=0.58,
    )

    ax = fig.add_subplot(gs[0, 0])
    local_metrics = ["local_retention", "trustworthiness", "label_neighbor_recall"]
    rows = []
    labels = []
    for method in config.ANCHOR_METHODS:
        vals = [float(local[(local["method"].eq(method)) & (local["metric"].eq(metric))]["value"].mean()) for metric in local_metrics]
        rows.append(vals)
        labels.append(method)
    im = ax.imshow(np.asarray(rows), vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(local_metrics)))
    ax.set_xticklabels(["local", "trust", "label"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    style.color_method_ticklabels(ax, "y")
    ax.set_title("Local gate")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.025)
    cbar.ax.set_ylabel("score", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 1])
    gmat = global_df.pivot(index="method", columns="dataset_id", values="value").reindex(config.ANCHOR_METHODS)
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
    gmat = gmat[dataset_order]
    im = ax.imshow(gmat.values, vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.axvline(1.5, color="white", lw=0.6)
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels([_short_dataset(x) for x in dataset_order], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(config.ANCHOR_METHODS)))
    ax.set_yticklabels(config.ANCHOR_METHODS, fontsize=5.2)
    style.color_method_ticklabels(ax, "y")
    ax.set_title("Global gate")
    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.025)
    cbar.ax.set_ylabel("Spearman rho", rotation=270, labelpad=9, fontsize=5)
    _panel_label(ax, "b")

    ax = fig.add_subplot(gs[0, 2])
    cwide = continuum.pivot(index="method", columns="metric", values="value").reindex(config.ANCHOR_METHODS)
    x = np.arange(len(cwide.index))
    width = 0.36
    ax.bar(x - width / 2, cwide["pseudotime_rank_corr"], width, color="#7A9CC6", label="rank corr.")
    ax.bar(x + width / 2, cwide["pseudotime_neighborhood_retention"], width, color="#59A14F", label="local retention")
    ax.axhline(0.45, color="#7A9CC6", lw=0.7, ls="--")
    ax.axhline(0.50, color="#59A14F", lw=0.7, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(cwide.index, rotation=35, ha="right")
    style.color_method_ticklabels(ax, "x")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Paul15 continuum score")
    ax.set_title("Continuum gate")
    ax.legend(loc="upper left", fontsize=4.9, frameon=False, title="bars; lines = cutoffs", title_fontsize=4.6)
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 0])
    dwide = donor.pivot(index="method", columns="metric", values="value").reindex(config.ANCHOR_METHODS)
    label_positions = {
        "PCA": (0.635, 0.670),
        "GLM-PCA": (0.620, 0.790),
        "scScope": (0.735, 0.235),
        "SAUCIE": (0.625, 0.865),
        "UMAP": (0.525, 0.982),
        "PHATE": (0.545, 0.925),
        "t-SNE": (0.438, 1.008),
        "PaCMAP": (0.408, 0.935),
    }
    for method, row in dwide.iterrows():
        ax.scatter(
            row["donor_entropy_norm"],
            row["cell_type_label_recall"],
            s=42,
            color=style.family_color(method),
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
        tx, ty = label_positions.get(method, (row["donor_entropy_norm"] + 0.012, row["cell_type_label_recall"]))
        ha = "right" if tx < row["donor_entropy_norm"] else "left"
        ax.annotate(
            method,
            xy=(row["donor_entropy_norm"], row["cell_type_label_recall"]),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=5.0,
            va="center",
            ha=ha,
            color=style.family_color(method),
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#7A7A7A", "shrinkA": 1.5, "shrinkB": 2.0},
        )
    ax.axhline(0.55, color="#777777", ls="--", lw=0.8)
    ax.axvline(0.50, color="#777777", ls="--", lw=0.8)
    ax.set_xlim(0.37, 0.80)
    ax.set_ylim(0.15, 1.04)
    ax.set_xlabel("donor entropy (local kNN)")
    ax.set_ylabel("cell-type label recall")
    ax.set_title("Donor-aware gate")
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1:3])
    plotted_gate = gate[gate["claim_gate"].isin(["local_neighbourhood_gate", "global_geometry_gate", "continuum_gate", "donor_aware_gate", "label_support_gate"])].copy()
    plotted_gate["pass"] = plotted_gate["support"].eq("pass").astype(float)
    summary_counts = (
        plotted_gate.groupby(["family", "claim_gate"])["pass"]
        .agg(pass_fraction="mean", pass_count="sum", n_cases="size")
        .reset_index()
    )
    summary = (
        summary_counts.pivot(index="family", columns="claim_gate", values="pass_fraction")
        .reindex(config.FAMILY_ORDER)
    )
    gate_order = ["label_support_gate", "local_neighbourhood_gate", "global_geometry_gate", "continuum_gate", "donor_aware_gate"]
    summary = summary[gate_order]
    summary_pass = summary_counts.pivot(index="family", columns="claim_gate", values="pass_count").reindex(config.FAMILY_ORDER)[gate_order]
    summary_n = summary_counts.pivot(index="family", columns="claim_gate", values="n_cases").reindex(config.FAMILY_ORDER)[gate_order]
    im = ax.imshow(summary.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(gate_order)))
    ax.set_xticklabels(["label", "local", "global", "continuum", "donor"], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(summary.index)))
    ax.set_yticklabels([config.FAMILY_LABELS[f] for f in summary.index])
    style.color_family_ticklabels(ax, "y")
    for i in range(summary.shape[0]):
        for j in range(summary.shape[1]):
            passed = int(round(float(summary_pass.iloc[i, j])))
            total = int(summary_n.iloc[i, j])
            frac = float(summary.iloc[i, j])
            ax.text(
                j,
                i,
                f"{passed}/{total}",
                ha="center",
                va="center",
                fontsize=5.7,
                color="white" if frac > 0.72 else "#17213A",
            )
    ax.set_title("Family gate pass")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.set_ylabel("pass fraction", rotation=270, labelpad=9, fontsize=5)
    _panel_label(ax, "e")

    ax = fig.add_subplot(gs[2, 0])
    valid_gate = gate.copy()
    valid_gate["pass_flag"] = valid_gate["support"].eq("pass").astype(float)
    dataset_gate = (
        valid_gate.groupby(["dataset_id", "claim_gate"], as_index=False)
        .agg(pass_fraction=("pass_flag", "mean"), n_cases=("pass_flag", "size"))
    )
    dataset_gate["label"] = dataset_gate["dataset_id"].map(_short_dataset) + " | " + dataset_gate["claim_gate"].map(
        {
            "label_support_gate": "label",
            "local_neighbourhood_gate": "local",
            "global_geometry_gate": "global",
            "continuum_gate": "continuum",
            "donor_aware_gate": "donor",
        }
    )
    dataset_gate = dataset_gate.sort_values("pass_fraction", ascending=True)
    y = np.arange(dataset_gate.shape[0])
    colors = [DATASET_COLORS[x] for x in dataset_gate["dataset_id"]]
    ax.barh(y, dataset_gate["pass_fraction"], color=colors, alpha=0.82)
    for yi, row in enumerate(dataset_gate.itertuples(index=False)):
        ax.text(row.pass_fraction + 0.025, yi, f"{row.pass_fraction:.2f}\nn={int(row.n_cases)}", va="center", fontsize=4.5)
    ax.set_yticks(y)
    ax.set_yticklabels(dataset_gate["label"], fontsize=4.8)
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("pass fraction")
    ax.set_title("Dataset gate support", loc="left")
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[2, 1])
    gate_order = ["label_support_gate", "local_neighbourhood_gate", "global_geometry_gate", "continuum_gate", "donor_aware_gate"]
    gate_labels = ["label", "local", "global", "continuum", "donor"]
    method_gate = (
        gate_by_method.pivot(index="method", columns="claim_gate", values="pass_fraction")
        .reindex(index=config.ANCHOR_METHODS, columns=gate_order)
    )
    im = ax.imshow(method_gate.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(gate_order)))
    ax.set_xticklabels(gate_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(config.ANCHOR_METHODS)))
    ax.set_yticklabels(config.ANCHOR_METHODS, fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, n_methods=len(config.ANCHOR_METHODS))
    for i in range(method_gate.shape[0]):
        for j in range(method_gate.shape[1]):
            val = method_gate.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.4, color="white" if val > 0.72 else "#17213A")
    ax.set_title("Method gate support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.025)
    cbar.ax.set_ylabel("pass fraction", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "g")

    ax = fig.add_subplot(gs[2, 2])
    marker_map = {"pbmc3k": "o", "paul15": "^", "heart_cell_atlas_subsampled": "s"}
    for dataset_id, part in local_tradeoff.groupby("dataset_id", sort=False):
        for row in part.itertuples(index=False):
            ax.scatter(
                row.local_retention,
                row.label_neighbor_recall,
                s=33,
                marker=marker_map.get(dataset_id, "o"),
                color=style.family_color(row.method),
                edgecolor="white",
                linewidth=0.35,
                alpha=0.86,
            )
    ax.axvline(0.30, color="#777777", lw=0.7, ls="--")
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.set_xlim(0, max(0.45, float(local_tradeoff["local_retention"].max()) + 0.08))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("local retention")
    ax.set_ylabel("label recall")
    ax.set_title("Local-label trade-off", loc="left")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="PBMC3k"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="Paul15"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.2, label="Heart"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=4.4, frameon=False)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[3, 0:2])
    threshold_frames = []
    for source_name, frame in [
        ("local", local),
        ("global", global_df),
        ("continuum", continuum),
        ("donor", donor),
    ]:
        part = frame[pd.notna(frame["threshold"])].copy()
        part["source_layer"] = source_name
        threshold_frames.append(part)
    margin = pd.concat(threshold_frames, ignore_index=True)
    metric_label = {
        "local_retention": "local",
        "trustworthiness": "trust",
        "label_neighbor_recall": "label",
        "global_rank_corr": "global",
        "pseudotime_rank_corr": "pseudo rank",
        "pseudotime_neighborhood_retention": "pseudo local",
        "cell_type_label_recall": "cell label",
        "donor_entropy_norm": "donor entropy",
    }
    metric_order = [
        "local_retention",
        "trustworthiness",
        "label_neighbor_recall",
        "global_rank_corr",
        "pseudotime_rank_corr",
        "pseudotime_neighborhood_retention",
        "cell_type_label_recall",
        "donor_entropy_norm",
    ]
    margin["threshold_margin"] = margin["value"].astype(float) - margin["threshold"].astype(float)
    margin_summary = (
        margin.groupby(["family", "metric"], as_index=False)
        .agg(mean_threshold_margin=("threshold_margin", "mean"), n_rows=("threshold_margin", "size"))
        .pivot(index="family", columns="metric", values="mean_threshold_margin")
        .reindex(index=config.FAMILY_ORDER, columns=metric_order)
    )
    margin_out = margin.groupby(["family", "metric"], as_index=False).agg(
        mean_threshold_margin=("threshold_margin", "mean"),
        min_threshold_margin=("threshold_margin", "min"),
        max_threshold_margin=("threshold_margin", "max"),
        n_rows=("threshold_margin", "size"),
    )
    margin_out.to_csv(SOURCE_DIR / "fig4_threshold_margin_by_family_metric.csv", index=False)
    max_abs = max(0.15, float(np.nanmax(np.abs(margin_summary.values))))
    im = ax.imshow(margin_summary.values, vmin=-max_abs, vmax=max_abs, cmap="RdBu", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_label[m] for m in metric_order], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(config.FAMILY_ORDER)))
    ax.set_yticklabels([config.FAMILY_LABELS[f] for f in config.FAMILY_ORDER])
    style.color_family_ticklabels(ax, "y")
    for i in range(margin_summary.shape[0]):
        for j in range(margin_summary.shape[1]):
            val = margin_summary.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=4.4, color="white" if abs(val) > 0.45 * max_abs else "#222222")
    ax.set_title("Threshold margins", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.018, ticks=[-round(max_abs, 2), 0, round(max_abs, 2)])
    cbar.ax.set_ylabel("value - threshold", rotation=270, labelpad=10, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "i", x=-0.055)

    ax = fig.add_subplot(gs[3, 2])
    gate_order = ["label_support_gate", "local_neighbourhood_gate", "global_geometry_gate", "continuum_gate", "donor_aware_gate"]
    gate_labels = ["label", "local", "global", "continuum", "donor"]
    family_range = (
        gate_by_method.groupby(["family", "claim_gate"], as_index=False)
        .agg(
            min_pass_fraction=("pass_fraction", "min"),
            max_pass_fraction=("pass_fraction", "max"),
            n_methods=("method", "nunique"),
        )
        .assign(pass_fraction_range=lambda df: df["max_pass_fraction"] - df["min_pass_fraction"])
    )
    family_range.to_csv(SOURCE_DIR / "fig4_family_internal_gate_range.csv", index=False)
    range_mat = family_range.pivot(index="family", columns="claim_gate", values="pass_fraction_range").reindex(index=config.FAMILY_ORDER, columns=gate_order)
    im = ax.imshow(range_mat.values, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(gate_order)))
    ax.set_xticklabels(gate_labels, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(config.FAMILY_ORDER)))
    family_short = {"factor": "Factor", "deep": "Deep", "graph": "Graph", "relational": "Relational"}
    ax.set_yticklabels([family_short[f] for f in config.FAMILY_ORDER], fontsize=5.0)
    for label, family in zip(ax.get_yticklabels(), config.FAMILY_ORDER):
        label.set_color(style.FAMILY_COLORS[family])
    for i in range(range_mat.shape[0]):
        for j in range(range_mat.shape[1]):
            val = range_mat.iloc[i, j]
            if abs(float(val)) < 1e-12:
                ax.text(j, i, "0", ha="center", va="center", fontsize=4.7, color="#9A8A67")
            else:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.7, color="white" if val > 0.55 else "#4A1F1F")
    ax.set_title("Within-family spread", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.025)
    cbar.ax.set_ylabel("max-min pass fraction", rotation=270, labelpad=8, fontsize=5)
    _panel_label(ax, "j")
    _save(fig, "Figure_4")


def main() -> None:
    build_fig3()
    build_fig4()
    files = sorted(path for path in OUTPUT_DIR.glob("Figure_*.*") if path.stem in {"Figure_3", "Figure_4"})
    manifest = pd.DataFrame(
        {
            "relative_path": [str(path.relative_to(ROOT)) for path in files],
            "size_bytes": [path.stat().st_size for path in files],
        }
    )
    manifest.to_csv(OUTPUT_DIR / "Figure_3_4_output_manifest.csv", index=False)
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()



