"""Build the source-data-backed Supplementary Figure S1."""

from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data"
REV_SOURCE_DIR = SOURCE_DIR / "generated"
OUTPUT_DIR = ROOT / "outputs" / "supplementary_figures"

METHOD_COLORS = {
    method: mpl.colors.to_hex(mpl.colormaps["tab10"](i))
    for i, method in enumerate(config.ANCHOR_METHODS)
}


def _method_color(method: str) -> str:
    return METHOD_COLORS.get(method, "#3B6F95")


def _color_method_labels(ax, axis: str) -> None:
    labels = ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()
    for label in labels:
        label.set_color(_method_color(label.get_text()))

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


def _save(fig: plt.Figure, name: str, aliases: tuple[str, ...] = ()) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for target_name in (name, *aliases):
        for ext in ["pdf", "svg", "png", "jpg"]:
            kwargs = {"bbox_inches": "tight"}
            if ext in {"png", "jpg"}:
                kwargs["dpi"] = 450
            target = OUTPUT_DIR / f"{target_name}.{ext}"
            target.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(target, **kwargs)
    plt.close(fig)


def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _short_dataset(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def build_fig2() -> None:
    """Figure 2: data contexts and diagnostic burden."""
    _apply_pub_style()
    lock = pd.read_csv(SOURCE_DIR / "supp_s1_dataset_summary.csv")
    composition = pd.read_csv(SOURCE_DIR / "supp_s1_dataset_composition.csv")
    label_burden = pd.read_csv(SOURCE_DIR / "supp_s1_label_burden_metrics.csv")
    expression_sparsity = pd.read_csv(SOURCE_DIR / "supp_s1_expression_sparsity_metrics.csv")
    rare_burden = pd.read_csv(SOURCE_DIR / "supp_s1_rare_label_burden.csv")
    pca_structure = pd.read_csv(SOURCE_DIR / "supp_s1_pca_variance_structure.csv")
    pc_separability = pd.read_csv(SOURCE_DIR / "supp_s1_pc_label_separability.csv")
    cell_complexity = pd.read_csv(SOURCE_DIR / "supp_s1_cell_complexity_metrics.csv")
    detection_quantiles = pd.read_csv(SOURCE_DIR / "supp_s1_detected_gene_quantiles.csv")
    count_depth = pd.read_csv(SOURCE_DIR / "supp_s1_total_count_depth.csv")

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
    ax.set_xlabel("analysed cells")
    ax.set_ylabel("analysed genes")
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
    ax.bar(x - width / 2, sep["pc_label_neighbor_recall"], width, color="#4C78A8", alpha=0.82, label="PC same-label fraction")
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

    _save(fig, "Supplementary_Figure_S1_dataset_context")



def main() -> None:
    build_fig2()


if __name__ == "__main__":
    main()
