"""Build source-data-backed main Figures 3 and 4."""

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
OUTPUT_DIR = ROOT / "outputs" / "main_figures"

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
            ax.set_title(method, color=_method_color(method), fontsize=6.7, pad=2)
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

    method_wide = pd.read_csv(SOURCE_DIR / "fig3_method_metric_wide.csv")
    metric_order = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
    metric_labels = {
        "local_retention": "local",
        "trustworthiness": "trust",
        "global_rank_corr": "global",
        "label_neighbor_recall": "same-label",
    }
    method_order = config.ANCHOR_METHODS
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
    REV_SOURCE_DIR.mkdir(parents=True, exist_ok=True)

    ax = fig.add_subplot(gs[3, 0:5])
    context_metric = (
        method_wide.groupby("dataset_id", as_index=False)[metric_order]
        .mean()
        .set_index("dataset_id")
        .reindex(dataset_order)
    )
    context_metric.to_csv(REV_SOURCE_DIR / "Fig3d_context_metric_means.csv")
    mat = context_metric[metric_order].values
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels([_short_dataset(x) for x in dataset_order], fontsize=5.6)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=5.0, color="white" if mat[i, j] < 0.25 else "#132A13")
    ax.set_title("Context-level diagnostic means", loc="left", pad=8)
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
    _color_method_labels(ax, "y")
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
        method_wide.groupby("method", as_index=False)[metric_order]
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
    _color_method_labels(ax, "y")
    for i in range(mm.shape[0]):
        for j in range(mm.shape[1]):
            val = mm.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.4, color="white" if val < 0.25 else "#132A13")
    ax.set_title("Method fingerprints", loc="left", pad=7)
    _panel_label(ax, "f")

    ax_g = fig.add_subplot(row4[0, 1])
    ax = ax_g
    between_method_range = (
        method_wide.groupby("dataset_id", as_index=False)[metric_order]
        .agg(lambda col: float(col.max() - col.min()))
        .set_index("dataset_id")
        .reindex(dataset_order)
    )
    between_method_range.to_csv(REV_SOURCE_DIR / "Fig3g_between_method_metric_range.csv")
    vmax_range = max(0.25, float(np.nanmax(between_method_range.values)))
    im = ax.imshow(between_method_range.values, vmin=0, vmax=vmax_range, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels([_short_dataset(x) for x in dataset_order], fontsize=5.2)
    for i in range(between_method_range.shape[0]):
        for j in range(between_method_range.shape[1]):
            val = float(between_method_range.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.7, color="white" if val > 0.72 * vmax_range else "#4A1F1F")
    ax.set_title("Between-method separation", loc="left", pad=7)
    _panel_label(ax, "g")

    ax_h = fig.add_subplot(row4[0, 2])
    ax = ax_h
    rank_vectors = {}
    for dataset_id in dataset_order:
        part = method_wide[method_wide["dataset_id"].eq(dataset_id)].set_index("method").reindex(method_order)
        ranked = part[metric_order].rank(axis=0, method="average", ascending=False)
        rank_vectors[dataset_id] = ranked.to_numpy().ravel()
    context_mat = pd.DataFrame(index=dataset_order, columns=dataset_order, dtype=float)
    for ds_a in dataset_order:
        for ds_b in dataset_order:
            context_mat.loc[ds_a, ds_b] = pd.Series(rank_vectors[ds_a]).corr(pd.Series(rank_vectors[ds_b]), method="spearman")
    context_mat.to_csv(REV_SOURCE_DIR / "Fig3h_context_rank_concordance.csv")
    im = ax.imshow(context_mat.values, vmin=-1, vmax=1, cmap="RdBu", aspect="equal")
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels([_short_dataset(x) for x in dataset_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(dataset_order)))
    ax.set_yticklabels([_short_dataset(x) for x in dataset_order], fontsize=5.2)
    for i in range(context_mat.shape[0]):
        for j in range(context_mat.shape[1]):
            val = float(context_mat.iloc[i, j])
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=4.9, color="white" if abs(val) > 0.72 else "#17213A")
    ax.set_title("Context rank concordance", loc="left", pad=7)
    cbar = fig.colorbar(im, ax=ax_h, fraction=0.045, pad=0.028, ticks=[-1, 0, 1])
    cbar.ax.set_ylabel("Spearman rho", rotation=270, labelpad=9, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[5, 0:4])
    context_range = (
        method_wide.groupby("method", as_index=False)[metric_order]
        .agg(lambda col: float(col.max() - col.min()))
        .set_index("method")
        .reindex(method_order)
    )
    context_range_out = context_range.reset_index()
    context_range_out.to_csv(REV_SOURCE_DIR / "Fig3i_method_context_metric_range.csv", index=False)
    cr = context_range[metric_order]
    context_range_vmax = max(0.25, float(np.nanmax(cr.values)))
    im = ax.imshow(cr.values, vmin=0, vmax=context_range_vmax, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_labels[m] for m in metric_order], rotation=28, ha="right")
    ax.set_yticks(np.arange(len(method_order)))
    ax.set_yticklabels(method_order, fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    _color_method_labels(ax, "y")
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
    tradeoff = method_wide[["dataset_id", "method", "local_retention", "global_rank_corr", "label_neighbor_recall", "trustworthiness"]].copy()
    tradeoff.to_csv(REV_SOURCE_DIR / "Fig3j_local_global_tradeoff.csv", index=False)
    markers = {"pbmc3k": "o", "paul15": "^", "heart_cell_atlas_subsampled": "s"}
    for dataset_id, part in tradeoff.groupby("dataset_id", sort=False):
        for row in part.itertuples(index=False):
            ax.scatter(
                row.global_rank_corr,
                row.local_retention,
                s=18 + 45 * float(row.label_neighbor_recall),
                marker=markers.get(dataset_id, "o"),
                color=_method_color(row.method),
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
    ax.text(0.02, 0.05, "point size: same-label fraction", transform=ax.transAxes, fontsize=4.8, color="#555555")
    _panel_label(ax, "j")

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=_method_color(method), markeredgecolor="white", markersize=4.2, label=method)
            for method in method_order
        ],
        loc="lower center",
        bbox_to_anchor=(0.50, 0.058),
        ncol=8,
        title="method",
        title_fontsize=5.4,
        fontsize=5.4,
        frameon=False,
    )
    _save(fig, "Figure_3", aliases=("Figure_3",))


def build_fig4() -> None:
    """Figure 4: claim-specific diagnostic gates."""
    _apply_pub_style()
    REV_SOURCE_DIR.mkdir(parents=True, exist_ok=True)
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
    _color_method_labels(ax, "y")
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
    _color_method_labels(ax, "y")
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
    _color_method_labels(ax, "x")
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
            color=_method_color(method),
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
            color=_method_color(method),
            arrowprops={"arrowstyle": "-", "lw": 0.35, "color": "#7A7A7A", "shrinkA": 1.5, "shrinkB": 2.0},
        )
    ax.axhline(0.55, color="#777777", ls="--", lw=0.8)
    ax.axvline(0.50, color="#777777", ls="--", lw=0.8)
    ax.set_xlim(0.37, 0.80)
    ax.set_ylim(0.15, 1.04)
    ax.set_xlabel("donor entropy (local kNN)")
    ax.set_ylabel("cell-type neighbour fraction")
    ax.set_title("Donor-aware gate")
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[1, 1:3])
    plotted_gate = gate[gate["claim_gate"].isin(["local_neighbourhood_gate", "global_geometry_gate", "continuum_gate", "donor_aware_gate", "label_support_gate"])].copy()
    plotted_gate["pass"] = plotted_gate["support"].eq("pass").astype(float)
    summary_counts = plotted_gate.groupby(["method", "dataset_id"], as_index=False).agg(
        pass_fraction=("pass", "mean"),
        pass_count=("pass", "sum"),
        n_cases=("pass", "size"),
    )
    summary_counts.to_csv(REV_SOURCE_DIR / "Fig4e_method_context_gate_breadth.csv", index=False)
    dataset_order = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
    summary = summary_counts.pivot(index="method", columns="dataset_id", values="pass_fraction").reindex(index=config.ANCHOR_METHODS, columns=dataset_order)
    summary_pass = summary_counts.pivot(index="method", columns="dataset_id", values="pass_count").reindex(index=config.ANCHOR_METHODS, columns=dataset_order)
    summary_n = summary_counts.pivot(index="method", columns="dataset_id", values="n_cases").reindex(index=config.ANCHOR_METHODS, columns=dataset_order)
    im = ax.imshow(summary.values, vmin=0, vmax=1, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(np.arange(len(dataset_order)))
    ax.set_xticklabels([_short_dataset(x) for x in dataset_order], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(config.ANCHOR_METHODS)))
    ax.set_yticklabels(config.ANCHOR_METHODS)
    _color_method_labels(ax, "y")
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
    ax.set_title("Method-context gate breadth")
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
    _color_method_labels(ax, "y")
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
                color=_method_color(row.method),
                edgecolor="white",
                linewidth=0.35,
                alpha=0.86,
            )
    ax.axvline(0.30, color="#777777", lw=0.7, ls="--")
    ax.axhline(0.55, color="#777777", lw=0.7, ls="--")
    ax.set_xlim(0, max(0.45, float(local_tradeoff["local_retention"].max()) + 0.08))
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("local retention")
    ax.set_ylabel("same-label neighbour fraction")
    ax.set_title("Local-identity trade-off", loc="left")
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
        "label_neighbor_recall": "same-label",
        "global_rank_corr": "global",
        "pseudotime_rank_corr": "pseudo rank",
        "pseudotime_neighborhood_retention": "pseudo local",
        "cell_type_label_recall": "cell identity",
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
        margin.groupby(["method", "metric"], as_index=False)
        .agg(mean_threshold_margin=("threshold_margin", "mean"), n_rows=("threshold_margin", "size"))
        .pivot(index="method", columns="metric", values="mean_threshold_margin")
        .reindex(index=config.ANCHOR_METHODS, columns=metric_order)
    )
    margin_out = margin.groupby(["method", "metric"], as_index=False).agg(
        mean_threshold_margin=("threshold_margin", "mean"),
        min_threshold_margin=("threshold_margin", "min"),
        max_threshold_margin=("threshold_margin", "max"),
        n_rows=("threshold_margin", "size"),
    )
    margin_out.to_csv(REV_SOURCE_DIR / "Fig4i_method_component_threshold_margins.csv", index=False)
    max_abs = max(0.15, float(np.nanmax(np.abs(margin_summary.values))))
    im = ax.imshow(margin_summary.values, vmin=-max_abs, vmax=max_abs, cmap="RdBu", aspect="auto")
    ax.set_xticks(np.arange(len(metric_order)))
    ax.set_xticklabels([metric_label[m] for m in metric_order], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(config.ANCHOR_METHODS)))
    ax.set_yticklabels(config.ANCHOR_METHODS)
    _color_method_labels(ax, "y")
    for i in range(margin_summary.shape[0]):
        for j in range(margin_summary.shape[1]):
            val = margin_summary.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=4.4, color="white" if abs(val) > 0.45 * max_abs else "#222222")
    ax.set_title("Boundary margins", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.026, pad=0.018, ticks=[-round(max_abs, 2), 0, round(max_abs, 2)])
    cbar.ax.set_ylabel("value - boundary", rotation=270, labelpad=10, fontsize=5)
    cbar.ax.tick_params(labelsize=4.8, pad=1)
    _panel_label(ax, "i", x=-0.055)

    ax = fig.add_subplot(gs[3, 2])
    method_range = gate_by_method.groupby("method", as_index=False).agg(
        minimum_gate_support=("pass_fraction", "min"),
        maximum_gate_support=("pass_fraction", "max"),
        n_gates=("claim_gate", "nunique"),
    )
    method_range["gate_support_range"] = method_range["maximum_gate_support"] - method_range["minimum_gate_support"]
    method_range = method_range.set_index("method").reindex(config.ANCHOR_METHODS).reset_index()
    method_range.to_csv(REV_SOURCE_DIR / "Fig4j_method_gate_inconsistency.csv", index=False)
    y = np.arange(len(method_range))
    ax.barh(y, method_range["gate_support_range"], color=[_method_color(m) for m in method_range["method"]], alpha=0.82)
    for yi, row in enumerate(method_range.itertuples(index=False)):
        ax.text(float(row.gate_support_range) + 0.02, yi, f"{float(row.gate_support_range):.2f}", va="center", fontsize=4.7)
    ax.set_yticks(y)
    ax.set_yticklabels(method_range["method"], fontsize=5.0)
    _color_method_labels(ax, "y")
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("max-min gate support")
    ax.set_title("Within-method gate inconsistency", loc="left")
    _panel_label(ax, "j")
    _save(fig, "Figure_4", aliases=("Figure_4",))


def main() -> None:
    build_fig3()
    build_fig4()

if __name__ == "__main__":
    main()
