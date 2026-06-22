"""Generate source-data-backed Supplementary Figs. S3-S6 for publication."""

from __future__ import annotations

from pathlib import Path
from string import ascii_lowercase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import config
from . import final_figure_style as fs


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data"
OUT_DIR = ROOT / "outputs" / "supplementary_figures"
META_DIR = ROOT / "metadata"
EXPORT_DPI = 600

METHOD_ORDER = config.ANCHOR_METHODS
FAMILY_ORDER = config.FAMILY_ORDER
DATASET_ORDER = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
DATASET_LABELS = {
    "pbmc3k": "PBMC3k",
    "paul15": "Paul15",
    "heart_cell_atlas_subsampled": "Heart atlas",
}
DATASET_COLORS = {
    "pbmc3k": "#4C78A8",
    "paul15": "#59A14F",
    "heart_cell_atlas_subsampled": "#A779A7",
}
GATE_ORDER = [
    "label_support_gate",
    "local_neighbourhood_gate",
    "global_geometry_gate",
    "continuum_gate",
    "donor_aware_gate",
]
GATE_LABELS = {
    "label_support_gate": "label",
    "local_neighbourhood_gate": "local",
    "global_geometry_gate": "global",
    "continuum_gate": "continuum",
    "donor_aware_gate": "donor",
}
METRIC_LABELS = {
    "local_retention": "local",
    "trustworthiness": "trust",
    "global_rank_corr": "global",
    "label_neighbor_recall": "label",
    "pseudotime_rank_corr": "pseudo rank",
    "pseudotime_neighborhood_retention": "pseudo local",
    "cell_type_label_recall": "cell recall",
    "donor_entropy_norm": "donor entropy",
    "donor_dominance": "donor dominance",
    "latent_distance_corr": "latent dist.",
    "rare_label_recall": "rare recall",
}


def _read(name: str) -> pd.DataFrame:
    path = SOURCE_DIR / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _save(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png", "jpg"]:
        fig.savefig(OUT_DIR / f"{stem}.{ext}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white")
    plt.close(fig)


def _apply_hq_style() -> None:
    fs.apply_final_style()
    mpl.rcParams.update(
        {
            "font.size": 8.0,
            "axes.labelsize": 8.2,
            "axes.titlesize": 9.2,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "figure.dpi": 180,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def _panel_label(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.06) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=12.5, fontweight="bold", clip_on=False)


def _family_color_for_method(method: str) -> str:
    return fs.FAMILY_COLORS[config.METHOD_FAMILY[method]]


def _short_label(value: str) -> str:
    replacements = {
        "discrete_cell_identity": "discrete identity",
        "trajectory_continuum": "continuum",
        "donor_variable_tissue_identity": "donor tissue",
        "rare_state_support": "rare state",
    }
    text = str(value)
    return replacements.get(text, fs.short_label(text))


def _ratio_heatmap(
    ax: plt.Axes,
    data: pd.DataFrame,
    *,
    title: str,
    cmap: str = "YlGnBu",
    vmin: float = 0,
    vmax: float = 1.6,
    cbar: bool = False,
    cbar_label: str = "value / threshold",
    annotate: bool = True,
):
    masked = np.ma.masked_invalid(data.to_numpy(dtype=float))
    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad("#F1F3F5")
    im = ax.imshow(masked, cmap=cmap_obj, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, loc="left")
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_xticklabels([_short_label(c) for c in data.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(data.shape[0]))
    ax.set_yticklabels([_short_label(i) for i in data.index])
    ax.tick_params(length=0, pad=1)
    if list(data.index) == METHOD_ORDER:
        for tick in ax.get_yticklabels():
            tick.set_color(_family_color_for_method(tick.get_text()))
    if annotate and data.size <= 100:
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data.iloc[i, j]
                if pd.isna(value):
                    ax.text(j, i, "n.a.", ha="center", va="center", fontsize=5.3, color="#6B7280")
                else:
                    color = "white" if value >= (vmax * 0.66) else fs.INK
                    ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=5.6, color=color)
    if cbar:
        cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
        cb.set_label(cbar_label, fontsize=7)
        cb.ax.tick_params(labelsize=6)
    return im


def _fraction_heatmap(ax: plt.Axes, data: pd.DataFrame, *, title: str, cbar: bool = False):
    im = _ratio_heatmap(
        ax,
        data,
        title=title,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        cbar=cbar,
        cbar_label="support fraction",
        annotate=True,
    )
    return im


def _style_axes(ax: plt.Axes, grid: bool = False) -> None:
    fs.clean_axis(ax, grid=grid)


def _embedding_colors(labels: pd.Series) -> dict[str, str]:
    categories = labels.astype(str).value_counts().index.tolist()
    palette = list(mpl.colormaps["tab20"].colors) + list(mpl.colormaps["tab20b"].colors)
    return {cat: mpl.colors.to_hex(palette[i % len(palette)]) for i, cat in enumerate(categories)}


def supplementary_fig_s3() -> None:
    """Full embedding atlas across three empirical contexts."""
    _apply_hq_style()
    files = {
        "pbmc3k": "fig3_pbmc3k_embedding_coordinates.csv",
        "paul15": "fig3_paul15_embedding_coordinates.csv",
        "heart_cell_atlas_subsampled": "fig3_heart_embedding_coordinates.csv",
    }
    data = {dataset: _read(path) for dataset, path in files.items()}
    colors = {dataset: _embedding_colors(df["label"]) for dataset, df in data.items()}

    fig = plt.figure(figsize=(18.5, 9.4))
    gs = fig.add_gridspec(3, 9, width_ratios=[1, 1, 1, 1, 1, 1, 1, 1, 1.52], hspace=0.26, wspace=0.055)
    axes = []
    for row, dataset in enumerate(DATASET_ORDER):
        df = data[dataset]
        row_axes = []
        for col, method in enumerate(METHOD_ORDER):
            ax = fig.add_subplot(gs[row, col])
            sub = df[df["method"].eq(method)]
            if sub.empty:
                raise ValueError(f"No embedding rows for {dataset} {method}")
            for label, lab_df in sub.groupby("label", sort=False):
                ax.scatter(lab_df["x"], lab_df["y"], s=2.0, lw=0, alpha=0.74, color=colors[dataset][str(label)], rasterized=True)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(method if row == 0 else "", color=_family_color_for_method(method), fontsize=8.6, pad=2)
            if col == 0:
                ax.set_ylabel(DATASET_LABELS[dataset], fontsize=9.2, fontweight="bold", color=DATASET_COLORS[dataset], labelpad=9)
                _panel_label(ax, ascii_lowercase[row], x=-0.23, y=1.04)
            for spine in ax.spines.values():
                spine.set_linewidth(0.35)
                spine.set_color("#C8CDD4")
            row_axes.append(ax)
        axes.append(row_axes)

        leg_ax = fig.add_subplot(gs[row, 8])
        leg_ax.axis("off")
        leg_ax.text(0, 1.0, f"{DATASET_LABELS[dataset]} labels", ha="left", va="top", fontsize=7.2, fontweight="bold")
        labels = list(colors[dataset].keys())
        ncol = 1 if len(labels) <= 12 else 2
        y0 = 0.90
        step = 0.072 if len(labels) <= 12 else 0.095
        per_col = int(np.ceil(len(labels) / ncol))
        for idx, label in enumerate(labels):
            c = idx // per_col
            r = idx % per_col
            x = c * 0.48
            y = y0 - r * step
            leg_ax.scatter([x], [y], s=16, color=colors[dataset][label], lw=0)
            leg_ax.text(x + 0.035, y, _short_label(label), fontsize=5.6, va="center", ha="left")
        leg_ax.set_xlim(0, 1)
        leg_ax.set_ylim(0, 1)

    family_handles = [
        Line2D([0], [0], color=fs.FAMILY_COLORS[fam], lw=2.5, label=fs.family_label(fam)) for fam in FAMILY_ORDER
    ]
    fig.legend(handles=family_handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.49, 0.005), title="method family", title_fontsize=7.2, fontsize=6.8)
    _save(fig, "Supplementary_Figure_S3_full_embedding_atlas")


def supplementary_fig_s4() -> None:
    """Diagnostic-gate support and metric-margin atlas."""
    _apply_hq_style()
    method_gate = _read("fig4_gate_pass_by_method.csv")
    family_gate = _read("fig4_gate_pass_by_family.csv")
    local = _read("fig4_local_gate_metrics.csv")
    global_df = _read("fig4_global_gate_metrics.csv")
    continuum = _read("fig4_paul15_continuum_metrics.csv")
    donor = _read("fig4_heart_donor_gate_metrics.csv")
    margins = _read("fig4_threshold_margin_by_family_metric.csv")
    spread = _read("fig4_family_internal_gate_range.csv")

    method_gate_mat = method_gate.pivot(index="method", columns="claim_gate", values="pass_fraction").reindex(index=METHOD_ORDER, columns=GATE_ORDER)
    method_gate_mat.columns = [GATE_LABELS[c] for c in method_gate_mat.columns]
    family_gate_mat = family_gate.pivot(index="family", columns="claim_gate", values="pass_fraction").reindex(index=FAMILY_ORDER, columns=GATE_ORDER)
    family_gate_mat.index = [fs.family_label(i) for i in family_gate_mat.index]
    family_gate_mat.columns = [GATE_LABELS[c] for c in family_gate_mat.columns]

    local_ratio = local.assign(ratio=lambda d: d["value"] / d["threshold"]).pivot_table(index="method", columns="metric", values="ratio", aggfunc="mean").reindex(METHOD_ORDER)
    local_ratio = local_ratio[[c for c in ["local_retention", "trustworthiness", "label_neighbor_recall"] if c in local_ratio.columns]]
    local_ratio.columns = [METRIC_LABELS[c] for c in local_ratio.columns]

    global_ratio = global_df.assign(ratio=lambda d: d["value"] / d["threshold"]).pivot_table(index="method", columns="dataset_id", values="ratio", aggfunc="mean").reindex(index=METHOD_ORDER, columns=DATASET_ORDER)
    global_ratio.columns = [DATASET_LABELS[c] for c in global_ratio.columns]

    continuum_ratio = continuum.assign(ratio=lambda d: d["value"] / d["threshold"]).pivot(index="method", columns="metric", values="ratio").reindex(METHOD_ORDER)
    continuum_ratio.columns = [METRIC_LABELS[c] for c in continuum_ratio.columns]

    donor_ratio = donor[donor["metric"].isin(["cell_type_label_recall", "donor_entropy_norm"])].assign(ratio=lambda d: d["value"] / d["threshold"]).pivot(index="method", columns="metric", values="ratio").reindex(METHOD_ORDER)
    donor_ratio.columns = [METRIC_LABELS[c] for c in donor_ratio.columns]

    margin_mat = margins.pivot(index="family", columns="metric", values="mean_threshold_margin").reindex(FAMILY_ORDER)
    margin_mat.index = [fs.family_label(i) for i in margin_mat.index]
    margin_mat = margin_mat[[c for c in ["local_retention", "trustworthiness", "label_neighbor_recall", "global_rank_corr", "pseudotime_rank_corr", "pseudotime_neighborhood_retention", "cell_type_label_recall", "donor_entropy_norm"] if c in margin_mat.columns]]
    margin_mat.columns = [METRIC_LABELS[c] for c in margin_mat.columns]

    spread_mat = spread.pivot(index="family", columns="claim_gate", values="pass_fraction_range").reindex(index=FAMILY_ORDER, columns=GATE_ORDER)
    spread_mat.index = [fs.family_label(i) for i in spread_mat.index]
    spread_mat.columns = [GATE_LABELS[c] for c in spread_mat.columns]

    fig, axes = plt.subplots(2, 4, figsize=(17.2, 9.4), gridspec_kw={"width_ratios": [1.36, 1.04, 0.90, 0.90]}, constrained_layout=False)
    plt.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.10, wspace=0.56, hspace=0.62)
    panels = [
        (axes[0, 0], method_gate_mat, "Method gate support", "fraction", True, "fraction"),
        (axes[0, 1], family_gate_mat, "Family gate support", "fraction", False, "fraction"),
        (axes[0, 2], local_ratio, "Local-gate ratios", "ratio", False, "ratio"),
        (axes[0, 3], global_ratio, "Global-rank ratios", "ratio", False, "ratio"),
        (axes[1, 0], continuum_ratio, "Paul15 continuum ratios", "ratio", False, "ratio"),
        (axes[1, 1], donor_ratio, "Heart donor-aware ratios", "ratio", False, "ratio"),
        (axes[1, 2], margin_mat, "Family threshold margins", "margin", True, "margin"),
        (axes[1, 3], spread_mat, "Within-family gate spread", "fraction", True, "fraction"),
    ]
    for idx, (ax, mat, title, mode, cbar, _) in enumerate(panels):
        if mode == "fraction":
            _fraction_heatmap(ax, mat, title=title, cbar=cbar)
        elif mode == "margin":
            _ratio_heatmap(ax, mat, title=title, cmap="RdBu_r", vmin=-0.45, vmax=0.45, cbar=cbar, cbar_label="mean value - threshold")
        else:
            _ratio_heatmap(ax, mat, title=title, cbar=cbar, cbar_label="value / threshold")
        _panel_label(ax, ascii_lowercase[idx])
    _save(fig, "Supplementary_Figure_S4_diagnostic_gate_metric_atlas")


def supplementary_fig_s5() -> None:
    """Biological-anchor evidence atlas."""
    _apply_hq_style()
    pbmc = _read("fig5_pbmc_marker_support.csv")
    marker_vs = _read("fig5_marker_vs_neighbour_support.csv")
    family_marker = _read("fig5_family_marker_support_summary.csv")
    paul = _read("fig5_paul15_lineage_marker_support.csv")
    heart = _read("fig5_heart_identity_donor_support.csv")
    rare = _read("fig5_rare_state_support.csv")
    rare_family = _read("fig5_rare_support_by_family.csv")
    summary = _read("fig5_evidence_support_summary.csv")

    labels = pbmc.drop_duplicates("label").sort_values("max_marker_z", ascending=False)
    recall_mat = marker_vs.pivot(index="method", columns="label", values="label_knn_recall").reindex(index=METHOD_ORDER, columns=labels["label"])
    recall_mat.columns = [_short_label(c) for c in recall_mat.columns]
    dual_mat = family_marker.pivot(index="family", columns="label", values="dual_support_fraction").reindex(index=FAMILY_ORDER, columns=labels["label"])
    dual_mat.index = [fs.family_label(i) for i in dual_mat.index]
    dual_mat.columns = [_short_label(c) for c in dual_mat.columns]

    rare["dataset_label"] = rare["dataset_id"].map(DATASET_LABELS).fillna(rare["dataset_id"])
    rare["rare_axis"] = rare["dataset_label"] + "\n" + rare["label"].map(_short_label)
    rare_order = rare.groupby("rare_axis")["rare_label_fraction"].mean().sort_values(ascending=False).index.tolist()
    rare_mat = rare.pivot_table(index="method", columns="rare_axis", values="label_knn_recall", aggfunc="mean").reindex(index=METHOD_ORDER, columns=rare_order)
    rare_family_mat = rare_family.pivot(index="family", columns="dataset_id", values="rare_pass_fraction").reindex(FAMILY_ORDER)
    rare_family_mat.index = [fs.family_label(i) for i in rare_family_mat.index]
    rare_family_mat.columns = [DATASET_LABELS.get(c, c) for c in rare_family_mat.columns]

    fig = plt.figure(figsize=(17.5, 9.8))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.08, 1.50, 1.18, 1.18], hspace=0.56, wspace=0.72)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(4)]

    ax = axes[0]
    ax.barh(np.arange(len(labels)), labels["max_marker_z"], color="#6B8FB9")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels([_short_label(x) for x in labels["label"]])
    ax.invert_yaxis()
    ax.set_xlabel("max marker z-score")
    ax.set_title("PBMC marker-program support", loc="left")
    _style_axes(ax, grid=True)
    _panel_label(ax, "a")

    _ratio_heatmap(axes[1], recall_mat, title="PBMC label-neighbour recall", cmap="YlGnBu", vmin=0, vmax=1, cbar=True, cbar_label="label kNN recall", annotate=False)
    _panel_label(axes[1], "b")

    _fraction_heatmap(axes[2], dual_mat, title="Family dual support", cbar=True)
    _panel_label(axes[2], "c")

    ax = axes[3]
    paul_sorted = paul.assign(program_label=lambda d: d["program"].map(_short_label)).sort_values("value")
    colors = ["#D1495B" if abs(v) < 0.25 else "#238B68" for v in paul_sorted["value"]]
    ax.barh(np.arange(len(paul_sorted)), paul_sorted["value"], color=colors)
    ax.axvline(0, color=fs.BORDER, lw=0.7)
    ax.axvline(0.25, color=fs.THRESHOLD_COLOR, lw=0.8, ls="--")
    ax.axvline(-0.25, color=fs.THRESHOLD_COLOR, lw=0.8, ls="--")
    ax.set_yticks(np.arange(len(paul_sorted)))
    ax.set_yticklabels(paul_sorted["program_label"])
    ax.set_xlabel("Spearman rho")
    ax.set_title("Paul15 lineage-anchor trend", loc="left")
    _style_axes(ax, grid=True)
    _panel_label(ax, "d")

    ax = axes[4]
    offsets = {
        "PCA": (0.010, 0.014),
        "GLM-PCA": (0.012, 0.016),
        "SAUCIE": (0.006, -0.018),
        "scScope": (0.010, 0.012),
        "UMAP": (-0.075, 0.012),
        "PHATE": (0.010, 0.012),
        "t-SNE": (0.010, -0.020),
        "PaCMAP": (0.010, -0.020),
    }
    for _, row in heart.iterrows():
        ax.scatter(row["cell_type_label_recall"], row["donor_entropy_norm"], s=42, color=_family_color_for_method(row["method"]), edgecolor="white", lw=0.4)
        dx, dy = offsets.get(row["method"], (0.01, 0.006))
        ax.text(row["cell_type_label_recall"] + dx, row["donor_entropy_norm"] + dy, row["method"], fontsize=5.8)
    ax.axvline(0.55, color=fs.THRESHOLD_COLOR, lw=0.8, ls="--")
    ax.axhline(0.50, color=fs.THRESHOLD_COLOR, lw=0.8, ls="--")
    ax.set_xlabel("cell-type recall")
    ax.set_ylabel("donor entropy")
    ax.set_title("Heart identity versus donor mixing", loc="left")
    _style_axes(ax, grid=True)
    _panel_label(ax, "e")

    _ratio_heatmap(axes[5], rare_mat, title="Rare-state recall by method", cmap="YlGnBu", vmin=0, vmax=1, cbar=True, cbar_label="rare-label recall", annotate=False)
    axes[5].tick_params(axis="x", labelsize=5.4)
    _panel_label(axes[5], "f")

    _fraction_heatmap(axes[6], rare_family_mat, title="Rare support by family", cbar=True)
    _panel_label(axes[6], "g")

    ax = axes[7]
    summary_order = summary.sort_values("support_summary", ascending=True)
    bars = ax.barh(np.arange(len(summary_order)), summary_order["support_summary"], color="#7DA7B8")
    ax.set_yticks(np.arange(len(summary_order)))
    ax.set_yticklabels([_short_label(x) for x in summary_order["claim_type"]])
    ax.set_xlim(0, 1)
    ax.set_xlabel("support fraction")
    ax.set_title("Independent-anchor synthesis", loc="left")
    for bar, val in zip(bars, summary_order["support_summary"]):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=6.2)
    _style_axes(ax, grid=True)
    _panel_label(ax, "h")

    _save(fig, "Supplementary_Figure_S5_biological_anchor_evidence_atlas")


def supplementary_fig_s6() -> None:
    """Robustness response atlas."""
    _apply_hq_style()
    output_dim = _read("fig6_output_dimension_response.csv")
    upstream = _read("fig6_upstream_pca_response.csv")
    dropout_noise = _read("fig6_dropout_noise_response.csv")
    dim_dataset = _read("fig6_dimension_failure_by_dataset_metric.csv")
    dim_method = _read("fig6_dimension_failure_by_method_metric.csv")
    upstream_fail = _read("fig6_upstream_failure_by_method_metric.csv")
    pert_fail = _read("fig6_perturbation_failure_by_metric.csv")
    worst = _read("fig6_worst_case_support.csv")

    metrics = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
    metric_colors = {
        "local_retention": "#4C78A8",
        "trustworthiness": "#F58518",
        "global_rank_corr": "#54A24B",
        "label_neighbor_recall": "#B279A2",
    }

    fig = plt.figure(figsize=(17.2, 9.8))
    gs = fig.add_gridspec(3, 3, hspace=0.68, wspace=0.50)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

    ax = axes[0]
    od = output_dim[output_dim["metric"].isin(metrics)].groupby(["output_dimension", "metric"], as_index=False)["value"].mean()
    for metric in metrics:
        sub = od[od["metric"].eq(metric)]
        ax.plot(sub["output_dimension"], sub["value"], marker="o", ms=3.5, color=metric_colors[metric], label=METRIC_LABELS[metric])
    ax.set_xlabel("output dimension")
    ax.set_ylabel("mean diagnostic score")
    ax.set_title("Output-dimension response", loc="left")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    _style_axes(ax, grid=True)
    _panel_label(ax, "a")

    mat = dim_dataset.pivot(index="dataset_id", columns="metric", values="failure_fraction").reindex(index=DATASET_ORDER, columns=metrics)
    mat.index = [DATASET_LABELS[i] for i in mat.index]
    mat.columns = [METRIC_LABELS[c] for c in mat.columns]
    _fraction_heatmap(axes[1], mat, title="Dimension failure by dataset", cbar=True)
    _panel_label(axes[1], "b")

    mat = dim_method.pivot(index="method", columns="metric", values="failure_fraction").reindex(index=[m for m in METHOD_ORDER if m in dim_method["method"].unique()], columns=metrics)
    mat.columns = [METRIC_LABELS[c] for c in mat.columns]
    _fraction_heatmap(axes[2], mat, title="Dimension failure by method", cbar=True)
    _panel_label(axes[2], "c")

    ax = axes[3]
    up = upstream[upstream["metric"].isin(metrics)].groupby(["upstream_pca_dimension", "metric"], as_index=False)["value"].mean()
    for metric in metrics:
        sub = up[up["metric"].eq(metric)]
        ax.plot(sub["upstream_pca_dimension"], sub["value"], marker="o", ms=3.5, color=metric_colors[metric], label=METRIC_LABELS[metric])
    ax.set_xlabel("upstream PCA dimension")
    ax.set_ylabel("mean diagnostic score")
    ax.set_title("Upstream-representation response", loc="left")
    _style_axes(ax, grid=True)
    _panel_label(ax, "d")

    mat = upstream_fail.pivot(index="method", columns="metric", values="failure_fraction").reindex(index=[m for m in METHOD_ORDER if m in upstream_fail["method"].unique()], columns=metrics)
    mat.columns = [METRIC_LABELS[c] for c in mat.columns]
    _fraction_heatmap(axes[4], mat, title="Upstream-PCA failure by method", cbar=True)
    _panel_label(axes[4], "e")

    ax = axes[5]
    dn = dropout_noise[dropout_noise["metric"].isin(metrics)].groupby(["perturbation", "level", "metric"], as_index=False)["value"].mean()
    for perturbation, ls in [("dropout", "-"), ("noise", "--")]:
        for metric in metrics:
            sub = dn[dn["perturbation"].eq(perturbation) & dn["metric"].eq(metric)]
            label = f"{METRIC_LABELS[metric]} ({perturbation})"
            ax.plot(sub["level"], sub["value"], marker="o", ms=2.8, lw=0.8, ls=ls, color=metric_colors[metric], label=label)
    ax.set_xlabel("perturbation level")
    ax.set_ylabel("mean diagnostic score")
    ax.set_title("Dropout/noise response", loc="left")
    _style_axes(ax, grid=True)
    _panel_label(ax, "f")

    mat = pert_fail.pivot(index="metric", columns="perturbation", values="failure_fraction").reindex(metrics)
    mat.index = [METRIC_LABELS[i] for i in mat.index]
    _fraction_heatmap(axes[6], mat, title="Perturbation failure by metric", cbar=True)
    _panel_label(axes[6], "g")

    worst_ratio = worst[worst["metric"].isin(metrics + ["latent_distance_corr", "rare_label_recall"])].assign(ratio=lambda d: d["worst_value"] / d["threshold"])
    available_worst_methods = [m for m in METHOD_ORDER if m in set(worst_ratio["method"])]
    metric_order = [
        metric
        for metric in ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall", "latent_distance_corr", "rare_label_recall"]
        if metric in set(worst_ratio["metric"])
    ]
    mat = worst_ratio.pivot(index="method", columns="metric", values="ratio").reindex(index=available_worst_methods, columns=metric_order)
    mat.columns = [METRIC_LABELS.get(c, c) for c in mat.columns]
    _ratio_heatmap(axes[7], mat, title="Worst-case support ratio", cbar=True, cbar_label="worst value / threshold")
    _panel_label(axes[7], "h")

    ax = axes[8]
    support_counts = worst.groupby(["method", "support"]).size().unstack(fill_value=0).reindex(available_worst_methods).fillna(0)
    passed = support_counts.get("pass", pd.Series(0, index=support_counts.index))
    failed = support_counts.get("below_threshold", pd.Series(0, index=support_counts.index))
    y = np.arange(len(support_counts))
    ax.barh(y, passed, color="#238B68", label="pass")
    ax.barh(y, failed, left=passed, color="#D1495B", label="below threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(support_counts.index)
    for tick in ax.get_yticklabels():
        tick.set_color(_family_color_for_method(tick.get_text()))
    ax.invert_yaxis()
    ax.set_xlabel("worst-case metrics")
    ax.set_title("Worst-case pass/fail burden", loc="left")
    ax.legend(frameon=False, fontsize=6.2, loc="upper right", bbox_to_anchor=(1.02, 1.18), ncol=2)
    _style_axes(ax, grid=True)
    _panel_label(ax, "i")

    _save(fig, "Supplementary_Figure_S6_robustness_response_atlas")


def build_supplementary_matrix() -> None:
    rows = [
        ("S3", "a-c", "Full embeddings for PBMC3k, Paul15 and heart atlas", "D2", "fig3_pbmc3k_embedding_coordinates.csv; fig3_paul15_embedding_coordinates.csv; fig3_heart_embedding_coordinates.csv", "figures/plot_supplementary_figures_s3_s6.py"),
        ("S4", "a-h", "Diagnostic-gate support, component ratios and margins", "D2", "fig4_gate_pass_by_method.csv; fig4_gate_pass_by_family.csv; fig4_local_gate_metrics.csv; fig4_global_gate_metrics.csv; fig4_paul15_continuum_metrics.csv; fig4_heart_donor_gate_metrics.csv; fig4_threshold_margin_by_family_metric.csv; fig4_family_internal_gate_range.csv", "figures/plot_supplementary_figures_s3_s6.py"),
        ("S5", "a-h", "Independent biological-anchor evidence", "D2", "fig5_pbmc_marker_support.csv; fig5_marker_vs_neighbour_support.csv; fig5_paul15_lineage_marker_support.csv; fig5_heart_identity_donor_support.csv; fig5_rare_state_support.csv; fig5_rare_support_by_family.csv; fig5_evidence_support_summary.csv", "figures/plot_supplementary_figures_s3_s6.py"),
        ("S6", "a-i", "Robustness response and worst-case support", "D2/D4", "fig6_output_dimension_response.csv; fig6_upstream_pca_response.csv; fig6_dropout_noise_response.csv; fig6_dimension_failure_by_dataset_metric.csv; fig6_dimension_failure_by_method_metric.csv; fig6_upstream_failure_by_method_metric.csv; fig6_perturbation_failure_by_metric.csv; fig6_worst_case_support.csv", "figures/plot_supplementary_figures_s3_s6.py"),
    ]
    META_DIR.mkdir(parents=True, exist_ok=True)
    out = META_DIR / "supplementary_figure_claim_to_evidence_matrix.tsv"
    pd.DataFrame(rows, columns=["figure", "panel_range", "main_evidence_role", "data_source_class", "source_data_files", "plotting_code"]).to_csv(out, sep="\t", index=False)


def main() -> None:
    supplementary_fig_s3()
    supplementary_fig_s4()
    supplementary_fig_s5()
    supplementary_fig_s6()
    build_supplementary_matrix()


if __name__ == "__main__":
    main()


