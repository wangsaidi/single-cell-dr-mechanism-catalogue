"""Rebuild Supplementary Figs S7-S8 from publication source tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .figure_style import (
    BORDER,
    DATASET_COLORS,
    DATASET_LABELS,
    FAIL_COLOR,
    FAMILY_COLORS,
    FAMILY_LABELS,
    FAMILY_ORDER,
    INK,
    METHOD_ORDER,
    PASS_COLOR,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    method_family,
    panel_label,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data" / "generated"
OUTPUT_DIR = ROOT / "outputs" / "supplementary_figures"

METHODS = list(METHOD_ORDER)
DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
BOUNDARY_SHIFTS = [-0.05, -0.025, 0.0, 0.025, 0.05]
AUDIT_SEED = 20260714
GATE_ORDER = [
    "label_support_gate",
    "local_neighbourhood_gate",
    "global_geometry_gate",
    "continuum_gate",
    "donor_aware_gate",
]
GATE_LABELS = {
    "label_support_gate": "Label",
    "local_neighbourhood_gate": "Local",
    "global_geometry_gate": "Global",
    "continuum_gate": "Continuum",
    "donor_aware_gate": "Donor-aware",
}
METRICS = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
METRIC_LABELS = {
    "local_retention": "Local retention",
    "trustworthiness": "Trustworthiness",
    "global_rank_corr": "Global rank",
    "label_neighbor_recall": "Label recall",
}


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png", "jpg"):
        path = OUTPUT_DIR / f"{stem}.{ext}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "jpg"}:
            kwargs["dpi"] = 600
        if ext == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(path, **kwargs)
    plt.close(fig)


def plot_s7() -> None:
    decisions = pd.read_csv(SOURCE_DIR / "SuppS7_boundary_sensitivity_decisions.csv")
    family_summary = pd.read_csv(SOURCE_DIR / "SuppS7_boundary_sensitivity_family_summary.csv")
    method_summary = pd.read_csv(SOURCE_DIR / "SuppS7_boundary_sensitivity_method_summary.csv")
    gate_stability = pd.read_csv(SOURCE_DIR / "SuppS7_boundary_sensitivity_gate_summary.csv")
    change_counts = pd.read_csv(SOURCE_DIR / "SuppS7_boundary_sensitivity_changes.csv")

    apply_final_style()
    fig = plt.figure(figsize=(7.2, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, left=0.10, right=0.98, bottom=0.075, top=0.975, wspace=0.36, hspace=0.46)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    axa, axb, axc, axd, axe, axf = axes

    for family in FAMILY_ORDER:
        sub = family_summary[family_summary["family"].eq(family)].sort_values("boundary_shift")
        axa.plot(sub["boundary_shift"], sub["pass_fraction"], marker="o", ms=3.8,
                 color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
    axa.axvline(0, color=TEXT_MUTED, lw=0.8, ls="--")
    axa.set(xlabel="Boundary shift", ylabel="Gate pass fraction", title="Navigation-group support", ylim=(-0.02, 1.02))
    axa.legend(frameon=False, loc="upper right", handlelength=1.5)
    clean_axis(axa)

    for method, marker in zip(METHODS, ["o", "s", "^", "D", "P", "v", "X", ">"]):
        sub = method_summary[method_summary["method"].eq(method)].sort_values("boundary_shift")
        axb.plot(sub["boundary_shift"], sub["pass_fraction"], marker=marker, ms=3.2,
                 color=FAMILY_COLORS[method_family(method)], alpha=0.92, label=method)
    axb.axvline(0, color=TEXT_MUTED, lw=0.8, ls="--")
    axb.set(xlabel="Boundary shift", ylabel="Gate pass fraction", title="Method support", ylim=(-0.02, 1.02))
    axb.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.8, handletextpad=0.4)
    clean_axis(axb)

    baseline = decisions[np.isclose(decisions["boundary_shift"], 0.0)].copy()
    row_order = [
        (dataset, gate)
        for dataset in DATASETS
        for gate in GATE_ORDER
        if ((baseline["dataset_id"] == dataset) & (baseline["claim_gate"] == gate)).any()
    ]
    pivot = baseline.assign(row_key=baseline["dataset_id"] + "|" + baseline["claim_gate"]).pivot(
        index="row_key", columns="method", values="limiting_margin"
    )
    arr = pivot.reindex(index=[f"{d}|{g}" for d, g in row_order], columns=METHODS).to_numpy(float)
    vmax = max(0.05, float(np.nanpercentile(np.abs(arr), 95)))
    im = axc.imshow(arr, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    axc.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right")
    axc.set_yticks(range(len(row_order)), [f"{DATASET_LABELS[d]}  {GATE_LABELS[g]}" for d, g in row_order])
    axc.set(xlabel="Method", ylabel="Dataset and claim gate", title="Baseline margins")
    cbar = fig.colorbar(im, ax=axc, fraction=0.045, pad=0.025)
    cbar.set_label("Limiting value minus boundary")
    cbar.ax.tick_params(labelsize=5.8)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            if np.isfinite(arr[y, x]):
                axc.text(x, y, f"{arr[y, x]:.2f}", ha="center", va="center", fontsize=4.2,
                         color="white" if abs(arr[y, x]) > vmax * 0.55 else INK)
    for spine in axc.spines.values():
        spine.set_visible(False)
    axc.tick_params(length=0)

    gate_colors = {
        "label_support_gate": "#4477AA", "local_neighbourhood_gate": "#EE6677",
        "global_geometry_gate": "#228833", "continuum_gate": "#CCBB44", "donor_aware_gate": "#AA3377",
    }
    for gate in GATE_ORDER:
        sub = gate_stability[gate_stability["claim_gate"].eq(gate)].sort_values("boundary_shift")
        axd.plot(sub["boundary_shift"], sub["unchanged_fraction"], marker="o", ms=3.2,
                 color=gate_colors[gate], label=GATE_LABELS[gate])
    axd.axvline(0, color=TEXT_MUTED, lw=0.8, ls="--")
    axd.set(xlabel="Boundary shift", ylabel="Decisions unchanged", title="Gate stability", ylim=(-0.02, 1.02))
    axd.legend(frameon=False, loc="lower right", ncol=2, columnspacing=0.8)
    clean_axis(axd)

    change_pivot = change_counts.pivot(index="boundary_shift", columns="decision_change", values="decision_count").fillna(0)
    shifts = np.asarray(BOUNDARY_SHIFTS)
    gained = change_pivot.reindex(shifts).get("gained_support", pd.Series(0, index=shifts)).to_numpy(float)
    lost = change_pivot.reindex(shifts).get("lost_support", pd.Series(0, index=shifts)).to_numpy(float)
    axe.bar(shifts, gained, width=0.014, color=PASS_COLOR, label="Gained support")
    axe.bar(shifts, -lost, width=0.014, color=FAIL_COLOR, label="Lost support")
    axe.axhline(0, color=INK, lw=0.65)
    axe.axvline(0, color=TEXT_MUTED, lw=0.8, ls="--")
    axe.set(xlabel="Boundary shift", ylabel="Decisions changed", title="Boundary transitions")
    axe.legend(frameon=False, loc="upper right")
    clean_axis(axe)

    base_method = method_summary[np.isclose(method_summary["boundary_shift"], 0.0)][
        ["method", "family", "pass_fraction"]
    ].rename(columns={"pass_fraction": "baseline_pass_fraction"})
    stable = method_summary[~np.isclose(method_summary["boundary_shift"], 0.0)].groupby(
        ["method", "family"], as_index=False
    )["unchanged_fraction"].min().rename(columns={"unchanged_fraction": "worst_shift_agreement"})
    method_plane = base_method.merge(stable, on=["method", "family"]).set_index("method").reindex(METHODS).reset_index()
    for y, row in enumerate(method_plane.itertuples(index=False)):
        axf.plot([row.baseline_pass_fraction, row.worst_shift_agreement], [y, y], color=BORDER, lw=1.2, zorder=1)
        axf.scatter(row.baseline_pass_fraction, y, s=34, color=FAMILY_COLORS[row.family],
                    edgecolor="white", linewidth=0.6, zorder=3)
        axf.scatter(row.worst_shift_agreement, y, s=38, marker="D", facecolor="white",
                    edgecolor=FAMILY_COLORS[row.family], linewidth=1.0, zorder=3)
    axf.set_yticks(range(len(METHODS)), METHODS)
    axf.invert_yaxis()
    axf.set(xlabel="Fraction or agreement", ylabel="Method", title="Support versus stability", xlim=(-0.02, 1.02))
    clean_axis(axf, grid=True)
    axf.legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=INK, markeredgecolor="white", markersize=5, label="Baseline pass fraction"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white", markeredgecolor=INK, markersize=4.5, label="Worst-shift agreement"),
    ], frameon=False, loc="lower left")

    for letter, ax in zip("abcdef", axes):
        panel_label(ax, letter, x=-0.14, y=1.10)
    save_figure(fig, "Supplementary_Figure_S7_boundary_sensitivity")


def plot_s8() -> None:
    metrics = pd.read_csv(SOURCE_DIR / "SuppS8_multiseed_metric_runs.csv")
    pairwise = pd.read_csv(SOURCE_DIR / "SuppS8_multiseed_embedding_stability.csv")
    dispersion = pd.read_csv(SOURCE_DIR / "SuppS8_multiseed_metric_dispersion.csv")
    method_summary = pd.read_csv(SOURCE_DIR / "SuppS8_multiseed_method_summary.csv")
    pair_summary = pairwise.groupby(["dataset_id", "method", "family"], as_index=False).agg(
        mean_knn_overlap=("knn_overlap", "mean"), mean_distance_rank=("distance_rank_concordance", "mean")
    )

    apply_final_style()
    fig = plt.figure(figsize=(7.4, 8.8), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, left=0.10, right=0.98, bottom=0.075, top=0.975, wspace=0.48, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]
    axa, axb, axc, axd, axe, axf = axes
    rng = np.random.default_rng(AUDIT_SEED)
    markers = {"pbmc3k": "o", "paul15": "s", "heart_cell_atlas_subsampled": "^"}

    for ax, field, ylabel, title in [
        (axa, "knn_overlap", "Pairwise kNN overlap", "Neighbourhood stability"),
        (axb, "distance_rank_concordance", "Distance-rank concordance", "Global-layout stability"),
    ]:
        for x, method in enumerate(METHODS):
            sub = pairwise[pairwise["method"].eq(method)]
            for dataset in DATASETS:
                vals = sub[sub["dataset_id"].eq(dataset)][field].to_numpy(float)
                ax.scatter(x + rng.uniform(-0.11, 0.11, len(vals)), vals, s=11, alpha=0.45,
                           color=DATASET_COLORS[dataset], marker=markers[dataset], linewidth=0)
            ax.scatter(x, sub[field].mean(), s=28, facecolor=FAMILY_COLORS[method_family(method)],
                       edgecolor=INK, linewidth=0.55, zorder=4)
        ax.set_xticks(range(len(METHODS)), METHODS, rotation=45, ha="right")
        ax.set(xlabel="Method", ylabel=ylabel, title=title, ylim=(-0.05 if field.startswith("distance") else -0.02, 1.02))
        clean_axis(ax, grid=True)
    axb.legend(handles=[Line2D([0], [0], marker=markers[d], color="none", markerfacecolor=DATASET_COLORS[d],
                               markeredgecolor="none", markersize=5, label=DATASET_LABELS[d]) for d in DATASETS],
               frameon=False, loc="lower left")

    range_table = dispersion.groupby(["method", "metric"], as_index=False)["value_range"].max().pivot(
        index="method", columns="metric", values="value_range"
    ).reindex(index=METHODS, columns=METRICS)
    arr = range_table.to_numpy(float)
    vmax = max(0.05, float(np.nanpercentile(arr, 95)))
    im = axc.imshow(arr, cmap="magma_r", vmin=0, vmax=vmax, aspect="auto", interpolation="nearest")
    _annotated_heatmap(fig, axc, im, arr, vmax, "Metric dispersion", "Max. seed range", absolute=False)

    margins = metrics.assign(threshold_margin=metrics["value"] - metrics["threshold"]).groupby(
        ["method", "metric"], as_index=False
    )["threshold_margin"].min().pivot(index="method", columns="metric", values="threshold_margin").reindex(
        index=METHODS, columns=METRICS
    )
    arr = margins.to_numpy(float)
    vmax_margin = max(0.05, float(np.nanpercentile(np.abs(arr), 95)))
    im = axd.imshow(arr, cmap="RdBu", vmin=-vmax_margin, vmax=vmax_margin, aspect="auto", interpolation="nearest")
    _annotated_heatmap(fig, axd, im, arr, vmax_margin, "Worst-seed margins", "Minimum value minus boundary", absolute=True)

    family_context = pair_summary.groupby(["dataset_id", "family"], as_index=False).agg(
        mean_knn_overlap=("mean_knn_overlap", "mean"),
        min_knn_overlap=("mean_knn_overlap", "min"),
        max_knn_overlap=("mean_knn_overlap", "max"),
    )
    x = np.arange(len(FAMILY_ORDER), dtype=float)
    for dataset, offset in zip(DATASETS, [-0.20, 0.0, 0.20]):
        sub = family_context[family_context["dataset_id"].eq(dataset)].set_index("family").reindex(FAMILY_ORDER)
        y = sub["mean_knn_overlap"].to_numpy(float)
        axe.errorbar(x + offset, y, yerr=np.vstack([y - sub["min_knn_overlap"], sub["max_knn_overlap"] - y]),
                     fmt=markers[dataset], ms=4.2, color=DATASET_COLORS[dataset], ecolor=DATASET_COLORS[dataset],
                     elinewidth=0.8, capsize=2, label=DATASET_LABELS[dataset])
    axe.set_xticks(x, [FAMILY_LABELS[f] for f in FAMILY_ORDER], rotation=30, ha="right")
    axe.set(xlabel="Navigation group", ylabel="Mean pairwise kNN overlap", title="Context dependence", ylim=(-0.02, 1.02))
    axe.legend(frameon=False, loc="lower left")
    clean_axis(axe, grid=True)

    label_offsets = {
        "PCA": (-4, 8, "right"), "t-SNE": (-4, -8, "right"), "GLM-PCA": (4, 8, "left"),
        "scScope": (4, -8, "left"), "PHATE": (4, 8, "left"), "SAUCIE": (4, -8, "left"),
        "PaCMAP": (4, 8, "left"), "UMAP": (4, -8, "left"),
    }
    for row in method_summary.itertuples(index=False):
        axf.scatter(row.worst_context_knn_overlap, row.worst_context_distance_rank, s=52,
                    color=FAMILY_COLORS[row.family], edgecolor="white", linewidth=0.7, alpha=0.95)
        dx, dy, ha = label_offsets[row.method]
        axf.annotate(row.method, (row.worst_context_knn_overlap, row.worst_context_distance_rank),
                     xytext=(dx, dy), textcoords="offset points", fontsize=5.4, color=INK, ha=ha, va="center")
    axf.set(xlabel="Worst-context kNN overlap", ylabel="Worst-context distance concordance",
            title="Worst-case stability", xlim=(-0.02, 1.04), ylim=(-0.05, 1.04))
    clean_axis(axf, grid=True)

    for letter, ax in zip("abcdef", axes):
        panel_label(ax, letter, x=-0.14, y=1.10)
    save_figure(fig, "Supplementary_Figure_S8_multiseed_stability")


def _annotated_heatmap(fig, ax, image, values, scale, title, colorbar_label, *, absolute: bool) -> None:
    ax.set_xticks(range(len(METRICS)), [METRIC_LABELS[m] for m in METRICS], rotation=35, ha="right")
    ax.set_yticks(range(len(METHODS)), METHODS)
    ax.set(xlabel="Diagnostic metric", ylabel="Method", title=title)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label(colorbar_label)
    cbar.ax.tick_params(labelsize=5.8)
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            intensity = abs(value) if absolute else value
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=4.8,
                    color="white" if intensity > scale * 0.55 else INK)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def main() -> None:
    plot_s7()
    plot_s8()
    print(f"Wrote Supplementary Figs S7-S8 to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
