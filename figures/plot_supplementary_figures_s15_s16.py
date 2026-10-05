"""Generate result-only supplementary panels from calibrated empirical data."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figures.figure_style import apply_final_style, DATASET_COLORS, DATASET_LABELS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_data/calibration"
OUTPUT = ROOT / "outputs/supplementary_figures"
METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
DATASETS = list(DATASET_COLORS)
METRICS = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
SHORT = ["Local", "Trust", "Global", "Label"]
ROOTS = ["stored_8Mk", "7MEP", "9GMP", "1Ery"]


def decorate(ax, letter, title):
    ax.set_title(title, loc="left", pad=8, fontsize=9)
    width = ax.get_position().width
    ax.text(-.040 / width, 1.07, letter, transform=ax.transAxes, weight="bold", fontsize=11)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", color="#e8e8e8", lw=.5, zorder=0)


def save(fig, name):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(OUTPUT / f"{name}.{suffix}", dpi=450, facecolor="white")
    plt.close(fig)


def geometry_panel(ax, frame, metric, letter, title):
    for di, ds in enumerate(DATASETS):
        data = frame[frame.dataset_id.eq(ds) & frame.metric.eq(metric)]
        for mi, method in enumerate(METHODS):
            rows = data[data.method.eq(method)]
            first = rows[rows.null_family.eq("annotation_unrestricted" if metric == METRICS[3] else "geometry_unrestricted")].iloc[0]
            conditional = rows[rows.null_family.eq("annotation_within_donor" if metric == METRICS[3] else "geometry_within_annotation")]
            y = mi + (di - 1) * .23
            colour = DATASET_COLORS[ds]
            ax.plot([first.null_q025, first.null_q975], [y, y], color="#777777", lw=1.0, zorder=2)
            ax.scatter(first.null_mean, y, marker="|", color="#777777", s=14, zorder=3)
            if len(conditional):
                row = conditional.iloc[0]
                ax.plot([row.null_q025, row.null_q975], [y, y], color=colour, lw=2.5, alpha=.38, zorder=2)
            ax.scatter(first.observed, y, color=colour, edgecolor="white", linewidth=.25, s=16, zorder=4)
    cutoff = float(frame[frame.metric.eq(metric)].operational_cutoff.iloc[0])
    ax.axvline(cutoff, color="#222222", ls="--", lw=.7)
    ax.set_yticks(range(9), METHODS)
    ax.set_ylim(8.65, -.65)
    ax.set_xlim(-.05 if metric == METRICS[2] else 0, 1.02)
    ax.set_xlabel({METRICS[0]: "Neighbour overlap (fraction)", METRICS[1]: "Trustworthiness (score)", METRICS[2]: "Distance-rank correlation (rho)", METRICS[3]: "Same-label neighbours (fraction)"}[metric])
    decorate(ax, letter, title)


def continuum_panel(ax, fig, data, metric, letter, title):
    rows = data[data.metric.eq(metric) & data.null_family.eq("continuum_within_lineage")]
    values = rows.pivot(index="method", columns="root", values="observed").reindex(index=METHODS, columns=ROOTS)
    q = rows.pivot(index="method", columns="root", values="q_bh").reindex(index=METHODS, columns=ROOTS)
    ax.grid(False)
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for y in range(9):
        for x in range(4):
            v = values.iloc[y, x]
            ax.text(x, y, f"{v:.2f}" + ("*" if q.iloc[y, x] < .05 else ""), ha="center", va="center", fontsize=6.8, color="white" if v < .55 else "#171717")
    ax.set_xticks(range(4), ["Stored 8Mk", "7MEP", "9GMP", "1Ery"])
    ax.set_yticks(range(9), METHODS)
    ax.set_xlabel("Expression-reference DPT root")
    decorate(ax, letter, title); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=.045, pad=.03, ticks=[0, .5, 1])
    cb.ax.tick_params(labelsize=7)
    cb.set_label("Observed score", fontsize=7)


def plot_s15():
    geometry = pd.read_csv(SOURCE / "calibration_geometry.csv")
    continuum = pd.read_csv(SOURCE / "calibration_continuum.csv")
    donor = pd.read_csv(SOURCE / "calibration_donor.csv")
    fig = plt.figure(figsize=(7.35, 9.3))
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, .82], left=.12, right=.975, bottom=.075, top=.975, hspace=.60, wspace=.37)
    titles = ["Local correspondence", "Neighbour intrusion", "Global distance order", "Annotation recovery"]
    for i, (metric, title) in enumerate(zip(METRICS, titles)):
        geometry_panel(fig.add_subplot(grid[i // 2, i % 2]), geometry, metric, chr(97 + i), title)
    continuum_panel(fig.add_subplot(grid[2, 0]), fig, continuum, "pseudotime_distance_correlation", "e", "Pseudotime distance order")
    continuum_panel(fig.add_subplot(grid[2, 1]), fig, continuum, "local_pseudotime_retention", "f", "Local pseudotime smoothness")
    ax = fig.add_subplot(grid[3, :])
    for mi, method in enumerate(METHODS):
        row = donor[donor.method.eq(method)].iloc[0]
        ax.plot([row.null_q025, row.null_q975], [mi, mi], color=DATASET_COLORS[DATASETS[2]], alpha=.4, lw=3)
        ax.scatter(row.observed, mi, color=DATASET_COLORS[DATASETS[2]], s=17, zorder=3)
    ax.set_yticks(range(9), METHODS); ax.set_ylim(8.65, -.65)
    ax.set_xlim(0, 1.02); ax.axvline(.50, ls="--", color="#222222", lw=.7)
    ax.set_xlabel("Donor entropy normalised by log(number of donors)")
    decorate(ax, "g", "Donor mixing within cell type")
    handles = [Line2D([], [], marker="o", ls="", color=DATASET_COLORS[ds], label=DATASET_LABELS[ds], markersize=4) for ds in DATASETS]
    handles += [Line2D([], [], color="#777777", label="Unrestricted null, central 95%"), Line2D([], [], color="#777777", lw=3, alpha=.4, label="Conditioned null, central 95%"), Line2D([], [], color="#222222", ls="--", label="Original operational cutoff")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=6.7, bbox_to_anchor=(.53, .002), columnspacing=1.4)
    save(fig, "Supplementary_Figure_S15_permutation_calibration")


def plot_s16():
    full = pd.read_csv(SOURCE / "scvi_geometry_full.csv")
    reference = pd.read_csv(SOURCE / "scvi_reference_sensitivity.csv")
    stability = pd.read_csv(SOURCE / "scvi_seed_stability.csv")
    fig = plt.figure(figsize=(7.35, 7.8))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.08, 1, 1], left=.095, right=.97, bottom=.09, top=.97, hspace=.65, wspace=.65)
    for di, ds in enumerate(DATASETS):
        ax = fig.add_subplot(grid[0, di])
        data = full[full.dataset_id.eq(ds) & full.output_dimension.eq(2) & full.seed.eq(0)]
        matrix = data.pivot(index="method", columns="metric", values="value").reindex(index=METHODS, columns=METRICS)
        ax.imshow(matrix, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for y in range(9):
            for x in range(4):
                value = matrix.iloc[y, x]
                ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=6, color="white" if value < .55 else "#171717")
        ax.set_yticks(range(9), METHODS); ax.set_xticks(range(4), SHORT, rotation=45, ha="right")
        ax.set_xlabel("Geometric diagnostic")
        decorate(ax, chr(97 + di), DATASET_LABELS[ds]); ax.grid(False)
        ax.get_yticklabels()[-1].set_weight("bold")
    ax = fig.add_subplot(grid[1, :])
    vi = full[full.method.eq("scVI")]
    wide = vi.pivot(index=["dataset_id", "seed", "metric"], columns="output_dimension", values="value")
    wide["change"] = wide[10] - wide[2]
    for di, ds in enumerate(DATASETS):
        for mi, metric in enumerate(METRICS):
            values = wide.xs((ds, metric), level=("dataset_id", "metric"))["change"].to_numpy()
            x = mi + (di - 1) * .20
            ax.plot([x, x], [values.min(), values.max()], color=DATASET_COLORS[ds], lw=1)
            ax.scatter(np.full(5, x) + np.linspace(-.04, .04, 5), values, color=DATASET_COLORS[ds], s=12, zorder=3)
            ax.scatter(x, np.median(values), facecolor="white", edgecolor=DATASET_COLORS[ds], s=30, zorder=4)
    ax.axhline(0, color="#777777", ls="--", lw=.6)
    ax.set_xticks(range(4), ["Local retention", "Trustworthiness", "Global rank", "Same-label fraction"])
    ax.set_ylabel("Score change, 10D minus 2D")
    decorate(ax, "d", "scVI latent-dimension response")
    ax.grid(axis="y", color="#e8e8e8", lw=.5)
    ax = fig.add_subplot(grid[2, :2])
    for ds in DATASETS:
        for dim in [2, 10]:
            data = reference[reference.dataset_id.eq(ds) & reference.output_dimension.eq(dim)]
            stats = data.groupby("reference_dimension").value.agg(["median", "min", "max"])
            ax.plot(stats.index, stats["median"], color=DATASET_COLORS[ds], ls="-" if dim == 2 else "--", marker="o" if dim == 2 else "s", ms=3)
            ax.fill_between(stats.index, stats["min"], stats["max"], color=DATASET_COLORS[ds], alpha=.1)
    ax.set_xticks([2, 5, 10, 20, 50]); ax.set_ylim(0, 1)
    ax.set_xlabel("PCA reference dimension"); ax.set_ylabel("Neighbour overlap (fraction)")
    decorate(ax, "e", "scVI reference dependence")
    ax = fig.add_subplot(grid[2, 2])
    for di, ds in enumerate(DATASETS):
        for dim in [2, 10]:
            data = stability[stability.dataset_id.eq(ds) & stability.output_dimension.eq(dim)]
            matrix = data.pivot(index=["seed_a", "seed_b"], columns="metric", values="value")
            ax.scatter(matrix.neighbour_overlap, matrix.distance_rank_stability, marker="o" if dim == 2 else "s", facecolors="none", edgecolors=DATASET_COLORS[ds], s=17)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.03)
    ax.set_xlabel("Neighbour overlap"); ax.set_ylabel("Distance-rank stability")
    decorate(ax, "f", "scVI seed stability")
    handles = [Line2D([], [], marker="o", color=DATASET_COLORS[ds], label=DATASET_LABELS[ds], markersize=4) for ds in DATASETS]
    handles += [Line2D([], [], marker="o", color="#444444", label="2D latent", ms=4), Line2D([], [], marker="s", ls="--", color="#444444", label="10D latent", ms=4)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=6.7, bbox_to_anchor=(.52, .001))
    # One shared heatmap scale is reserved outside the data panels.
    cax = fig.add_axes([.18, .682, .68, .009])
    fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap="viridis"), cax=cax, orientation="horizontal", ticks=[0, .5, 1])
    cax.tick_params(labelsize=6, pad=1)
    cax.set_xlabel("Observed diagnostic score", fontsize=6.5, labelpad=1)
    save(fig, "Supplementary_Figure_S16_scvi_geometry")


if __name__ == "__main__":
    apply_final_style()
    plot_s15()
    plot_s16()
