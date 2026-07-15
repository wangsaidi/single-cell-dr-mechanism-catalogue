"""Plot source-data-backed supplementary figures for the manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, style

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "source_data"
SUPP_FIG_DIR = ROOT / "outputs" / "supplementary_figures"
EXPORT_DPI = 600

METHOD_ORDER = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
SCENARIO_ORDER = [
    "linear_low_rank",
    "nonlinear_manifold",
    "branching_trajectory",
    "dropout_stress",
    "batch_shift",
    "rare_population",
]
SCENARIO_LABELS = {
    "linear_low_rank": "Linear low-rank",
    "nonlinear_manifold": "Nonlinear manifold",
    "branching_trajectory": "Branching trajectory",
    "dropout_stress": "Dropout stress",
    "batch_shift": "Batch shift",
    "rare_population": "Rare population",
}
METRIC_ORDER = [
    "truth_local_retention",
    "truth_trustworthiness",
    "latent_distance_corr",
    "label_neighbor_recall",
    "pseudotime_distance_corr",
    "batch_entropy_norm",
    "rare_label_recall",
]
METRIC_LABELS = {
    "truth_local_retention": "local\nretention",
    "truth_trustworthiness": "trust",
    "latent_distance_corr": "latent\ndist.",
    "label_neighbor_recall": "same-label\nfraction",
    "pseudotime_distance_corr": "pseudo-\ntime",
    "batch_entropy_norm": "batch\nentropy",
    "rare_label_recall": "rare-state\nfraction",
}


def _apply_style() -> None:
    style.apply_style()
    mpl.rcParams.update(
        {
            "font.size": 7.8,
            "axes.titlesize": 8.6,
            "axes.labelsize": 7.4,
            "xtick.labelsize": 6.4,
            "ytick.labelsize": 6.4,
            "figure.dpi": 180,
            "savefig.dpi": EXPORT_DPI,
        }
    )


def _save(fig: plt.Figure, name: str) -> None:
    SUPP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png", "jpg"]:
        fig.savefig(SUPP_FIG_DIR / f"{name}.{ext}", bbox_inches="tight", dpi=EXPORT_DPI, facecolor="white")
    plt.close(fig)


def _panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.10, 1.08, letter, transform=ax.transAxes, ha="left", va="top", fontsize=10.5, fontweight="bold")


def build_supp_fig_s2() -> None:
    """Supplementary Fig. S2: metric-level mechanism simulation summary."""
    _apply_style()
    sim = pd.read_csv(SOURCE_DIR / "fig6_mechanism_simulation_suite.csv")
    sim = sim[sim["support"].isin(["pass", "below_threshold"])].copy()
    sim["threshold_ratio"] = (sim["value"].astype(float) / sim["threshold"].astype(float)).clip(lower=-0.25, upper=1.6)

    fig, axes = plt.subplots(2, 3, figsize=(9.5, 7.2), constrained_layout=False)
    plt.subplots_adjust(left=0.075, right=0.91, top=0.96, bottom=0.08, wspace=0.36, hspace=0.50)
    cmap = mpl.colormaps["YlGnBu"].copy()
    norm = mpl.colors.Normalize(vmin=0, vmax=1.6)

    for idx, scenario in enumerate(SCENARIO_ORDER):
        ax = axes.flat[idx]
        sub = sim[sim["scenario"].eq(scenario)]
        metric_order = [metric for metric in METRIC_ORDER if metric in set(sub["metric"])]
        ratio = (
            sub.pivot_table(index="method", columns="metric", values="threshold_ratio", aggfunc="mean")
            .reindex(index=METHOD_ORDER, columns=metric_order)
        )
        support = (
            sub.assign(pass_flag=sub["support"].eq("pass").astype(float))
            .pivot_table(index="method", columns="metric", values="pass_flag", aggfunc="mean")
            .reindex(index=METHOD_ORDER, columns=metric_order)
        )
        values = ratio.values.astype(float)
        if np.isnan(values).any():
            raise ValueError(f"Missing metric values in Supplementary Fig. S2 scenario: {scenario}")
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(SCENARIO_LABELS[scenario], loc="left")
        ax.set_xticks(np.arange(len(metric_order)))
        ax.set_xticklabels([METRIC_LABELS[m] for m in metric_order], rotation=0)
        ax.set_yticks(np.arange(len(METHOD_ORDER)))
        ax.set_yticklabels(METHOD_ORDER)
        style.color_method_ticklabels(ax, "y")
        style.add_method_group_dividers(ax, n_methods=len(METHOD_ORDER))
        ax.tick_params(length=0, pad=1)
        for i in range(ratio.shape[0]):
            for j in range(ratio.shape[1]):
                val = ratio.iloc[i, j]
                mark = f"{support.iloc[i, j]:.1f}"
                color = "white" if val > 1.05 else "#172A3A"
                ax.text(j, i, mark, ha="center", va="center", fontsize=5.2, fontweight="bold", color=color)
        _panel_label(ax, chr(ord("a") + idx))

    cax = fig.add_axes([0.93, 0.20, 0.018, 0.62])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("value / operational boundary", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    _save(fig, "Supplementary_Figure_S2_mechanism_simulation_metric_summary")


def main() -> None:
    build_supp_fig_s2()


if __name__ == "__main__":
    main()
