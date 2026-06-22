"""Build main Figure 2 evidence-synthesis panels.

This script builds the adopted source-data-backed main-text Figure 2.
Supplementary Figure S1 is provided as final static output in
``outputs/supplementary_figures`` and is not regenerated here.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SOURCE_DIR = DATA_DIR / "source_data"
OUTPUT_DIR = ROOT / "outputs" / "main_figures"
METADATA_DIR = ROOT / "metadata"

METHOD_ORDER = config.ANCHOR_METHODS
FAMILY_ORDER = config.FAMILY_ORDER
FAMILY_LABELS = config.FAMILY_LABELS
EVIDENCE_LAYERS = ["empirical gates", "biological anchors", "mechanism simulations"]
CLAIM_GATE_ORDER = [
    "label_support_gate",
    "local_neighbourhood_gate",
    "global_geometry_gate",
    "continuum_gate",
    "donor_aware_gate",
]
CLAIM_GATE_LABELS = ["label", "local", "global", "continuum", "donor"]
CLAIM_AXIS_ORDER = [
    "truth_geometry",
    "cell_state_support",
    "continuum_support",
    "rare_state_support",
    "batch_biology_tradeoff",
]
CLAIM_AXIS_LABELS = ["truth\ngeom.", "cell\nstate", "continuum", "rare\nstate", "batch\nbiology"]
METRIC_ORDER = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
METRIC_LABELS = {
    "local_retention": "local",
    "trustworthiness": "trust",
    "global_rank_corr": "global",
    "label_neighbor_recall": "label",
}
DATASET_LABELS = {
    "pbmc3k": "PBMC3k",
    "paul15": "Paul15",
    "heart_cell_atlas_subsampled": "Heart atlas",
}


def _apply_pub_style() -> None:
    style.apply_style()
    mpl.rcParams.update(
        {
            "font.size": 7.0,
            "axes.titlesize": 7.7,
            "axes.labelsize": 6.7,
            "xtick.labelsize": 5.6,
            "ytick.labelsize": 5.6,
            "legend.fontsize": 5.3,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.titlepad": 4,
        }
    )


def _panel_label(ax: plt.Axes, letter: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _save(fig: plt.Figure, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png", "jpg"]:
        kwargs = {"bbox_inches": "tight"}
        if ext in {"png", "jpg"}:
            kwargs["dpi"] = 450
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", **kwargs)
    plt.close(fig)


def _family_short(family: str) -> str:
    return {"factor": "Factor", "deep": "Deep", "graph": "Graph", "relational": "Relational"}[family]


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = SOURCE_DIR / name
    df.to_csv(path, index=False, float_format="%.10g")
    return path


def _support_matrix(
    df: pd.DataFrame,
    index: str,
    columns: str,
    values: str = "support_fraction",
    *,
    row_order: list[str] | None = None,
    col_order: list[str] | None = None,
    allow_missing: bool = False,
) -> pd.DataFrame:
    mat = df.pivot(index=index, columns=columns, values=values)
    if row_order is not None:
        mat = mat.reindex(index=row_order)
    if col_order is not None:
        mat = mat.reindex(columns=col_order)
    if mat.isna().any().any() and not allow_missing:
        missing = mat.isna().sum().sum()
        raise ValueError(f"Missing values in matrix {index} x {columns}: {missing}")
    return mat


def _heatmap_values(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: str = "YlGnBu",
    vmin: float = 0,
    vmax: float = 1,
    fmt: str = ".2f",
    value_fontsize: float = 4.6,
) -> mpl.image.AxesImage:
    im = ax.imshow(matrix.values, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = float(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                format(val, fmt),
                ha="center",
                va="center",
                fontsize=value_fontsize,
                color="white" if val > (vmin + 0.72 * (vmax - vmin)) else "#17213A",
            )
    return im


def _heatmap_values_with_na(
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    cmap: str = "YlGnBu",
    vmin: float = 0,
    vmax: float = 1,
    fmt: str = ".2f",
    value_fontsize: float = 4.6,
) -> mpl.image.AxesImage:
    arr = matrix.values.astype(float)
    masked = np.ma.masked_invalid(arr)
    cmap_obj = mpl.colormaps[cmap].copy()
    cmap_obj.set_bad("#F2F2F2")
    im = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap_obj, vmin=vmin, vmax=vmax)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix.iloc[i, j]
            if pd.isna(val):
                ax.text(j, i, "n.a.", ha="center", va="center", fontsize=value_fontsize + 0.5, color="#7A7A7A")
            else:
                val = float(val)
                ax.text(
                    j,
                    i,
                    format(val, fmt),
                    ha="center",
                    va="center",
                    fontsize=value_fontsize,
                    color="white" if val > (vmin + 0.72 * (vmax - vmin)) else "#17213A",
                )
    return im


def _barh_with_values(ax: plt.Axes, values: pd.Series, labels: list[str], colors: list[str]) -> None:
    y = np.arange(len(values))
    ax.barh(y, values, color=colors, alpha=0.86)
    for yi, val in enumerate(values):
        ax.text(float(val) + 0.015, yi, f"{float(val):.2f}", va="center", fontsize=4.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 1.08)
    ax.invert_yaxis()


def build_fig2_source_tables() -> dict[str, Path]:
    """Build source-data tables for each panel from locked existing results."""
    gate = pd.read_csv(SOURCE_DIR / "fig4_gate_pass_matrix.csv")
    method_evidence = pd.read_csv(SOURCE_DIR / "integrated_method_evidence_layers.csv")
    family_evidence = pd.read_csv(SOURCE_DIR / "integrated_family_evidence_layers.csv")
    bottleneck = pd.read_csv(SOURCE_DIR / "integrated_method_evidence_bottleneck.csv")
    bio_audit = pd.read_csv(SOURCE_DIR / "fig5_independent_evidence_audit.csv")
    sim = pd.read_csv(SOURCE_DIR / "fig6_mechanism_simulation_suite.csv")
    pert_fail = pd.read_csv(SOURCE_DIR / "fig6_perturbation_failure_by_metric.csv")

    panel_a = method_evidence.copy()
    panel_a["source_definition"] = (
        "support_fraction combines empirical diagnostic gates, independent biological-anchor rows, "
        "or mechanism-simulation rows as defined in the upstream source tables"
    )

    panel_b = family_evidence.copy()
    panel_b["source_definition"] = "family mean of method-level evidence-layer support fractions"

    panel_c = (
        gate.assign(pass_bool=gate["support"].eq("pass").astype(float))
        .groupby(["family", "claim_gate"], as_index=False)
        .agg(support_fraction=("pass_bool", "mean"), n_rows=("pass_bool", "size"))
    )
    panel_c["source_definition"] = "fraction of valid method-dataset gate rows passing predeclared thresholds"

    panel_d = (
        gate.assign(pass_bool=gate["support"].eq("pass").astype(float))
        .groupby(["dataset_id", "claim_gate"], as_index=False)
        .agg(support_fraction=("pass_bool", "mean"), n_rows=("pass_bool", "size"))
    )
    panel_d["dataset_label"] = panel_d["dataset_id"].map(DATASET_LABELS)
    panel_d["source_definition"] = "dataset-specific pass fraction for valid diagnostic gates"

    panel_e = bio_audit.copy()
    panel_e["claim_label"] = panel_e["claim_type"].map(
        {
            "discrete_cell_identity": "cell identity",
            "trajectory_continuum": "continuum",
            "donor_variable_tissue_identity": "donor tissue",
            "rare_state_support": "rare state",
        }
    )
    panel_e["source_definition"] = "independent biological-anchor support fraction from marker, lineage, donor, or rare-state analyses"

    sim_valid = sim[sim["support"].isin(["pass", "below_threshold"])].copy()
    panel_f = (
        sim_valid.assign(pass_bool=sim_valid["support"].eq("pass").astype(float))
        .groupby(["family", "claim_axis"], as_index=False)
        .agg(support_fraction=("pass_bool", "mean"), n_rows=("pass_bool", "size"))
    )
    panel_f["source_definition"] = "family-level pass fraction in the compact known-truth mechanism simulation suite"

    panel_g = method_evidence.pivot(index="method", columns="evidence_layer", values="support_fraction").reset_index()
    panel_g["family"] = panel_g["method"].map(config.METHOD_FAMILY)
    panel_g["source_definition"] = "method-level cross-layer concordance; point size in the figure encodes mechanism-simulation support"

    panel_h = (
        sim_valid.assign(pass_bool=sim_valid["support"].eq("pass").astype(float))
        .groupby(["method", "family", "claim_axis"], as_index=False)
        .agg(support_fraction=("pass_bool", "mean"), n_rows=("pass_bool", "size"))
    )
    panel_h["source_definition"] = "method-level pass fraction for each known-truth simulation claim axis"

    panel_i = pert_fail.copy()
    panel_i["source_definition"] = "fraction of dropout or noise stress rows falling below the predeclared metric threshold"

    panel_j = bottleneck.copy()
    panel_j["source_definition"] = "weakest evidence layer per method, computed from empirical, biological-anchor, and simulation support fractions"

    integrated = method_evidence.groupby("method", as_index=False).agg(
        mean_support=("support_fraction", "mean"),
        minimum_support=("support_fraction", "min"),
        n_rows=("n_rows", "sum"),
    )
    integrated["family"] = integrated["method"].map(config.METHOD_FAMILY)
    integrated["support_gap"] = integrated["mean_support"] - integrated["minimum_support"]
    integrated["source_definition"] = "integrated method support and bottleneck gap across all three evidence layers"

    paths = {
        "a": _write_csv(panel_a, "Figure_2_panel_a_method_evidence_layers.csv"),
        "b": _write_csv(panel_b, "Figure_2_panel_b_family_evidence_layers.csv"),
        "c": _write_csv(panel_c, "Figure_2_panel_c_family_gate_support.csv"),
        "d": _write_csv(panel_d, "Figure_2_panel_d_dataset_gate_support.csv"),
        "e": _write_csv(panel_e, "Figure_2_panel_e_biological_anchor_support.csv"),
        "f": _write_csv(panel_f, "Figure_2_panel_f_simulation_claim_support_by_family.csv"),
        "g": _write_csv(panel_g, "Figure_2_panel_g_cross_layer_concordance.csv"),
        "h": _write_csv(panel_h, "Figure_2_panel_h_method_simulation_claim_support.csv"),
        "i": _write_csv(panel_i, "Figure_2_panel_i_perturbation_failure_by_metric.csv"),
        "j": _write_csv(panel_j, "Figure_2_panel_j_method_evidence_bottleneck.csv"),
        "integrated": _write_csv(integrated, "Figure_2_integrated_method_support.csv"),
    }
    manifest = pd.DataFrame(
        [
            {
                "panel": panel,
                "source_data_file": str(path.relative_to(ROOT)),
            }
            for panel, path in sorted(paths.items())
        ]
    )
    paths["manifest"] = _write_csv(manifest, "Figure_2_source_manifest.csv")
    return paths


def write_claim_matrix(paths: dict[str, Path]) -> Path:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("a", "Do methods have matched support across empirical diagnostics, biological anchors and mechanism simulations?", "D5 hybrid empirical-simulation synthesis", paths["a"]),
        ("b", "Do mechanism-family averages hide evidence-layer dependence?", "D5 hybrid empirical-simulation synthesis", paths["b"]),
        ("c", "Which biological claim gates are supported by each mechanism family?", "D2 empirical-derived diagnostic synthesis", paths["c"]),
        ("d", "Which data contexts make particular claim gates hardest to satisfy?", "D2 empirical-derived diagnostic synthesis", paths["d"]),
        ("e", "Which independent biological anchors support the visual-interpretation claims?", "D2 empirical-derived biological-anchor synthesis", paths["e"]),
        ("f", "Which mechanism families pass known-truth simulation claim axes?", "D4 simulation-derived synthesis", paths["f"]),
        ("g", "Are empirical gates and biological anchors concordant at the method level?", "D5 hybrid empirical-simulation synthesis", paths["g"]),
        ("h", "Which known-truth simulation claim axes are supported by each method?", "D4 simulation-derived synthesis", paths["h"]),
        ("i", "Which diagnostic metrics fail under dropout or noise stress?", "D4 simulation-derived perturbation synthesis", paths["i"]),
        ("j", "Which methods are constrained by their weakest evidence layer?", "D5 hybrid empirical-simulation synthesis", paths["j"]),
    ]
    df = pd.DataFrame(
        [
            {
                "figure_id": "Fig2",
                "figure_main_claim": "Mechanism-aware interpretation requires evidence across empirical diagnostic gates, independent biological anchors, known-truth simulations and robustness stress tests rather than a universal visual ranking.",
                "figure_claim_type": "hybrid empirical-simulation evidence synthesis",
                "panel_id": panel,
                "panel_scientific_question": question,
                "data_source_class": cls,
                "evidence_role": "source-data-backed synthesis panel",
                "counts_toward_evidence_panel_target": "TRUE",
                "source_data_file": str(path.relative_to(ROOT)),
                "plotting_code_path": "figures/plot_figure_2.py",
                "statistical_test_or_model": "Descriptive support fractions, failure fractions, threshold margins, or cross-layer summaries; no inferential p-value test is reported.",
                "n_definition": "n is the number of source-data rows contributing to the relevant support or failure fraction, as recorded in the panel source table.",
                "reviewer_risk": "Moderate; this is an integrated synthesis and must be read with detailed source panels in Figs. 3-7 and Supplementary Fig. S1.",
            }
            for panel, question, cls, path in rows
        ]
    )
    out = METADATA_DIR / "Figure_2_claim_to_evidence_matrix.tsv"
    df.to_csv(out, sep="\t", index=False)
    return out


def build_fig2() -> None:
    _apply_pub_style()
    paths = build_fig2_source_tables()
    write_claim_matrix(paths)

    method_evidence = pd.read_csv(paths["a"])
    family_evidence = pd.read_csv(paths["b"])
    family_gate = pd.read_csv(paths["c"])
    dataset_gate = pd.read_csv(paths["d"])
    bio_anchor = pd.read_csv(paths["e"])
    sim_family = pd.read_csv(paths["f"])
    cross_layer = pd.read_csv(paths["g"])
    method_sim = pd.read_csv(paths["h"])
    perturb_fail = pd.read_csv(paths["i"])
    bottleneck = pd.read_csv(paths["j"])
    integrated = pd.read_csv(paths["integrated"])

    fig = plt.figure(figsize=(7.9, 10.9))
    gs = fig.add_gridspec(
        5,
        4,
        height_ratios=[1.02, 0.98, 1.02, 0.94, 0.95],
        width_ratios=[1.05, 1.05, 1.08, 0.96],
        hspace=0.86,
        wspace=0.72,
    )

    ax = fig.add_subplot(gs[0, 0:3])
    method_mat = _support_matrix(
        method_evidence,
        "method",
        "evidence_layer",
        row_order=METHOD_ORDER,
        col_order=EVIDENCE_LAYERS,
    )
    im = _heatmap_values(ax, method_mat, value_fontsize=4.9)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["empirical\ngates", "biological\nanchors", "known-truth\nsimulations"], rotation=0)
    ax.set_yticks(np.arange(len(METHOD_ORDER)))
    ax.set_yticklabels(METHOD_ORDER)
    style.color_method_ticklabels(ax, "y")
    style.add_method_group_dividers(ax, len(METHOD_ORDER))
    ax.set_title("Cross-layer method support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.010)
    cbar.ax.set_ylabel("support fraction", rotation=270, labelpad=8, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "a")

    ax = fig.add_subplot(gs[0, 3])
    family_mat = _support_matrix(
        family_evidence,
        "family",
        "evidence_layer",
        row_order=FAMILY_ORDER,
        col_order=EVIDENCE_LAYERS,
    )
    im = _heatmap_values(ax, family_mat, value_fontsize=4.8)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["emp.", "bio", "sim"], rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels([_family_short(f) for f in FAMILY_ORDER])
    for tick, fam in zip(ax.get_yticklabels(), FAMILY_ORDER):
        tick.set_color(style.FAMILY_COLORS[fam])
    ax.set_title("Family support", loc="left")
    _panel_label(ax, "b", x=-0.22)

    ax = fig.add_subplot(gs[1, 0:2])
    gate_mat = _support_matrix(
        family_gate,
        "family",
        "claim_gate",
        row_order=FAMILY_ORDER,
        col_order=CLAIM_GATE_ORDER,
    )
    im = _heatmap_values(ax, gate_mat, value_fontsize=4.7)
    ax.set_xticks(np.arange(len(CLAIM_GATE_ORDER)))
    ax.set_xticklabels(CLAIM_GATE_LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels([_family_short(f) for f in FAMILY_ORDER])
    for tick, fam in zip(ax.get_yticklabels(), FAMILY_ORDER):
        tick.set_color(style.FAMILY_COLORS[fam])
    ax.set_title("Claim-gate support", loc="left")
    _panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 2:4])
    dg = dataset_gate.copy()
    dg = dg[dg["dataset_label"].isin(["PBMC3k", "Paul15", "Heart atlas"])]
    dataset_mat = _support_matrix(
        dg,
        "dataset_label",
        "claim_gate",
        row_order=["PBMC3k", "Paul15", "Heart atlas"],
        col_order=CLAIM_GATE_ORDER,
        allow_missing=True,
    )
    im = _heatmap_values_with_na(ax, dataset_mat, value_fontsize=4.7)
    ax.set_xticks(np.arange(len(CLAIM_GATE_ORDER)))
    ax.set_xticklabels(CLAIM_GATE_LABELS, rotation=25, ha="right")
    ax.set_yticks(np.arange(dataset_mat.shape[0]))
    ax.set_yticklabels(dataset_mat.index)
    ax.set_title("Context-gate support", loc="left")
    ax.text(
        0.99,
        -0.28,
        "grey = gate not defined for context",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.8,
        color="#666666",
    )
    _panel_label(ax, "d")

    ax = fig.add_subplot(gs[2, 0])
    bio_order = ["cell identity", "continuum", "donor tissue", "rare state"]
    bio_plot = bio_anchor.set_index("claim_label").reindex(bio_order)
    _barh_with_values(
        ax,
        bio_plot["support_summary"],
        bio_order,
        ["#4C78A8", "#59A14F", "#B07AA1", "#D98CB7"],
    )
    ax.set_xlabel("support fraction")
    ax.set_title("Biological anchors", loc="left")
    for yi, n in enumerate(bio_plot["n_method_label_rows"]):
        ax.text(0.02, yi + 0.28, f"n={int(n)}", fontsize=4.4, color="#555555")
    _panel_label(ax, "e", x=-0.28)

    ax = fig.add_subplot(gs[2, 1:4])
    sim_mat = _support_matrix(
        sim_family,
        "family",
        "claim_axis",
        row_order=FAMILY_ORDER,
        col_order=CLAIM_AXIS_ORDER,
    )
    im = _heatmap_values(ax, sim_mat, value_fontsize=4.7)
    ax.set_xticks(np.arange(len(CLAIM_AXIS_ORDER)))
    ax.set_xticklabels(CLAIM_AXIS_LABELS, rotation=0)
    ax.set_yticks(np.arange(len(FAMILY_ORDER)))
    ax.set_yticklabels([_family_short(f) for f in FAMILY_ORDER])
    for tick, fam in zip(ax.get_yticklabels(), FAMILY_ORDER):
        tick.set_color(style.FAMILY_COLORS[fam])
    ax.set_title("Known-truth simulation support", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.010)
    cbar.ax.set_ylabel("support fraction", rotation=270, labelpad=8, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "f")

    ax = fig.add_subplot(gs[3, 0:2])
    for _, row in cross_layer.iterrows():
        method = str(row["method"])
        empirical_support = float(row["empirical gates"])
        biological_support = float(row["biological anchors"])
        simulation_support = float(row["mechanism simulations"])
        ax.scatter(
            empirical_support,
            biological_support,
            s=36 + 150 * simulation_support,
            color=style.family_color(method),
            edgecolor="white",
            linewidth=0.35,
            alpha=0.88,
        )
        label_offsets = {
            "PCA": (-40, -10),
            "GLM-PCA": (18, -18),
            "scScope": (8, 8),
            "SAUCIE": (14, -14),
            "UMAP": (-42, -12),
            "PHATE": (18, 18),
            "t-SNE": (22, 6),
            "PaCMAP": (-48, 6),
        }
        dx, dy = label_offsets.get(method, (8, 8))
        ax.annotate(
            method,
            xy=(empirical_support, biological_support),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=4.4,
            color=style.family_color(method),
            ha="left" if dx >= 0 else "right",
            va="center",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
            arrowprops=dict(arrowstyle="-", lw=0.35, color="#888888", shrinkA=0, shrinkB=2),
            clip_on=False,
        )
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.05, 1.16)
    ax.set_xlabel("empirical gate support")
    ax.set_ylabel("biological anchor support")
    ax.set_title("Cross-layer concordance", loc="left")
    ax.text(
        0.98,
        0.05,
        "point size: simulation support",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.5,
        color="#555555",
    )
    _panel_label(ax, "g")

    ax = fig.add_subplot(gs[3, 2:4])
    method_sim_mat = _support_matrix(
        method_sim,
        "method",
        "claim_axis",
        row_order=METHOD_ORDER,
        col_order=CLAIM_AXIS_ORDER,
    )
    im = _heatmap_values(ax, method_sim_mat, cmap="YlGnBu", fmt=".2f", value_fontsize=4.45)
    ax.set_xticks(np.arange(len(CLAIM_AXIS_ORDER)))
    ax.set_xticklabels(CLAIM_AXIS_LABELS, rotation=0)
    ax.set_yticks(np.arange(method_sim_mat.shape[0]))
    ax.set_yticklabels(method_sim_mat.index)
    style.color_method_ticklabels(ax, "y")
    ax.set_title("Simulation support by method", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.020, pad=0.010)
    cbar.ax.set_ylabel("support fraction", rotation=270, labelpad=8, fontsize=5)
    cbar.ax.tick_params(labelsize=5)
    _panel_label(ax, "h")

    ax = fig.add_subplot(gs[4, 0:2])
    pert_mat = _support_matrix(
        perturb_fail,
        "metric",
        "perturbation",
        values="failure_fraction",
        row_order=METRIC_ORDER,
        col_order=["dropout", "noise"],
    )
    im = _heatmap_values(ax, pert_mat, cmap="Reds", fmt=".2f", value_fontsize=4.9)
    ax.set_xticks(np.arange(pert_mat.shape[1]))
    ax.set_xticklabels(["dropout", "noise"], rotation=15, ha="right")
    ax.set_yticks(np.arange(len(METRIC_ORDER)))
    ax.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax.set_title("Perturbation failure", loc="left")
    _panel_label(ax, "i")

    ax = fig.add_subplot(gs[4, 2:4])
    integrated = integrated.set_index("method").reindex(METHOD_ORDER).reset_index()
    for row in integrated.itertuples(index=False):
        ax.plot([row.minimum_support, row.mean_support], [row.method, row.method], color=style.family_color(row.method), lw=1.1, alpha=0.85)
        ax.scatter(row.minimum_support, row.method, s=26, color="white", edgecolor=style.family_color(row.method), linewidth=1.0, zorder=3)
        ax.scatter(row.mean_support, row.method, s=30, color=style.family_color(row.method), edgecolor="white", linewidth=0.35, zorder=3)
    ax.set_xlim(0, 1.04)
    ax.set_xlabel("support fraction")
    ax.set_ylabel("")
    ax.set_title("Mean versus bottleneck", loc="left")
    style.color_method_ticklabels(ax, "y")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", markersize=4.5, label="minimum"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markeredgecolor="white", markersize=4.5, label="mean"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=4.8,
    )
    _panel_label(ax, "j")

    handles = style.family_legend_handles()
    fig.legend(
        handles=handles,
        title="mechanism family",
        loc="lower center",
        bbox_to_anchor=(0.52, 0.006),
        ncol=4,
        frameon=False,
        fontsize=5.2,
        title_fontsize=5.4,
    )
    fig.subplots_adjust(bottom=0.065, top=0.985, left=0.075, right=0.975)
    _save(fig, "Figure_2")


def main() -> None:
    build_fig2()
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.pdf'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.svg'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.png'}")
    print(f"Wrote {METADATA_DIR / 'Figure_2_claim_to_evidence_matrix.tsv'}")


if __name__ == "__main__":
    main()



