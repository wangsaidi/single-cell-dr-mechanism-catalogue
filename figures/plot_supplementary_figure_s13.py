"""Summarise independent generated-dataset replication in the simulation suite."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures import config  # noqa: E402
from figures.figure_style import (  # noqa: E402
    BORDER,
    FAMILY_COLORS,
    GRID,
    INK,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    panel_label,
)


BASE_SOURCE = ROOT / "data" / "source_data"
SOURCE = ROOT / "data" / "source_data" / "generated"
FIGURE_DIR = ROOT / "outputs" / "supplementary_figures"
META_DIR = ROOT / "metadata"

METHOD_ORDER = list(config.ANCHOR_METHODS)
SCENARIO_ORDER = [
    "linear_low_rank",
    "nonlinear_manifold",
    "branching_trajectory",
    "dropout_stress",
    "batch_shift",
    "rare_population",
]
SCENARIO_LABELS = {
    "linear_low_rank": "Linear",
    "nonlinear_manifold": "Nonlinear",
    "branching_trajectory": "Branching",
    "dropout_stress": "Dropout",
    "batch_shift": "Batch shift",
    "rare_population": "Rare state",
}
SCENARIO_COLORS = {
    "linear_low_rank": "#4C78A8",
    "nonlinear_manifold": "#72A0C1",
    "branching_trajectory": "#59A14F",
    "dropout_stress": "#E0A458",
    "batch_shift": "#B279A2",
    "rare_population": "#E15759",
}
CLAIM_LABELS = {
    "truth_geometry": "Truth geometry",
    "cell_state_support": "Cell state",
    "continuum_support": "Continuum",
    "batch_biology_tradeoff": "Batch-biology",
    "rare_state_support": "Rare state",
}
METRIC_LABELS = {
    "truth_local_retention": "Truth local",
    "truth_trustworthiness": "Truth trust",
    "latent_distance_corr": "Latent distance",
    "label_neighbor_recall": "Label recall",
    "pseudotime_distance_corr": "Pseudotime",
    "batch_entropy_norm": "Batch entropy",
    "rare_label_recall": "Rare recall",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = SOURCE / name
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def build_tables() -> dict[str, pd.DataFrame]:
    suite = pd.read_csv(BASE_SOURCE / "fig6_mechanism_simulation_suite.csv")
    valid = suite[suite["support"].isin(["pass", "below_threshold"])].copy()
    if set(valid["replicate"].unique()) != set(range(5)):
        raise AssertionError("Simulation replication requires generated-dataset replicates 0-4")
    expected_runs = len(SCENARIO_ORDER) * len(METHOD_ORDER) * 5
    observed_runs = valid[["scenario", "replicate", "method"]].drop_duplicates()
    if len(observed_runs) != expected_runs:
        raise AssertionError(f"Expected {expected_runs} complete method-scenario-replicate runs, found {len(observed_runs)}")
    if valid["value"].isna().any() or not np.isfinite(valid["value"]).all():
        raise AssertionError("Simulation replication contains non-finite metric values")

    valid["passed"] = valid["support"].eq("pass")
    valid["threshold_margin"] = valid["value"] - valid["threshold"]
    valid["n_definition"] = (
        "one method-scenario-generated-dataset-metric row; five replicates are independently generated simulated datasets"
    )

    run_summary = (
        valid.groupby(["scenario", "replicate", "method", "family", "seed"], as_index=False)
        .agg(
            n_metrics=("metric", "size"),
            fraction_pass=("passed", "mean"),
            mean_threshold_margin=("threshold_margin", "mean"),
            minimum_threshold_margin=("threshold_margin", "min"),
        )
    )
    run_summary["n_definition"] = "one method-scenario-generated-dataset run"

    method_scenario = (
        run_summary.groupby(["scenario", "method", "family"], as_index=False)
        .agg(
            mean_fraction_pass=("fraction_pass", "mean"),
            sd_fraction_pass=("fraction_pass", "std"),
            minimum_fraction_pass=("fraction_pass", "min"),
            maximum_fraction_pass=("fraction_pass", "max"),
            replicate_range=("fraction_pass", lambda values: float(values.max() - values.min())),
            n_generated_datasets=("replicate", "nunique"),
        )
    )
    method_scenario["n_definition"] = "five independently generated datasets per method-scenario combination"

    agreement_rows: list[dict[str, object]] = []
    for keys, group in valid.groupby(["scenario", "method", "family", "metric", "claim_axis"], sort=False):
        group = group.sort_values("replicate")
        decisions = dict(zip(group["replicate"].astype(int), group["passed"].astype(bool)))
        if set(decisions) != set(range(5)):
            raise AssertionError(f"Incomplete generated-dataset decisions for {keys}")
        pair_values = [decisions[first] == decisions[second] for first, second in combinations(range(5), 2)]
        values = group["value"].astype(float)
        agreement_rows.append(
            {
                "scenario": keys[0],
                "method": keys[1],
                "family": keys[2],
                "metric": keys[3],
                "claim_axis": keys[4],
                "decision_agreement": float(np.mean(pair_values)),
                "n_replicate_pairs": len(pair_values),
                "value_mean": float(values.mean()),
                "value_sd": float(values.std(ddof=1)),
                "value_minimum": float(values.min()),
                "value_maximum": float(values.max()),
                "value_range": float(values.max() - values.min()),
                "pass_fraction": float(group["passed"].mean()),
                "n_generated_datasets": 5,
                "n_definition": "five independently generated datasets and ten generated-dataset pairs",
            }
        )
    agreement = pd.DataFrame(agreement_rows)

    method_claim = (
        valid.groupby(["method", "family", "claim_axis"], as_index=False)
        .agg(
            pass_fraction=("passed", "mean"),
            mean_margin=("threshold_margin", "mean"),
            minimum_margin=("threshold_margin", "min"),
            n_metric_rows=("passed", "size"),
            n_generated_datasets=("replicate", "nunique"),
            n_scenarios=("scenario", "nunique"),
        )
    )
    method_claim["n_definition"] = "all applicable metric rows from five independently generated datasets per scenario"

    metric_replication = (
        agreement.groupby(["metric", "claim_axis"], as_index=False)
        .agg(
            median_value_range=("value_range", "median"),
            mean_value_range=("value_range", "mean"),
            maximum_value_range=("value_range", "max"),
            mean_decision_agreement=("decision_agreement", "mean"),
            minimum_decision_agreement=("decision_agreement", "min"),
            n_method_scenario_metrics=("decision_agreement", "size"),
        )
    )
    metric_replication["n_definition"] = "method-scenario-metric combinations, each evaluated in five generated datasets"

    method_summary = (
        run_summary.groupby(["method", "family"], as_index=False)
        .agg(
            mean_run_support=("fraction_pass", "mean"),
            worst_run_support=("fraction_pass", "min"),
            best_run_support=("fraction_pass", "max"),
            run_support_range=("fraction_pass", lambda values: float(values.max() - values.min())),
            n_method_scenario_generated_dataset_runs=("fraction_pass", "size"),
        )
    )
    method_summary["n_definition"] = "30 method-scenario-generated-dataset runs per method"

    return {
        "metric_rows": valid,
        "run_summary": run_summary,
        "method_scenario": method_scenario,
        "decision_agreement": agreement,
        "method_claim": method_claim,
        "metric_replication": metric_replication,
        "method_summary": method_summary,
    }


def _dot_heatmap(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_column: str,
    x_order: list[str],
    y_column: str,
    y_order: list[str],
    value_column: str,
    annotation_column: str | None = None,
) -> None:
    cmap = mpl.colormaps["viridis"]
    for row in frame.itertuples(index=False):
        x_value = getattr(row, x_column)
        y_value = getattr(row, y_column)
        if x_value not in x_order or y_value not in y_order:
            continue
        x = x_order.index(x_value)
        y = y_order.index(y_value)
        value = float(getattr(row, value_column))
        ax.scatter(x, y, s=54, c=[value], cmap=cmap, vmin=0, vmax=1, edgecolor="white", linewidth=0.6)
        if annotation_column:
            text = getattr(row, annotation_column)
            ax.text(x, y, f"{float(text):.1f}", ha="center", va="center", fontsize=4.5, color="white" if value < 0.55 else INK)
    ax.set_xticks(range(len(x_order)), [SCENARIO_LABELS.get(value, CLAIM_LABELS.get(value, value)) for value in x_order], rotation=34, ha="right")
    ax.set_yticks(range(len(y_order)), y_order)
    ax.set_xlim(-0.5, len(x_order) - 0.5)
    ax.set_ylim(len(y_order) - 0.5, -0.5)
    for value in np.arange(-0.5, len(x_order), 1):
        ax.axvline(value, color=GRID, linewidth=0.45, zorder=0)
    for value in np.arange(-0.5, len(y_order), 1):
        ax.axhline(value, color=GRID, linewidth=0.45, zorder=0)
    clean_axis(ax, grid=False)


def plot_figure(tables: dict[str, pd.DataFrame]) -> list[Path]:
    apply_final_style()
    fig = plt.figure(figsize=(7.2, 6.8), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, left=0.08, right=0.985, top=0.965, bottom=0.10, wspace=0.42, hspace=0.48)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    axa, axb, axc, axd, axe, axf = axes
    rng = np.random.default_rng(20260714)

    runs = tables["run_summary"]
    for x, scenario in enumerate(SCENARIO_ORDER):
        values = runs.loc[runs["scenario"].eq(scenario), "fraction_pass"].to_numpy()
        jitter = rng.uniform(-0.18, 0.18, len(values))
        axa.scatter(x + jitter, values, s=7, alpha=0.38, color=SCENARIO_COLORS[scenario], edgecolor="none")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axa.vlines(x, q1, q3, color=INK, linewidth=2.2)
        axa.scatter(x, median, s=30, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
    axa.set_xticks(range(len(SCENARIO_ORDER)), [SCENARIO_LABELS[value] for value in SCENARIO_ORDER], rotation=32, ha="right")
    axa.set_ylabel("Run-level pass fraction")
    axa.set_ylim(-0.03, 1.03)
    axa.set_title("Support varies across scenarios", loc="left")
    clean_axis(axa)

    method_scenario = tables["method_scenario"]
    _dot_heatmap(
        axb,
        method_scenario,
        x_column="scenario",
        x_order=SCENARIO_ORDER,
        y_column="method",
        y_order=METHOD_ORDER,
        value_column="replicate_range",
        annotation_column="replicate_range",
    )
    axb.set_title("Generated-dataset support range", loc="left")

    method_claim = tables["method_claim"]
    claim_order = [claim for claim in CLAIM_LABELS if claim in set(method_claim["claim_axis"])]
    _dot_heatmap(
        axc,
        method_claim,
        x_column="claim_axis",
        x_order=claim_order,
        y_column="method",
        y_order=METHOD_ORDER,
        value_column="pass_fraction",
        annotation_column="pass_fraction",
    )
    axc.set_xticklabels([CLAIM_LABELS[value] for value in claim_order], rotation=34, ha="right")
    axc.set_title("Claim-axis support", loc="left")

    agreement = tables["decision_agreement"]
    for x, claim in enumerate(claim_order):
        values = agreement.loc[agreement["claim_axis"].eq(claim), "decision_agreement"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, len(values))
        axd.scatter(x + jitter, values, s=8, alpha=0.42, color="#4C78A8", edgecolor="none")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axd.vlines(x, q1, q3, color=INK, linewidth=2.2)
        axd.scatter(x, median, s=28, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
    axd.set_xticks(range(len(claim_order)), [CLAIM_LABELS[value] for value in claim_order], rotation=32, ha="right")
    axd.set_ylabel("Replicate-pair decision agreement")
    axd.set_ylim(-0.03, 1.03)
    axd.set_title("Claim decisions across replicates", loc="left")
    clean_axis(axd)

    metric_order = [metric for metric in METRIC_LABELS if metric in set(agreement["metric"])]
    for x, metric in enumerate(metric_order):
        values = agreement.loc[agreement["metric"].eq(metric), "value_range"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, len(values))
        axe.scatter(x + jitter, values, s=8, alpha=0.40, color="#C56A2D", edgecolor="none")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axe.vlines(x, q1, q3, color=INK, linewidth=2.2)
        axe.scatter(x, median, s=28, facecolor="white", edgecolor=INK, linewidth=0.9, zorder=4)
    axe.set_xticks(range(len(metric_order)), [METRIC_LABELS[value] for value in metric_order], rotation=36, ha="right")
    axe.set_ylabel("Max minus min across replicates")
    axe.set_title("Metric variation across replicates", loc="left")
    clean_axis(axe)

    method_summary = tables["method_summary"].sort_values("mean_run_support").reset_index(drop=True)
    label_x = float(method_summary["mean_run_support"].max()) + 0.065
    label_y = np.linspace(0.18, 0.53, len(method_summary))
    for row, y_text in zip(method_summary.itertuples(index=False), label_y):
        color = FAMILY_COLORS[row.family]
        axf.scatter(row.mean_run_support, row.worst_run_support, s=36, color=color, edgecolor="white", linewidth=0.6)
        axf.plot(
            [row.mean_run_support + 0.006, label_x - 0.008],
            [row.worst_run_support, y_text],
            color=BORDER,
            linewidth=0.55,
            zorder=1,
        )
        axf.text(label_x, y_text, row.method, ha="left", va="center", fontsize=5.0, color=INK)
    low = min(float(method_summary["worst_run_support"].min()), float(method_summary["mean_run_support"].min())) - 0.04
    diagonal_high = min(0.56, float(method_summary["mean_run_support"].max()) + 0.02)
    axf.plot([low, diagonal_high], [low, diagonal_high], color=BORDER, linestyle="--", linewidth=0.8)
    axf.set_xlim(low, label_x + 0.11)
    axf.set_ylim(0.15, 0.57)
    axf.set_xlabel("Mean run-level support")
    axf.set_ylabel("Worst run-level support")
    axf.set_title("Average support can hide a weak replicate", loc="left")
    clean_axis(axf)

    for letter, ax in zip("abcdef", axes):
        panel_label(ax, letter, x=-0.16, y=1.10)

    color_handles = [
        mpl.lines.Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=mpl.colormaps["viridis"](value), markeredgecolor="white", markersize=5, label=label)
        for value, label in [(0.0, "0"), (0.5, "0.5"), (1.0, "1")]
    ]
    axc.legend(handles=color_handles, title="Fraction or range", frameon=False, fontsize=5.2, title_fontsize=5.4, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))

    outputs = []
    for ext in ("pdf", "svg", "png", "jpg"):
        path = FIGURE_DIR / f"Supplementary_Figure_S13_simulation_replication.{ext}"
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if ext in {"png", "jpg"}:
            kwargs["dpi"] = 600
        if ext == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    table_names = (
        "metric_rows", "run_summary", "method_scenario", "decision_agreement",
        "method_claim", "metric_replication", "method_summary",
    )
    tables = {name: pd.read_csv(SOURCE / f"SuppS13_{name}.csv") for name in table_names}
    plot_figure(tables)

if __name__ == "__main__":
    main()
