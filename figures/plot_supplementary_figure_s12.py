"""Resolve robustness results into baseline-to-condition support transitions."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.figure_style import (  # noqa: E402
    BORDER,
    FAIL_COLOR,
    GRID,
    INK,
    PASS_COLOR,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    panel_label,
)


BASE_SOURCE = ROOT / "data" / "source_data"
SOURCE = ROOT / "data" / "source_data" / "generated"
FIGURE_DIR = ROOT / "outputs" / "supplementary_figures"
META_DIR = ROOT / "metadata"

AXIS_LABELS = {
    "output_dimension": "Output dimension",
    "upstream_pca": "Upstream PCA",
    "expression_perturbation": "Expression perturbation",
}
AXIS_COLORS = {
    "output_dimension": "#3572A5",
    "upstream_pca": "#7A5195",
    "expression_perturbation": "#C56A2D",
}
TRANSITION_ORDER = ["pass_to_pass", "pass_to_fail", "fail_to_pass", "fail_to_fail"]
TRANSITION_LABELS = {
    "pass_to_pass": "Pass to pass",
    "pass_to_fail": "Pass to fail",
    "fail_to_pass": "Fail to pass",
    "fail_to_fail": "Fail to fail",
}
TRANSITION_COLORS = {
    "pass_to_pass": PASS_COLOR,
    "pass_to_fail": FAIL_COLOR,
    "fail_to_pass": "#4C78A8",
    "fail_to_fail": "#B8BDC6",
}
METRIC_LABELS = {
    "local_retention": "Local retention",
    "trustworthiness": "Trustworthiness",
    "global_rank_corr": "Global rank",
    "label_neighbor_recall": "Label recall",
}
METHOD_ORDER = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = SOURCE / name
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def _transition_frame(
    frame: pd.DataFrame,
    *,
    analysis_axis: str,
    condition_column: str,
    baseline_value: float,
    extra_key: list[str] | None = None,
) -> pd.DataFrame:
    extra_key = extra_key or []
    key = ["dataset_id", "method", "family", "metric", *extra_key]
    baseline_mask = np.isclose(frame[condition_column].astype(float), baseline_value)
    baseline = frame.loc[baseline_mask, key + ["value", "threshold", "support"]].copy()
    if baseline.duplicated(key).any():
        raise AssertionError(f"Duplicate baseline rows for {analysis_axis}")
    baseline = baseline.rename(
        columns={
            "value": "baseline_value",
            "threshold": "baseline_threshold",
            "support": "baseline_support",
        }
    )
    conditions = frame.loc[~baseline_mask].copy()
    merged = conditions.merge(baseline, on=key, how="left", validate="many_to_one")
    if merged["baseline_support"].isna().any():
        raise AssertionError(f"Missing matched baseline rows for {analysis_axis}")
    if not np.allclose(merged["threshold"], merged["baseline_threshold"], rtol=0, atol=1e-12):
        raise AssertionError(f"Threshold changed between baseline and conditions for {analysis_axis}")

    merged["analysis_axis_resolved"] = analysis_axis
    merged["condition_parameter"] = condition_column
    merged["condition_numeric"] = merged[condition_column].astype(float)
    if analysis_axis == "output_dimension":
        merged["condition_label"] = merged[condition_column].map(lambda value: f"d={int(value)}")
    elif analysis_axis == "upstream_pca":
        merged["condition_label"] = merged[condition_column].map(lambda value: f"PCA{int(value)}")
    else:
        merged["condition_label"] = merged.apply(
            lambda row: f"{str(row['perturbation']).capitalize()} {float(row[condition_column]):.1f}",
            axis=1,
        )
    merged["baseline_pass"] = merged["baseline_support"].eq("pass")
    merged["condition_pass"] = merged["support"].eq("pass")
    merged["transition"] = np.select(
        [
            merged["baseline_pass"] & merged["condition_pass"],
            merged["baseline_pass"] & ~merged["condition_pass"],
            ~merged["baseline_pass"] & merged["condition_pass"],
        ],
        ["pass_to_pass", "pass_to_fail", "fail_to_pass"],
        default="fail_to_fail",
    )
    merged["baseline_margin"] = merged["baseline_value"] - merged["baseline_threshold"]
    merged["condition_margin"] = merged["value"] - merged["threshold"]
    merged["margin_change"] = merged["condition_margin"] - merged["baseline_margin"]
    merged["n_definition"] = (
        "one method-dataset-metric-condition comparison relative to its matched baseline; "
        "comparisons are computational conditions, not independent biological replicates"
    )
    return merged


def build_transitions() -> pd.DataFrame:
    output = pd.read_csv(BASE_SOURCE / "fig6_output_dimension_response.csv")
    upstream = pd.read_csv(BASE_SOURCE / "fig6_upstream_pca_response.csv")
    perturbation = pd.read_csv(BASE_SOURCE / "fig6_dropout_noise_response.csv")
    frames = [
        _transition_frame(
            output,
            analysis_axis="output_dimension",
            condition_column="output_dimension",
            baseline_value=2,
        ),
        _transition_frame(
            upstream,
            analysis_axis="upstream_pca",
            condition_column="upstream_pca_dimension",
            baseline_value=50,
        ),
        _transition_frame(
            perturbation,
            analysis_axis="expression_perturbation",
            condition_column="level",
            baseline_value=0,
            extra_key=["perturbation", "replicate"],
        ),
    ]
    transitions = pd.concat(frames, ignore_index=True, sort=False)
    transitions = transitions.sort_values(
        ["analysis_axis_resolved", "method", "dataset_id", "metric", "condition_label"]
    ).reset_index(drop=True)
    return transitions


def _summaries(transitions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    transition_summary = (
        transitions.groupby(["analysis_axis_resolved", "transition"], as_index=False)
        .size()
        .rename(columns={"size": "n_comparisons"})
    )
    totals = transition_summary.groupby("analysis_axis_resolved")["n_comparisons"].transform("sum")
    transition_summary["fraction"] = transition_summary["n_comparisons"] / totals

    def loss_summary(group_columns: list[str]) -> pd.DataFrame:
        rows = []
        for keys, group in transitions.groupby(group_columns, sort=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            eligible = group["baseline_pass"]
            lost = group["transition"].eq("pass_to_fail")
            recovered_eligible = ~group["baseline_pass"]
            recovered = group["transition"].eq("fail_to_pass")
            row = dict(zip(group_columns, keys))
            row.update(
                {
                    "n_conditions": int(len(group)),
                    "n_baseline_supported": int(eligible.sum()),
                    "n_pass_to_fail": int(lost.sum()),
                    "support_loss_fraction": float(lost.sum() / eligible.sum()) if eligible.sum() else np.nan,
                    "n_baseline_unsupported": int(recovered_eligible.sum()),
                    "n_fail_to_pass": int(recovered.sum()),
                    "support_recovery_fraction": (
                        float(recovered.sum() / recovered_eligible.sum()) if recovered_eligible.sum() else np.nan
                    ),
                    "n_definition": (
                        "non-baseline computational conditions matched to the corresponding baseline; "
                        "conditions are not independent biological replicates"
                    ),
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    return {
        "transition_summary": transition_summary,
        "axis_loss": loss_summary(["analysis_axis_resolved"]),
        "method_loss": loss_summary(["analysis_axis_resolved", "method", "family"]),
        "metric_loss": loss_summary(["analysis_axis_resolved", "metric"]),
        "condition_loss": loss_summary(["analysis_axis_resolved", "condition_label", "condition_numeric"]),
    }


def _dot_matrix(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    row_column: str,
    row_order: list[str],
    row_labels: dict[str, str] | None = None,
) -> None:
    axes = list(AXIS_LABELS)
    row_labels = row_labels or {}
    shown = frame[frame["n_baseline_supported"].gt(0)].copy()
    for row in shown.itertuples(index=False):
        y = row_order.index(getattr(row, row_column))
        x = axes.index(row.analysis_axis_resolved)
        fraction = float(row.support_loss_fraction)
        size = 25 + 95 * min(1.0, row.n_baseline_supported / max(1, shown["n_baseline_supported"].max()))
        ax.scatter(
            x,
            y,
            s=size,
            c=[fraction],
            cmap="magma_r",
            vmin=0,
            vmax=1,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.text(x, y, f"{row.n_pass_to_fail}/{row.n_baseline_supported}", ha="center", va="center", fontsize=5.2, color="white" if fraction > 0.38 else INK)
    ineligible = frame[frame["n_baseline_supported"].eq(0)]
    for row in ineligible.itertuples(index=False):
        y = row_order.index(getattr(row, row_column))
        x = axes.index(row.analysis_axis_resolved)
        ax.text(x, y, "0 eligible", ha="center", va="center", fontsize=4.9, color=TEXT_MUTED)
    ax.set_xticks(range(len(axes)), [AXIS_LABELS[value] for value in axes], rotation=22, ha="right")
    ax.set_yticks(range(len(row_order)), [row_labels.get(value, value) for value in row_order])
    ax.set_xlim(-0.55, len(axes) - 0.45)
    ax.set_ylim(len(row_order) - 0.5, -0.5)
    ax.grid(False)
    for x in np.arange(-0.5, len(axes), 1):
        ax.axvline(x, color=GRID, linewidth=0.45, zorder=0)
    for y in np.arange(-0.5, len(row_order), 1):
        ax.axhline(y, color=GRID, linewidth=0.45, zorder=0)
    clean_axis(ax, grid=False)


def plot_figure(transitions: pd.DataFrame, summaries: dict[str, pd.DataFrame]) -> list[Path]:
    apply_final_style()
    fig = plt.figure(figsize=(7.2, 6.7), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, top=0.965, bottom=0.085, wspace=0.42, hspace=0.54)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]
    axa, axb, axc, axd, axe, axf = axes

    composition = summaries["transition_summary"]
    axis_order = list(AXIS_LABELS)
    bottom = np.zeros(len(axis_order))
    for transition in TRANSITION_ORDER:
        values = []
        for axis in axis_order:
            match = composition[
                composition["analysis_axis_resolved"].eq(axis) & composition["transition"].eq(transition)
            ]
            values.append(float(match["fraction"].iloc[0]) if len(match) else 0.0)
        axa.bar(range(len(axis_order)), values, bottom=bottom, color=TRANSITION_COLORS[transition], width=0.68, label=TRANSITION_LABELS[transition])
        bottom += np.asarray(values)
    axa.set_xticks(range(len(axis_order)), [AXIS_LABELS[value] for value in axis_order], rotation=22, ha="right")
    axa.set_ylabel("Fraction of matched conditions")
    axa.set_ylim(0, 1)
    axa.legend(frameon=False, fontsize=5.6, ncol=2, loc="upper center", bbox_to_anchor=(0.50, -0.29))
    axa.set_title("Baseline-to-condition transitions", loc="left")
    clean_axis(axa)

    axis_loss = summaries["axis_loss"].set_index("analysis_axis_resolved").reindex(axis_order)
    bars = axb.bar(
        range(len(axis_order)),
        axis_loss["support_loss_fraction"],
        color=[AXIS_COLORS[value] for value in axis_order],
        width=0.65,
    )
    for bar, row in zip(bars, axis_loss.itertuples()):
        axb.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, f"{row.n_pass_to_fail}/{row.n_baseline_supported}", ha="center", va="bottom", fontsize=6)
    axb.set_xticks(range(len(axis_order)), [AXIS_LABELS[value] for value in axis_order], rotation=22, ha="right")
    axb.set_ylabel("Pass-to-fail fraction")
    axb.set_ylim(0, min(1.05, max(0.25, float(axis_loss["support_loss_fraction"].max()) + 0.16)))
    axb.set_title("Loss among supported baselines", loc="left")
    clean_axis(axb)

    method_loss = summaries["method_loss"]
    method_order = [method for method in METHOD_ORDER if method in set(method_loss["method"])]
    _dot_matrix(axc, method_loss, row_column="method", row_order=method_order)
    axc.set_title("Method-specific support loss", loc="left")

    metric_loss = summaries["metric_loss"]
    metric_order = [metric for metric in METRIC_LABELS if metric in set(metric_loss["metric"])]
    _dot_matrix(axd, metric_loss, row_column="metric", row_order=metric_order, row_labels=METRIC_LABELS)
    axd.set_title("Diagnostic-specific support loss", loc="left")

    condition_loss = summaries["condition_loss"].copy()
    condition_loss = condition_loss[condition_loss["n_baseline_supported"].gt(0)]
    condition_loss["sort_axis"] = condition_loss["analysis_axis_resolved"].map({value: i for i, value in enumerate(axis_order)})
    condition_loss = condition_loss.sort_values(["sort_axis", "condition_numeric", "condition_label"]).reset_index(drop=True)
    x = np.arange(len(condition_loss))
    axe.plot(x, condition_loss["support_loss_fraction"], color=BORDER, linewidth=0.7, zorder=1)
    for axis in axis_order:
        mask = condition_loss["analysis_axis_resolved"].eq(axis)
        axe.scatter(x[mask], condition_loss.loc[mask, "support_loss_fraction"], s=30, color=AXIS_COLORS[axis], edgecolor="white", linewidth=0.6, label=AXIS_LABELS[axis], zorder=2)
    axe.set_xticks(x, condition_loss["condition_label"], rotation=38, ha="right")
    axe.set_ylabel("Pass-to-fail fraction")
    axe.set_ylim(-0.03, 1.03)
    axe.set_title("Loss across non-baseline settings", loc="left")
    clean_axis(axe)

    positions = np.arange(len(axis_order))
    rng = np.random.default_rng(20260714)
    for position, axis in zip(positions, axis_order):
        values = transitions.loc[transitions["analysis_axis_resolved"].eq(axis), "margin_change"].astype(float).to_numpy()
        jitter = rng.uniform(-0.14, 0.14, size=len(values))
        axf.scatter(position + jitter, values, s=5, alpha=0.24, color=AXIS_COLORS[axis], edgecolor="none")
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axf.vlines(position, q1, q3, color=INK, linewidth=2.2, zorder=3)
        axf.scatter(position, median, s=28, color="white", edgecolor=INK, linewidth=0.9, zorder=4)
    axf.axhline(0, color=BORDER, linewidth=0.8, linestyle="--")
    axf.set_xticks(positions, [AXIS_LABELS[value] for value in axis_order], rotation=22, ha="right")
    axf.set_ylabel("Change in threshold margin")
    axf.set_title("Condition-induced margin shifts", loc="left")
    clean_axis(axf)

    for letter, ax in zip("abcdef", axes):
        panel_label(ax, letter, x=-0.16, y=1.10)

    dot_legend = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=mpl.colormaps["magma_r"](value), markeredgecolor="white", markersize=5, label=label)
        for value, label in [(0.0, "No loss"), (0.5, "50% loss"), (1.0, "Complete loss")]
    ]
    axc.legend(handles=dot_legend, frameon=False, fontsize=5.4, loc="upper center", bbox_to_anchor=(0.50, -0.27), ncol=3)
    axd.text(0.0, -0.29, "Labels show lost / baseline-supported comparisons", transform=axd.transAxes, fontsize=5.5, color=TEXT_MUTED, ha="left")

    outputs = []
    for ext in ("pdf", "svg", "png", "jpg"):
        path = FIGURE_DIR / f"Supplementary_Figure_S12_robustness_transitions.{ext}"
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
    transitions = pd.read_csv(SOURCE / "SuppS12_robustness_transitions.csv")
    summaries = {
        "transition_summary": pd.read_csv(SOURCE / "SuppS12_transition_summary.csv"),
        "axis_loss": pd.read_csv(SOURCE / "SuppS12_axis_loss.csv"),
        "method_loss": pd.read_csv(SOURCE / "SuppS12_method_loss.csv"),
        "metric_loss": pd.read_csv(SOURCE / "SuppS12_metric_loss.csv"),
        "condition_loss": pd.read_csv(SOURCE / "SuppS12_condition_loss.csv"),
    }
    plot_figure(transitions, summaries)

if __name__ == "__main__":
    main()
