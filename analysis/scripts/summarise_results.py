"""Create traceable panel-level summaries for the empirical analyses.

The summaries retain continuous endpoints. Resolution-grid areas are used to
avoid selecting a label-optimal Leiden resolution. Exact permutation tests in
the diagnostic-to-outcome analysis describe alignment within the nine-method
panel and are not used to infer a wider population of algorithms.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.paths import RESULTS_DIR

BASE = RESULTS_DIR
OUT = BASE / "panel_source_data"

CLUSTER_DIR = BASE / "downstream_clustering"
MARKER_DIR = BASE / "marker_concordance"
TRAJECTORY_DIR = BASE / "trajectory"
DONOR_DIR = BASE / "donor_predictability"
DIAGNOSTIC_DIR = BASE / "diagnostics"
DIMENSION_DIR = BASE / "dimension_sensitivity"

PRIMARY_K = 15


def normalized_auc(frame: pd.DataFrame, x: str, y: str) -> float:
    ordered = frame.sort_values(x)
    x_values = ordered[x].to_numpy(dtype=float)
    y_values = ordered[y].to_numpy(dtype=float)
    if x_values.size < 2 or np.isclose(x_values.max(), x_values.min()):
        return float(np.mean(y_values))
    return float(np.trapezoid(y_values, x_values) / (x_values.max() - x_values.min()))


def summarize_population_recovery() -> pd.DataFrame:
    detail = pd.read_csv(CLUSTER_DIR / "population_recovery_by_resolution.csv")
    detail = detail[detail["graph_k"].eq(PRIMARY_K)].copy()
    rows = []
    for keys, values in detail.groupby(
        ["dataset_id", "method", "embedding_seed", "label", "n_cells_label", "prevalence"], sort=False
    ):
        dataset_id, method, seed, label, n_cells, prevalence = keys
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "embedding_seed": seed,
                "label": label,
                "n_cells_label": int(n_cells),
                "prevalence": float(prevalence),
                "precision_resolution_auc": normalized_auc(values, "resolution", "precision"),
                "recall_resolution_auc": normalized_auc(values, "resolution", "recall"),
                "f1_resolution_auc": normalized_auc(values, "resolution", "f1"),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "population_recovery_resolution_auc.csv", index=False)
    return output


def summarize_marker_concordance() -> pd.DataFrame:
    detail = pd.read_csv(MARKER_DIR / "marker_concordance_by_resolution.csv")
    rows = []
    metrics = [
        "cluster_weighted_marker_concordance",
        "unweighted_marker_concordance",
        "median_expected_program_smd",
        "median_expected_program_rank",
    ]
    for keys, values in detail.groupby(["dataset_id", "method", "embedding_seed"], sort=False):
        dataset_id, method, seed = keys
        row = {"dataset_id": dataset_id, "method": method, "embedding_seed": seed}
        for metric in metrics:
            row[f"{metric}_resolution_auc"] = normalized_auc(values, "resolution", metric)
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "marker_concordance_resolution_auc.csv", index=False)
    return output


def summarize_clustering() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(CLUSTER_DIR / "clustering_resolution_summaries.csv")
    seed_summary = (
        summary.groupby(["dataset_id", "method"], as_index=False)
        .agg(
            ari_auc_median=("adjusted_rand_index_resolution_auc", "median"),
            ari_auc_min=("adjusted_rand_index_resolution_auc", "min"),
            ari_auc_max=("adjusted_rand_index_resolution_auc", "max"),
            nmi_auc_median=("normalized_mutual_information_resolution_auc", "median"),
            nmi_auc_min=("normalized_mutual_information_resolution_auc", "min"),
            nmi_auc_max=("normalized_mutual_information_resolution_auc", "max"),
            macro_f1_auc_median=("macro_f1_after_cluster_majority_mapping_resolution_auc", "median"),
            macro_f1_auc_min=("macro_f1_after_cluster_majority_mapping_resolution_auc", "min"),
            macro_f1_auc_max=("macro_f1_after_cluster_majority_mapping_resolution_auc", "max"),
            n_embedding_seeds=("embedding_seed", "nunique"),
        )
    )
    seed_summary.to_csv(OUT / "clustering_seed_summary.csv", index=False)

    primary_seed = (
        summary[summary["embedding_seed"].eq(0)]
        .rename(
            columns={
                "adjusted_rand_index_resolution_auc": "ari_resolution_auc",
                "normalized_mutual_information_resolution_auc": "nmi_resolution_auc",
                "macro_f1_after_cluster_majority_mapping_resolution_auc": "macro_f1_resolution_auc",
            }
        )
        [
            [
                "dataset_id",
                "method",
                "embedding_seed",
                "ari_resolution_auc",
                "nmi_resolution_auc",
                "macro_f1_resolution_auc",
            ]
        ]
        .copy()
    )
    primary_seed.to_csv(OUT / "clustering_primary_seed_summary.csv", index=False)

    detail = pd.read_csv(CLUSTER_DIR / "clustering_metrics_by_resolution.csv")
    detail = detail[detail["embedding_seed"].eq(0)].copy()
    graph_rows = []
    for keys, values in detail.groupby(["dataset_id", "method", "graph_k"], sort=False):
        dataset_id, method, graph_k = keys
        graph_rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "graph_k": int(graph_k),
                "ari_resolution_auc": normalized_auc(values, "resolution", "adjusted_rand_index"),
                "nmi_resolution_auc": normalized_auc(values, "resolution", "normalized_mutual_information"),
                "macro_f1_resolution_auc": normalized_auc(
                    values, "resolution", "macro_f1_after_cluster_majority_mapping"
                ),
            }
        )
    graph_summary = pd.DataFrame(graph_rows)
    graph_summary.to_csv(OUT / "graph_k_sensitivity.csv", index=False)
    return primary_seed, seed_summary


def summarize_trajectory() -> pd.DataFrame:
    detail = pd.read_csv(TRAJECTORY_DIR / "trajectory_outcomes.csv")
    summary = (
        detail.groupby(["dataset_id", "method"], as_index=False)
        .agg(
            dpt_spearman_median=("reference_pseudotime_spearman", "median"),
            dpt_spearman_min=("reference_pseudotime_spearman", "min"),
            dpt_spearman_max=("reference_pseudotime_spearman", "max"),
            dpt_kendall_median=("reference_pseudotime_kendall", "median"),
            branch_neighbour_fraction_median=("branch_neighbour_fraction", "median"),
            n_embedding_seeds=("embedding_seed", "nunique"),
            dpt_spearman_finite_only_median=("reference_pseudotime_spearman_finite_only", "median"),
            dpt_spearman_finite_only_min=("reference_pseudotime_spearman_finite_only", "min"),
            dpt_spearman_finite_only_max=("reference_pseudotime_spearman_finite_only", "max"),
            n_total=("n_total", "first"), n_finite_min=("n_finite", "min"),
            n_finite_max=("n_finite", "max"), n_common_min=("n_common", "min"),
            n_common_max=("n_common", "max"), coverage_min=("coverage", "min"),
            coverage_max=("coverage", "max"), n_disconnected_min=("n_disconnected", "min"),
            n_disconnected_max=("n_disconnected", "max"),
        )
    )
    summary["correlation_mask_definition"] = "nine-method common finite cells within each embedding seed"
    summary.to_csv(OUT / "trajectory_seed_summary.csv", index=False)

    markers = pd.read_csv(TRAJECTORY_DIR / "lineage_marker_monotonicity.csv")
    marker_summary = (
        markers.groupby(["method", "lineage"], as_index=False)
        .agg(
            lineage_program_spearman_median=("lineage_program_spearman", "median"),
            lineage_program_spearman_min=("lineage_program_spearman", "min"),
            lineage_program_spearman_max=("lineage_program_spearman", "max"),
            progenitor_program_spearman_median=("progenitor_program_spearman", "median"),
            n_embedding_seeds=("embedding_seed", "nunique"),
            lineage_program_spearman_finite_only_median=("lineage_program_spearman_finite_only", "median"),
            progenitor_program_spearman_finite_only_median=("progenitor_program_spearman_finite_only", "median"),
            n_lineage_cells=("n_lineage_cells", "first"),
            n_lineage_finite_min=("n_lineage_finite", "min"), n_lineage_finite_max=("n_lineage_finite", "max"),
            n_lineage_common_min=("n_lineage_common", "min"), n_lineage_common_max=("n_lineage_common", "max"),
        )
    )
    marker_summary["correlation_mask_definition"] = "lineage cells in nine-method common finite mask and finite marker scores"
    marker_summary.to_csv(OUT / "lineage_marker_seed_summary.csv", index=False)
    return summary


def exact_spearman_permutation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    x_rank = rankdata(np.asarray(x, dtype=float))
    y_rank = rankdata(np.asarray(y, dtype=float))
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if not np.isfinite(denominator) or denominator <= 0:
        return float("nan"), float("nan"), 0
    observed = float(np.dot(x_centered, y_centered) / denominator)
    extreme = 0
    total = 0
    tolerance = 1e-12
    for permutation in itertools.permutations(y_centered.tolist()):
        value = float(np.dot(x_centered, np.asarray(permutation)) / denominator)
        extreme += abs(value) >= abs(observed) - tolerance
        total += 1
    return observed, float(extreme / total), total


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full_like(p_values, np.nan)
    finite = np.isfinite(p_values)
    if not finite.any():
        return adjusted
    finite_values = p_values[finite]
    finite_indices = np.where(finite)[0]
    order = np.argsort(finite_values)
    finite_adjusted = np.empty_like(finite_values)
    running = 1.0
    for reverse_rank, index in enumerate(order[::-1], start=1):
        rank = finite_values.size - reverse_rank + 1
        running = min(running, finite_values[index] * finite_values.size / rank)
        finite_adjusted[index] = running
    adjusted[finite_indices] = finite_adjusted
    return np.clip(adjusted, 0.0, 1.0)


def matched_seed_zero(frame: pd.DataFrame, seed_column: str, keys: list[str]) -> pd.DataFrame:
    selected = frame[frame[seed_column].eq(0)].copy()
    if selected.empty or selected.duplicated(keys).any():
        raise ValueError(f"Missing or duplicate seed-zero association inputs: {keys}")
    return selected


def validate_association_panel(frame: pd.DataFrame, columns: list[str]) -> None:
    expected = {"PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"}
    if len(frame) != 9 or set(frame.method) != expected or not np.isfinite(frame[columns].to_numpy()).all():
        raise ValueError("Association must have one finite matched seed-zero value per each of nine methods")


def diagnostic_outcome_links(
    cluster_primary_seed: pd.DataFrame,
    marker_summary: pd.DataFrame,
    trajectory_summary: pd.DataFrame,
) -> None:
    geometry = pd.read_csv(DIAGNOSTIC_DIR / "geometry_metrics_all_methods.csv")
    geometry = matched_seed_zero(geometry[geometry["output_dimension"].eq(2)], "seed", ["dataset_id", "method", "metric"])
    geometry_wide = geometry.pivot(index=["dataset_id", "method"], columns="metric", values="value").reset_index()

    cluster = matched_seed_zero(cluster_primary_seed, "embedding_seed", ["dataset_id", "method"])
    marker = matched_seed_zero(marker_summary, "embedding_seed", ["dataset_id", "method"])
    links: list[dict[str, object]] = []
    scatter_rows: list[pd.DataFrame] = []

    definitions = [
        ("local_retention_vs_ari", "local_retention", "ari_resolution_auc", ["pbmc3k", "heart_cell_atlas_subsampled"]),
        (
            "label_neighbour_vs_macro_f1",
            "label_neighbor_recall",
            "macro_f1_resolution_auc",
            ["pbmc3k", "heart_cell_atlas_subsampled"],
        ),
    ]
    for comparison, diagnostic, outcome, datasets in definitions:
        for dataset_id in datasets:
            merged = geometry_wide[geometry_wide["dataset_id"].eq(dataset_id)].merge(
                cluster[cluster["dataset_id"].eq(dataset_id)], on=["dataset_id", "method"], how="inner"
            )
            validate_association_panel(merged, [diagnostic, outcome])
            coefficient, p_value, permutations = exact_spearman_permutation(
                merged[diagnostic].to_numpy(), merged[outcome].to_numpy()
            )
            links.append(
                {
                    "comparison": comparison,
                    "dataset_id": dataset_id,
                    "diagnostic": diagnostic,
                    "outcome": outcome,
                    "spearman_rho": coefficient,
                    "exact_two_sided_permutation_p": p_value,
                    "n_methods": int(merged.shape[0]),
                    "n_exact_permutations": permutations,
                }
            )
            scatter_rows.append(
                merged[["dataset_id", "method", diagnostic, outcome]].rename(
                    columns={diagnostic: "diagnostic_value", outcome: "outcome_value"}
                ).assign(comparison=comparison, diagnostic=diagnostic, outcome=outcome)
            )

    for dataset_id in ["pbmc3k", "heart_cell_atlas_subsampled"]:
        merged = marker[marker["dataset_id"].eq(dataset_id)].merge(
            cluster[cluster["dataset_id"].eq(dataset_id)], on=["dataset_id", "method"], how="inner"
        )
        diagnostic = "cluster_weighted_marker_concordance_resolution_auc"
        outcome = "macro_f1_resolution_auc"
        validate_association_panel(merged, [diagnostic, outcome])
        coefficient, p_value, permutations = exact_spearman_permutation(
            merged[diagnostic].to_numpy(), merged[outcome].to_numpy()
        )
        links.append(
            {
                "comparison": "marker_concordance_vs_macro_f1",
                "dataset_id": dataset_id,
                "diagnostic": diagnostic,
                "outcome": outcome,
                "spearman_rho": coefficient,
                "exact_two_sided_permutation_p": p_value,
                "n_methods": int(merged.shape[0]),
                "n_exact_permutations": permutations,
            }
        )
        scatter_rows.append(
            merged[["dataset_id", "method", diagnostic, outcome]].rename(
                columns={diagnostic: "diagnostic_value", outcome: "outcome_value"}
            ).assign(
                comparison="marker_concordance_vs_macro_f1",
                diagnostic=diagnostic,
                outcome=outcome,
            )
        )

    trajectory_geometry = pd.read_csv(DIAGNOSTIC_DIR / "trajectory_geometry_metrics_all_methods.csv")
    trajectory_geometry = matched_seed_zero(
        trajectory_geometry[trajectory_geometry["metric"].eq("pseudotime_rank_corr")], "seed", ["method"]
    ).rename(columns={"value": "pseudotime_rank_corr"})
    trajectory_detail = matched_seed_zero(
        pd.read_csv(TRAJECTORY_DIR / "trajectory_outcomes.csv"), "embedding_seed", ["method"]
    )
    if trajectory_detail["n_common"].nunique() != 1 or not trajectory_detail["n_comparison_methods"].eq(9).all():
        raise ValueError("Paul15 association requires the same nine-method common-cell comparison")
    outcome = "dpt_spearman_seed0_common"
    trajectory_detail = trajectory_detail.rename(columns={"reference_pseudotime_spearman_common": outcome})
    merged = trajectory_geometry[["method", "pseudotime_rank_corr"]].merge(
        trajectory_detail[["method", outcome, "n_common"]], on="method", how="inner", validate="one_to_one"
    )
    validate_association_panel(merged, ["pseudotime_rank_corr", outcome])
    coefficient, p_value, permutations = exact_spearman_permutation(
        merged["pseudotime_rank_corr"].to_numpy(), merged[outcome].to_numpy()
    )
    links.append(
        {
            "comparison": "trajectory_geometry_vs_dpt_order",
            "dataset_id": "paul15",
            "diagnostic": "pseudotime_rank_corr",
            "outcome": outcome,
            "spearman_rho": coefficient,
            "exact_two_sided_permutation_p": p_value,
            "n_methods": int(merged.shape[0]),
            "n_exact_permutations": permutations,
            "n_outcome_common_cells": int(merged.n_common.iloc[0]),
            "outcome_mask_definition": "finite intersection across all nine seed-zero methods and reference",
        }
    )
    scatter_rows.append(
        merged[["method", "pseudotime_rank_corr", outcome, "n_common"]]
        .rename(columns={"pseudotime_rank_corr": "diagnostic_value", outcome: "outcome_value"})
        .assign(
            dataset_id="paul15",
            comparison="trajectory_geometry_vs_dpt_order",
            diagnostic="pseudotime_rank_corr",
            outcome=outcome,
        )
    )

    link_table = pd.DataFrame(links)
    if len(link_table) != 7:
        raise ValueError("Expected exactly seven planned association tests")
    link_table["embedding_seed"] = 0
    link_table["diagnostic_seed"] = 0
    link_table["outcome_seed"] = 0
    link_table["q_bh_7_planned_tests"] = benjamini_hochberg(
        link_table["exact_two_sided_permutation_p"].to_numpy()
    )
    link_table["inference_scope"] = "descriptive alignment within the nine evaluated implementations"
    link_table.to_csv(OUT / "diagnostic_outcome_associations.csv", index=False)
    pd.concat(scatter_rows, ignore_index=True).assign(embedding_seed=0, diagnostic_seed=0, outcome_seed=0).to_csv(
        OUT / "diagnostic_outcome_scatter_data.csv", index=False
    )


def copy_direct_summaries() -> None:
    for source, target in [
        (DONOR_DIR / "heart_donor_bootstrap_summary.csv", OUT / "heart_donor_bootstrap_summary.csv"),
        (DONOR_DIR / "heart_cell_type_predictability_by_donor.csv", OUT / "heart_predictability_by_donor.csv"),
        (DONOR_DIR / "heart_predictability_summary.csv", OUT / "heart_predictability_seed_summary.csv"),
        (TRAJECTORY_DIR / "trajectory_root_sensitivity.csv", OUT / "trajectory_root_sensitivity.csv"),
        (CLUSTER_DIR / "clustering_seed_stability.csv", OUT / "clustering_seed_stability.csv"),
        (DIMENSION_DIR / "output_dimension_clustering_summary.csv", OUT / "output_dimension_clustering_summary.csv"),
        (DIMENSION_DIR / "output_dimension_trajectory_outcomes.csv", OUT / "output_dimension_trajectory_outcomes.csv"),
    ]:
        pd.read_csv(source).to_csv(target, index=False)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    population = summarize_population_recovery()
    marker = summarize_marker_concordance()
    cluster_primary_seed, _ = summarize_clustering()
    trajectory = summarize_trajectory()
    diagnostic_outcome_links(cluster_primary_seed, marker, trajectory)
    copy_direct_summaries()
    manifest = {
        "primary_graph_k": PRIMARY_K,
        "resolution_summary": "normalised trapezoidal area across the fixed resolution grid",
        "population_rows": int(population.shape[0]),
        "marker_rows": int(marker.shape[0]),
        "association_family": "seven planned diagnostic-to-outcome comparisons",
        "association_multiplicity": "Benjamini-Hochberg correction across seven exact permutation tests",
        "association_embedding_seed": 0,
        "trajectory_correlation_mask": "finite common-cell intersection across nine methods per seed/root; finite-only sensitivity retained",
    }
    (OUT / "panel_source_data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
