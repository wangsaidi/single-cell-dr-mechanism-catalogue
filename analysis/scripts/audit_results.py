"""Audit completeness and numerical validity of the major-revision results."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import LOGS_DIR, RESULTS_DIR

BASE = RESULTS_DIR
LOGS = LOGS_DIR
METHODS = {"PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"}
DATASETS = {"pbmc3k", "paul15", "heart_cell_atlas_subsampled"}
ROOTS = {"7MEP_centroid", "9GMP_centroid", "1Ery_centroid", "stored_8Mk_root"}
SEEDS = set(range(5))


def root_grid_errors(frame: pd.DataFrame) -> list[str]:
    columns = ["method", "embedding_seed", "root_definition"]
    if not set(columns).issubset(frame.columns):
        return ["root sensitivity: missing method/seed/root key columns"]
    expected = set(product(METHODS, SEEDS, ROOTS))
    observed = set(frame[columns].itertuples(index=False, name=None))
    errors = []
    if len(frame) != len(expected):
        errors.append(f"root sensitivity rows {len(frame)} != {len(expected)}")
    if frame.duplicated(columns).any():
        errors.append("root sensitivity: duplicate method/seed/root keys")
    if observed != expected:
        errors.append(f"root sensitivity: {len(expected-observed)} missing and {len(observed-expected)} unexpected keys")
    return errors


def read(relative: str) -> pd.DataFrame:
    path = BASE / relative
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def finite_columns(frame: pd.DataFrame, columns: list[str], name: str, errors: list[str]) -> None:
    for column in columns:
        if column not in frame:
            errors.append(f"{name}: missing column {column}")
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            errors.append(f"{name}: non-finite values in {column}")


def check_methods(frame: pd.DataFrame, name: str, errors: list[str]) -> None:
    observed = set(frame["method"].dropna().astype(str))
    if observed != METHODS:
        errors.append(f"{name}: methods {sorted(observed)} != {sorted(METHODS)}")


def main() -> None:
    errors: list[str] = []
    checks: dict[str, object] = {}

    clustering = read("downstream_clustering/clustering_metrics_by_resolution.csv")
    checks["clustering_rows"] = int(clustering.shape[0])
    if clustering.shape[0] != 1728:
        errors.append(f"clustering rows {clustering.shape[0]} != 1728")
    check_methods(clustering, "clustering", errors)
    if set(clustering["dataset_id"].astype(str)) != DATASETS:
        errors.append("clustering: incomplete dataset coverage")
    finite_columns(
        clustering,
        [
            "adjusted_rand_index",
            "normalized_mutual_information",
            "macro_f1_after_cluster_majority_mapping",
            "weighted_f1_after_cluster_majority_mapping",
        ],
        "clustering",
        errors,
    )

    cluster_summary = read("downstream_clustering/clustering_resolution_summaries.csv")
    checks["clustering_summary_rows"] = int(cluster_summary.shape[0])
    if cluster_summary.shape[0] != 135:
        errors.append(f"clustering summary rows {cluster_summary.shape[0]} != 135")

    stability = read("downstream_clustering/clustering_seed_stability.csv")
    checks["clustering_seed_stability_rows"] = int(stability.shape[0])
    if stability.shape[0] != 2160:
        errors.append(f"clustering seed-stability rows {stability.shape[0]} != 2160")
    finite_columns(
        stability,
        ["partition_adjusted_rand_index", "partition_normalized_mutual_information"],
        "clustering stability",
        errors,
    )

    population = read("downstream_clustering/population_recovery_by_resolution.csv")
    checks["population_rows"] = int(population.shape[0])
    check_methods(population, "population recovery", errors)
    finite_columns(population, ["precision", "recall", "f1", "prevalence"], "population recovery", errors)

    marker = read("marker_concordance/marker_concordance_by_resolution.csv")
    checks["marker_summary_rows"] = int(marker.shape[0])
    if marker.shape[0] != 144:
        errors.append(f"marker summary rows {marker.shape[0]} != 144")
    check_methods(marker, "marker concordance", errors)
    finite_columns(
        marker,
        [
            "cluster_weighted_marker_concordance",
            "unweighted_marker_concordance",
            "median_expected_program_smd",
            "median_expected_program_rank",
        ],
        "marker concordance",
        errors,
    )

    trajectory = read("trajectory/trajectory_outcomes.csv")
    checks["trajectory_rows"] = int(trajectory.shape[0])
    if trajectory.shape[0] != 45:
        errors.append(f"trajectory rows {trajectory.shape[0]} != 45")
    check_methods(trajectory, "trajectory", errors)
    finite_columns(
        trajectory,
        ["reference_pseudotime_spearman", "reference_pseudotime_kendall", "branch_neighbour_fraction"],
        "trajectory",
        errors,
    )

    lineage = read("trajectory/lineage_marker_monotonicity.csv")
    checks["lineage_marker_rows"] = int(lineage.shape[0])
    if lineage.shape[0] != 180:
        errors.append(f"lineage marker rows {lineage.shape[0]} != 180")
    finite_columns(
        lineage,
        ["lineage_program_spearman", "progenitor_program_spearman"],
        "lineage markers",
        errors,
    )

    roots = read("trajectory/trajectory_root_sensitivity.csv")
    checks["root_sensitivity_rows"] = int(roots.shape[0])
    errors.extend(root_grid_errors(roots))
    finite_columns(
        roots,
        ["reference_pseudotime_spearman", "primary_root_pseudotime_concordance"],
        "root sensitivity",
        errors,
    )

    donor_folds = read("donor_predictability/heart_predictability_folds.csv")
    checks["donor_fold_rows"] = int(donor_folds.shape[0])
    if donor_folds.shape[0] != 450:
        errors.append(f"donor fold rows {donor_folds.shape[0]} != 450")
    check_methods(donor_folds, "donor predictability", errors)
    finite_columns(
        donor_folds,
        ["balanced_accuracy", "balanced_accuracy_above_chance", "macro_f1"],
        "donor predictability",
        errors,
    )

    donor_level = read("donor_predictability/heart_cell_type_predictability_by_donor.csv")
    checks["donor_level_rows"] = int(donor_level.shape[0])
    checks["n_donors"] = int(donor_level["donor"].nunique())
    expected_donor_rows = 9 * 5 * int(donor_level["donor"].nunique())
    if donor_level.shape[0] != expected_donor_rows:
        errors.append(f"donor-level rows {donor_level.shape[0]} != {expected_donor_rows}")
    finite_columns(
        donor_level,
        ["cell_type_balanced_accuracy", "cell_type_macro_f1"],
        "donor-level predictability",
        errors,
    )

    dimension_cluster = read("dimension_sensitivity/output_dimension_clustering_summary.csv")
    checks["dimension_clustering_rows"] = int(dimension_cluster.shape[0])
    if dimension_cluster.shape[0] != 78:
        errors.append(f"dimension clustering rows {dimension_cluster.shape[0]} != 78")
    finite_columns(
        dimension_cluster,
        [
            "adjusted_rand_index_resolution_auc",
            "normalized_mutual_information_resolution_auc",
            "macro_f1_after_cluster_majority_mapping_resolution_auc",
        ],
        "dimension clustering",
        errors,
    )

    dimension_trajectory = read("dimension_sensitivity/output_dimension_trajectory_outcomes.csv")
    checks["dimension_trajectory_rows"] = int(dimension_trajectory.shape[0])
    if dimension_trajectory.shape[0] != 26:
        errors.append(f"dimension trajectory rows {dimension_trajectory.shape[0]} != 26")
    finite_columns(
        dimension_trajectory,
        ["reference_pseudotime_spearman", "reference_pseudotime_kendall", "branch_neighbour_fraction"],
        "dimension trajectory",
        errors,
    )

    associations = read("panel_source_data/diagnostic_outcome_associations.csv")
    checks["association_rows"] = int(associations.shape[0])
    if associations.shape[0] != 7:
        errors.append(f"association rows {associations.shape[0]} != 7")
    if not (associations["n_methods"] == 9).all():
        errors.append("association tests do not all contain nine methods")
    estimable = associations["n_exact_permutations"].gt(0)
    if not (associations.loc[estimable, "n_exact_permutations"] == math.factorial(9)).all():
        errors.append("estimable association tests do not use all 9! permutations")
    checks["estimable_associations"] = int(estimable.sum())

    for relative in [
        "trajectory/trajectory_failures.json",
        "dimension_sensitivity/output_dimension_trajectory_failures.json",
    ]:
        failures = json.loads((BASE / relative).read_text(encoding="utf-8"))
        checks[f"{Path(relative).stem}_count"] = len(failures)
        if failures:
            errors.append(f"{relative}: {len(failures)} recorded failures")

    result = {"audit_passed": not errors, "errors": errors, "checks": checks}
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / "revision_result_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
