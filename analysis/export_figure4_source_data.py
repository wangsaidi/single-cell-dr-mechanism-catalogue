"""Export recomputed downstream outcomes to the Figure 4 source tables."""

from __future__ import annotations

import pandas as pd

from analysis.paths import PUBLIC_SOURCE_DATA_DIR, RESULTS_DIR


METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "scVI", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
OUTPUT_DIR = PUBLIC_SOURCE_DATA_DIR / "figure4_downstream"


def write_panel(letter: str, frame: pd.DataFrame) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / f"Figure4_panel_{letter}.csv", index=False, float_format="%.10g")


def resolution_summary(metric: str) -> pd.DataFrame:
    data = pd.read_csv(RESULTS_DIR / "downstream_clustering" / "clustering_metrics_by_resolution.csv")
    data = data[
        data["dataset_id"].eq("pbmc3k")
        & data["graph_k"].eq(15)
        & data["method"].isin(METHODS)
    ]
    return (
        data.groupby(["method", "resolution"], as_index=False)[metric]
        .agg(median="median", low="min", high="max", n_embedding_seeds="count")
    )


def main() -> None:
    write_panel("a", resolution_summary("adjusted_rand_index"))
    write_panel("b", resolution_summary("macro_f1_after_cluster_majority_mapping"))

    population = pd.read_csv(RESULTS_DIR / "panel_source_data" / "population_recovery_resolution_auc.csv")
    population = population[
        population["dataset_id"].eq("pbmc3k") & population["method"].isin(METHODS)
    ].copy()
    write_panel("c", population)

    trajectory = pd.read_csv(RESULTS_DIR / "trajectory" / "trajectory_outcomes.csv")
    trajectory = trajectory[
        trajectory["root_definition"].eq("7MEP_centroid") & trajectory["method"].isin(METHODS)
    ]
    trajectory_summary = (
        trajectory.groupby("method", as_index=False)
        .agg(
            dpt_median=("reference_pseudotime_spearman", "median"),
            dpt_min=("reference_pseudotime_spearman", "min"),
            dpt_max=("reference_pseudotime_spearman", "max"),
            lineage_median=("branch_neighbour_fraction", "median"),
            lineage_min=("branch_neighbour_fraction", "min"),
            lineage_max=("branch_neighbour_fraction", "max"),
            n_embedding_seeds=("embedding_seed", "nunique"),
            n_total=("n_total", "first"),
            n_finite_min=("n_finite", "min"),
            n_finite_max=("n_finite", "max"),
            n_common_min=("n_common", "min"),
            n_common_max=("n_common", "max"),
            n_lineage_focal=("n_branch_evaluable_cells", "first"),
        )
        .set_index("method")
        .reindex(METHODS)
        .reset_index()
    )
    trajectory_summary["order_mask_definition"] = "nine-method finite intersection within each seed and root"
    write_panel("d", trajectory_summary)

    heart = pd.read_csv(RESULTS_DIR / "panel_source_data" / "heart_predictability_seed_summary.csv")
    heart = heart[heart["method"].isin(METHODS)]
    heart_panel = (
        heart.pivot_table(
            index=["method", "embedding_seed"],
            columns="endpoint",
            values="balanced_accuracy_mean",
        )
        .reset_index()
        .rename(
            columns={
                "cell_type_grouped_by_donor": "cell_type_accuracy",
                "donor_stratified_cell_split": "donor_accuracy",
            }
        )
    )
    write_panel("e", heart_panel)

    associations = pd.read_csv(
        RESULTS_DIR / "panel_source_data" / "diagnostic_outcome_associations.csv"
    )
    labels = {
        "local_retention_vs_ari": "Local retention and partition agreement",
        "label_neighbour_vs_macro_f1": "Label-neighbour agreement and population recovery",
        "marker_concordance_vs_macro_f1": "Marker concordance and population recovery",
        "trajectory_geometry_vs_dpt_order": "Continuum geometry and inferred developmental order",
    }
    dataset_labels = {
        "pbmc3k": "PBMC3k",
        "heart_cell_atlas_subsampled": "Heart atlas",
        "paul15": "Paul15",
    }
    associations["label"] = (
        associations["comparison"].map(labels)
        + " | "
        + associations["dataset_id"].map(dataset_labels)
    )
    associations = associations.sort_values(["comparison", "dataset_id"], kind="stable").reset_index(drop=True)
    write_panel("f", associations)
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
