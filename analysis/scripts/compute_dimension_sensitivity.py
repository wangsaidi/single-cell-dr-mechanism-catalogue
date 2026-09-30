"""Compare downstream results across available output dimensions.

PCA, UMAP, PHATE and PaCMAP use the audited seed-zero representations already
generated for the submitted dimensionality analysis. scVI contributes direct
two- and ten-dimensional latents over five seeds. The unequal design is kept
explicit and is used as a sensitivity analysis, not as a method ranking.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from analysis.paths import MANIFEST_PATH, REPO_ROOT, RESULTS_DIR, ROBUSTNESS_EMBEDDINGS_DIR
from analysis.scripts.compute_clustering import build_graph, evaluate_partition, resolution_auc
from analysis.scripts.compute_trajectory import (
    K,
    PRIMARY_ROOT,
    PROC_PATH,
    branch_neighbour_fraction,
    coarse_branches,
    reference_dpt,
    root_definitions,
    run_dpt,
    safe_kendall,
    safe_spearman,
)


ROOT = REPO_ROOT
MANIFEST = MANIFEST_PATH
EMBEDDINGS = RESULTS_DIR / "embeddings"
ROBUST_EMBEDDINGS = ROBUSTNESS_EMBEDDINGS_DIR
OUT_DIR = RESULTS_DIR / "dimension_sensitivity"

DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP", "scVI"]
DIMENSIONS = [2, 5, 10, 20]
SEEDS = [0, 1, 2, 3, 4]
RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
GRAPH_K = 15
GRAPH_SEED = 0
LEIDEN_ITERATIONS = -1


def available_seeds(method: str, dimension: int, requested: list[int]) -> list[int]:
    if method == "scVI":
        return requested if dimension in [2, 10] else []
    return [0]


def embedding_path(dataset_id: str, method: str, dimension: int, seed: int) -> Path:
    if method == "scVI":
        return EMBEDDINGS / f"{dataset_id}_scvi_dim{dimension}_seed{seed}.npy"
    return ROBUST_EMBEDDINGS / f"dim_{dataset_id}_{method}_{dimension}.npy"


def run_clustering(datasets: list[str], methods: list[str], dimensions: list[int], seeds: list[int]) -> None:
    manifest = pd.read_csv(MANIFEST).set_index("dataset_id")
    rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []

    for dataset_id in datasets:
        record = manifest.loc[dataset_id]
        proc = sc.read_h5ad(ROOT / str(record["analysis_proc_path"]), backed="r")
        labels = proc.obs[str(record["label_field"])].astype(str).to_numpy()
        proc.file.close()
        for method in methods:
            for dimension in dimensions:
                for seed in available_seeds(method, dimension, seeds):
                    coords = np.load(embedding_path(dataset_id, method, dimension, seed)).astype(np.float32)
                    if coords.shape != (labels.size, dimension):
                        raise ValueError(
                            f"Unexpected shape for {dataset_id} {method} dim={dimension} seed={seed}: {coords.shape}"
                        )
                    work = build_graph(coords, GRAPH_K)
                    for resolution in RESOLUTIONS:
                        key = f"leiden_{resolution:g}"
                        sc.tl.leiden(
                            work,
                            resolution=resolution,
                            key_added=key,
                            random_state=GRAPH_SEED,
                            flavor="igraph",
                            directed=False,
                            n_iterations=LEIDEN_ITERATIONS,
                        )
                        clusters = work.obs[key].astype(str).to_numpy()
                        global_row, per_population = evaluate_partition(
                            dataset_id,
                            method,
                            seed,
                            GRAPH_K,
                            resolution,
                            clusters,
                            labels,
                        )
                        global_row["output_dimension"] = dimension
                        for row in per_population:
                            row["output_dimension"] = dimension
                        rows.append(global_row)
                        population_rows.extend(per_population)
                    print(
                        f"completed dimension clustering {dataset_id} {method} dim={dimension} seed={seed}",
                        flush=True,
                    )

    detail = pd.DataFrame(rows)
    population = pd.DataFrame(population_rows)
    detail.to_csv(OUT_DIR / "output_dimension_clustering_by_resolution.csv", index=False)
    population.to_csv(OUT_DIR / "output_dimension_population_recovery.csv", index=False)

    summary_rows = []
    metrics = [
        "adjusted_rand_index",
        "normalized_mutual_information",
        "macro_f1_after_cluster_majority_mapping",
        "weighted_f1_after_cluster_majority_mapping",
    ]
    for keys, values in detail.groupby(
        ["dataset_id", "method", "output_dimension", "embedding_seed"], sort=False
    ):
        dataset_id, method, dimension, seed = keys
        row = {
            "dataset_id": dataset_id,
            "method": method,
            "output_dimension": dimension,
            "embedding_seed": seed,
            "graph_k": GRAPH_K,
        }
        for metric in metrics:
            row[f"{metric}_resolution_auc"] = resolution_auc(values, metric)
            row[f"{metric}_resolution_median"] = float(values[metric].median())
        summary_rows.append(row)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "output_dimension_clustering_summary.csv", index=False)


def run_trajectory(methods: list[str], dimensions: list[int], seeds: list[int]) -> None:
    proc = sc.read_h5ad(PROC_PATH)
    labels = proc.obs["paul15_clusters"].astype(str).to_numpy()
    branches = coarse_branches(labels)
    roots = root_definitions(proc)
    reference = reference_dpt(proc, roots[PRIMARY_ROOT])
    evaluable = ~np.isin(branches, ["other", "progenitor"])
    rows = []
    failures = []
    for method in methods:
        for dimension in dimensions:
            for seed in available_seeds(method, dimension, seeds):
                coords = np.load(embedding_path("paul15", method, dimension, seed)).astype(np.float32)
                try:
                    pseudotime, _ = run_dpt(coords, roots[PRIMARY_ROOT])
                except Exception as exc:
                    failures.append(
                        {"method": method, "output_dimension": dimension, "seed": seed, "error": repr(exc)}
                    )
                    continue
                rows.append(
                    {
                        "dataset_id": "paul15",
                        "method": method,
                        "output_dimension": dimension,
                        "embedding_seed": seed,
                        "root_definition": PRIMARY_ROOT,
                        "reference_pseudotime_spearman": safe_spearman(reference, pseudotime),
                        "reference_pseudotime_kendall": safe_kendall(reference, pseudotime),
                        "branch_neighbour_fraction": branch_neighbour_fraction(coords, branches, K),
                        "n_cells": int(proc.n_obs),
                        "n_branch_evaluable_cells": int(evaluable.sum()),
                    }
                )
                print(
                    f"completed dimension trajectory {method} dim={dimension} seed={seed}", flush=True
                )
    pd.DataFrame(rows).to_csv(OUT_DIR / "output_dimension_trajectory_outcomes.csv", index=False)
    (OUT_DIR / "output_dimension_trajectory_failures.json").write_text(
        json.dumps(failures, indent=2), encoding="utf-8"
    )


def run(datasets: list[str], methods: list[str], dimensions: list[int], seeds: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_clustering(datasets, methods, dimensions, seeds)
    run_trajectory(methods, dimensions, seeds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--dimensions", nargs="+", type=int, choices=DIMENSIONS, default=DIMENSIONS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.datasets, args.methods, args.dimensions, args.seeds)


if __name__ == "__main__":
    main()
