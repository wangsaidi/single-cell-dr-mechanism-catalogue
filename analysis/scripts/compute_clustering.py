"""Evaluate clustering and population recovery from each two-dimensional representation.

Labels are never used to construct the neighbourhood graph or the Leiden
partition. They are used only after clustering to calculate agreement and
population-recovery endpoints. Results are reported over a fixed resolution
grid instead of selecting the best label-informed resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
from itertools import combinations
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
    v_measure_score,
)

from analysis.paths import (
    ANCHOR_EMBEDDINGS_DIR,
    LOGS_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    RESULTS_DIR,
)


ROOT = REPO_ROOT
MANIFEST = MANIFEST_PATH
EXISTING_EMBEDDINGS = ANCHOR_EMBEDDINGS_DIR
SCVI_EMBEDDINGS = RESULTS_DIR / "embeddings"
RESULTS = RESULTS_DIR / "downstream_clustering"
LOGS = LOGS_DIR

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
SEEDS = [0, 1, 2, 3, 4]
RESOLUTIONS = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
PRIMARY_K = 15
SENSITIVITY_K = [5, 30, 50]
GRAPH_SEED = 0
LEIDEN_ITERATIONS = -1


def stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def embedding_path(dataset_id: str, method: str, seed: int) -> Path:
    if method == "scVI":
        return SCVI_EMBEDDINGS / f"{dataset_id}_scvi_dim2_seed{seed}.npy"
    return EXISTING_EMBEDDINGS / f"{dataset_id}_{stem(method)}_seed{seed}.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def majority_label_prediction(clusters: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
    table = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(labels, name="label"))
    mapping = {str(cluster): str(table.loc[cluster].idxmax()) for cluster in table.index}
    predicted = np.asarray([mapping[str(cluster)] for cluster in clusters], dtype=object)
    return predicted, mapping


def resolution_auc(values: pd.DataFrame, metric: str) -> float:
    ordered = values.sort_values("resolution")
    x = ordered["resolution"].to_numpy(dtype=float)
    y = ordered[metric].to_numpy(dtype=float)
    if x.size < 2 or np.isclose(x.max(), x.min()):
        return float(np.mean(y))
    return float(np.trapezoid(y, x) / (x.max() - x.min()))


def build_graph(coords: np.ndarray, k: int) -> ad.AnnData:
    if not np.isfinite(coords).all():
        raise ValueError("Embedding contains non-finite coordinates")
    work = ad.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    work.obsm["X_representation"] = np.asarray(coords, dtype=np.float32)
    sc.pp.neighbors(
        work,
        n_neighbors=k,
        use_rep="X_representation",
        metric="euclidean",
        random_state=GRAPH_SEED,
    )
    return work


def evaluate_partition(
    dataset_id: str,
    method: str,
    embedding_seed: int,
    k: int,
    resolution: float,
    clusters: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    predicted, mapping = majority_label_prediction(clusters, labels)
    label_order = sorted(np.unique(labels).astype(str))
    precision, recall, per_label_f1, support = precision_recall_fscore_support(
        labels,
        predicted,
        labels=label_order,
        zero_division=0,
    )
    global_row = {
        "dataset_id": dataset_id,
        "method": method,
        "embedding_seed": embedding_seed,
        "graph_k": k,
        "graph_seed": GRAPH_SEED,
        "resolution": resolution,
        "n_cells": int(labels.size),
        "n_reference_labels": int(np.unique(labels).size),
        "n_leiden_clusters": int(np.unique(clusters).size),
        "adjusted_rand_index": float(adjusted_rand_score(labels, clusters)),
        "normalized_mutual_information": float(normalized_mutual_info_score(labels, clusters)),
        "homogeneity": float(homogeneity_score(labels, clusters)),
        "completeness": float(completeness_score(labels, clusters)),
        "v_measure": float(v_measure_score(labels, clusters)),
        "macro_f1_after_cluster_majority_mapping": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1_after_cluster_majority_mapping": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "labels_used_in_graph_or_clustering": False,
        "cluster_to_label_mapping": json.dumps(mapping, sort_keys=True),
    }
    population_rows = []
    for label, p, r, score, n_cells in zip(label_order, precision, recall, per_label_f1, support):
        population_rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "embedding_seed": embedding_seed,
                "graph_k": k,
                "resolution": resolution,
                "label": label,
                "n_cells_label": int(n_cells),
                "prevalence": float(n_cells / labels.size),
                "low_prevalence_200_cell_flag": bool(n_cells <= 200),
                "precision": float(p),
                "recall": float(r),
                "f1": float(score),
                "mapping_definition": "each Leiden cluster assigned its majority reference label after unsupervised clustering",
            }
        )
    return global_row, population_rows


def run(datasets: list[str], methods: list[str], seeds: list[int]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST).set_index("dataset_id")
    global_rows: list[dict[str, object]] = []
    population_rows: list[dict[str, object]] = []
    assignment_rows: list[pd.DataFrame] = []
    partition_cache: dict[tuple[str, str, int, int, float], np.ndarray] = {}
    input_rows: list[dict[str, object]] = []

    for dataset_id in datasets:
        record = manifest.loc[dataset_id]
        proc_path = ROOT / str(record["analysis_proc_path"])
        proc = sc.read_h5ad(proc_path, backed="r")
        label_field = str(record["label_field"])
        labels = proc.obs[label_field].astype(str).to_numpy()
        cell_ids = proc.obs_names.astype(str).to_numpy()
        proc.file.close()

        for method in methods:
            for embedding_seed in seeds:
                path = embedding_path(dataset_id, method, embedding_seed)
                if not path.exists():
                    raise FileNotFoundError(f"Missing embedding: {path}")
                coords = np.load(path).astype(np.float32)
                if coords.shape != (labels.size, 2):
                    raise ValueError(f"Unexpected shape for {dataset_id} {method} seed {embedding_seed}: {coords.shape}")
                input_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": method,
                        "embedding_seed": embedding_seed,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                        "n_cells": int(coords.shape[0]),
                        "n_dimensions": int(coords.shape[1]),
                    }
                )

                k_values = [PRIMARY_K] + (SENSITIVITY_K if embedding_seed == 0 else [])
                for k in k_values:
                    work = build_graph(coords, k)
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
                        partition_cache[(dataset_id, method, embedding_seed, k, resolution)] = clusters
                        global_row, per_population = evaluate_partition(
                            dataset_id,
                            method,
                            embedding_seed,
                            k,
                            resolution,
                            clusters,
                            labels,
                        )
                        global_rows.append(global_row)
                        population_rows.extend(per_population)
                        if embedding_seed == 0 and k == PRIMARY_K:
                            assignment_rows.append(
                                pd.DataFrame(
                                    {
                                        "dataset_id": dataset_id,
                                        "method": method,
                                        "embedding_seed": embedding_seed,
                                        "graph_k": k,
                                        "resolution": resolution,
                                        "cell_id": cell_ids,
                                        "reference_label": labels,
                                        "leiden_cluster": clusters,
                                    }
                                )
                            )
                    del work
                print(f"completed {dataset_id} {method} seed={embedding_seed}", flush=True)

    global_df = pd.DataFrame(global_rows)
    population_df = pd.DataFrame(population_rows)
    assignments = pd.concat(assignment_rows, ignore_index=True)

    summary_rows = []
    primary = global_df[global_df["graph_k"].eq(PRIMARY_K)]
    summary_metrics = [
        "adjusted_rand_index",
        "normalized_mutual_information",
        "macro_f1_after_cluster_majority_mapping",
        "weighted_f1_after_cluster_majority_mapping",
    ]
    for keys, values in primary.groupby(["dataset_id", "method", "embedding_seed"], sort=False):
        row = dict(zip(["dataset_id", "method", "embedding_seed"], keys))
        row["graph_k"] = PRIMARY_K
        for metric in summary_metrics:
            row[f"{metric}_resolution_auc"] = resolution_auc(values, metric)
            row[f"{metric}_resolution_median"] = float(values[metric].median())
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)

    stability_rows = []
    for dataset_id in datasets:
        for method in methods:
            for resolution in RESOLUTIONS:
                for first_seed, second_seed in combinations(seeds, 2):
                    first = partition_cache[(dataset_id, method, first_seed, PRIMARY_K, resolution)]
                    second = partition_cache[(dataset_id, method, second_seed, PRIMARY_K, resolution)]
                    stability_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": method,
                            "graph_k": PRIMARY_K,
                            "resolution": resolution,
                            "seed_a": first_seed,
                            "seed_b": second_seed,
                            "partition_adjusted_rand_index": float(adjusted_rand_score(first, second)),
                            "partition_normalized_mutual_information": float(normalized_mutual_info_score(first, second)),
                        }
                    )

    global_df.to_csv(RESULTS / "clustering_metrics_by_resolution.csv", index=False)
    population_df.to_csv(RESULTS / "population_recovery_by_resolution.csv", index=False)
    summary_df.to_csv(RESULTS / "clustering_resolution_summaries.csv", index=False)
    pd.DataFrame(stability_rows).to_csv(RESULTS / "clustering_seed_stability.csv", index=False)
    assignments.to_csv(RESULTS / "primary_cluster_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame(input_rows).to_csv(LOGS / "clustering_embedding_inputs.csv", index=False)

    metadata = {
        "methods": methods,
        "datasets": datasets,
        "embedding_seeds": seeds,
        "primary_k": PRIMARY_K,
        "sensitivity_k_seed0_only": SENSITIVITY_K,
        "resolutions": RESOLUTIONS,
        "graph_seed": GRAPH_SEED,
        "leiden_iterations": LEIDEN_ITERATIONS,
        "labels_used_during_graph_or_clustering": False,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "scanpy": importlib.metadata.version("scanpy"),
            "anndata": importlib.metadata.version("anndata"),
            "igraph": importlib.metadata.version("igraph"),
            "leidenalg": importlib.metadata.version("leidenalg"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    (LOGS / "downstream_clustering_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.datasets, args.methods, args.seeds)


if __name__ == "__main__":
    main()
