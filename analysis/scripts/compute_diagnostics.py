"""Calculate matched geometric diagnostics for the revised method panel.

The eight submitted implementations are retained from the audited source table.
scVI is evaluated with the same PCA50 reference, neighbourhood size and sampled
pair definition. Two-dimensional results enter the primary cross-analysis; the
ten-dimensional scVI latent is reported only as a dimensionality sensitivity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness
from scipy.stats import spearmanr
from analysis.scripts.compute_trajectory import PRIMARY_ROOT, reference_dpt, root_definitions

from analysis.paths import (
    ANALYSIS_OBJECTS_DIR,
    ANCHOR_EMBEDDINGS_DIR,
    LOGS_DIR as ANALYSIS_LOGS_DIR,
    MANIFEST_PATH,
    PUBLIC_SOURCE_DATA_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    SOURCE_RESULTS_DIR,
)

ROOT = REPO_ROOT
MANIFEST = MANIFEST_PATH
GENERATED_BASELINE = SOURCE_RESULTS_DIR / "fig3_family_local_label_metrics.csv"
EXISTING = (
    GENERATED_BASELINE
    if GENERATED_BASELINE.exists()
    else PUBLIC_SOURCE_DATA_DIR / "fig3_family_local_label_metrics.csv"
)
SCVI_DIR = RESULTS_DIR / "embeddings"
EXISTING_EMBEDDINGS = ANCHOR_EMBEDDINGS_DIR
OUT_DIR = RESULTS_DIR / "diagnostics"
LOG_DIR = ANALYSIS_LOGS_DIR

DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
SEEDS = [0, 1, 2, 3, 4]
DIMENSIONS = [2, 10]
METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
K = 15
RANK_PAIRS = 5000


def method_stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def primary_embedding_path(dataset_id: str, method: str, seed: int) -> Path:
    if method == "scVI":
        return SCVI_DIR / f"{dataset_id}_scvi_dim2_seed{seed}.npy"
    return EXISTING_EMBEDDINGS / f"{dataset_id}_{method_stem(method)}_seed{seed}.npy"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def knn_idx(values: np.ndarray, k: int = K) -> np.ndarray:
    model = NearestNeighbors(n_neighbors=k).fit(values)
    return model.kneighbors(X=None, n_neighbors=k, return_distance=False)


def local_retention(high: np.ndarray, low: np.ndarray, k: int = K) -> float:
    high_idx = knn_idx(high, k)
    low_idx = knn_idx(low, k)
    overlap = [len(set(high_idx[i]) & set(low_idx[i])) / k for i in range(high.shape[0])]
    return float(np.mean(overlap))


def label_neighbour_recall(low: np.ndarray, labels: np.ndarray, k: int = K) -> float:
    low_idx = knn_idx(low, k)
    per_cell = [(labels[low_idx[i]] == labels[i]).mean() for i in range(labels.size)]
    return float(np.mean(per_cell))


def global_rank(high: np.ndarray, low: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, high.shape[0], RANK_PAIRS)
    j = rng.integers(0, high.shape[0], RANK_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    high_distance = np.linalg.norm(high[i] - high[j], axis=1)
    low_distance = np.linalg.norm(low[i] - low[j], axis=1)
    value = spearmanr(high_distance, low_distance).statistic
    return float(value) if np.isfinite(value) else float("nan")


def one_dimensional_rank(reference: np.ndarray, coords: np.ndarray, seed: int) -> float:
    rng = np.random.default_rng(seed)
    i = rng.integers(0, reference.size, RANK_PAIRS)
    j = rng.integers(0, reference.size, RANK_PAIRS)
    keep = i != j
    i, j = i[keep], j[keep]
    reference_distance = np.abs(reference[i] - reference[j])
    representation_distance = np.linalg.norm(coords[i] - coords[j], axis=1)
    value = spearmanr(reference_distance, representation_distance).statistic
    return float(value) if np.isfinite(value) else float("nan")


def paul15_reference_pseudotime(proc) -> tuple[np.ndarray, int]:
    root = root_definitions(proc)[PRIMARY_ROOT]
    return reference_dpt(proc, root), root


def write_trajectory_geometry(methods: list[str], seeds: list[int]) -> None:
    proc = sc.read_h5ad(ANALYSIS_OBJECTS_DIR / "paul15_proc.h5ad")
    pseudotime, root = paul15_reference_pseudotime(proc)
    rng = np.random.default_rng(0)
    random_i = rng.integers(0, pseudotime.size, RANK_PAIRS)
    random_j = rng.integers(0, pseudotime.size, RANK_PAIRS)
    nonself = random_i != random_j
    random_mean = float(np.mean(np.abs(pseudotime[random_i[nonself]] - pseudotime[random_j[nonself]])))
    rows = []
    for method in methods:
        for seed in seeds:
            coords = np.load(primary_embedding_path("paul15", method, seed)).astype(np.float32)
            neighbours = knn_idx(coords, K)
            neighbour_delta = np.array(
                [np.mean(np.abs(pseudotime[neighbours[i]] - pseudotime[i])) for i in range(coords.shape[0])]
            )
            retention = 1.0 - float(neighbour_delta.mean() / random_mean)
            rows.extend(
                [
                    {
                        "dataset_id": "paul15",
                        "method": method,
                        "seed": seed,
                        "metric": "pseudotime_rank_corr",
                        "value": one_dimensional_rank(pseudotime, coords, seed),
                        "reference_root_cell": str(proc.obs_names[root]),
                        "reference_root_definition": "7MEP centroid in the expression-derived PCA reference",
                        "n_cells": int(proc.n_obs),
                        "n_reference_finite": int(np.isfinite(pseudotime).sum()),
                        "reference_missingness_policy": "non-finite reference raises an error; no imputation",
                    },
                    {
                        "dataset_id": "paul15",
                        "method": method,
                        "seed": seed,
                        "metric": "pseudotime_neighbourhood_retention",
                        "value": retention,
                        "reference_root_cell": str(proc.obs_names[root]),
                        "reference_root_definition": "7MEP centroid in the expression-derived PCA reference",
                        "n_cells": int(proc.n_obs),
                        "n_reference_finite": int(np.isfinite(pseudotime).sum()),
                        "reference_missingness_policy": "non-finite reference raises an error; no imputation",
                    },
                ]
            )
    pd.DataFrame(rows).to_csv(OUT_DIR / "trajectory_geometry_metrics_all_methods.csv", index=False)


def normalised_entropy(values: np.ndarray, categories: np.ndarray) -> float:
    if categories.size <= 1:
        return float("nan")
    counts = np.asarray([(values == category).sum() for category in categories], dtype=float)
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(categories.size))


def write_heart_neighbourhood_metrics(methods: list[str], seeds: list[int]) -> None:
    proc = sc.read_h5ad(
        ANALYSIS_OBJECTS_DIR / "heart_cell_atlas_subsampled_proc.h5ad", backed="r"
    )
    labels = proc.obs["cell_type"].astype(str).to_numpy()
    donors = proc.obs["donor"].astype(str).to_numpy()
    donor_categories = np.unique(donors)
    n_cells = int(proc.n_obs)
    proc.file.close()
    rows = []
    for method in methods:
        for seed in seeds:
            coords = np.load(primary_embedding_path("heart_cell_atlas_subsampled", method, seed)).astype(np.float32)
            neighbours = knn_idx(coords, K)
            label_recall = float(np.mean([(labels[neighbours[i]] == labels[i]).mean() for i in range(n_cells)]))
            donor_entropy = float(
                np.mean([normalised_entropy(donors[neighbours[i]], donor_categories) for i in range(n_cells)])
            )
            donor_dominance = float(
                np.mean(
                    [
                        pd.Series(donors[neighbours[i]]).value_counts(normalize=True).iloc[0]
                        for i in range(n_cells)
                    ]
                )
            )
            for metric, value in [
                ("cell_type_label_neighbour_recall", label_recall),
                ("donor_entropy_normalised", donor_entropy),
                ("donor_dominance", donor_dominance),
            ]:
                rows.append(
                    {
                        "dataset_id": "heart_cell_atlas_subsampled",
                        "method": method,
                        "seed": seed,
                        "metric": metric,
                        "value": value,
                        "k": K,
                        "n_cells": n_cells,
                    }
                )
    pd.DataFrame(rows).to_csv(OUT_DIR / "heart_neighbourhood_metrics_all_methods.csv", index=False)


def run(datasets: list[str], seeds: list[int], dimensions: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST).set_index("dataset_id")
    rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    neighbour_cache: dict[tuple[str, int, int], np.ndarray] = {}

    for dataset_id in datasets:
        record = manifest.loc[dataset_id]
        proc_path = ROOT / str(record["analysis_proc_path"])
        proc = sc.read_h5ad(proc_path, backed="r")
        labels = proc.obs[str(record["label_field"])].astype(str).to_numpy()
        high = np.asarray(proc.obsm["X_pca_ref"][:, :50], dtype=np.float32)
        n_cells = int(proc.n_obs)
        proc.file.close()

        for dimension in dimensions:
            for seed in seeds:
                path = SCVI_DIR / f"{dataset_id}_scvi_dim{dimension}_seed{seed}.npy"
                if not path.exists():
                    raise FileNotFoundError(path)
                coords = np.load(path).astype(np.float32)
                if coords.shape != (n_cells, dimension):
                    raise ValueError(f"Unexpected shape for {path}: {coords.shape}")
                input_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": "scVI",
                        "seed": seed,
                        "output_dimension": dimension,
                        "path": str(path.relative_to(ROOT)),
                        "sha256": sha256_file(path),
                    }
                )
                neighbour_cache[(dataset_id, dimension, seed)] = knn_idx(coords, K)
                values = {
                    "local_retention": local_retention(high, coords, K),
                    "trustworthiness": float(trustworthiness(high, coords, n_neighbors=K)),
                    "global_rank_corr": global_rank(high, coords, seed),
                    "label_neighbor_recall": label_neighbour_recall(coords, labels, K),
                }
                for metric, value in values.items():
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": "scVI",
                            "family": "deep",
                            "seed": seed,
                            "output_dimension": dimension,
                            "metric": metric,
                            "value": value,
                            "k": K if metric != "global_rank_corr" else np.nan,
                            "n_cells": n_cells,
                            "n_pcs_reference": high.shape[1],
                            "n_sampled_pairs": RANK_PAIRS if metric == "global_rank_corr" else np.nan,
                        }
                    )
                print(f"completed diagnostics {dataset_id} scVI dim={dimension} seed={seed}", flush=True)

    scvi = pd.DataFrame(rows)
    existing = pd.read_csv(EXISTING).copy()
    existing["output_dimension"] = 2
    existing["n_sampled_pairs"] = np.where(existing["metric"].eq("global_rank_corr"), RANK_PAIRS, np.nan)
    shared = [
        "dataset_id",
        "method",
        "family",
        "seed",
        "output_dimension",
        "metric",
        "value",
        "k",
        "n_cells",
        "n_pcs_reference",
        "n_sampled_pairs",
    ]
    combined = pd.concat([existing[shared], scvi[shared]], ignore_index=True)
    combined.to_csv(OUT_DIR / "geometry_metrics_all_methods.csv", index=False)
    scvi.to_csv(OUT_DIR / "scvi_dimension_diagnostics.csv", index=False)

    stability_rows = []
    for dataset_id in datasets:
        for dimension in dimensions:
            for first, second in combinations(seeds, 2):
                first_idx = neighbour_cache[(dataset_id, dimension, first)]
                second_idx = neighbour_cache[(dataset_id, dimension, second)]
                overlap = [len(set(first_idx[i]) & set(second_idx[i])) / K for i in range(first_idx.shape[0])]
                stability_rows.append(
                    {
                        "dataset_id": dataset_id,
                        "method": "scVI",
                        "output_dimension": dimension,
                        "seed_a": first,
                        "seed_b": second,
                        "knn_overlap": float(np.mean(overlap)),
                        "k": K,
                    }
                )
    pd.DataFrame(stability_rows).to_csv(OUT_DIR / "scvi_dimension_seed_stability.csv", index=False)
    pd.DataFrame(input_rows).to_csv(LOG_DIR / "revision_diagnostic_inputs.csv", index=False)
    write_trajectory_geometry(METHODS, seeds)
    write_heart_neighbourhood_metrics(METHODS, seeds)

    metadata = {
        "primary_output_dimension": 2,
        "sensitivity_output_dimension": 10,
        "reference": "first 50 components of the audited expression-derived PCA reference",
        "neighbourhood_size": K,
        "sampled_pairs": RANK_PAIRS,
        "labels_used_during_representation_fitting": False,
        "labels_used_only_for_label_neighbour_recall": True,
    }
    (LOG_DIR / "revision_diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--dimensions", nargs="+", type=int, choices=DIMENSIONS, default=DIMENSIONS)
    args = parser.parse_args()
    run(args.datasets, args.seeds, args.dimensions)


if __name__ == "__main__":
    main()
