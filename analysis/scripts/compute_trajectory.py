"""Infer Paul15 trajectories from each representation and compare outcomes.

The primary root cell is selected once from the 7MEP centroid in the PCA50
expression reference and then held fixed across methods. Labels are not used
to fit any representation. Reference DPT, branch annotations and lineage
marker programs are reported as separate consistency references rather than
as biological ground truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import kendalltau, spearmanr
from sklearn.neighbors import NearestNeighbors

from analysis.paths import (
    ANALYSIS_OBJECTS_DIR,
    ANCHOR_EMBEDDINGS_DIR,
    EXPRESSION_OBJECTS_DIR,
    RESULTS_DIR,
)

PROC_PATH = ANALYSIS_OBJECTS_DIR / "paul15_proc.h5ad"
EXPRESSION_PATH = EXPRESSION_OBJECTS_DIR / "paul15_expression.h5ad"
EXISTING_EMBEDDINGS = ANCHOR_EMBEDDINGS_DIR
SCVI_EMBEDDINGS = RESULTS_DIR / "embeddings"
OUT_DIR = RESULTS_DIR / "trajectory"

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
SEEDS = [0, 1, 2, 3, 4]
K = 15
N_DCS = 10
PRIMARY_ROOT = "7MEP_centroid"

LINEAGE_CLUSTERS = {
    "erythroid": ["7MEP", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
    "megakaryocytic": ["7MEP", "8Mk"],
    "basophil": ["7MEP", "12Baso", "13Baso"],
    "myeloid": ["9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"],
}

MARKERS = {
    "erythroid": ["Gata1", "Klf1", "Hba-a1", "Hba-a2", "Hbb-b1", "Hbb-b2"],
    "megakaryocytic": ["Pf4", "Ppbp", "Itga2b"],
    "basophil": ["Mcpt8", "Prss34", "Cpa3"],
    "myeloid": ["Lyz2", "S100a8", "S100a9", "Csf1r", "Mpo"],
    "progenitor": ["Kit", "Gata2", "Meis1", "Hlf"],
}


def stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def embedding_path(method: str, seed: int) -> Path:
    if method == "scVI":
        return SCVI_EMBEDDINGS / f"paul15_scvi_dim2_seed{seed}.npy"
    return EXISTING_EMBEDDINGS / f"paul15_{stem(method)}_seed{seed}.npy"


def safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(first, second, nan_policy="omit").correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    value = kendalltau(first, second, nan_policy="omit").correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def root_index(proc, cluster: str) -> int:
    labels = proc.obs["paul15_clusters"].astype(str).to_numpy()
    candidates = np.where(labels == cluster)[0]
    if candidates.size == 0:
        raise ValueError(f"Missing root cluster: {cluster}")
    reference = np.asarray(proc.obsm["X_pca_ref"][candidates, :20], dtype=float)
    centroid = reference.mean(axis=0)
    return int(candidates[np.argmin(np.linalg.norm(reference - centroid, axis=1))])


def root_definitions(proc) -> dict[str, int]:
    return {
        PRIMARY_ROOT: root_index(proc, "7MEP"),
        "9GMP_centroid": root_index(proc, "9GMP"),
        "1Ery_centroid": root_index(proc, "1Ery"),
        "stored_8Mk_root": int(proc.uns["iroot"]),
    }


def run_dpt(coords: np.ndarray, root: int) -> tuple[np.ndarray, np.ndarray]:
    work = ad.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    work.obsm["X_representation"] = np.asarray(coords, dtype=np.float32)
    sc.pp.neighbors(work, n_neighbors=K, use_rep="X_representation", metric="euclidean", random_state=0)
    sc.tl.diffmap(work, n_comps=15)
    work.uns["iroot"] = int(root)
    sc.tl.dpt(work, n_dcs=N_DCS)
    pseudotime = work.obs["dpt_pseudotime"].to_numpy(dtype=float)
    pseudotime = np.nan_to_num(pseudotime, nan=np.nanmedian(pseudotime), posinf=1.0, neginf=0.0)
    order = work.obs["dpt_order"].to_numpy(dtype=float) if "dpt_order" in work.obs else np.argsort(np.argsort(pseudotime))
    return pseudotime, order


def branch_neighbour_fraction(coords: np.ndarray, branches: np.ndarray, k: int = K) -> float:
    neighbours = NearestNeighbors(n_neighbors=k).fit(coords).kneighbors(
        X=None, n_neighbors=k, return_distance=False
    )
    evaluable = ~np.isin(branches, ["other", "progenitor"])
    fractions = [
        float(np.mean(branches[neighbours[index]] == branches[index]))
        for index in np.where(evaluable)[0]
    ]
    return float(np.mean(fractions)) if fractions else float("nan")


def reference_dpt(proc, root: int) -> np.ndarray:
    work = proc.copy()
    sc.pp.neighbors(work, n_neighbors=K, use_rep="X_pca_ref", random_state=0)
    sc.tl.diffmap(work, n_comps=15)
    work.uns["iroot"] = int(root)
    sc.tl.dpt(work, n_dcs=N_DCS)
    values = work.obs["dpt_pseudotime"].to_numpy(dtype=float)
    return np.nan_to_num(values, nan=np.nanmedian(values), posinf=1.0, neginf=0.0)


def coarse_branches(labels: np.ndarray) -> np.ndarray:
    coarse = np.asarray(["other"] * labels.size, dtype=object)
    for branch, clusters in LINEAGE_CLUSTERS.items():
        mask = np.isin(labels, clusters)
        replace = mask & (coarse == "other")
        coarse[replace] = branch
    coarse[labels == "7MEP"] = "progenitor"
    coarse[labels == "19Lymph"] = "lymphoid"
    return coarse.astype(str)


def marker_scores(cell_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    expression = sc.read_h5ad(EXPRESSION_PATH)
    missing_cells = sorted(set(cell_ids) - set(map(str, expression.obs_names)))
    if missing_cells:
        raise ValueError(f"Paul15 expression object misses {len(missing_cells)} cells")
    expression = expression[cell_ids, :].copy()
    available = set(map(str, expression.var_names))
    rows = []
    scores = {}
    for program, genes in MARKERS.items():
        present = [gene for gene in genes if gene in available]
        missing = [gene for gene in genes if gene not in available]
        if not present:
            raise ValueError(f"No available genes for {program}")
        values = expression[:, present].X
        if sparse.issparse(values):
            values = values.toarray()
        scores[program] = np.log1p(np.asarray(values, dtype=np.float32)).mean(axis=1)
        rows.append(
            {
                "program": program,
                "genes_requested": ";".join(genes),
                "genes_present": ";".join(present),
                "genes_missing": ";".join(missing),
            }
        )
    return pd.DataFrame(scores, index=cell_ids), pd.DataFrame(rows)


def run(methods: list[str], seeds: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = sc.read_h5ad(PROC_PATH)
    labels = proc.obs["paul15_clusters"].astype(str).to_numpy()
    cell_ids = proc.obs_names.astype(str).to_numpy()
    branches = coarse_branches(labels)
    roots = root_definitions(proc)
    reference_by_root = {name: reference_dpt(proc, index) for name, index in roots.items()}
    primary_reference = reference_by_root[PRIMARY_ROOT]
    scores, marker_inputs = marker_scores(cell_ids)
    marker_inputs.to_csv(OUT_DIR / "lineage_marker_programs.csv", index=False)

    metric_rows = []
    marker_rows = []
    root_rows = []
    assignment_rows = []
    failure_rows = []

    for method in methods:
        for seed in seeds:
            path = embedding_path(method, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            coords = np.load(path).astype(np.float32)
            if coords.shape != (proc.n_obs, 2):
                raise ValueError(f"Unexpected shape for {method} seed {seed}: {coords.shape}")
            try:
                pseudotime, order = run_dpt(coords, roots[PRIMARY_ROOT])
            except Exception as exc:
                failure_rows.append({"method": method, "seed": seed, "root": PRIMARY_ROOT, "error": repr(exc)})
                continue

            evaluable = ~np.isin(branches, ["other", "progenitor"])
            metric_rows.append(
                {
                    "dataset_id": "paul15",
                    "method": method,
                    "embedding_seed": seed,
                    "root_definition": PRIMARY_ROOT,
                    "reference_pseudotime_spearman": safe_spearman(primary_reference, pseudotime),
                    "reference_pseudotime_kendall": safe_kendall(primary_reference, pseudotime),
                    "branch_neighbour_fraction": branch_neighbour_fraction(coords, branches, K),
                    "n_cells": int(proc.n_obs),
                    "n_branch_evaluable_cells": int(evaluable.sum()),
                    "reference_definition": "PCA50 DPT with the same independently selected 7MEP centroid root; analytical consistency reference, not biological ground truth",
                }
            )

            for lineage, clusters in LINEAGE_CLUSTERS.items():
                mask = np.isin(labels, clusters)
                lineage_pt = pseudotime[mask]
                lineage_score = scores.loc[mask, lineage].to_numpy(dtype=float)
                progenitor_score = scores.loc[mask, "progenitor"].to_numpy(dtype=float)
                marker_rows.append(
                    {
                        "dataset_id": "paul15",
                        "method": method,
                        "embedding_seed": seed,
                        "lineage": lineage,
                        "n_lineage_cells": int(mask.sum()),
                        "lineage_program_spearman": safe_spearman(lineage_pt, lineage_score),
                        "progenitor_program_spearman": safe_spearman(lineage_pt, progenitor_score),
                        "marker_consistency_definition": "lineage program is expected to rise and progenitor program to decline along inferred pseudotime",
                    }
                )

            if seed == 0:
                assignment_rows.append(
                    pd.DataFrame(
                        {
                            "cell_id": cell_ids,
                            "method": method,
                            "embedding_seed": seed,
                            "root_definition": PRIMARY_ROOT,
                            "reference_cluster": labels,
                            "coarse_branch": branches,
                            "reference_dpt_pseudotime": primary_reference,
                            "method_dpt_pseudotime": pseudotime,
                            "method_dpt_order": order,
                        }
                    )
                )

                for root_name, root in roots.items():
                    try:
                        root_pt, _ = run_dpt(coords, root)
                        root_rows.append(
                            {
                                "dataset_id": "paul15",
                                "method": method,
                                "embedding_seed": seed,
                                "root_definition": root_name,
                                "root_cell_id": cell_ids[root],
                                "root_cluster": labels[root],
                                "reference_pseudotime_spearman": safe_spearman(reference_by_root[root_name], root_pt),
                                "primary_root_pseudotime_concordance": safe_spearman(pseudotime, root_pt),
                            }
                        )
                    except Exception as exc:
                        failure_rows.append({"method": method, "seed": seed, "root": root_name, "error": repr(exc)})
            print(f"completed Paul15 trajectory {method} seed={seed}", flush=True)

    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "trajectory_outcomes.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(OUT_DIR / "lineage_marker_monotonicity.csv", index=False)
    pd.DataFrame(root_rows).to_csv(OUT_DIR / "trajectory_root_sensitivity.csv", index=False)
    pd.concat(assignment_rows, ignore_index=True).to_csv(
        OUT_DIR / "trajectory_assignments_seed0.csv.gz", index=False, compression="gzip"
    )
    (OUT_DIR / "trajectory_failures.json").write_text(json.dumps(failure_rows, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.methods, args.seeds)


if __name__ == "__main__":
    main()
