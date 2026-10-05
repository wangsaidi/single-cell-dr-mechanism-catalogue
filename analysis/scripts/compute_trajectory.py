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
from scipy.sparse.csgraph import connected_components
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
    mask = finite_pair_mask(first, second)
    if mask.sum() < 2 or np.unique(first[mask]).size < 2 or np.unique(second[mask]).size < 2:
        return float("nan")
    value = spearmanr(first[mask], second[mask]).correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def safe_kendall(first: np.ndarray, second: np.ndarray) -> float:
    mask = finite_pair_mask(first, second)
    if mask.sum() < 2 or np.unique(first[mask]).size < 2 or np.unique(second[mask]).size < 2:
        return float("nan")
    value = kendalltau(first[mask], second[mask]).correlation
    return float(value) if value is not None and np.isfinite(value) else float("nan")


def finite_pair_mask(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    if first.shape != second.shape or first.ndim != 1:
        raise ValueError("Correlation inputs must be aligned one-dimensional arrays")
    return np.isfinite(first) & np.isfinite(second)


def preserve_missing_pseudotime(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).copy()
    values[~np.isfinite(values)] = np.nan
    return values


def require_finite_reference(values: np.ndarray, context: str) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    missing = int((~np.isfinite(values)).sum())
    if missing:
        raise ValueError(f"Non-finite reference pseudotime for {context}: {missing}/{values.size}")
    return values


def common_finite_mask(reference: np.ndarray, pseudotimes: dict[str, np.ndarray], context: str) -> np.ndarray:
    require_finite_reference(reference, context)
    if not pseudotimes:
        raise ValueError(f"No representations for common-cell comparison: {context}")
    mask = np.ones(reference.size, dtype=bool)
    for name, values in pseudotimes.items():
        if values.shape != reference.shape:
            raise ValueError(f"Misaligned pseudotime for {context}: {name}")
        mask &= np.isfinite(values)
    if mask.sum() < 2:
        counts = {name: int(np.isfinite(values).sum()) for name, values in pseudotimes.items()}
        raise ValueError(f"Insufficient finite common cells for {context}: n_common={mask.sum()}, n_finite={counts}")
    return mask


def coverage_fields(pseudotime: np.ndarray, common: np.ndarray, metadata: dict) -> dict:
    finite = np.isfinite(pseudotime)
    disconnected = metadata["disconnected_mask"]
    return {
        "n_total": int(pseudotime.size), "n_finite": int(finite.sum()),
        "n_common": int(common.sum()), "coverage": float(finite.mean()),
        "common_coverage": float(common.mean()), "n_disconnected": int(disconnected.sum()),
        "n_nonfinite_other": int((~finite & ~disconnected).sum()),
        "n_connected_components": metadata["n_connected_components"],
    }


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


def run_dpt_roots(coords: np.ndarray, roots: dict[str, int]) -> dict:
    if coords.ndim != 2 or not np.isfinite(coords).all():
        raise ValueError("DPT representation must be a finite two-dimensional array")
    work = ad.AnnData(X=np.zeros((coords.shape[0], 1), dtype=np.float32))
    work.obsm["X_representation"] = np.asarray(coords, dtype=np.float32)
    sc.pp.neighbors(work, n_neighbors=K, use_rep="X_representation", metric="euclidean", random_state=0)
    sc.tl.diffmap(work, n_comps=15)
    n_components, components = connected_components(work.obsp["connectivities"], directed=False)
    results = {}
    for name, root in roots.items():
        if not 0 <= root < coords.shape[0]:
            raise ValueError(f"Invalid DPT root {name}: {root}")
        work.uns["iroot"] = int(root)
        sc.tl.dpt(work, n_dcs=N_DCS)
        pseudotime = preserve_missing_pseudotime(work.obs["dpt_pseudotime"].to_numpy(dtype=float))
        disconnected = components != components[root]
        pseudotime[disconnected] = np.nan
        finite = np.isfinite(pseudotime)
        order = np.full(pseudotime.size, np.nan)
        order[finite] = np.argsort(np.argsort(pseudotime[finite], kind="stable"), kind="stable")
        results[name] = (pseudotime, order, {
            "disconnected_mask": disconnected, "n_connected_components": int(n_components),
        })
    return results


def run_dpt(coords: np.ndarray, root: int) -> tuple[np.ndarray, np.ndarray]:
    pseudotime, order, _ = run_dpt_roots(coords, {"root": root})["root"]
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
    return require_finite_reference(values, f"root index {root}")


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
    if len(methods) != len(METHODS) or set(methods) != set(METHODS):
        raise ValueError("Primary trajectory comparison requires all nine distinct methods")
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
    membership_rows = []
    evaluable = ~np.isin(branches, ["other", "progenitor"])
    for seed in seeds:
        coordinates, fitted = {}, {}
        for method in methods:
            path = embedding_path(method, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            coords = np.load(path).astype(np.float32)
            if coords.shape != (proc.n_obs, 2):
                raise ValueError(f"Unexpected shape for {method} seed {seed}: {coords.shape}")
            coordinates[method] = coords
            fitted[method] = run_dpt_roots(coords, roots)
        common_by_root = {
            name: common_finite_mask(reference_by_root[name], {
                method: fitted[method][name][0] for method in methods
            }, f"seed {seed}, root {name}") for name in roots
        }
        for name, root_common in common_by_root.items():
            membership_rows.append(pd.DataFrame({
                "cell_id": cell_ids, "embedding_seed": seed, "root_definition": name,
                "root_cell_id": cell_ids[roots[name]], "in_common_finite_mask": root_common,
                "n_comparison_methods": len(methods),
            }))
        common = common_by_root[PRIMARY_ROOT]
        for method in methods:
            coords = coordinates[method]
            pseudotime, order, metadata = fitted[method][PRIMARY_ROOT]
            finite = np.isfinite(pseudotime)
            common_spearman = safe_spearman(primary_reference[common], pseudotime[common])
            common_kendall = safe_kendall(primary_reference[common], pseudotime[common])
            metric_rows.append(
                {
                    "dataset_id": "paul15",
                    "method": method,
                    "embedding_seed": seed,
                    "root_definition": PRIMARY_ROOT,
                    "root_cell_id": cell_ids[roots[PRIMARY_ROOT]],
                    "reference_pseudotime_spearman": common_spearman,
                    "reference_pseudotime_kendall": common_kendall,
                    "reference_pseudotime_spearman_common": common_spearman,
                    "reference_pseudotime_kendall_common": common_kendall,
                    "reference_pseudotime_spearman_finite_only": safe_spearman(primary_reference[finite], pseudotime[finite]),
                    "reference_pseudotime_kendall_finite_only": safe_kendall(primary_reference[finite], pseudotime[finite]),
                    "branch_neighbour_fraction": branch_neighbour_fraction(coords, branches, K),
                    "n_cells": int(proc.n_obs),
                    "n_branch_evaluable_cells": int(evaluable.sum()),
                    "reference_definition": "PCA50 DPT with the same independently selected 7MEP centroid root; analytical consistency reference, not biological ground truth",
                    "correlation_mask_definition": "finite intersection of reference and all nine methods within embedding seed",
                    "n_comparison_methods": len(methods),
                    **coverage_fields(pseudotime, common, metadata),
                }
            )

            for lineage, clusters in LINEAGE_CLUSTERS.items():
                mask = np.isin(labels, clusters)
                lineage_score = scores[lineage].to_numpy(dtype=float)
                progenitor_score = scores["progenitor"].to_numpy(dtype=float)
                expression_finite = np.isfinite(lineage_score) & np.isfinite(progenitor_score)
                marker_finite = mask & finite & expression_finite
                marker_common = mask & common & expression_finite
                marker_rows.append(
                    {
                        "dataset_id": "paul15",
                        "method": method,
                        "embedding_seed": seed,
                        "lineage": lineage,
                        "root_definition": PRIMARY_ROOT,
                        "root_cell_id": cell_ids[roots[PRIMARY_ROOT]],
                        "n_lineage_cells": int(mask.sum()),
                        "n_lineage_finite": int(marker_finite.sum()),
                        "n_lineage_common": int(marker_common.sum()),
                        "n_lineage_disconnected": int((mask & metadata["disconnected_mask"]).sum()),
                        "lineage_program_spearman": safe_spearman(pseudotime[marker_common], lineage_score[marker_common]),
                        "progenitor_program_spearman": safe_spearman(pseudotime[marker_common], progenitor_score[marker_common]),
                        "lineage_program_spearman_finite_only": safe_spearman(pseudotime[marker_finite], lineage_score[marker_finite]),
                        "progenitor_program_spearman_finite_only": safe_spearman(pseudotime[marker_finite], progenitor_score[marker_finite]),
                        "correlation_mask_definition": "lineage cells in nine-method common finite mask and finite marker scores",
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
                            "method_dpt_finite": finite,
                            "method_dpt_disconnected": metadata["disconnected_mask"],
                            "in_common_finite_mask": common,
                        }
                    )
                )

            for root_name, root in roots.items():
                root_pt, _, root_metadata = fitted[method][root_name]
                root_common = common_by_root[root_name]
                pair_common = root_common & common
                if pair_common.sum() < 2:
                    raise ValueError(f"Insufficient common cells across roots for seed {seed}, root {root_name}")
                root_finite = np.isfinite(root_pt)
                pair_finite = root_finite & finite
                root_rows.append({
                    "dataset_id": "paul15", "method": method, "embedding_seed": seed,
                    "root_definition": root_name, "root_cell_id": cell_ids[root], "root_cluster": labels[root],
                    "reference_pseudotime_spearman": safe_spearman(reference_by_root[root_name][root_common], root_pt[root_common]),
                    "reference_pseudotime_kendall": safe_kendall(reference_by_root[root_name][root_common], root_pt[root_common]),
                    "reference_pseudotime_spearman_finite_only": safe_spearman(reference_by_root[root_name][root_finite], root_pt[root_finite]),
                    "primary_root_pseudotime_concordance": safe_spearman(pseudotime[pair_common], root_pt[pair_common]),
                    "primary_root_pseudotime_concordance_finite_only": safe_spearman(pseudotime[pair_finite], root_pt[pair_finite]),
                    "n_common_root_pair": int(pair_common.sum()), "n_finite_root_pair": int(pair_finite.sum()),
                    "correlation_mask_definition": "nine-method common finite cells per seed/root; root-pair concordance intersects both common masks",
                    "n_comparison_methods": len(methods),
                    **coverage_fields(root_pt, root_common, root_metadata),
                })
            print(f"completed Paul15 trajectory {method} seed={seed}", flush=True)

    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "trajectory_outcomes.csv", index=False)
    pd.DataFrame(marker_rows).to_csv(OUT_DIR / "lineage_marker_monotonicity.csv", index=False)
    pd.DataFrame(root_rows).to_csv(OUT_DIR / "trajectory_root_sensitivity.csv", index=False)
    pd.concat(assignment_rows, ignore_index=True).to_csv(
        OUT_DIR / "trajectory_assignments_seed0.csv.gz", index=False, compression="gzip"
    )
    pd.concat(membership_rows, ignore_index=True).to_csv(
        OUT_DIR / "trajectory_common_cell_membership.csv.gz", index=False, compression="gzip"
    )
    (OUT_DIR / "trajectory_failures.json").write_text(json.dumps([], indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.methods, args.seeds)


if __name__ == "__main__":
    main()
