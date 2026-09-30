"""Quantify marker-program concordance of representation-derived Leiden clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from analysis.paths import EXPRESSION_OBJECTS_DIR, RESULTS_DIR

CLUSTER_DIR = RESULTS_DIR / "downstream_clustering"
OUT_DIR = RESULTS_DIR / "marker_concordance"

PBMC_MARKERS = {
    "T_cell": ["CD3D", "CD3E", "IL7R"],
    "B_cell": ["MS4A1", "CD79A", "CD79B"],
    "CD14_monocyte": ["CD14", "LYZ", "S100A8"],
    "FCGR3A_monocyte": ["FCGR3A", "MS4A7"],
    "NK_cell": ["GNLY", "NKG7", "KLRD1"],
    "dendritic": ["FCER1A", "CST3"],
    "megakaryocyte": ["PPBP"],
}

PBMC_EXPECTED = {
    "CD4 T cells": ["T_cell"],
    "CD8 T cells": ["T_cell"],
    "B cells": ["B_cell"],
    "CD14+ Monocytes": ["CD14_monocyte"],
    "FCGR3A+ Monocytes": ["FCGR3A_monocyte"],
    "NK cells": ["NK_cell"],
    "Dendritic cells": ["dendritic"],
    "Megakaryocytes": ["megakaryocyte"],
}

HEART_MARKERS = {
    "cardiomyocyte": ["TNNT2", "MYH6", "MYH7", "ACTC1", "TTN"],
    "endothelial": ["PECAM1", "VWF", "KDR", "EMCN"],
    "fibroblast": ["COL1A1", "COL1A2", "DCN", "LUM"],
    "pericyte": ["RGS5", "PDGFRB", "MCAM", "CSPG4"],
    "myeloid": ["LST1", "C1QA", "C1QB", "LYZ"],
    "lymphoid": ["CD3D", "CD3E", "NKG7", "MS4A1"],
    "smooth_muscle": ["ACTA2", "MYH11", "TAGLN"],
    "adipocyte": ["PLIN1", "ADIPOQ", "LPL"],
    "neuronal": ["NRXN1", "RBFOX3", "SYT1"],
    "mesothelial": ["MSLN", "WT1", "UPK3B"],
}

HEART_EXPECTED = {
    "Atrial_Cardiomyocyte": ["cardiomyocyte"],
    "Ventricular_Cardiomyocyte": ["cardiomyocyte"],
    "Endothelial": ["endothelial"],
    "Fibroblast": ["fibroblast"],
    "Pericytes": ["pericyte"],
    "Myeloid": ["myeloid"],
    "Lymphoid": ["lymphoid"],
    "Smooth_muscle_cells": ["smooth_muscle"],
    "Adipocytes": ["adipocyte"],
    "Neuronal": ["neuronal"],
    "Mesothelial": ["mesothelial"],
}


def dense_columns(adata, genes: list[str]) -> np.ndarray:
    values = adata[:, genes].X
    if sparse.issparse(values):
        values = values.toarray()
    return np.asarray(values, dtype=np.float32)


def module_scores(adata, marker_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    scores = {}
    available = set(map(str, adata.var_names))
    for program, genes in marker_sets.items():
        present = [gene for gene in genes if gene in available]
        missing = [gene for gene in genes if gene not in available]
        if not present:
            raise ValueError(f"No genes available for marker program {program}")
        values = np.log1p(dense_columns(adata, present))
        scores[program] = values.mean(axis=1)
        rows.append(
            {
                "program": program,
                "genes_requested": ";".join(genes),
                "genes_present": ";".join(present),
                "genes_missing": ";".join(missing),
                "n_genes_present": len(present),
            }
        )
    score_df = pd.DataFrame(scores, index=adata.obs_names.astype(str))
    return score_df, pd.DataFrame(rows)


def standardised_mean_difference(inside: np.ndarray, outside: np.ndarray) -> float:
    if inside.size < 2 or outside.size < 2:
        return float("nan")
    numerator = float(np.mean(inside) - np.mean(outside))
    pooled = np.sqrt(
        ((inside.size - 1) * np.var(inside, ddof=1) + (outside.size - 1) * np.var(outside, ddof=1))
        / max(1, inside.size + outside.size - 2)
    )
    return numerator / pooled if pooled > 0 else 0.0


def load_expression(dataset_id: str, cell_ids: list[str]):
    if dataset_id == "pbmc3k":
        adata = sc.read_h5ad(EXPRESSION_OBJECTS_DIR / "pbmc3k_expression.h5ad")
        marker_sets = PBMC_MARKERS
        expected = PBMC_EXPECTED
    elif dataset_id == "heart_cell_atlas_subsampled":
        adata = sc.read_h5ad(EXPRESSION_OBJECTS_DIR / "heart_cell_atlas_expression.h5ad")
        marker_sets = HEART_MARKERS
        expected = HEART_EXPECTED
    else:
        raise KeyError(dataset_id)
    missing = sorted(set(cell_ids) - set(map(str, adata.obs_names)))
    if missing:
        raise ValueError(f"{dataset_id} expression object misses {len(missing)} clustered cells")
    return adata[cell_ids, :].copy(), marker_sets, expected


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assignments = pd.read_csv(CLUSTER_DIR / "primary_cluster_assignments.csv.gz")
    assignments = assignments[assignments["dataset_id"].isin(["pbmc3k", "heart_cell_atlas_subsampled"])].copy()
    cluster_rows = []
    summary_rows = []
    program_input_rows = []

    for dataset_id, dataset_assignments in assignments.groupby("dataset_id", sort=False):
        cell_ids = dataset_assignments["cell_id"].drop_duplicates().astype(str).tolist()
        adata, marker_sets, expected = load_expression(dataset_id, cell_ids)
        scores, program_inputs = module_scores(adata, marker_sets)
        program_inputs.insert(0, "dataset_id", dataset_id)
        program_input_rows.append(program_inputs)

        for (method, resolution), frame in dataset_assignments.groupby(["method", "resolution"], sort=False):
            frame = frame.set_index("cell_id").loc[cell_ids].reset_index()
            cluster_results = []
            for cluster, cluster_frame in frame.groupby("leiden_cluster", sort=False):
                member_ids = cluster_frame["cell_id"].astype(str)
                member_mask = scores.index.isin(member_ids)
                majority_label = cluster_frame["reference_label"].value_counts().idxmax()
                expected_programs = expected.get(str(majority_label), [])
                effects = {
                    program: standardised_mean_difference(
                        scores.loc[member_mask, program].to_numpy(dtype=float),
                        scores.loc[~member_mask, program].to_numpy(dtype=float),
                    )
                    for program in marker_sets
                }
                ordered = sorted(effects, key=lambda program: (-np.nan_to_num(effects[program], nan=-np.inf), program))
                top_program = ordered[0]
                expected_effects = [effects[program] for program in expected_programs if program in effects]
                expected_ranks = [ordered.index(program) + 1 for program in expected_programs if program in ordered]
                result = {
                    "dataset_id": dataset_id,
                    "method": method,
                    "embedding_seed": 0,
                    "graph_k": 15,
                    "resolution": resolution,
                    "leiden_cluster": cluster,
                    "cluster_n_cells": int(cluster_frame.shape[0]),
                    "cluster_fraction": float(cluster_frame.shape[0] / frame.shape[0]),
                    "majority_reference_label": majority_label,
                    "majority_label_fraction": float(cluster_frame["reference_label"].value_counts(normalize=True).iloc[0]),
                    "expected_programs": ";".join(expected_programs),
                    "top_marker_program": top_program,
                    "top_program_smd": float(effects[top_program]),
                    "expected_program_max_smd": float(np.nanmax(expected_effects)) if expected_effects else float("nan"),
                    "expected_program_best_rank": int(min(expected_ranks)) if expected_ranks else np.nan,
                    "top_program_matches_expected": bool(top_program in expected_programs),
                    "program_smd_json": json.dumps(effects, sort_keys=True),
                    "marker_definition": "mean log1p marker-program expression inside versus outside the cluster, reported as a standardised mean difference",
                }
                cluster_rows.append(result)
                cluster_results.append(result)

            cluster_table = pd.DataFrame(cluster_results)
            weights = cluster_table["cluster_n_cells"].to_numpy(dtype=float)
            concordance = cluster_table["top_program_matches_expected"].astype(float).to_numpy()
            summary_rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "embedding_seed": 0,
                    "graph_k": 15,
                    "resolution": resolution,
                    "n_clusters": int(cluster_table.shape[0]),
                    "cluster_weighted_marker_concordance": float(np.average(concordance, weights=weights)),
                    "unweighted_marker_concordance": float(np.mean(concordance)),
                    "median_expected_program_smd": float(cluster_table["expected_program_max_smd"].median()),
                    "median_expected_program_rank": float(cluster_table["expected_program_best_rank"].median()),
                }
            )

    pd.DataFrame(cluster_rows).to_csv(OUT_DIR / "cluster_marker_concordance.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "marker_concordance_by_resolution.csv", index=False)
    pd.concat(program_input_rows, ignore_index=True).to_csv(OUT_DIR / "marker_program_definitions.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run()


if __name__ == "__main__":
    main()
