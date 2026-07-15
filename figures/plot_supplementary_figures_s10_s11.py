"""Calibrate local-neighbour and Paul15 continuum diagnostics.

Supplementary Fig. S10 varies neighbourhood size and reference dimension and
compares observed local overlap with permutation nulls. Supplementary Fig. S11
tests Paul15 continuum summaries across DPT roots and lineage-restricted
analyses. Every plotted value is derived from the processed empirical objects
and fitted method coordinates.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures import config  # noqa: E402
from figures.figure_style import (  # noqa: E402
    BORDER,
    DATASET_COLORS,
    DATASET_LABELS,
    FAIL_COLOR,
    GRID,
    INK,
    PASS_COLOR,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    panel_label,
)


SOURCE = ROOT / "data" / "source_data" / "generated"
FIGURES = ROOT / "outputs" / "supplementary_figures"
METADATA = ROOT / "metadata"
ANALYSIS = ROOT / "data"
EMBEDDINGS = ROOT / "data"

METHODS = list(config.ANCHOR_METHODS)
DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
PCA_DIMS = [2, 5, 10, 20, 50]
K_VALUES = [5, 15, 30, 50]
EVALUATION_SEED = 20260714
NULL_PERMUTATIONS = 100
PAIR_SAMPLE_N = 5000

METHOD_COLORS = {
    method: mpl.colors.to_hex(mpl.colormaps["tab10"](index))
    for index, method in enumerate(METHODS)
}


def _stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = SOURCE / name
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def _evaluation_indices(adata, dataset_id: str) -> np.ndarray:
    label_field = str(adata.uns.get("ijbs_label_field", ""))
    if label_field not in adata.obs:
        raise RuntimeError(f"Label field is unavailable for {dataset_id}: {label_field}")
    labels = adata.obs[label_field].astype(str).to_numpy()
    target_n = min(1000, adata.n_obs)
    rng = np.random.default_rng(EVALUATION_SEED)
    groups = {label: np.flatnonzero(labels == label) for label in np.unique(labels)}
    raw = {label: target_n * len(indices) / adata.n_obs for label, indices in groups.items()}
    quota = {label: min(len(groups[label]), max(1, int(np.floor(raw[label])))) for label in groups}
    while sum(quota.values()) > target_n:
        candidates = [label for label in groups if quota[label] > 1]
        label = min(candidates, key=lambda value: raw[value] - quota[value])
        quota[label] -= 1
    while sum(quota.values()) < target_n:
        candidates = [label for label in groups if quota[label] < len(groups[label])]
        label = max(candidates, key=lambda value: raw[value] - quota[value])
        quota[label] += 1
    selected: list[int] = []
    for label in sorted(groups):
        selected.extend(rng.choice(groups[label], size=quota[label], replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)


def _mean_overlap(first: np.ndarray, second: np.ndarray) -> float:
    k = first.shape[1]
    return float(np.mean([len(set(first[i]) & set(second[i])) / k for i in range(first.shape[0])]))


def run_local_calibration() -> dict[str, Path]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    detail_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    rng = np.random.default_rng(EVALUATION_SEED)

    for dataset_id in DATASETS:
        adata = sc.read_h5ad(ANALYSIS / f"{dataset_id}_proc.h5ad")
        selected = _evaluation_indices(adata, dataset_id)
        pca = np.asarray(adata.obsm["X_pca_ref"][selected, :50], dtype=np.float32)
        coordinates = {
            method: np.load(EMBEDDINGS / f"{dataset_id}_{_stem(method)}_seed0.npy").astype(np.float32)[selected]
            for method in METHODS
        }
        reference_neighbors = {
            (dimension, k): diagnostics.knn_idx(pca[:, :dimension], k)
            for dimension in PCA_DIMS
            for k in K_VALUES
        }
        embedding_neighbors = {
            (method, k): diagnostics.knn_idx(coordinates[method], k)
            for method in METHODS
            for k in K_VALUES
        }
        for dimension in PCA_DIMS:
            for k in K_VALUES:
                null_expectation = k / (selected.size - 1)
                ref_idx = reference_neighbors[(dimension, k)]
                for method in METHODS:
                    emb_idx = embedding_neighbors[(method, k)]
                    observed = _mean_overlap(ref_idx, emb_idx)
                    detail_rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": method,
                            "navigation_group": config.METHOD_FAMILY[method],
                            "reference_pca_dimensions": dimension,
                            "k": k,
                            "local_retention": observed,
                            "analytic_random_expectation": null_expectation,
                            "normalised_excess_overlap": (observed - null_expectation) / (1 - null_expectation),
                            "fixed_operational_boundary": 0.30,
                            "meets_boundary": observed >= 0.30,
                            "evaluation_cells": int(selected.size),
                            "n_definition": "one method-dataset-reference-dimension-k analysis on the fixed 1,000-cell evaluation subset",
                        }
                    )
                    if dimension == 50 and k == 15:
                        for permutation_index in range(NULL_PERMUTATIONS):
                            permutation = rng.permutation(selected.size)
                            inverse = np.empty_like(permutation)
                            inverse[permutation] = np.arange(selected.size)
                            permuted_reference = inverse[ref_idx[permutation]]
                            null_rows.append(
                                {
                                    "dataset_id": dataset_id,
                                    "method": method,
                                    "permutation": permutation_index,
                                    "reference_pca_dimensions": dimension,
                                    "k": k,
                                    "null_local_retention": _mean_overlap(permuted_reference, emb_idx),
                                    "evaluation_cells": int(selected.size),
                                    "permutation_seed": EVALUATION_SEED,
                                }
                            )

    detail = pd.DataFrame(detail_rows)
    null = pd.DataFrame(null_rows)
    rank_rows = []
    baseline = detail[detail["k"].eq(15)]
    for dataset_id in DATASETS:
        matrix = baseline[baseline["dataset_id"].eq(dataset_id)].pivot(
            index="method", columns="reference_pca_dimensions", values="local_retention"
        ).reindex(index=METHODS, columns=PCA_DIMS)
        for first, second in combinations(PCA_DIMS, 2):
            rho = spearmanr(matrix[first], matrix[second]).correlation
            rank_rows.append(
                {
                    "dataset_id": dataset_id,
                    "reference_dimension_a": first,
                    "reference_dimension_b": second,
                    "method_rank_spearman_rho": float(rho),
                    "n_methods": len(METHODS),
                }
            )
    rank = pd.DataFrame(rank_rows)
    threshold_counts = (
        detail.groupby(["reference_pca_dimensions", "k"], as_index=False)
        .agg(
            analyses=("meets_boundary", "size"),
            below_boundary=("meets_boundary", lambda values: int((~values).sum())),
            mean_local_retention=("local_retention", "mean"),
        )
    )
    threshold_counts["below_boundary_fraction"] = threshold_counts["below_boundary"] / threshold_counts["analyses"]
    paths = {
        "detail": _write_csv(detail, "SuppS10_local_reference_sensitivity.csv"),
        "null": _write_csv(null, "SuppS10_local_permutation_null.csv"),
        "rank": _write_csv(rank, "SuppS10_reference_rank_stability.csv"),
        "threshold": _write_csv(threshold_counts, "SuppS10_boundary_count_sensitivity.csv"),
    }
    return paths


def _root_index(adata, cluster: str) -> int:
    labels = adata.obs["paul15_clusters"].astype(str).to_numpy()
    candidates = np.where(labels == cluster)[0]
    if candidates.size == 0:
        raise RuntimeError(f"Paul15 cluster {cluster} is absent")
    points = np.asarray(adata.obsm["X_pca_ref"][candidates, :20], dtype=float)
    centroid = points.mean(axis=0)
    return int(candidates[np.argmin(np.linalg.norm(points - centroid, axis=1))])


def _pair_indices(n: int, count: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    first = rng.integers(0, n, count * 2)
    second = rng.integers(0, n, count * 2)
    keep = first != second
    return first[keep][:count], second[keep][:count]


def _safe_spearman(first: np.ndarray, second: np.ndarray) -> float:
    value = spearmanr(first, second).correlation
    return 0.0 if value is None or np.isnan(value) else float(value)


def run_continuum_calibration() -> dict[str, Path]:
    adata = sc.read_h5ad(ANALYSIS / "paul15_proc.h5ad")
    labels = adata.obs["paul15_clusters"].astype(str).to_numpy()
    stored_root = int(adata.uns["iroot"])
    roots = {
        "stored_8Mk_root": stored_root,
        "7MEP_centroid": _root_index(adata, "7MEP"),
        "9GMP_centroid": _root_index(adata, "9GMP"),
        "1Ery_centroid": _root_index(adata, "1Ery"),
    }
    work = adata.copy()
    sc.pp.neighbors(work, n_neighbors=15, use_rep="X_pca_ref", random_state=0)
    sc.tl.diffmap(work, n_comps=15)
    pseudotimes: dict[str, np.ndarray] = {}
    root_rows = []
    for root_name, root_index in roots.items():
        work.uns["iroot"] = int(root_index)
        sc.tl.dpt(work, n_dcs=10)
        values = work.obs["dpt_pseudotime"].to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=np.nanmedian(values))
        pseudotimes[root_name] = values
        root_rows.append(
            {
                "root_definition": root_name,
                "root_index": int(root_index),
                "root_cell_id": str(work.obs_names[root_index]),
                "root_cluster": str(labels[root_index]),
                "n_cells": int(work.n_obs),
                "n_dcs": 10,
                "neighbourhood_k": 15,
            }
        )

    root_concordance_rows = []
    for first in roots:
        for second in roots:
            root_concordance_rows.append(
                {
                    "root_a": first,
                    "root_b": second,
                    "pseudotime_spearman_rho": _safe_spearman(pseudotimes[first], pseudotimes[second]),
                    "n_cells": int(work.n_obs),
                }
            )

    coordinates = {
        method: np.load(EMBEDDINGS / f"paul15_{_stem(method)}_seed0.npy").astype(np.float32)
        for method in METHODS
    }
    embedding_neighbors = {method: diagnostics.knn_idx(points, 15) for method, points in coordinates.items()}
    pair_i, pair_j = _pair_indices(work.n_obs, PAIR_SAMPLE_N, EVALUATION_SEED)
    metric_rows = []
    for root_name, pseudotime in pseudotimes.items():
        random_delta = np.abs(pseudotime[pair_i] - pseudotime[pair_j])
        random_mean = float(random_delta.mean())
        for method, points in coordinates.items():
            distance = np.linalg.norm(points[pair_i] - points[pair_j], axis=1)
            rank_corr = _safe_spearman(random_delta, distance)
            neighbors = embedding_neighbors[method]
            local_delta = np.array(
                [np.mean(np.abs(pseudotime[neighbors[cell]] - pseudotime[cell])) for cell in range(work.n_obs)]
            )
            local_retention = 1 - float(local_delta.mean() / random_mean)
            metric_rows.extend(
                [
                    {
                        "root_definition": root_name,
                        "method": method,
                        "navigation_group": config.METHOD_FAMILY[method],
                        "metric": "pseudotime_distance_correlation",
                        "value": rank_corr,
                        "boundary": 0.45,
                        "meets_boundary": rank_corr >= 0.45,
                        "n_cells": int(work.n_obs),
                        "sampled_pairs": int(pair_i.size),
                    },
                    {
                        "root_definition": root_name,
                        "method": method,
                        "navigation_group": config.METHOD_FAMILY[method],
                        "metric": "local_pseudotime_retention",
                        "value": local_retention,
                        "boundary": 0.50,
                        "meets_boundary": local_retention >= 0.50,
                        "n_cells": int(work.n_obs),
                        "sampled_pairs": int(pair_i.size),
                    },
                ]
            )

    branch_definitions = {
        "erythroid": ["7MEP", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
        "megakaryocytic": ["7MEP", "8Mk"],
        "basophil": ["7MEP", "12Baso", "13Baso"],
        "myeloid": ["9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"],
    }
    branch_roots = {
        "erythroid": "7MEP_centroid",
        "megakaryocytic": "7MEP_centroid",
        "basophil": "7MEP_centroid",
        "myeloid": "9GMP_centroid",
    }
    branch_rows = []
    for branch, clusters in branch_definitions.items():
        indices = np.where(np.isin(labels, clusters))[0]
        root_name = branch_roots[branch]
        pseudotime = pseudotimes[root_name]
        local_i, local_j = _pair_indices(indices.size, min(3000, indices.size * 5), EVALUATION_SEED + len(branch))
        cell_i, cell_j = indices[local_i], indices[local_j]
        reference_delta = np.abs(pseudotime[cell_i] - pseudotime[cell_j])
        for method, points in coordinates.items():
            distance = np.linalg.norm(points[cell_i] - points[cell_j], axis=1)
            branch_rows.append(
                {
                    "lineage": branch,
                    "root_definition": root_name,
                    "method": method,
                    "navigation_group": config.METHOD_FAMILY[method],
                    "within_lineage_pseudotime_distance_correlation": _safe_spearman(reference_delta, distance),
                    "n_lineage_cells": int(indices.size),
                    "sampled_within_lineage_pairs": int(cell_i.size),
                    "cluster_membership": ";".join(clusters),
                }
            )

    coarse = np.array(["other"] * work.n_obs, dtype=object)
    for cell, cluster in enumerate(labels):
        if cluster.endswith("Ery"):
            coarse[cell] = "erythroid"
        elif cluster == "8Mk":
            coarse[cell] = "megakaryocytic"
        elif "Baso" in cluster:
            coarse[cell] = "basophil"
        elif cluster in {"9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"}:
            coarse[cell] = "myeloid"
        elif cluster == "19Lymph":
            coarse[cell] = "lymphoid"
        elif cluster == "7MEP":
            coarse[cell] = "progenitor"
    branch_neighbor_rows = []
    for method, neighbors in embedding_neighbors.items():
        per_cell = np.mean(coarse[neighbors] == coarse[:, None], axis=1)
        branch_neighbor_rows.append(
            {
                "method": method,
                "navigation_group": config.METHOD_FAMILY[method],
                "same_branch_neighbor_fraction": float(per_cell.mean()),
                "n_cells": int(work.n_obs),
                "k": 15,
                "branch_definition": "coarse Paul15 annotation branch; 7MEP retained as progenitor",
            }
        )

    metric = pd.DataFrame(metric_rows)
    rank_rows = []
    rank_metric = metric[metric["metric"].eq("pseudotime_distance_correlation")].pivot(
        index="method", columns="root_definition", values="value"
    ).reindex(index=METHODS, columns=list(roots))
    for first, second in combinations(roots, 2):
        rank_rows.append(
            {
                "root_a": first,
                "root_b": second,
                "method_rank_spearman_rho": _safe_spearman(rank_metric[first], rank_metric[second]),
                "n_methods": len(METHODS),
            }
        )

    paths = {
        "roots": _write_csv(pd.DataFrame(root_rows), "SuppS11_dpt_root_definitions.csv"),
        "root_concordance": _write_csv(pd.DataFrame(root_concordance_rows), "SuppS11_dpt_root_concordance.csv"),
        "metrics": _write_csv(metric, "SuppS11_root_sensitive_continuum_metrics.csv"),
        "branches": _write_csv(pd.DataFrame(branch_rows), "SuppS11_lineage_restricted_metrics.csv"),
        "branch_neighbors": _write_csv(pd.DataFrame(branch_neighbor_rows), "SuppS11_same_branch_neighbour_fraction.csv"),
        "rank": _write_csv(pd.DataFrame(rank_rows), "SuppS11_root_method_rank_stability.csv"),
    }
    return paths


def _heatmap(ax, matrix: pd.DataFrame, *, vmin: float, vmax: float, cmap: str, fmt: str = ".2f") -> None:
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(matrix.shape[0]))
    ax.set_yticklabels(matrix.index)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = float(matrix.iloc[row, column])
            ax.text(column, row, format(value, fmt), ha="center", va="center", fontsize=4.7, color="white" if abs(value) > (vmax - vmin) * 0.62 + vmin else INK)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _save(fig: plt.Figure, name: str) -> None:
    stem = FIGURES / name
    for extension in ("pdf", "svg", "png", "jpg"):
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in {"png", "jpg"}:
            kwargs["dpi"] = 600
        if extension == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)


def plot_local(paths: dict[str, Path]) -> None:
    apply_final_style()
    detail = pd.read_csv(paths["detail"])
    null = pd.read_csv(paths["null"])
    rank = pd.read_csv(paths["rank"])
    threshold = pd.read_csv(paths["threshold"])
    fig = plt.figure(figsize=(7.3, 9.0))
    grid = fig.add_gridspec(4, 2, hspace=0.74, wspace=0.46)

    ax = fig.add_subplot(grid[0, 0])
    matrix = detail[detail["k"].eq(15)].groupby(["method", "reference_pca_dimensions"])["local_retention"].mean().unstack().reindex(index=METHODS, columns=PCA_DIMS)
    image = _heatmap(ax, matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("PCA reference dimensions")
    ax.set_title("Reference dimension changes overlap", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="mean local retention")
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    matrix = detail[detail["reference_pca_dimensions"].eq(50)].groupby(["method", "k"])["local_retention"].mean().unstack().reindex(index=METHODS, columns=K_VALUES)
    image = _heatmap(ax, matrix, vmin=0, vmax=1, cmap="viridis")
    ax.set_xlabel("neighbourhood size k")
    ax.set_title("Neighbourhood size changes overlap", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="mean local retention")
    panel_label(ax, "b")

    ax = fig.add_subplot(grid[1, 0])
    paired = detail[detail["k"].eq(15) & detail["reference_pca_dimensions"].isin([2, 50])].pivot(index=["dataset_id", "method"], columns="reference_pca_dimensions", values="local_retention").reset_index()
    for method in METHODS:
        sub = paired[paired["method"].eq(method)]
        ax.scatter(sub[2], sub[50], s=22, color=METHOD_COLORS[method], label=method, alpha=0.9)
    ax.plot([0, 1], [0, 1], color=GRID, lw=0.8, ls="--")
    ax.set_xlabel("local retention vs PCA2")
    ax.set_ylabel("local retention vs PCA50")
    ax.set_title("Dimension matching changes the reference", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "c")

    ax = fig.add_subplot(grid[1, 1])
    observed = detail[detail["reference_pca_dimensions"].eq(50) & detail["k"].eq(15)][["dataset_id", "method", "local_retention"]]
    null_summary = null.groupby(["dataset_id", "method"])["null_local_retention"].agg(["mean", lambda values: values.quantile(0.025), lambda values: values.quantile(0.975)]).reset_index()
    null_summary.columns = ["dataset_id", "method", "null_mean", "null_q025", "null_q975"]
    calibrated = observed.merge(null_summary, on=["dataset_id", "method"], validate="one_to_one")
    for index, method in enumerate(METHODS):
        sub = calibrated[calibrated["method"].eq(method)]
        x = np.full(len(sub), index) + np.linspace(-0.12, 0.12, len(sub))
        ax.vlines(x, sub["null_q025"], sub["null_q975"], color="#BDBDBD", lw=1.0)
        ax.scatter(x, sub["local_retention"], s=20, color=METHOD_COLORS[method], zorder=3)
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("local retention")
    ax.set_title("Observed overlap exceeds permutation null", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "d")

    ax = fig.add_subplot(grid[2, 0])
    matrix = threshold.pivot(index="reference_pca_dimensions", columns="k", values="below_boundary").reindex(index=PCA_DIMS, columns=K_VALUES)
    image = _heatmap(ax, matrix, vmin=0, vmax=24, cmap="magma_r", fmt=".0f")
    ax.set_xlabel("neighbourhood size k")
    ax.set_ylabel("PCA reference dimensions")
    ax.set_title("The 0.30 count is design-dependent", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="analyses below 0.30")
    panel_label(ax, "e")

    ax = fig.add_subplot(grid[2, 1])
    rank_matrix = pd.DataFrame(np.eye(len(PCA_DIMS)), index=PCA_DIMS, columns=PCA_DIMS)
    pooled = detail[detail["k"].eq(15)].groupby(["method", "reference_pca_dimensions"])["local_retention"].mean().unstack().reindex(index=METHODS, columns=PCA_DIMS)
    for first in PCA_DIMS:
        for second in PCA_DIMS:
            rank_matrix.loc[first, second] = _safe_spearman(pooled[first], pooled[second])
    image = _heatmap(ax, rank_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xlabel("PCA reference dimensions")
    ax.set_ylabel("PCA reference dimensions")
    ax.set_title("Method ordering varies across references", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="method-rank rho")
    panel_label(ax, "f")

    ax = fig.add_subplot(grid[3, 0])
    baseline = detail[detail["reference_pca_dimensions"].eq(50) & detail["k"].eq(15)]
    for dataset_id in DATASETS:
        sub = baseline[baseline["dataset_id"].eq(dataset_id)].set_index("method").reindex(METHODS)
        ax.plot(np.arange(len(METHODS)), sub["local_retention"], marker="o", ms=3.4, lw=0.9, color=DATASET_COLORS[dataset_id], label=DATASET_LABELS[dataset_id])
    ax.axhline(0.30, color=FAIL_COLOR, ls="--", lw=0.8)
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("local retention, PCA50 and k=15")
    ax.set_title("Context remains visible at fixed settings", loc="left")
    ax.legend(frameon=False, fontsize=5.2, loc="upper left")
    clean_axis(ax, grid=True)
    panel_label(ax, "g")

    ax = fig.add_subplot(grid[3, 1])
    excess = baseline.groupby("method")["normalised_excess_overlap"].agg(["mean", "min", "max"]).reindex(METHODS)
    x = np.arange(len(METHODS))
    ax.vlines(x, excess["min"], excess["max"], color=[METHOD_COLORS[m] for m in METHODS], lw=1.2)
    ax.scatter(x, excess["mean"], s=24, color=[METHOD_COLORS[m] for m in METHODS], zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("overlap above random expectation")
    ax.set_title("Random-normalised overlap remains continuous", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "h")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.07)
    _save(fig, "Supplementary_Figure_S10_local_calibration")


def plot_continuum(paths: dict[str, Path]) -> None:
    apply_final_style()
    roots = pd.read_csv(paths["roots"])
    concordance = pd.read_csv(paths["root_concordance"])
    metrics = pd.read_csv(paths["metrics"])
    branches = pd.read_csv(paths["branches"])
    branch_neighbors = pd.read_csv(paths["branch_neighbors"])
    rank = pd.read_csv(paths["rank"])
    root_order = roots["root_definition"].tolist()
    root_labels = [value.replace("_", " ") for value in root_order]

    fig = plt.figure(figsize=(7.3, 9.0))
    grid = fig.add_gridspec(4, 2, hspace=0.72, wspace=0.46)

    ax = fig.add_subplot(grid[0, 0])
    matrix = concordance.pivot(index="root_a", columns="root_b", values="pseudotime_spearman_rho").reindex(index=root_order, columns=root_order)
    matrix.index = root_labels
    matrix.columns = root_labels
    image = _heatmap(ax, matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("DPT ordering depends on root choice", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="cell-rank rho")
    panel_label(ax, "a")

    for panel_index, metric_name in enumerate(["pseudotime_distance_correlation", "local_pseudotime_retention"]):
        ax = fig.add_subplot(grid[0, 1] if panel_index == 0 else grid[1, 0])
        sub = metrics[metrics["metric"].eq(metric_name)]
        for method in METHODS:
            values = sub[sub["method"].eq(method)].set_index("root_definition").reindex(root_order)
            ax.plot(np.arange(len(root_order)), values["value"], marker="o", ms=3.1, lw=0.9, color=METHOD_COLORS[method], label=method)
        boundary = 0.45 if panel_index == 0 else 0.50
        ax.axhline(boundary, color=FAIL_COLOR, ls="--", lw=0.8)
        ax.set_xticks(np.arange(len(root_order)))
        ax.set_xticklabels(root_labels, rotation=35, ha="right")
        ax.set_ylabel("correlation" if panel_index == 0 else "local retention")
        ax.set_title("Pooled distance agreement across roots" if panel_index == 0 else "Local pseudotime retention across roots", loc="left")
        clean_axis(ax, grid=True)
        panel_label(ax, "b" if panel_index == 0 else "c")

    ax = fig.add_subplot(grid[1, 1])
    branch_matrix = branches.pivot(index="method", columns="lineage", values="within_lineage_pseudotime_distance_correlation").reindex(index=METHODS, columns=["erythroid", "megakaryocytic", "basophil", "myeloid"])
    image = _heatmap(ax, branch_matrix, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_title("Lineage-restricted distance agreement", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02, label="within-lineage rho")
    panel_label(ax, "d")

    ax = fig.add_subplot(grid[2, 0])
    values = branch_neighbors.set_index("method").reindex(METHODS)
    x = np.arange(len(METHODS))
    ax.bar(x, values["same_branch_neighbor_fraction"], color=[METHOD_COLORS[m] for m in METHODS], width=0.68)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("same-branch neighbour fraction")
    ax.set_title("Branch labels provide a distinct endpoint", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "e")

    ax = fig.add_subplot(grid[2, 1])
    stored = metrics[metrics["metric"].eq("pseudotime_distance_correlation") & metrics["root_definition"].eq("stored_8Mk_root")].set_index("method")
    branch_median = branches.groupby("method")["within_lineage_pseudotime_distance_correlation"].median()
    for method in METHODS:
        ax.scatter(stored.loc[method, "value"], branch_median.loc[method], s=28, color=METHOD_COLORS[method])
        ax.text(stored.loc[method, "value"] + 0.006, branch_median.loc[method], method, fontsize=4.6, color=METHOD_COLORS[method])
    ax.axvline(0.45, color=FAIL_COLOR, ls="--", lw=0.7)
    ax.axhline(0.45, color=FAIL_COLOR, ls="--", lw=0.7)
    ax.set_xlabel("pooled rho, stored root")
    ax.set_ylabel("median within-lineage rho")
    ax.set_title("Pooled and branch-aware endpoints diverge", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "f")

    ax = fig.add_subplot(grid[3, 0])
    pass_fraction = metrics[metrics["metric"].eq("pseudotime_distance_correlation")].groupby("method")["meets_boundary"].mean().reindex(METHODS)
    ax.bar(np.arange(len(METHODS)), pass_fraction, color=[METHOD_COLORS[m] for m in METHODS], width=0.68)
    ax.set_xticks(np.arange(len(METHODS)))
    ax.set_xticklabels(METHODS, rotation=45, ha="right")
    ax.set_ylabel("fraction of DPT roots meeting 0.45")
    ax.set_ylim(0, 1.05)
    ax.set_title("Continuum calls vary with the root", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "g")

    ax = fig.add_subplot(grid[3, 1])
    rank_values = rank["method_rank_spearman_rho"].to_numpy(dtype=float)
    ax.scatter(np.arange(len(rank_values)), rank_values, s=27, color="#4C78A8")
    labels = [f"{a.split('_')[0]} / {b.split('_')[0]}" for a, b in zip(rank["root_a"], rank["root_b"])]
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("method-rank rho")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Method ordering is root-sensitive", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "h")

    fig.subplots_adjust(left=0.12, right=0.98, top=0.98, bottom=0.07)
    _save(fig, "Supplementary_Figure_S11_continuum_calibration")


def main() -> None:
    local_paths = {
        "detail": SOURCE / "SuppS10_local_reference_sensitivity.csv",
        "null": SOURCE / "SuppS10_local_permutation_null.csv",
        "rank": SOURCE / "SuppS10_reference_rank_stability.csv",
        "threshold": SOURCE / "SuppS10_boundary_count_sensitivity.csv",
    }
    continuum_paths = {
        "roots": SOURCE / "SuppS11_dpt_root_definitions.csv",
        "root_concordance": SOURCE / "SuppS11_dpt_root_concordance.csv",
        "metrics": SOURCE / "SuppS11_root_sensitive_continuum_metrics.csv",
        "branches": SOURCE / "SuppS11_lineage_restricted_metrics.csv",
        "branch_neighbors": SOURCE / "SuppS11_same_branch_neighbour_fraction.csv",
        "rank": SOURCE / "SuppS11_root_method_rank_stability.csv",
    }
    plot_local(local_paths)
    plot_continuum(continuum_paths)

if __name__ == "__main__":
    main()
