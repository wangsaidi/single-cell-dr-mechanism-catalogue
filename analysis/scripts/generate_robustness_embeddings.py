"""Generate dimension and perturbation robustness source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from analysis import config, diagnostics
from analysis.paths import (
    ANALYSIS_OBJECTS_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    ROBUSTNESS_EMBEDDINGS_DIR,
    SOURCE_RESULTS_DIR,
    ensure_analysis_dirs,
)


ROOT = REPO_ROOT
SOURCE_DIR = SOURCE_RESULTS_DIR
ANALYSIS_DIR = ANALYSIS_OBJECTS_DIR
ROBUST_DIR = ROBUSTNESS_EMBEDDINGS_DIR

OUTPUT_DIMS = [2, 3, 5, 10, 20]
UPSTREAM_PCS = [20, 50, 100]
ROBUST_METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP"]
UPSTREAM_METHODS = ["UMAP", "PHATE", "t-SNE", "PaCMAP"]
SIM_METHODS = ["PCA", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
PERTURB_LEVELS = [0.0, 0.2, 0.4]
PERTURB_REPLICATES = [0, 1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense(x):
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x)


def _read_manifest() -> pd.DataFrame:
    return pd.read_csv(MANIFEST_PATH)


def _load_proc(dataset_id: str):
    manifest = _read_manifest()
    row = manifest.loc[manifest["dataset_id"] == dataset_id].iloc[0]
    return sc.read_h5ad(ROOT / row["analysis_proc_path"]), row


def _compute_pca_space(proc, n_pcs: int) -> np.ndarray:
    key = f"X_pca_ref_{n_pcs}"
    if key in proc.obsm:
        return np.asarray(proc.obsm[key], dtype=np.float32)
    work = proc.copy()
    n_comps = min(n_pcs, work.n_obs - 1, work.n_vars - 1)
    sc.tl.pca(work, n_comps=n_comps, svd_solver="arpack", random_state=config.SEED)
    return np.asarray(work.obsm["X_pca"][:, :n_comps], dtype=np.float32)


def _embed_from_reference(X_ref: np.ndarray, method: str, n_components: int, seed: int) -> np.ndarray:
    if method == "PCA":
        if X_ref.shape[1] < n_components:
            raise ValueError(f"PCA reference has {X_ref.shape[1]} dimensions, requested {n_components}")
        return np.asarray(X_ref[:, :n_components], dtype=np.float32)
    if method == "UMAP":
        import umap

        reducer = umap.UMAP(
            n_neighbors=config.N_NEIGHBORS,
            min_dist=0.5,
            random_state=seed,
            n_components=n_components,
        )
        return np.asarray(reducer.fit_transform(X_ref), dtype=np.float32)
    if method == "PHATE":
        import phate

        reducer = phate.PHATE(
            n_components=n_components,
            knn=config.N_NEIGHBORS,
            n_pca=None,
            random_state=seed,
            n_jobs=1,
            verbose=0,
        )
        return np.asarray(reducer.fit_transform(X_ref), dtype=np.float32)
    if method == "PaCMAP":
        import pacmap

        reducer = pacmap.PaCMAP(
            n_components=n_components,
            n_neighbors=config.N_NEIGHBORS,
            random_state=seed,
            apply_pca=False,
            verbose=False,
        )
        return np.asarray(reducer.fit_transform(X_ref), dtype=np.float32)
    if method == "t-SNE":
        if n_components != 2:
            raise ValueError("t-SNE robustness source data are restricted to n_components=2")
        work = sc.AnnData(X_ref)
        work.obsm["X_pca_ref"] = X_ref
        sc.tl.tsne(work, use_rep="X_pca_ref", perplexity=config.TSNE_PERPLEXITY, random_state=seed, n_jobs=1)
        return np.asarray(work.obsm["X_tsne"], dtype=np.float32)
    raise KeyError(method)


def _label_recall(Z: np.ndarray, labels, k: int = config.N_NEIGHBORS) -> float:
    idx = diagnostics.knn_idx(Z, k)
    labels = np.asarray(labels).astype(str)
    return float(np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(labels.shape[0])]))


def _metric_bundle(dataset_id: str, method: str, labels, X_ref: np.ndarray, Z: np.ndarray, seed: int, extra: dict) -> list[dict[str, object]]:
    local_mean, _ = diagnostics.local_retention(X_ref, Z, k=config.N_NEIGHBORS)
    rows = []
    for metric, value, threshold in [
        ("local_retention", local_mean, config.SUPPORT_THRESHOLDS["local_retention"]),
        ("trustworthiness", diagnostics.trust(X_ref, Z, k=config.N_NEIGHBORS), config.SUPPORT_THRESHOLDS["trustworthiness"]),
        ("global_rank_corr", diagnostics.global_rank_corr(X_ref, Z, seed=seed), config.SUPPORT_THRESHOLDS["global_rank_corr"]),
        ("label_neighbor_recall", _label_recall(Z, labels), config.SUPPORT_THRESHOLDS["label_recall"]),
    ]:
        row = {
            "dataset_id": dataset_id,
            "method": method,
            "family": config.METHOD_FAMILY.get(method, "relational"),
            "seed": seed,
            "metric": metric,
            "value": value,
            "threshold": threshold,
            "support": "pass" if value >= threshold else "below_threshold",
            "n_cells": int(Z.shape[0]),
            "n_unit": "method-dataset-condition run; cells compute metrics and are not independent replicates",
        }
        row.update(extra)
        rows.append(row)
    return rows


def write_output_dimension_response() -> Path:
    rows = []
    for dataset_id in ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]:
        proc, meta = _load_proc(dataset_id)
        label_field = meta["label_field"]
        labels = proc.obs[label_field].astype(str).to_numpy()
        X_ref = np.asarray(proc.obsm["X_pca_ref"], dtype=np.float32)
        for method in ROBUST_METHODS:
            for dim in OUTPUT_DIMS:
                try:
                    cache = ROBUST_DIR / f"dim_{dataset_id}_{method}_{dim}.npy"
                    if cache.exists():
                        Z = np.load(cache)
                    else:
                        Z = _embed_from_reference(X_ref, method, dim, config.SEED)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache, Z.astype(np.float32))
                    rows.extend(
                        _metric_bundle(
                            dataset_id,
                            method,
                            labels,
                            X_ref,
                            Z,
                            config.SEED,
                            {"analysis_axis": "output_embedding_dimension", "output_dimension": dim},
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": method,
                            "family": config.METHOD_FAMILY.get(method, ""),
                            "seed": config.SEED,
                            "metric": "run_status",
                            "value": np.nan,
                            "threshold": np.nan,
                            "support": "not_run",
                            "n_cells": int(proc.n_obs),
                            "n_unit": "method-dataset-condition run",
                            "analysis_axis": "output_embedding_dimension",
                            "output_dimension": dim,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    out = SOURCE_DIR / "fig6_output_dimension_response.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_upstream_pca_response() -> Path:
    rows = []
    for dataset_id in ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]:
        proc, meta = _load_proc(dataset_id)
        label_field = meta["label_field"]
        labels = proc.obs[label_field].astype(str).to_numpy()
        for n_pcs in UPSTREAM_PCS:
            X_ref = _compute_pca_space(proc, n_pcs)
            for method in UPSTREAM_METHODS:
                try:
                    cache = ROBUST_DIR / f"upstream_{dataset_id}_{method}_pca{n_pcs}.npy"
                    if cache.exists():
                        Z = np.load(cache)
                    else:
                        Z = _embed_from_reference(X_ref, method, 2, config.SEED)
                        cache.parent.mkdir(parents=True, exist_ok=True)
                        np.save(cache, Z.astype(np.float32))
                    rows.extend(
                        _metric_bundle(
                            dataset_id,
                            method,
                            labels,
                            X_ref,
                            Z,
                            config.SEED,
                            {"analysis_axis": "upstream_pca_dimension", "upstream_pca_dimension": n_pcs},
                        )
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "dataset_id": dataset_id,
                            "method": method,
                            "family": config.METHOD_FAMILY.get(method, ""),
                            "seed": config.SEED,
                            "metric": "run_status",
                            "value": np.nan,
                            "threshold": np.nan,
                            "support": "not_run",
                            "n_cells": int(proc.n_obs),
                            "n_unit": "method-dataset-condition run",
                            "analysis_axis": "upstream_pca_dimension",
                            "upstream_pca_dimension": n_pcs,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
    out = SOURCE_DIR / "fig6_upstream_pca_response.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _preprocess_counts_to_pca(counts: np.ndarray) -> np.ndarray:
    adata = sc.AnnData(counts)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)
    sc.tl.pca(adata, n_comps=min(config.N_PCS, adata.n_obs - 1, adata.n_vars - 1), svd_solver="arpack", random_state=config.SEED)
    return np.asarray(adata.obsm["X_pca"], dtype=np.float32)


def write_dropout_noise_response() -> Path:
    proc, meta = _load_proc("pbmc3k")
    counts = sc.read_h5ad(ANALYSIS_DIR / "pbmc3k_counts_hvg.h5ad")
    base_counts = _dense(counts.X).astype(np.float32)
    labels = proc.obs[meta["label_field"]].astype(str).to_numpy()
    rows = []
    for perturbation in ["dropout", "noise"]:
        for level in PERTURB_LEVELS:
            for replicate in PERTURB_REPLICATES:
                rng = np.random.default_rng(config.SEED + replicate + int(level * 1000) + (0 if perturbation == "dropout" else 10000))
                perturbed = base_counts.copy()
                if perturbation == "dropout" and level > 0:
                    nonzero = perturbed > 0
                    mask = rng.random(perturbed.shape) < level
                    perturbed[nonzero & mask] = 0
                elif perturbation == "noise" and level > 0:
                    noise = rng.normal(0, level * np.std(np.log1p(base_counts), axis=0, keepdims=True))
                    perturbed = np.maximum(0, np.expm1(np.log1p(perturbed) + noise)).astype(np.float32)
                X_ref = _preprocess_counts_to_pca(perturbed)
                for method in ROBUST_METHODS:
                    try:
                        Z = _embed_from_reference(X_ref, method, 2, config.SEED + replicate)
                        rows.extend(
                            _metric_bundle(
                                "pbmc3k",
                                method,
                                labels,
                                X_ref,
                                Z,
                                config.SEED + replicate,
                                {
                                    "analysis_axis": "dropout_noise_perturbation",
                                    "perturbation": perturbation,
                                    "level": level,
                                    "replicate": replicate,
                                    "perturbation_definition": "real PBMC3k-derived count matrix with nonzero dropout or log-scale Gaussian noise",
                                },
                            )
                        )
                    except Exception as exc:
                        rows.append(
                            {
                                "dataset_id": "pbmc3k",
                                "method": method,
                                "family": config.METHOD_FAMILY.get(method, ""),
                                "seed": config.SEED + replicate,
                                "metric": "run_status",
                                "value": np.nan,
                                "threshold": np.nan,
                                "support": "not_run",
                                "n_cells": int(proc.n_obs),
                                "n_unit": "method-perturbation-replicate run",
                                "analysis_axis": "dropout_noise_perturbation",
                                "perturbation": perturbation,
                                "level": level,
                                "replicate": replicate,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
    out = SOURCE_DIR / "fig6_dropout_noise_response.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def _simulate_counts(seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    n_cells = 1600
    n_genes = 600
    labels = np.repeat(["state_A", "state_B", "state_C", "rare_state"], [520, 500, 500, 80])
    batch = rng.choice(["batch_1", "batch_2"], size=n_cells, p=[0.52, 0.48])
    centers = {
        "state_A": np.array([-2.0, 0.0]),
        "state_B": np.array([0.0, 1.5]),
        "state_C": np.array([2.0, 0.0]),
        "rare_state": np.array([0.0, -2.2]),
    }
    latent = np.vstack([centers[label] + rng.normal(0, 0.45, size=2) for label in labels])
    gene_loadings = rng.normal(0, 0.35, size=(2, n_genes))
    base = rng.normal(-0.8, 0.4, size=n_genes)
    batch_shift = (batch == "batch_2").astype(float)[:, None] * rng.normal(0.0, 0.18, size=(1, n_genes))
    eta = base + latent @ gene_loadings + batch_shift
    mu = np.exp(np.clip(eta, -5, 4))
    counts = rng.poisson(mu).astype(np.float32)
    dropout_prob = 1 / (1 + np.exp(np.log1p(mu) - 1.0))
    counts[rng.random(counts.shape) < dropout_prob * 0.35] = 0
    return counts, labels, batch, latent


def write_known_truth_simulation() -> Path:
    rows = []
    counts, labels, batch, latent = _simulate_counts()
    X_ref = _preprocess_counts_to_pca(counts)
    for method in SIM_METHODS:
        try:
            Z = _embed_from_reference(X_ref, method, 2, config.SEED)
            local_rows = _metric_bundle(
                "known_truth_count_simulation",
                method,
                labels,
                X_ref,
                Z,
                config.SEED,
                {"analysis_axis": "known_truth_simulation", "simulation_replicate": 0},
            )
            rows.extend(local_rows)
            latent_corr = diagnostics.global_rank_corr(latent, Z, seed=config.SEED)
            rows.append(
                {
                    "dataset_id": "known_truth_count_simulation",
                    "method": method,
                    "family": config.METHOD_FAMILY.get(method, "relational"),
                    "seed": config.SEED,
                    "metric": "latent_distance_corr",
                    "value": latent_corr,
                    "threshold": config.SUPPORT_THRESHOLDS["global_rank_corr"],
                    "support": "pass" if latent_corr >= config.SUPPORT_THRESHOLDS["global_rank_corr"] else "below_threshold",
                    "n_cells": int(counts.shape[0]),
                    "n_unit": "simulation cells generated from one known-truth replicate",
                    "analysis_axis": "known_truth_simulation",
                    "simulation_replicate": 0,
                    "simulation_definition": "realistic count-like simulation with known latent geometry, batch labels and rare state",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "dataset_id": "known_truth_count_simulation",
                    "method": method,
                    "family": config.METHOD_FAMILY.get(method, ""),
                    "seed": config.SEED,
                    "metric": "run_status",
                    "value": np.nan,
                    "threshold": np.nan,
                    "support": "not_run",
                    "n_cells": int(counts.shape[0]),
                    "n_unit": "simulation method run",
                    "analysis_axis": "known_truth_simulation",
                    "simulation_replicate": 0,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    sim_meta = SOURCE_DIR / "fig6_known_truth_simulation_design.json"
    sim_meta.write_text(
        json.dumps(
            {
                "seed": config.SEED,
                "n_cells": int(counts.shape[0]),
                "n_genes": int(counts.shape[1]),
                "labels": pd.Series(labels).value_counts().to_dict(),
                "batches": pd.Series(batch).value_counts().to_dict(),
                "definition": "count-like simulation calibrated for source-data stress testing, not experimental biology",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    out = SOURCE_DIR / "fig6_known_truth_simulation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def write_synthesis(paths: dict[str, Path]) -> tuple[Path, Path]:
    frames = []
    for axis, path in paths.items():
        df = pd.read_csv(path)
        df["source_axis"] = axis
        frames.append(df)
    all_df = pd.concat(frames, ignore_index=True, sort=False)
    metric_df = all_df[all_df["metric"].isin(["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall", "latent_distance_corr"])].copy()
    worst = (
        metric_df.groupby(["method", "family", "metric"], as_index=False)
        .agg(worst_value=("value", "min"), n_conditions=("value", "count"))
    )
    threshold_map = {
        "local_retention": config.SUPPORT_THRESHOLDS["local_retention"],
        "trustworthiness": config.SUPPORT_THRESHOLDS["trustworthiness"],
        "global_rank_corr": config.SUPPORT_THRESHOLDS["global_rank_corr"],
        "label_neighbor_recall": config.SUPPORT_THRESHOLDS["label_recall"],
        "latent_distance_corr": config.SUPPORT_THRESHOLDS["global_rank_corr"],
    }
    worst["threshold"] = worst["metric"].map(threshold_map)
    worst["support"] = np.where(worst["worst_value"] >= worst["threshold"], "pass", "below_threshold")
    worst["support_definition"] = "minimum metric value across predeclared dimension, perturbation and simulation conditions"
    worst_out = SOURCE_DIR / "fig6_worst_case_support.csv"
    worst.to_csv(worst_out, index=False)

    practice_rows = []
    mechanism_path = SOURCE_DIR / "fig6_mechanism_simulation_suite.csv"
    if "simulation" in paths and mechanism_path.exists():
        paths = dict(paths)
        paths["simulation"] = mechanism_path
    for axis, path in paths.items():
        df = pd.read_csv(path)
        valid = df[df["support"].isin(["pass", "below_threshold"])]
        practice_rows.append(
            {
                "analysis_axis": "mechanism_simulation" if axis == "simulation" and path.name == "fig6_mechanism_simulation_suite.csv" else axis,
                "source_data": path.name,
                "n_metric_rows": int(valid.shape[0]),
                "fraction_below_threshold": float((valid["support"] == "below_threshold").mean()) if valid.shape[0] else np.nan,
                "practice_check": {
                    "output_dimension": "Check whether the conclusion changes when the output dimension is increased beyond two.",
                    "upstream_pca": "Check whether graph or relational layouts depend on the supplied PCA reference space.",
                    "perturbation": "Check whether dropout or noise stress changes the claim-support diagnostics.",
                    "simulation": "Use the mechanism simulation suite as a ground-truth stress test, not as a universal method ranking.",
                }[axis],
            }
        )
    practice_out = SOURCE_DIR / "fig6_practice_map.csv"
    pd.DataFrame(practice_rows).to_csv(practice_out, index=False)
    return worst_out, practice_out


def write_method_coverage() -> Path:
    rows = []
    for method in config.ANCHOR_METHODS:
        rows.append(
            {
                "method": method,
                "family": config.METHOD_FAMILY[method],
                "output_dimension_response": method in ROBUST_METHODS,
                "upstream_pca_response": method in UPSTREAM_METHODS,
                "dropout_noise_response": method in ROBUST_METHODS,
                "known_truth_simulation": method in SIM_METHODS,
                "coverage_note": "False means unsupported or outside the intended robustness axis; no values are imputed",
            }
        )
    out = SOURCE_DIR / "fig6_method_coverage.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def main() -> None:
    ensure_analysis_dirs()
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    ROBUST_DIR.mkdir(parents=True, exist_ok=True)
    dim_out = write_output_dimension_response()
    upstream_out = write_upstream_pca_response()
    perturb_out = write_dropout_noise_response()
    sim_out = write_known_truth_simulation()
    coverage_out = write_method_coverage()
    worst_out, practice_out = write_synthesis(
        {
            "output_dimension": dim_out,
            "upstream_pca": upstream_out,
            "perturbation": perturb_out,
            "simulation": sim_out,
        }
    )
    paths = [dim_out, upstream_out, perturb_out, sim_out, coverage_out, worst_out, practice_out, SOURCE_DIR / "fig6_known_truth_simulation_design.json"]
    hash_out = SOURCE_DIR / "fig6_robustness_source_data.sha256.txt"
    hash_out.write_text("\n".join(f"{_sha256(path)}  {path.relative_to(ROOT)}" for path in paths) + "\n", encoding="utf-8")
    for path in paths:
        print(f"wrote {path}")
    print(f"wrote {hash_out}")


if __name__ == "__main__":
    main()
