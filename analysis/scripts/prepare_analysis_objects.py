"""Prepare the three reproducible empirical analysis objects.

The locked datasets preserve public-data provenance. These analysis objects
standardize cells, genes, labels and reference PCA spaces for the eight-method
figure pipeline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from analysis import config
from analysis.data_loader import load_heart_atlas, load_paul15, load_pbmc3k
from analysis.paths import (
    ANALYSIS_OBJECTS_DIR,
    DATA_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    ensure_analysis_dirs,
)


ROOT = REPO_ROOT
ANALYSIS_DIR = ANALYSIS_OBJECTS_DIR
SOURCE_DIR = DATA_DIR

TARGET_HEART_N = 8000
RARE_KEEP_THRESHOLD = 200


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts_copy(x):
    return x.copy() if sparse.issparse(x) else np.asarray(x).copy()


def _integer_fraction(x, n_rows: int = 200, n_cols: int = 200) -> float:
    sample = x[: min(x.shape[0], n_rows), : min(x.shape[1], n_cols)]
    if sparse.issparse(sample):
        sample = sample.toarray()
    arr = np.asarray(sample)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(np.isclose(finite, np.round(finite))))


def _stratified_heart_indices(adata, target_n: int = TARGET_HEART_N) -> np.ndarray:
    rng = np.random.default_rng(config.SEED)
    obs = adata.obs[["cell_type", "donor"]].astype(str).copy()
    obs["row_index"] = np.arange(adata.n_obs)
    label_counts = obs["cell_type"].value_counts()
    rare_labels = set(label_counts[label_counts <= RARE_KEEP_THRESHOLD].index)

    keep = set(obs.loc[obs["cell_type"].isin(rare_labels), "row_index"].tolist())
    remaining = obs.loc[~obs["row_index"].isin(keep)].copy()
    slots = max(0, target_n - len(keep))
    if slots <= 0:
        return np.array(sorted(keep), dtype=int)

    remaining["stratum"] = remaining["cell_type"] + "||" + remaining["donor"]
    stratum_counts = remaining["stratum"].value_counts()
    raw_alloc = stratum_counts / stratum_counts.sum() * slots
    alloc = np.floor(raw_alloc).astype(int)
    alloc[alloc == 0] = 1
    alloc = np.minimum(alloc, stratum_counts)

    while int(alloc.sum()) > slots:
        candidates = alloc[alloc > 1].index
        if len(candidates) == 0:
            break
        fractional = raw_alloc.loc[candidates] - np.floor(raw_alloc.loc[candidates])
        alloc.loc[fractional.sort_values().index[0]] -= 1
    while int(alloc.sum()) < slots:
        room = (stratum_counts - alloc)
        candidates = room[room > 0].index
        if len(candidates) == 0:
            break
        fractional = raw_alloc.loc[candidates] - np.floor(raw_alloc.loc[candidates])
        alloc.loc[fractional.sort_values(ascending=False).index[0]] += 1

    sampled: list[int] = list(keep)
    for stratum, n_take in alloc.items():
        choices = remaining.loc[remaining["stratum"] == stratum, "row_index"].to_numpy()
        if n_take >= len(choices):
            sampled.extend(choices.tolist())
        else:
            sampled.extend(rng.choice(choices, size=int(n_take), replace=False).tolist())

    sampled = np.array(sorted(set(sampled)), dtype=int)
    if sampled.shape[0] > target_n:
        non_rare = [idx for idx in sampled if idx not in keep]
        trim = sampled.shape[0] - target_n
        drop = set(rng.choice(non_rare, size=trim, replace=False).tolist())
        sampled = np.array([idx for idx in sampled if idx not in drop], dtype=int)
    return sampled


def _preprocess_counts(counts, dataset_id: str, label_field: str, batch_field: str = "") -> tuple[object, object, dict]:
    counts = counts.copy()
    counts.var_names_make_unique()
    counts.obs_names_make_unique()
    counts.obs["analysis_dataset_id"] = dataset_id
    counts.uns["analysis_label_field"] = label_field
    counts.uns["analysis_batch_field"] = batch_field
    counts.uns["counts_integer_fraction"] = _integer_fraction(counts.X)

    proc = counts.copy()
    proc.layers["counts"] = _counts_copy(counts.X)
    sc.pp.filter_genes(proc, min_cells=3)
    if proc.n_vars > config.N_HVG:
        sc.pp.normalize_total(proc, target_sum=1e4)
        sc.pp.log1p(proc)
        sc.pp.highly_variable_genes(proc, n_top_genes=config.N_HVG, flavor="seurat")
        proc = proc[:, proc.var["highly_variable"].to_numpy()].copy()
    else:
        sc.pp.normalize_total(proc, target_sum=1e4)
        sc.pp.log1p(proc)
        proc.var["highly_variable"] = True

    counts_hvg = counts[:, proc.var_names].copy()
    proc.raw = proc
    sc.pp.scale(proc, max_value=10)
    n_comps = min(config.N_PCS, proc.n_obs - 1, proc.n_vars - 1)
    sc.tl.pca(proc, n_comps=n_comps, svd_solver="arpack", random_state=config.SEED)
    proc.obsm["X_pca_ref"] = np.asarray(proc.obsm["X_pca"], dtype=np.float32)
    if "variance_ratio" in proc.uns.get("pca", {}):
        proc.uns["pca_ref_variance_ratio"] = np.asarray(proc.uns["pca"]["variance_ratio"], dtype=np.float32)
    proc.uns["analysis_label_field"] = label_field
    proc.uns["analysis_batch_field"] = batch_field
    counts_hvg.uns["analysis_label_field"] = label_field
    counts_hvg.uns["analysis_batch_field"] = batch_field

    summary = {
        "n_obs": int(proc.n_obs),
        "n_vars": int(proc.n_vars),
        "n_pcs": int(n_comps),
        "label_field": label_field,
        "batch_field": batch_field,
        "counts_integer_fraction": counts.uns["counts_integer_fraction"],
    }
    return proc, counts_hvg, summary


def _write_pair(dataset_id: str, proc, counts, source_cache: str, subset_strategy: str, summary: dict) -> dict[str, object]:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    proc_path = ANALYSIS_DIR / f"{dataset_id}_proc.h5ad"
    counts_path = ANALYSIS_DIR / f"{dataset_id}_counts_hvg.h5ad"
    proc.write_h5ad(proc_path, compression="gzip")
    counts.write_h5ad(counts_path, compression="gzip")
    return {
        "dataset_id": dataset_id,
        "source_cache": source_cache,
        "analysis_proc_path": str(proc_path.relative_to(ROOT)),
        "analysis_counts_path": str(counts_path.relative_to(ROOT)),
        "subset_strategy": subset_strategy,
        "n_obs": summary["n_obs"],
        "n_vars": summary["n_vars"],
        "n_pcs": summary["n_pcs"],
        "label_field": summary["label_field"],
        "batch_field": summary["batch_field"],
        "counts_integer_fraction": summary["counts_integer_fraction"],
        "proc_sha256": _sha256(proc_path),
        "counts_sha256": _sha256(counts_path),
    }


def prepare_pbmc3k() -> dict[str, object]:
    adata_proc, adata_counts = load_pbmc3k()
    adata_counts = adata_counts.copy()
    adata_counts.obs["louvain"] = adata_proc.obs["louvain"].astype(str).values
    proc, counts, summary = _preprocess_counts(adata_counts, "pbmc3k", "louvain")
    return _write_pair(
        "pbmc3k",
        proc,
        counts,
        "scanpy.datasets.pbmc3k counts matched to scanpy.datasets.pbmc3k_processed labels",
        "full matched PBMC3k cells and processed genes",
        summary,
    )


def prepare_public_dataset(
    dataset_id: str,
    adata,
    label_field: str,
    batch_field: str,
    source_description: str,
) -> tuple[dict[str, object], pd.DataFrame | None]:
    subset_table = None
    subset_strategy = "full dataset"

    if dataset_id == "heart_cell_atlas_subsampled":
        selected = _stratified_heart_indices(adata)
        before = adata.obs.copy()
        before["selected_for_analysis"] = False
        before.iloc[selected, before.columns.get_loc("selected_for_analysis")] = True
        subset_table = pd.DataFrame(
            {
                "cell_id": before.index.astype(str),
                "dataset_id": dataset_id,
                "selected_for_analysis": before["selected_for_analysis"].to_numpy(),
                "cell_type": before["cell_type"].astype(str).to_numpy(),
                "donor": before["donor"].astype(str).to_numpy(),
            }
        )
        adata = adata[selected, :].copy()
        subset_strategy = (
            f"deterministic stratified subset, target_n={TARGET_HEART_N}, "
            f"all cell_type levels with <= {RARE_KEEP_THRESHOLD} cells retained, "
            "remaining cells sampled by cell_type x donor"
        )

    proc, counts, summary = _preprocess_counts(
        adata,
        dataset_id,
        label_field,
        batch_field,
    )
    out = _write_pair(dataset_id, proc, counts, source_description, subset_strategy, summary)
    return out, subset_table


def main() -> None:
    ensure_analysis_dirs()
    manifest: list[dict[str, object]] = [prepare_pbmc3k()]
    subset_tables: list[pd.DataFrame] = []

    public_datasets = [
        (
            "paul15",
            load_paul15(),
            "paul15_clusters",
            "",
            "scanpy.datasets.paul15",
        ),
        (
            "heart_cell_atlas_subsampled",
            load_heart_atlas(),
            "cell_type",
            "donor",
            "scvi.data.heart_cell_atlas_subsampled",
        ),
    ]
    for dataset_id, adata, label_field, batch_field, source_description in public_datasets:
        out, subset_table = prepare_public_dataset(
            dataset_id, adata, label_field, batch_field, source_description
        )
        manifest.append(out)
        if subset_table is not None:
            subset_tables.append(subset_table)

    manifest_df = pd.DataFrame(manifest)
    manifest_out = MANIFEST_PATH
    manifest_df.to_csv(manifest_out, index=False)

    if subset_tables:
        subset_out = SOURCE_DIR / "analysis_subset_membership.csv"
        pd.concat(subset_tables, ignore_index=True).to_csv(subset_out, index=False)
    else:
        subset_out = None

    hashes = [f"{_sha256(manifest_out)}  {manifest_out.name}"]
    if subset_out is not None:
        hashes.append(f"{_sha256(subset_out)}  {subset_out.name}")
    (SOURCE_DIR / "analysis_object_manifest.sha256.txt").write_text("\n".join(hashes) + "\n", encoding="utf-8")

    config_out = ANALYSIS_DIR / "analysis_object_config.json"
    config_out.write_text(
        json.dumps(
            {
                "seed": config.SEED,
                "n_hvg": config.N_HVG,
                "n_pcs": config.N_PCS,
                "target_heart_n": TARGET_HEART_N,
                "rare_keep_threshold": RARE_KEEP_THRESHOLD,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(manifest_df.to_string(index=False))
    print(f"\nwrote {manifest_out}")
    if subset_out is not None:
        print(f"wrote {subset_out}")


if __name__ == "__main__":
    main()
