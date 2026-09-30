"""Acquire the three public datasets used in the empirical analyses."""

from __future__ import annotations

import scanpy as sc

from analysis.paths import RAW_CACHE_DIR, ensure_analysis_dirs


def load_pbmc3k():
    """Return Scanpy PBMC3k processed data and matched integer counts."""
    ensure_analysis_dirs()
    proc_path = RAW_CACHE_DIR / "pbmc3k_processed.h5ad"
    counts_path = RAW_CACHE_DIR / "pbmc3k_counts.h5ad"
    if proc_path.exists() and counts_path.exists():
        return sc.read_h5ad(proc_path), sc.read_h5ad(counts_path)

    processed = sc.datasets.pbmc3k_processed().copy()
    counts_all = sc.datasets.pbmc3k().copy()
    processed.var_names_make_unique()
    counts_all.var_names_make_unique()
    common_cells = processed.obs_names.intersection(counts_all.obs_names)
    common_genes = processed.var_names.intersection(counts_all.var_names)
    if len(common_cells) != processed.n_obs or len(common_genes) < processed.n_vars:
        raise ValueError("The PBMC3k processed and count objects do not align as expected.")
    counts = counts_all[processed.obs_names, processed.var_names].copy()
    counts.obs["louvain"] = processed.obs["louvain"].astype(str).values
    processed.write_h5ad(proc_path, compression="gzip")
    counts.write_h5ad(counts_path, compression="gzip")
    return processed, counts


def load_paul15():
    """Return the public Paul15 count object distributed through Scanpy."""
    ensure_analysis_dirs()
    path = RAW_CACHE_DIR / "paul15.h5ad"
    if not path.exists():
        sc.datasets.paul15().write_h5ad(path, compression="gzip")
    return sc.read_h5ad(path)


def load_heart_atlas():
    """Return the public heart-atlas subset distributed through scvi-tools."""
    ensure_analysis_dirs()
    path = RAW_CACHE_DIR / "heart_cell_atlas_subsampled.h5ad"
    if path.exists():
        return sc.read_h5ad(path)
    import scvi

    adata = scvi.data.heart_cell_atlas_subsampled(save_path=str(RAW_CACHE_DIR))
    adata.write_h5ad(path, compression="gzip")
    return adata
