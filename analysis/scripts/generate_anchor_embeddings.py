"""Generate the empirical anchor embeddings and preservation diagnostics.

This script creates plot-ready source data for Fig. 3 and the first layer of
diagnostic metrics. Failed methods are logged and left missing; no placeholder
coordinates are generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from sklearn.manifold import TSNE

from analysis import config, diagnostics
from analysis.paths import (
    ANALYSIS_OBJECTS_DIR,
    ANALYSIS_ROOT,
    ANCHOR_EMBEDDINGS_DIR,
    LEGACY_CACHE_DIR,
    LOGS_DIR as ANALYSIS_LOGS_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    SOURCE_RESULTS_DIR,
    ensure_analysis_dirs,
)


ROOT = REPO_ROOT
SOURCE_DIR = SOURCE_RESULTS_DIR
ANALYSIS_DIR = ANALYSIS_OBJECTS_DIR
EMB_DIR = ANCHOR_EMBEDDINGS_DIR
LEGACY_DIR = LEGACY_CACHE_DIR
LOG_DIR = ANALYSIS_LOGS_DIR
MANIFEST = MANIFEST_PATH

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
DEEP_METHODS = {"scScope", "SAUCIE"}


def _ensure_dirs() -> None:
    for path in (SOURCE_DIR, EMB_DIR, LEGACY_DIR, LOG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _stem(value: str) -> str:
    return value.lower().replace("-", "").replace(" ", "_")


def _dense_float32(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_hash(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _embedding_path(dataset_id: str, method: str, seed: int) -> Path:
    return EMB_DIR / f"{dataset_id}_{_stem(method)}_seed{seed}.npy"


def _load_or_none(path: Path):
    if path.exists():
        return np.load(path).astype(np.float32)
    return None


def _save(path: Path, arr: np.ndarray) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr, dtype=np.float32)
    np.save(path, arr)
    return arr


def _counts_matrix(counts) -> np.ndarray:
    return _dense_float32(counts.X)


def _pca(proc, seed: int) -> np.ndarray:
    _ = seed
    return np.asarray(proc.obsm["X_pca_ref"][:, :2], dtype=np.float32)


def _glmpca(counts, seed: int) -> np.ndarray:
    from glmpca.glmpca import glmpca

    config.set_seeds(seed)
    result = glmpca(
        _counts_matrix(counts).T,
        2,
        fam="poi",
        ctl={"maxIter": config.GLMPCA_MAX_ITER, "eps": 1e-4},
        verbose=False,
    )
    return np.asarray(result["factors"], dtype=np.float32)


def _tsne(proc, seed: int) -> np.ndarray:
    config.set_seeds(seed)
    work = proc.copy()
    sc.tl.tsne(
        work,
        use_rep="X_pca_ref",
        perplexity=config.TSNE_PERPLEXITY,
        random_state=seed,
        n_jobs=1,
    )
    return np.asarray(work.obsm["X_tsne"], dtype=np.float32)


def _umap(proc, seed: int) -> np.ndarray:
    import umap

    config.set_seeds(seed)
    reducer = umap.UMAP(
        n_neighbors=config.N_NEIGHBORS,
        min_dist=0.5,
        random_state=seed,
        n_components=2,
    )
    return np.asarray(reducer.fit_transform(proc.obsm["X_pca_ref"]), dtype=np.float32)


def _phate(proc, seed: int) -> np.ndarray:
    import phate

    config.set_seeds(seed)
    reducer = phate.PHATE(
        n_components=2,
        knn=config.N_NEIGHBORS,
        n_pca=None,
        random_state=seed,
        n_jobs=1,
        verbose=0,
    )
    return np.asarray(reducer.fit_transform(proc.obsm["X_pca_ref"]), dtype=np.float32)


def _pacmap(proc, seed: int) -> np.ndarray:
    import pacmap

    config.set_seeds(seed)
    reducer = pacmap.PaCMAP(
        n_components=2,
        n_neighbors=config.N_NEIGHBORS,
        random_state=seed,
        apply_pca=False,
        verbose=False,
    )
    return np.asarray(reducer.fit_transform(proc.obsm["X_pca_ref"]), dtype=np.float32)


def _legacy_input_path(dataset_id: str, method: str) -> Path:
    if method == "scScope":
        return LEGACY_DIR / f"{dataset_id}_scscope_library_normalised_input.npz"
    return LEGACY_DIR / f"{dataset_id}_log1p_counts_input.npz"


def _prepare_legacy_input(dataset_id: str, method: str, counts, label_field: str) -> Path:
    raw_counts = _counts_matrix(counts).astype(np.float32)
    if method == "scScope":
        library_size = raw_counts.sum(axis=1)
        positive = library_size > 0
        target_library_size = float(np.median(library_size[positive]))
        X = np.zeros_like(raw_counts, dtype=np.float32)
        X[positive] = raw_counts[positive] * (target_library_size / library_size[positive, None])
        transform = (
            "per-cell library-size normalisation to the median positive library size on analysis-object HVGs; "
            "no log transform, following the official scScope demonstration"
        )
    else:
        X = np.log1p(raw_counts).astype(np.float32)
        target_library_size = None
        transform = "log1p(raw counts) on analysis-object HVGs"
    cells = np.asarray(counts.obs_names.astype(str))
    genes = np.asarray(counts.var_names.astype(str))
    labels = np.asarray(counts.obs[label_field].astype(str)) if label_field in counts.obs else np.asarray([""] * counts.n_obs)
    path = _legacy_input_path(dataset_id, method)
    meta_path = path.with_suffix(".json")
    digest = _array_hash(X)
    if path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("matrix_sha256") == digest and meta.get("shape") == list(X.shape):
                return path
        except json.JSONDecodeError:
            pass

    np.savez_compressed(path, X=X, cells=cells, genes=genes, labels=labels)
    meta_path.write_text(
        json.dumps(
            {
                "dataset_id": dataset_id,
                "shape": list(X.shape),
                "matrix_sha256": digest,
                "method": method,
                "transform": transform,
                "target_library_size": target_library_size,
                "label_field": label_field,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _legacy_output_path(dataset_id: str, method: str, seed: int) -> Path:
    tag = f"analysis_{dataset_id}_recommended_v1" if method == "scScope" else f"analysis_{dataset_id}"
    return LEGACY_DIR / f"{_stem(method)}_seed{seed}_{tag}.npy"


def _project_scscope_latent(dataset_id: str, latent: np.ndarray, seed: int) -> np.ndarray:
    output = LEGACY_DIR / f"scscope_seed{seed}_analysis_{dataset_id}_recommended_v1_tsne2.npy"
    cached = _load_or_none(output)
    if cached is not None:
        return cached
    coords = TSNE(
        n_components=2,
        perplexity=config.TSNE_PERPLEXITY,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(latent).astype(np.float32)
    _save(output, coords)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "scScope",
                "representation": "official-style latent representation projected to two dimensions with t-SNE",
                "latent_shape": list(latent.shape),
                "embedding_shape": list(coords.shape),
                "latent_sha256": _array_hash(latent),
                "seed": seed,
                "tsne_perplexity": config.TSNE_PERPLEXITY,
                "tsne_init": "pca",
                "tsne_learning_rate": "auto",
                "tsne_max_iter": 1000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return coords


def _run_legacy(dataset_id: str, method: str, counts, label_field: str, seed: int) -> np.ndarray:
    out = _legacy_output_path(dataset_id, method, seed)
    latent = _load_or_none(out)

    if latent is None:
        configured_python = os.environ.get("SCDR_LEGACY_PYTHON", "")
        python_exe = Path(configured_python) if configured_python else Path()
        if not python_exe.exists():
            raise FileNotFoundError(
                "Set SCDR_LEGACY_PYTHON to the Python executable in the recorded Python 3.7 environment."
            )

        input_path = _prepare_legacy_input(dataset_id, method, counts, label_field)
        runner = ANALYSIS_ROOT / "legacy_deep_runner.py"
        saucie_parent = Path(
            os.environ.get("SCDR_SAUCIE_SOURCE", ANALYSIS_ROOT / "external" / "SAUCIE")
        )
        tag = f"analysis_{dataset_id}_recommended_v1" if method == "scScope" else f"analysis_{dataset_id}"
        cmd = [
            str(python_exe),
            str(runner),
            "--input",
            str(input_path),
            "--out-dir",
            str(LEGACY_DIR),
            "--methods",
            method,
            "--seed",
            str(seed),
            "--tag",
            tag,
            "--batch-size",
            "64",
            "--latent-dim",
            "50" if method == "scScope" else "2",
            "--scscope-epochs",
            "300" if method == "scScope" else "20",
            "--scscope-recurrence",
            "2" if method == "scScope" else "1",
            "--saucie-steps",
            "300",
            "--saucie-parent",
            str(saucie_parent),
        ]
        env = os.environ.copy()
        env["PATH"] = os.pathsep.join(
            [
                str(python_exe.parent),
                str(python_exe.parent / "Library" / "bin"),
                str(python_exe.parent / "Library" / "mingw-w64" / "bin"),
                str(python_exe.parent / "DLLs"),
                str(python_exe.parent / "Scripts"),
                env.get("PATH", ""),
            ]
        )
        env["PYTHONPATH"] = str(saucie_parent)
        env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        subprocess.run(cmd, check=True, env=env)
        latent = _load_or_none(out)
        if latent is None:
            raise FileNotFoundError(f"Legacy runner did not create {out}")

    if method == "scScope":
        return _project_scscope_latent(dataset_id, latent, seed)
    return latent


def compute_embedding(dataset_id: str, method: str, proc, counts, label_field: str, seed: int, force: bool = False) -> np.ndarray:
    path = _embedding_path(dataset_id, method, seed)
    if not force:
        cached = _load_or_none(path)
        if cached is not None:
            return cached
    if method == "PCA":
        coords = _pca(proc, seed)
    elif method == "GLM-PCA":
        coords = _glmpca(counts, seed)
    elif method == "UMAP":
        coords = _umap(proc, seed)
    elif method == "PHATE":
        coords = _phate(proc, seed)
    elif method == "t-SNE":
        coords = _tsne(proc, seed)
    elif method == "PaCMAP":
        coords = _pacmap(proc, seed)
    elif method in DEEP_METHODS:
        coords = _run_legacy(dataset_id, method, counts, label_field, seed)
    else:
        raise KeyError(method)
    if coords.shape != (proc.n_obs, 2):
        raise ValueError(f"{dataset_id} {method} expected {(proc.n_obs, 2)}, got {coords.shape}")
    return _save(path, coords)


def _label_recall(Z: np.ndarray, labels, k: int = config.N_NEIGHBORS) -> float:
    idx = diagnostics.knn_idx(Z, k)
    lab = np.asarray(labels).astype(str)
    per_cell = np.array([(lab[idx[i]] == lab[i]).mean() for i in range(lab.shape[0])], dtype=float)
    return float(np.nanmean(per_cell))


def _metric_rows(dataset_id: str, method: str, proc, labels, coords: np.ndarray, seed: int) -> list[dict[str, object]]:
    X_high = np.asarray(proc.obsm["X_pca_ref"][:, : min(config.N_PCS, proc.obsm["X_pca_ref"].shape[1])], dtype=np.float32)
    local_mean, _ = diagnostics.local_retention(X_high, coords, k=config.N_NEIGHBORS)
    rows = [
        {
            "dataset_id": dataset_id,
            "method": method,
            "family": config.METHOD_FAMILY[method],
            "seed": seed,
            "metric": "local_retention",
            "value": local_mean,
            "k": config.N_NEIGHBORS,
            "n_cells": int(proc.n_obs),
            "n_pcs_reference": int(X_high.shape[1]),
            "n_definition": "method-dataset-seed run; cells compute the metric but are not independent replicates",
        },
        {
            "dataset_id": dataset_id,
            "method": method,
            "family": config.METHOD_FAMILY[method],
            "seed": seed,
            "metric": "trustworthiness",
            "value": diagnostics.trust(X_high, coords, k=config.N_NEIGHBORS),
            "k": config.N_NEIGHBORS,
            "n_cells": int(proc.n_obs),
            "n_pcs_reference": int(X_high.shape[1]),
            "n_definition": "method-dataset-seed run; cells compute the metric but are not independent replicates",
        },
        {
            "dataset_id": dataset_id,
            "method": method,
            "family": config.METHOD_FAMILY[method],
            "seed": seed,
            "metric": "global_rank_corr",
            "value": diagnostics.global_rank_corr(X_high, coords, n_pairs=config.RANK_PAIRS, seed=seed),
            "k": np.nan,
            "n_cells": int(proc.n_obs),
            "n_pcs_reference": int(X_high.shape[1]),
            "n_definition": f"{config.RANK_PAIRS} sampled cell pairs per method-dataset-seed run; pairs are computational subsamples",
        },
        {
            "dataset_id": dataset_id,
            "method": method,
            "family": config.METHOD_FAMILY[method],
            "seed": seed,
            "metric": "label_neighbor_recall",
            "value": _label_recall(coords, labels, k=config.N_NEIGHBORS),
            "k": config.N_NEIGHBORS,
            "n_cells": int(proc.n_obs),
            "n_pcs_reference": int(X_high.shape[1]),
            "n_definition": "method-dataset-seed run; per-cell neighbour labels compute the metric",
        },
    ]
    return rows


def _support_rows(metrics: pd.DataFrame) -> pd.DataFrame:
    threshold = {
        "local_retention": config.SUPPORT_THRESHOLDS["local_retention"],
        "trustworthiness": config.SUPPORT_THRESHOLDS["trustworthiness"],
        "global_rank_corr": config.SUPPORT_THRESHOLDS["global_rank_corr"],
        "label_neighbor_recall": config.SUPPORT_THRESHOLDS["label_recall"],
    }
    rows = []
    for (dataset_id, method, family), sub in metrics.groupby(["dataset_id", "method", "family"]):
        passed = []
        for metric, cut in threshold.items():
            value = float(sub.loc[sub["metric"] == metric, "value"].mean())
            passed.append(value >= cut)
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "method": method,
                    "family": family,
                    "claim_context": "cross_context_embedding_support",
                    "metric": metric,
                    "value": value,
                    "threshold": cut,
                    "support": "pass" if value >= cut else "below_threshold",
                    "support_definition": "diagnostic threshold support, not a universal method ranking",
                }
            )
        rows.append(
            {
                "dataset_id": dataset_id,
                "method": method,
                "family": family,
                "claim_context": "cross_context_embedding_support",
                "metric": "all_four_metrics",
                "value": int(sum(passed)),
                "threshold": 4,
                "support": "strong" if all(passed) else ("partial" if any(passed) else "unsupported"),
                "support_definition": "number of predeclared Fig3 diagnostic thresholds passed",
            }
        )
    return pd.DataFrame(rows)


def _write_dataset_coordinates(dataset_id: str, proc, labels, batch_field: str, embeddings: dict[str, np.ndarray], seed: int) -> Path:
    rows = []
    batch_values = proc.obs[batch_field].astype(str).to_numpy() if batch_field and batch_field in proc.obs else np.asarray([""] * proc.n_obs)
    cell_ids = proc.obs_names.astype(str).to_numpy()
    for method, coords in embeddings.items():
        rows.append(
            pd.DataFrame(
                {
                    "dataset_id": dataset_id,
                    "cell_id": cell_ids,
                    "method": method,
                    "family": config.METHOD_FAMILY[method],
                    "seed": seed,
                    "x": coords[:, 0],
                    "y": coords[:, 1],
                    "label": np.asarray(labels).astype(str),
                    "batch_or_donor": batch_values,
                    "n_cells": int(proc.n_obs),
                    "source": "empirical_embedding_coordinates",
                }
            )
        )
    out = SOURCE_DIR / f"fig3_{'heart' if dataset_id == 'heart_cell_atlas_subsampled' else dataset_id}_embedding_coordinates.csv"
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    return out


def _load_dataset(record) -> tuple[object, object]:
    proc = sc.read_h5ad(ROOT / record.analysis_proc_path)
    counts = sc.read_h5ad(ROOT / record.analysis_counts_path)
    return proc, counts


def run(methods: list[str], datasets: list[str], seed: int, force: bool = False) -> None:
    manifest = pd.read_csv(MANIFEST)
    all_metric_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    coordinate_files: list[Path] = []

    for record in manifest.itertuples(index=False):
        if datasets and record.dataset_id not in datasets:
            continue
        proc, counts = _load_dataset(record)
        label_field = str(record.label_field)
        batch_field = "" if pd.isna(record.batch_field) else str(record.batch_field)
        labels = proc.obs[label_field].astype(str).to_numpy()
        embeddings: dict[str, np.ndarray] = {}
        for method in methods:
            started = time.time()
            try:
                coords = compute_embedding(record.dataset_id, method, proc, counts, label_field, seed, force=force)
                embeddings[method] = coords
                all_metric_rows.extend(_metric_rows(record.dataset_id, method, proc, labels, coords, seed))
                print(f"{record.dataset_id} {method} OK {coords.shape} {time.time() - started:.1f}s")
            except Exception as exc:
                failures.append(
                    {
                        "dataset_id": record.dataset_id,
                        "method": method,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                print(f"{record.dataset_id} {method} FAILED {type(exc).__name__}: {exc}")

        if embeddings:
            coordinate_files.append(_write_dataset_coordinates(record.dataset_id, proc, labels, batch_field, embeddings, seed))

    metrics = pd.DataFrame(all_metric_rows)
    metrics_out = SOURCE_DIR / "fig3_family_local_label_metrics.csv"
    metrics.to_csv(metrics_out, index=False)

    support_out = SOURCE_DIR / "fig3_cross_context_support_matrix.csv"
    _support_rows(metrics).to_csv(support_out, index=False)

    failures_out = LOG_DIR / "anchor_embedding_failures.json"
    failures_out.write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")

    versions_out = LOG_DIR / "anchor_embedding_run_metadata.json"
    versions_out.write_text(
        json.dumps(
            {
                "seed": seed,
                "methods": methods,
                "datasets": datasets or "all",
                "python": sys.version,
                "platform": platform.platform(),
                "scanpy": sc.__version__,
                "numpy": np.__version__,
                "config": {
                    "n_neighbors": config.N_NEIGHBORS,
                    "n_pcs": config.N_PCS,
                    "tsne_perplexity": config.TSNE_PERPLEXITY,
                    "glmpca_max_iter": config.GLMPCA_MAX_ITER,
                    "rank_pairs": config.RANK_PAIRS,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    hash_lines = []
    for path in [*coordinate_files, metrics_out, support_out, failures_out, versions_out]:
        hash_lines.append(f"{_sha256(path)}  {path.relative_to(ROOT)}")
    (SOURCE_DIR / "fig3_embedding_source_data.sha256.txt").write_text("\n".join(hash_lines) + "\n", encoding="utf-8")

    print(f"wrote {metrics_out}")
    print(f"wrote {support_out}")
    print(f"failures: {len(failures)} recorded in {failures_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=METHODS)
    parser.add_argument("--datasets", nargs="*", default=[])
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    unknown = [m for m in args.methods if m not in METHODS]
    if unknown:
        raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
    ensure_analysis_dirs()
    _ensure_dirs()
    run(args.methods, args.datasets, args.seed, force=args.force)


if __name__ == "__main__":
    main()
