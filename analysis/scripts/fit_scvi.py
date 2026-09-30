"""Fit direct scVI representations for the expanded downstream analysis.

The primary representation is a two-dimensional scVI latent space, not a
post-hoc UMAP projection. A ten-dimensional latent space is retained as a
secondary dimensionality-sensitivity analysis. No labels or donor identifiers
are supplied to the model.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from scipy import sparse
from scvi.model import SCVI

from analysis.paths import LOGS_DIR as ANALYSIS_LOGS_DIR, MANIFEST_PATH, REPO_ROOT, RESULTS_DIR

ROOT = REPO_ROOT
MANIFEST = MANIFEST_PATH
OUT_DIR = RESULTS_DIR / "embeddings"
LOG_DIR = ANALYSIS_LOGS_DIR

DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
LATENT_DIMS = [2, 10]
SEEDS = [0, 1, 2, 3, 4]

MODEL_CONFIG = {
    "n_layers": 1,
    "n_hidden": 128,
    "dropout_rate": 0.1,
    "dispersion": "gene",
    "gene_likelihood": "nb",
    "latent_distribution": "normal",
}

TRAIN_CONFIG = {
    "max_epochs": 200,
    "train_size": 0.9,
    "validation_size": 0.1,
    "batch_size": 128,
    "early_stopping": True,
    "early_stopping_patience": 20,
    "accelerator": "cpu",
    "devices": 1,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def counts_are_integer(adata: ad.AnnData) -> float:
    values = adata.X.data if sparse.issparse(adata.X) else np.asarray(adata.X).ravel()
    if values.size == 0:
        return 1.0
    return float(np.mean(np.isclose(values, np.round(values))))


def output_path(dataset_id: str, latent_dim: int, seed: int) -> Path:
    return OUT_DIR / f"{dataset_id}_scvi_dim{latent_dim}_seed{seed}.npy"


def history_path(dataset_id: str, latent_dim: int, seed: int) -> Path:
    return OUT_DIR / f"{dataset_id}_scvi_dim{latent_dim}_seed{seed}_history.csv"


def metadata_path(dataset_id: str, latent_dim: int, seed: int) -> Path:
    return OUT_DIR / f"{dataset_id}_scvi_dim{latent_dim}_seed{seed}.json"


def serialise_history(model: SCVI, path: Path) -> int:
    history = model.history
    frames: list[pd.DataFrame] = []
    for metric, values in history.items():
        frame = values.copy()
        if isinstance(frame, pd.Series):
            frame = frame.to_frame(name="value")
        frame = frame.reset_index().rename(columns={frame.index.name or "index": "epoch"})
        value_columns = [column for column in frame.columns if column != "epoch"]
        if len(value_columns) == 1:
            frame = frame.rename(columns={value_columns[0]: "value"})
        frame.insert(0, "metric", metric)
        frames.append(frame[["metric", "epoch", "value"]])
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["metric", "epoch", "value"])
    combined.to_csv(path, index=False)
    trainer_epoch = int(getattr(getattr(model, "trainer", None), "current_epoch", 0))
    trainer_epoch = min(TRAIN_CONFIG["max_epochs"], max(0, trainer_epoch))
    if combined.empty:
        return max(0, trainer_epoch)
    history_epoch = int(pd.to_numeric(combined["epoch"], errors="coerce").max()) + 1
    return min(TRAIN_CONFIG["max_epochs"], max(history_epoch, trainer_epoch))


def fit_one(
    dataset_id: str,
    counts_path: Path,
    input_sha256: str,
    latent_dim: int,
    seed: int,
    force: bool,
) -> dict[str, object]:
    out = output_path(dataset_id, latent_dim, seed)
    history_out = history_path(dataset_id, latent_dim, seed)
    metadata_out = metadata_path(dataset_id, latent_dim, seed)
    if out.exists() and history_out.exists() and metadata_out.exists() and not force:
        metadata = json.loads(metadata_out.read_text(encoding="utf-8"))
        metadata["status"] = "cached"
        return metadata

    adata = sc.read_h5ad(counts_path)
    integer_fraction = counts_are_integer(adata)
    if integer_fraction < 0.999999:
        raise ValueError(f"{dataset_id} count input is not integer-valued: {integer_fraction:.6f}")

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    scvi.settings.seed = seed
    SCVI.setup_anndata(adata)

    model = SCVI(adata, n_latent=latent_dim, **MODEL_CONFIG)
    started = time.time()
    model.train(
        max_epochs=TRAIN_CONFIG["max_epochs"],
        train_size=TRAIN_CONFIG["train_size"],
        validation_size=TRAIN_CONFIG["validation_size"],
        batch_size=TRAIN_CONFIG["batch_size"],
        early_stopping=TRAIN_CONFIG["early_stopping"],
        early_stopping_patience=TRAIN_CONFIG["early_stopping_patience"],
        accelerator=TRAIN_CONFIG["accelerator"],
        devices=TRAIN_CONFIG["devices"],
        enable_progress_bar=False,
        logger=False,
    )
    elapsed = time.time() - started
    latent = np.asarray(model.get_latent_representation(), dtype=np.float32)
    if latent.shape != (adata.n_obs, latent_dim):
        raise ValueError(f"Unexpected latent shape for {dataset_id}: {latent.shape}")

    np.save(out, latent)
    epochs_trained = serialise_history(model, history_out)
    metadata = {
        "status": "completed",
        "dataset_id": dataset_id,
        "input_path": str(counts_path.relative_to(ROOT)),
        "input_sha256": input_sha256,
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "integer_count_fraction": integer_fraction,
        "method": "scVI",
        "labels_used_in_fit": False,
        "batch_key": None,
        "latent_dim": latent_dim,
        "seed": seed,
        "model_config": MODEL_CONFIG,
        "train_config": TRAIN_CONFIG,
        "epochs_trained": epochs_trained,
        "elapsed_seconds": elapsed,
        "latent_sha256": sha256_array(latent),
        "latent_finite_fraction": float(np.isfinite(latent).mean()),
        "latent_axis_variance": np.var(latent, axis=0).astype(float).tolist(),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "scanpy": importlib.metadata.version("scanpy"),
            "anndata": importlib.metadata.version("anndata"),
            "scvi_tools": importlib.metadata.version("scvi-tools"),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    metadata_out.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run(datasets: list[str], latent_dims: list[int], seeds: list[int], force: bool) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST).set_index("dataset_id")
    rows: list[dict[str, object]] = []
    for dataset_id in datasets:
        if dataset_id not in manifest.index:
            raise KeyError(f"Dataset is absent from the analysis manifest: {dataset_id}")
        record = manifest.loc[dataset_id]
        counts_path = ROOT / str(record["analysis_counts_path"])
        input_sha256 = sha256_file(counts_path)
        expected_sha256 = str(record["counts_sha256"])
        if input_sha256 != expected_sha256:
            raise ValueError(f"Input checksum mismatch for {dataset_id}")
        for latent_dim in latent_dims:
            for seed in seeds:
                print(f"Fitting {dataset_id} scVI dim={latent_dim} seed={seed}", flush=True)
                metadata = fit_one(dataset_id, counts_path, input_sha256, latent_dim, seed, force)
                rows.append(metadata)
                print(
                    f"  {metadata['status']} epochs={metadata.get('epochs_trained')} "
                    f"elapsed={float(metadata.get('elapsed_seconds', 0.0)):.1f}s",
                    flush=True,
                )
    summary = pd.json_normalize(rows, sep=".")
    summary.to_csv(LOG_DIR / "scvi_run_manifest.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--latent-dims", nargs="+", type=int, choices=LATENT_DIMS, default=LATENT_DIMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run(args.datasets, args.latent_dims, args.seeds, args.force)


if __name__ == "__main__":
    main()
