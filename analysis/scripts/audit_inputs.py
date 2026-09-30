"""Audit empirical analysis objects and representation files before revision analyses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import (
    ANCHOR_EMBEDDINGS_DIR,
    LOGS_DIR,
    MANIFEST_PATH,
    REPO_ROOT,
    RESULTS_DIR,
    ROBUSTNESS_EMBEDDINGS_DIR,
)

ROOT = REPO_ROOT
MANIFEST = MANIFEST_PATH
EXISTING = ANCHOR_EMBEDDINGS_DIR
ROBUST = ROBUSTNESS_EMBEDDINGS_DIR
SCVI = RESULTS_DIR / "embeddings"
LOGS = LOGS_DIR

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
SEEDS = [0, 1, 2, 3, 4]
DIMENSION_METHODS = ["PCA", "UMAP", "PHATE", "PaCMAP"]
DIMENSIONS = [2, 5, 10, 20]


def stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run() -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST)
    object_rows = []
    embedding_rows = []
    dimension_rows = []
    errors = []

    for record in manifest.itertuples(index=False):
        for role, relative, expected in [
            ("processed", record.analysis_proc_path, record.proc_sha256),
            ("counts", record.analysis_counts_path, record.counts_sha256),
        ]:
            path = ROOT / str(relative)
            observed = sha256_file(path) if path.exists() else ""
            valid = path.exists() and observed == str(expected)
            object_rows.append(
                {
                    "dataset_id": record.dataset_id,
                    "role": role,
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                    "expected_sha256": expected,
                    "observed_sha256": observed,
                    "checksum_valid": valid,
                }
            )
            if not valid:
                errors.append(f"Invalid analysis object: {record.dataset_id} {role}")

        for method in METHODS:
            for seed in SEEDS:
                path = EXISTING / f"{record.dataset_id}_{stem(method)}_seed{seed}.npy"
                row = {
                    "dataset_id": record.dataset_id,
                    "method": method,
                    "seed": seed,
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                }
                if path.exists():
                    values = np.load(path)
                    row.update(
                        {
                            "shape": "x".join(map(str, values.shape)),
                            "shape_valid": values.shape == (int(record.n_obs), 2),
                            "finite_fraction": float(np.isfinite(values).mean()),
                            "axis_variance_min": float(np.var(values, axis=0).min()),
                            "sha256": sha256_file(path),
                        }
                    )
                    if not row["shape_valid"] or row["finite_fraction"] != 1.0 or row["axis_variance_min"] <= 0:
                        errors.append(f"Invalid embedding array: {record.dataset_id} {method} seed {seed}")
                else:
                    errors.append(f"Missing embedding: {record.dataset_id} {method} seed {seed}")
                embedding_rows.append(row)

        for method in DIMENSION_METHODS:
            for dimension in DIMENSIONS:
                path = ROBUST / f"dim_{record.dataset_id}_{method}_{dimension}.npy"
                row = {
                    "dataset_id": record.dataset_id,
                    "method": method,
                    "output_dimension": dimension,
                    "seed": 0,
                    "path": str(path.relative_to(ROOT)),
                    "exists": path.exists(),
                }
                if path.exists():
                    values = np.load(path)
                    row.update(
                        {
                            "shape": "x".join(map(str, values.shape)),
                            "shape_valid": values.shape == (int(record.n_obs), dimension),
                            "finite_fraction": float(np.isfinite(values).mean()),
                            "axis_variance_min": float(np.var(values, axis=0).min()),
                            "sha256": sha256_file(path),
                        }
                    )
                    if not row["shape_valid"] or row["finite_fraction"] != 1.0 or row["axis_variance_min"] <= 0:
                        errors.append(
                            f"Invalid dimensionality array: {record.dataset_id} {method} dimension {dimension}"
                        )
                else:
                    errors.append(f"Missing dimensionality array: {record.dataset_id} {method} dimension {dimension}")
                dimension_rows.append(row)

    pd.DataFrame(object_rows).to_csv(LOGS / "analysis_object_audit.csv", index=False)
    pd.DataFrame(embedding_rows).to_csv(LOGS / "existing_embedding_audit.csv", index=False)
    pd.DataFrame(dimension_rows).to_csv(LOGS / "existing_dimension_embedding_audit.csv", index=False)
    summary = {
        "analysis_objects": len(object_rows),
        "existing_embedding_arrays_expected": len(manifest) * len(METHODS) * len(SEEDS),
        "existing_embedding_arrays_found": int(sum(row["exists"] for row in embedding_rows)),
        "existing_dimension_arrays_expected": len(manifest) * len(DIMENSION_METHODS) * len(DIMENSIONS),
        "existing_dimension_arrays_found": int(sum(row["exists"] for row in dimension_rows)),
        "errors": errors,
        "audit_passed": not errors,
    }
    (LOGS / "input_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
