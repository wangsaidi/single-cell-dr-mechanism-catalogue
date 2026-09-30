"""Verify scVI outputs and normalise the recorded epoch count.

Lightning reports ``current_epoch`` as the next epoch after a max-epoch stop in
some versions. The first revision run therefore recorded 201 for a fixed limit
of 200. This audit preserves the raw value and caps only the reported number of
completed epochs at the configured maximum.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import LOGS_DIR as ANALYSIS_LOGS_DIR, RESULTS_DIR

OUT_DIR = RESULTS_DIR / "embeddings"
LOG_DIR = ANALYSIS_LOGS_DIR
DATASETS = {
    "pbmc3k": 2638,
    "paul15": 2730,
    "heart_cell_atlas_subsampled": 8000,
}
DIMENSIONS = [2, 10]
SEEDS = [0, 1, 2, 3, 4]
MAX_EPOCHS = 200


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    failures = []
    metadata_rows = []
    for dataset_id, n_cells in DATASETS.items():
        for dimension in DIMENSIONS:
            for seed in SEEDS:
                stem = f"{dataset_id}_scvi_dim{dimension}_seed{seed}"
                array_path = OUT_DIR / f"{stem}.npy"
                metadata_path = OUT_DIR / f"{stem}.json"
                history_path = OUT_DIR / f"{stem}_history.csv"
                for path in [array_path, metadata_path, history_path]:
                    if not path.exists():
                        failures.append(f"missing {path}")
                if not array_path.exists() or not metadata_path.exists():
                    continue
                array = np.load(array_path)
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                expected_shape = (n_cells, dimension)
                if array.shape != expected_shape:
                    failures.append(f"shape {array_path}: {array.shape} != {expected_shape}")
                if not np.isfinite(array).all():
                    failures.append(f"non-finite values {array_path}")
                if np.any(np.var(array, axis=0) <= 0):
                    failures.append(f"zero-variance axis {array_path}")

                raw_epochs = int(metadata.get("epochs_trained", 0))
                if raw_epochs > MAX_EPOCHS:
                    metadata["epochs_recorded_raw"] = raw_epochs
                    metadata["epochs_trained"] = MAX_EPOCHS
                    metadata["epochs_normalization_note"] = (
                        "Capped at the configured maximum because this Lightning version reports "
                        "current_epoch as the next epoch after a max-epoch stop."
                    )
                    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "latent_dimension": dimension,
                        "seed": seed,
                        "n_cells": int(array.shape[0]),
                        "n_dimensions": int(array.shape[1]),
                        "finite_fraction": float(np.isfinite(array).mean()),
                        "minimum_axis_variance": float(np.var(array, axis=0).min()),
                        "epochs_trained": int(metadata.get("epochs_trained", 0)),
                        "embedding_sha256": sha256_file(array_path),
                        "metadata_sha256": sha256_file(metadata_path),
                    }
                )
                metadata_rows.append(metadata)

    audit = pd.DataFrame(rows)
    audit.to_csv(LOG_DIR / "scvi_output_audit.csv", index=False)
    pd.json_normalize(metadata_rows, sep=".").to_csv(LOG_DIR / "scvi_run_manifest.csv", index=False)
    summary = {
        "expected_outputs": len(DATASETS) * len(DIMENSIONS) * len(SEEDS),
        "audited_outputs": int(audit.shape[0]),
        "failures": failures,
        "audit_passed": not failures and audit.shape[0] == len(DATASETS) * len(DIMENSIONS) * len(SEEDS),
    }
    (LOG_DIR / "scvi_output_audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not summary["audit_passed"]:
        raise SystemExit(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
