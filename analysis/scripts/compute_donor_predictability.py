"""Separate cell-type and donor information retained by heart representations."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis.paths import ANALYSIS_OBJECTS_DIR, ANCHOR_EMBEDDINGS_DIR, RESULTS_DIR

PROC_PATH = ANALYSIS_OBJECTS_DIR / "heart_cell_atlas_subsampled_proc.h5ad"
EXISTING_EMBEDDINGS = ANCHOR_EMBEDDINGS_DIR
SCVI_EMBEDDINGS = RESULTS_DIR / "embeddings"
OUT_DIR = RESULTS_DIR / "donor_predictability"

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
SEEDS = [0, 1, 2, 3, 4]
CV_SEED = 0
BOOTSTRAP_REPLICATES = 10000


def stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def embedding_path(method: str, seed: int) -> Path:
    if method == "scVI":
        return SCVI_EMBEDDINGS / f"heart_cell_atlas_subsampled_scvi_dim2_seed{seed}.npy"
    return EXISTING_EMBEDDINGS / f"heart_cell_atlas_subsampled_{stem(method)}_seed{seed}.npy"


def classifier() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=CV_SEED,
                ),
            ),
        ]
    )


def chance_normalised(score: float, n_classes: int) -> float:
    chance = 1.0 / n_classes
    return float((score - chance) / (1.0 - chance))


def class_balanced_scores(observed: np.ndarray, predicted: np.ndarray) -> tuple[float, float, int]:
    class_order = sorted(np.unique(observed).astype(str))
    _, recall, per_class_f1, _ = precision_recall_fscore_support(
        observed,
        predicted,
        labels=class_order,
        zero_division=np.nan,
    )
    return float(np.nanmean(recall)), float(np.nanmean(per_class_f1)), len(class_order)


def donor_bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(values.mean()), float(low), float(high)


def evaluate_cv(
    coords: np.ndarray,
    target: np.ndarray,
    splits,
    endpoint: str,
    method: str,
    seed: int,
) -> tuple[list[dict[str, object]], np.ndarray]:
    rows = []
    prediction = np.asarray([""] * target.size, dtype=object)
    class_order = sorted(np.unique(target).astype(str))
    for fold, (train, test) in enumerate(splits):
        model = classifier()
        model.fit(coords[train], target[train])
        predicted = model.predict(coords[test]).astype(str)
        prediction[test] = predicted
        _, recall, per_class_f1, support = precision_recall_fscore_support(
            target[test], predicted, labels=class_order, zero_division=np.nan
        )
        balanced = float(np.nanmean(recall))
        macro_f1 = float(np.nanmean(per_class_f1))
        n_classes = int(np.unique(target[test]).size)
        rows.append(
            {
                "dataset_id": "heart_cell_atlas_subsampled",
                "method": method,
                "embedding_seed": seed,
                "endpoint": endpoint,
                "fold": fold,
                "n_train": int(train.size),
                "n_test": int(test.size),
                "n_test_classes": n_classes,
                "balanced_accuracy": balanced,
                "balanced_accuracy_above_chance": chance_normalised(balanced, n_classes),
                "macro_f1": macro_f1,
            }
        )
    return rows, prediction


def run(methods: list[str], seeds: list[int]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(PROC_PATH, backed="r")
    cell_ids = adata.obs_names.astype(str).to_numpy()
    cell_type = adata.obs["cell_type"].astype(str).to_numpy()
    donor = adata.obs["donor"].astype(str).to_numpy()
    adata.file.close()

    cell_type_cv = list(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(
            np.zeros(cell_type.size), cell_type, groups=donor
        )
    )
    donor_cv = list(
        StratifiedKFold(n_splits=5, shuffle=True, random_state=CV_SEED).split(
            np.zeros(donor.size), donor
        )
    )

    fold_rows = []
    prediction_rows = []
    per_donor_rows = []
    for method in methods:
        for seed in seeds:
            path = embedding_path(method, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            coords = np.load(path).astype(np.float32)
            if coords.shape != (cell_ids.size, 2):
                raise ValueError(f"Unexpected heart embedding shape for {method} seed {seed}: {coords.shape}")

            type_rows, type_prediction = evaluate_cv(
                coords, cell_type, cell_type_cv, "cell_type_grouped_by_donor", method, seed
            )
            donor_fold_rows, donor_prediction = evaluate_cv(
                coords, donor, donor_cv, "donor_stratified_cell_split", method, seed
            )
            fold_rows.extend(type_rows)
            fold_rows.extend(donor_fold_rows)
            for donor_id in sorted(np.unique(donor)):
                mask = donor == donor_id
                balanced, macro_f1, n_classes = class_balanced_scores(
                    cell_type[mask], type_prediction[mask]
                )
                per_donor_rows.append(
                    {
                        "dataset_id": "heart_cell_atlas_subsampled",
                        "method": method,
                        "embedding_seed": seed,
                        "donor": donor_id,
                        "n_cells": int(mask.sum()),
                        "n_cell_types_present": n_classes,
                        "cell_type_balanced_accuracy": balanced,
                        "cell_type_macro_f1": macro_f1,
                        "statistical_unit": "donor",
                    }
                )
            if seed == 0:
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "cell_id": cell_ids,
                            "method": method,
                            "embedding_seed": seed,
                            "cell_type": cell_type,
                            "cell_type_prediction_leave_donors_out": type_prediction,
                            "donor": donor,
                            "donor_prediction_stratified_cell_split": donor_prediction,
                        }
                    )
                )
            print(f"completed heart predictability {method} seed={seed}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    donor_df = pd.DataFrame(per_donor_rows)
    summary = (
        fold_df.groupby(["dataset_id", "method", "embedding_seed", "endpoint"], as_index=False)
        .agg(
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            balanced_accuracy_min=("balanced_accuracy", "min"),
            balanced_accuracy_max=("balanced_accuracy", "max"),
            balanced_accuracy_above_chance_mean=("balanced_accuracy_above_chance", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            n_folds=("fold", "size"),
        )
    )
    fold_df.to_csv(OUT_DIR / "heart_predictability_folds.csv", index=False)
    summary.to_csv(OUT_DIR / "heart_predictability_summary.csv", index=False)
    donor_df.to_csv(OUT_DIR / "heart_cell_type_predictability_by_donor.csv", index=False)
    bootstrap_rows = []
    for (method, seed), values in donor_df.groupby(["method", "embedding_seed"], sort=False):
        for metric in ["cell_type_balanced_accuracy", "cell_type_macro_f1"]:
            mean, low, high = donor_bootstrap(values[metric].to_numpy(dtype=float), CV_SEED + int(seed))
            bootstrap_rows.append(
                {
                    "dataset_id": "heart_cell_atlas_subsampled",
                    "method": method,
                    "embedding_seed": seed,
                    "metric": metric,
                    "mean_across_donors": mean,
                    "bootstrap_95_ci_low": low,
                    "bootstrap_95_ci_high": high,
                    "n_donors": int(values.shape[0]),
                    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                    "statistical_unit": "donor",
                }
            )
    pd.DataFrame(bootstrap_rows).to_csv(OUT_DIR / "heart_donor_bootstrap_summary.csv", index=False)
    pd.concat(prediction_rows, ignore_index=True).to_csv(
        OUT_DIR / "heart_predictions_seed0.csv.gz", index=False, compression="gzip"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    args = parser.parse_args()
    run(args.methods, args.seeds)


if __name__ == "__main__":
    main()
