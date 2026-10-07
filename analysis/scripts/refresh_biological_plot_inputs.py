"""Refresh merged biological plot inputs from the canonical non-self diagnostics."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/source_data"


def refresh_heart_donor_table(source: Path = SOURCE) -> pd.DataFrame:
    path = source / "fig5_heart_identity_donor_support.csv"
    merged = pd.read_csv(path)
    components = pd.read_csv(source / "fig4_heart_donor_gate_metrics.csv")
    wide = components.pivot(index=["dataset_id", "method", "seed"],
                            columns="metric", values="value")
    keys = ["dataset_id", "method", "seed"]
    indexed = merged.set_index(keys)
    if not indexed.index.is_unique or not wide.index.is_unique:
        raise ValueError("Donor sources do not have unique method-context-seed keys.")
    for column in ("cell_type_label_recall", "donor_entropy_norm"):
        values = wide[column].reindex(indexed.index)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing canonical donor component: {column}")
        indexed[column] = values
    updated = indexed.reset_index().reindex(columns=merged.columns)
    old_pass = (merged.cell_type_label_recall >= .55) & (merged.donor_entropy_norm >= .50)
    new_pass = (updated.cell_type_label_recall >= .55) & (updated.donor_entropy_norm >= .50)
    if not np.array_equal(old_pass, new_pass):
        raise ValueError("Donor support decisions changed; review dependent conclusions.")
    updated.to_csv(path, index=False)
    return updated


if __name__ == "__main__":
    result = refresh_heart_donor_table()
    print(f"Refreshed {len(result)} canonical donor-aware method rows.")
