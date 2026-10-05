"""Compare newly recomputed calibration with publication reference results."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.paths import PUBLIC_SOURCE_DATA_DIR, RESULTS_DIR

NAMES = ["calibration_geometry.csv", "calibration_continuum.csv", "calibration_donor.csv", "scvi_geometry_full.csv", "scvi_reference_sensitivity.csv", "scvi_seed_stability.csv", "calibration_evaluation_cells.csv", "permutation_draw_index.csv"]


def verify(output):
    for name in NAMES:
        expected = pd.read_csv(PUBLIC_SOURCE_DATA_DIR / "calibration" / name)
        observed = pd.read_csv(output / name)
        if list(expected.columns) != list(observed.columns) or expected.shape != observed.shape:
            raise AssertionError(f"Schema mismatch: {name}")
        for column in expected.columns:
            if pd.api.types.is_numeric_dtype(expected[column]):
                if not np.allclose(expected[column], observed[column], rtol=5e-8, atol=5e-9, equal_nan=True):
                    raise AssertionError(f"Numeric mismatch: {name} {column}")
            elif not expected[column].fillna("").equals(observed[column].fillna("")):
                raise AssertionError(f"Record mismatch: {name} {column}")
        print(f"PASS {name}: {len(observed)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "calibration")
    verify(parser.parse_args().output)
