"""Verify fixed inputs and compare recomputed tables with archived references."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paths import (
    ANALYSIS_ROOT,
    LOGS_DIR,
    REFERENCE_RESULTS_DIR,
    RESULTS_DIR,
    ensure_analysis_dirs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> list[str]:
    manifest = ANALYSIS_ROOT / "ARCHIVE_SHA256SUMS.txt"
    if not manifest.exists():
        return [f"missing checksum manifest: {manifest}"]
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        path = ANALYSIS_ROOT / relative
        if not path.exists():
            errors.append(f"missing input: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")
    return errors


def compare_frames(observed_path: Path, reference_path: Path) -> str | None:
    observed = pd.read_csv(observed_path)
    reference = pd.read_csv(reference_path)
    if list(observed.columns) != list(reference.columns):
        return "column mismatch"
    if observed.shape != reference.shape:
        return f"shape mismatch {observed.shape} != {reference.shape}"
    for column in observed.columns:
        left = observed[column]
        right = reference[column]
        left_num = pd.to_numeric(left, errors="coerce")
        right_num = pd.to_numeric(right, errors="coerce")
        numeric = left.notna().sum() == left_num.notna().sum() and right.notna().sum() == right_num.notna().sum()
        if numeric:
            if not np.allclose(left_num.to_numpy(float), right_num.to_numpy(float), rtol=1e-7, atol=1e-9, equal_nan=True):
                return f"numeric mismatch in {column}"
        else:
            if not left.fillna("<NA>").astype(str).equals(right.fillna("<NA>").astype(str)):
                return f"text mismatch in {column}"
    return None


def verify_results() -> list[str]:
    errors = []
    reference_files = sorted([*REFERENCE_RESULTS_DIR.rglob("*.csv"),
                              *REFERENCE_RESULTS_DIR.rglob("*.csv.gz")])
    for reference in reference_files:
        relative = reference.relative_to(REFERENCE_RESULTS_DIR)
        observed = RESULTS_DIR / relative
        if not observed.exists():
            errors.append(f"missing result: {relative.as_posix()}")
            continue
        message = compare_frames(observed, reference)
        if message:
            errors.append(f"{relative.as_posix()}: {message}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    ensure_analysis_dirs()
    errors = verify_inputs()
    if not args.inputs_only:
        errors.extend(verify_results())
    report = {
        "inputs_checked": True,
        "results_checked": not args.inputs_only,
        "reference_tables_compared": 0 if args.inputs_only else len([
            *REFERENCE_RESULTS_DIR.rglob("*.csv"), *REFERENCE_RESULTS_DIR.rglob("*.csv.gz")]),
        "errors": errors,
        "passed": not errors,
    }
    out = LOGS_DIR / "reproducibility_verification.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
