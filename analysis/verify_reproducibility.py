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
    DATA_DIR,
    SCVI_EMBEDDINGS_DIR,
    PUBLIC_SOURCE_DATA_DIR,
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


def verify_inputs(data_only: bool = False) -> list[str]:
    manifest = ANALYSIS_ROOT / "ARCHIVE_SHA256SUMS.txt"
    if not manifest.exists():
        return [f"missing checksum manifest: {manifest}"]
    errors = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        relative_path = Path(relative)
        if relative_path.parts[0] == "data":
            path = DATA_DIR.joinpath(*relative_path.parts[1:])
        elif relative_path.parts[:2] == ("results", "embeddings"):
            if data_only:
                continue
            path = SCVI_EMBEDDINGS_DIR.joinpath(*relative_path.parts[2:])
        else:
            errors.append(f"unexpected archive manifest entry: {relative}")
            continue
        if not path.exists():
            errors.append(f"missing input: {relative}")
        elif sha256(path) != expected:
            errors.append(f"checksum mismatch: {relative}")
    table_manifest = ANALYSIS_ROOT / "ARCHIVED_TABLE_SHA256SUMS.txt"
    if not table_manifest.is_file():
        errors.append(f"missing table checksum manifest: {table_manifest}")
    else:
        for line in table_manifest.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, relative = line.split("  ", 1)
            path = PUBLIC_SOURCE_DATA_DIR / relative
            # Publication CSVs are Git text: only CRLF versus LF is canonicalised.
            observed = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() if path.is_file() else None
            if observed != expected:
                errors.append(f"missing or changed archived publication table: {relative}")
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
    if not reference_files:
        return [f"no archived reference tables found: {REFERENCE_RESULTS_DIR}"]
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
    parser.add_argument("--data-only", action="store_true", help="Verify archive data/tables, excluding newly refitted scVI arrays.")
    args = parser.parse_args()
    ensure_analysis_dirs()
    errors = verify_inputs(data_only=args.data_only)
    if not args.inputs_only:
        errors.extend(verify_results())
    report = {
        "inputs_checked": True,
        "refitted_scvi_excluded_from_archive_hash_check": args.data_only,
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
