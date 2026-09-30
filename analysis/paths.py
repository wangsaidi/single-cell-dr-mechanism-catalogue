"""Portable paths shared by the public analysis workflow."""

from __future__ import annotations

import os
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_ROOT.parent
DATA_DIR = Path(os.environ.get("SCDR_ANALYSIS_DATA", ANALYSIS_ROOT / "data")).resolve()
RESULTS_DIR = Path(os.environ.get("SCDR_ANALYSIS_RESULTS", ANALYSIS_ROOT / "results")).resolve()
LOGS_DIR = Path(os.environ.get("SCDR_ANALYSIS_LOGS", ANALYSIS_ROOT / "logs")).resolve()

ANALYSIS_OBJECTS_DIR = DATA_DIR / "analysis_objects"
ANCHOR_EMBEDDINGS_DIR = DATA_DIR / "anchor_embeddings"
ROBUSTNESS_EMBEDDINGS_DIR = DATA_DIR / "robustness_embeddings"
EXPRESSION_OBJECTS_DIR = DATA_DIR / "expression_objects"
LEGACY_CACHE_DIR = DATA_DIR / "legacy_method_cache"
SIMULATION_CACHE_DIR = DATA_DIR / "simulation_cache"
RAW_CACHE_DIR = DATA_DIR / "raw_cache"
REFERENCE_RESULTS_DIR = DATA_DIR / "reference_results"

MANIFEST_PATH = DATA_DIR / "analysis_object_manifest.csv"
SOURCE_RESULTS_DIR = RESULTS_DIR / "source_data"
PUBLIC_SOURCE_DATA_DIR = REPO_ROOT / "data" / "source_data"


def ensure_analysis_dirs() -> None:
    for path in (
        DATA_DIR,
        RESULTS_DIR,
        LOGS_DIR,
        ANALYSIS_OBJECTS_DIR,
        ANCHOR_EMBEDDINGS_DIR,
        ROBUSTNESS_EMBEDDINGS_DIR,
        EXPRESSION_OBJECTS_DIR,
        LEGACY_CACHE_DIR,
        SIMULATION_CACHE_DIR,
        RAW_CACHE_DIR,
        SOURCE_RESULTS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
