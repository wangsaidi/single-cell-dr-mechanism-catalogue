"""Global configuration for the manuscript figure pipeline."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np

SEED = 0
STABILITY_SEEDS = list(range(10))
PERTURB_REPLICATES = list(range(5))

N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 15
TSNE_PERPLEXITY = 30
DIFFMAP_NCOMPS = 15
SCVI_NLATENT = 10
SCVI_MAX_EPOCHS = 200
VAE_NLATENT = 10
VAE_MAX_EPOCHS = 120
GLMPCA_MAX_ITER = 120
KNN_GRID = [15, 30, 50]
RANK_PAIRS = 5000
MDS_MAX_ITER = 120

DROPOUT_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
BATCH_BETA = [0.0, 0.5, 1.0, 1.5, 2.0]
NOISE_LEVEL = [0.0, 0.25, 0.5, 0.75, 1.0]
KSWEEP_GRID = [5, 10, 15, 30, 50, 100]

MM = 1 / 25.4
COL1_W = 89 * MM
COL2_W = 183 * MM
DPI = 300
FORMATS = ["pdf", "svg", "png"]

PATHS = {
    "cache": "cache",
    "legacy_deep": "cache/legacy_deep_embeddings",
    "source": "source_data",
    "out": "output",
}

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / PATHS["cache"]
LEGACY_DEEP_DIR = ROOT / PATHS["legacy_deep"]
SOURCE_DIR = ROOT / PATHS["source"]
OUTPUT_DIR = ROOT / PATHS["out"]

METHOD_FAMILY = {
    "PCA": "factor",
    "GLM-PCA": "factor",
    "ZIFA": "factor",
    "scVI": "deep",
    "VAE": "deep",
    "scScope": "deep",
    "SAUCIE": "deep",
    "UMAP": "graph",
    "PHATE": "graph",
    "Diffusion": "graph",
    "t-SNE": "relational",
    "TriMap": "relational",
    "PaCMAP": "relational",
    "IVIS": "relational",
}

AVAILABLE_26_METHOD_PACKAGES = {
    "PCA": "sklearn.decomposition.PCA",
    "GLM-PCA": "glmpca",
    "ZIFA": "ZIFA",
    "VAE": "torch/scvi-tools backend",
    "scScope": "conda env scdr-deep-py37",
    "SAUCIE": "GitHub source + conda env scdr-deep-py37",
    "UMAP": "umap-learn",
    "PHATE": "phate",
    "t-SNE": "sklearn.manifold.TSNE / scanpy.tl.tsne",
    "TriMap": "trimap",
    "PaCMAP": "pacmap",
    "IVIS": "conda env scdr-ivis-tf2-py310",
}

# Locked 8-tool backbone for Figures 2-6: two representative methods per
# mechanism family. Deep tools are generated through legacy environment bridges.
ANCHOR_METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
LEGACY_DEEP_METHODS = {"scScope", "SAUCIE"}
OPTIONAL_EXTENDED_METHODS = ["ZIFA", "Diffusion", "TriMap", "IVIS", "scVI"]
FAMILY_ORDER = ["factor", "deep", "graph", "relational"]
FAMILY_LABELS = {
    "factor": "Linear / count factor",
    "deep": "Deep latent",
    "graph": "Graph / diffusion",
    "relational": "Relational structure",
}

EMBEDDING_METHODS = ANCHOR_METHODS
STOCHASTIC_METHODS = {"t-SNE", "UMAP", "PaCMAP", "scScope", "SAUCIE"}

# The perturbation panel intentionally uses methods that can be recomputed many
# times on PBMC3k while keeping every plotted value source-data-backed.
PERTURB_METHODS = ["PCA", "UMAP", "Diffusion"]

SUPPORT_THRESHOLDS = {
    "local_retention": 0.30,
    "trustworthiness": 0.90,
    "global_rank_corr": 0.45,
    "label_recall": 0.55,
    "seed_stability": 0.45,
}


def ensure_dirs() -> None:
    for path in (CACHE_DIR, LEGACY_DEEP_DIR, SOURCE_DIR, OUTPUT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass
    try:
        import scanpy as sc

        sc.settings.verbosity = 1
    except ImportError:
        pass


