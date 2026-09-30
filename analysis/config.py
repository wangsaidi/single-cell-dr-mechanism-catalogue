"""Recorded constants for the empirical and controlled analyses."""

from __future__ import annotations

import os
import random

import numpy as np


SEED = 0
N_HVG = 2000
N_PCS = 50
N_NEIGHBORS = 15
TSNE_PERPLEXITY = 30
GLMPCA_MAX_ITER = 120
RANK_PAIRS = 5000

ANCHOR_METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
METHOD_FAMILY = {
    "PCA": "factor",
    "GLM-PCA": "factor",
    "scScope": "deep",
    "SAUCIE": "deep",
    "UMAP": "graph",
    "PHATE": "graph",
    "t-SNE": "relational",
    "PaCMAP": "relational",
    "scVI": "deep",
}
SUPPORT_THRESHOLDS = {
    "local_retention": 0.30,
    "trustworthiness": 0.90,
    "global_rank_corr": 0.45,
    "label_recall": 0.55,
}


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
