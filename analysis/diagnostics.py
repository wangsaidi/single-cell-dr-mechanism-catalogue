"""Shared diagnostic metrics for single-cell embedding interpretation."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from . import config


def knn_idx(X, k: int):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"Expected a two-dimensional feature matrix, found shape {X.shape}")
    if not 1 <= k < X.shape[0]:
        raise ValueError(f"k must satisfy 1 <= k < n_samples; received k={k}, n_samples={X.shape[0]}")
    # With X=None, scikit-learn treats the fitted samples as the query set and
    # removes each sample itself before returning exactly k neighbours.
    nn = NearestNeighbors(n_neighbors=k).fit(X)
    return nn.kneighbors(X=None, n_neighbors=k, return_distance=False)


def local_retention(X_high, Z_low, k: int = 15):
    hi, lo = knn_idx(X_high, k), knn_idx(Z_low, k)
    per_cell = np.array([len(set(hi[i]) & set(lo[i])) / k for i in range(X_high.shape[0])])
    return float(per_cell.mean()), per_cell


def trust(X_high, Z_low, k: int = 15):
    return float(trustworthiness(X_high, Z_low, n_neighbors=k))


def global_rank_corr(X_high, Z_low, n_pairs: int = config.RANK_PAIRS, seed: int = config.SEED):
    rng = np.random.default_rng(seed)
    n = X_high.shape[0]
    i, j = rng.integers(0, n, n_pairs), rng.integers(0, n, n_pairs)
    m = i != j
    i, j = i[m], j[m]
    dh = np.linalg.norm(X_high[i] - X_high[j], axis=1)
    dl = np.linalg.norm(Z_low[i] - Z_low[j], axis=1)
    value = spearmanr(dh, dl).correlation
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def seed_stability(embed_fn, seeds, k: int = 15):
    """Mean +/- std of pairwise kNN overlap of the same method across seeds."""
    idxs = [knn_idx(embed_fn(s), k) for s in seeds]
    n = idxs[0].shape[0]
    vals = [
        np.mean([len(set(idxs[a][i]) & set(idxs[b][i])) / k for i in range(n)])
        for a in range(len(seeds))
        for b in range(a + 1, len(seeds))
    ]
    return float(np.mean(vals)), float(np.std(vals))


def label_knn_recall(Z_low, labels, target, k: int = 15):
    idx = knn_idx(Z_low, k)
    lab = np.asarray(labels).astype(str)
    mask = lab == str(target)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean([(lab[idx[i]] == str(target)).mean() for i in np.where(mask)[0]]))
