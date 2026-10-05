"""Matched permutation references and supplementary scVI geometry.

Run from the repository root with the archived analysis inputs available.
The analysis compares random correspondence with the original cutoffs; it
does not fit cutoffs retrospectively or claim externally calibrated standards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import sklearn
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from analysis.paths import ANALYSIS_OBJECTS_DIR, ANCHOR_EMBEDDINGS_DIR, REPO_ROOT, RESULTS_DIR

DATASETS = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
METRICS = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
CUTOFF = dict(zip(METRICS, [0.30, 0.90, 0.45, 0.55]))
K = 15
SEED = 20261005
SUBSET_SEED = 20260714


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_cells(labels, target=1000):
    rng = np.random.default_rng(SUBSET_SEED)
    groups = {x: np.flatnonzero(labels == x) for x in np.unique(labels)}
    target = min(target, len(labels))
    raw = {x: target * len(y) / len(labels) for x, y in groups.items()}
    quota = {x: min(len(groups[x]), max(1, int(np.floor(raw[x])))) for x in groups}
    while sum(quota.values()) > target:
        x = min((x for x in groups if quota[x] > 1), key=lambda x: raw[x] - quota[x])
        quota[x] -= 1
    while sum(quota.values()) < target:
        x = max((x for x in groups if quota[x] < len(groups[x])), key=lambda x: raw[x] - quota[x])
        quota[x] += 1
    return np.sort(np.concatenate([rng.choice(groups[x], quota[x], replace=False) for x in sorted(groups)]))


def neighbours(x):
    return NearestNeighbors(n_neighbors=K).fit(x).kneighbors(X=None, return_distance=False)


def pair_indices(n):
    rng = np.random.default_rng(SEED)
    i, j = rng.integers(0, n, (2, 6000))
    keep = i != j
    return i[keep][:5000], j[keep][:5000]


def ranked(x):
    y = rankdata(x).astype(float)
    y -= y.mean()
    norm = np.linalg.norm(y)
    if norm == 0:
        raise ValueError("Constant rank vector; diagnostic is undefined, not zero.")
    return y / norm


def permutation(rng, n, groups=None):
    if groups is None:
        return rng.permutation(n)
    p = np.arange(n)
    for ix in groups:
        p[ix] = rng.permutation(ix)
    return p


def groups_for(values):
    return [np.flatnonzero(values == x) for x in np.unique(values)]


def null_summary(observed, null, cutoff, family, tail="upper", **keys):
    null = np.asarray(null, float)
    if not np.isfinite(null).all() or not np.isfinite(observed):
        raise ValueError("Non-finite null or observed diagnostic.")
    extreme = int(np.count_nonzero(null >= observed) if tail == "upper" else np.count_nonzero(null <= observed))
    p = (extreme + 1) / (len(null) + 1)
    return {
        **keys, "null_family": family, "tail": tail, "observed": float(observed),
        "operational_cutoff": cutoff, "original_cutoff_met": bool(observed >= cutoff),
        "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
        "null_q025": float(np.quantile(null, .025)), "null_q05": float(np.quantile(null, .05)),
        "null_q95": float(np.quantile(null, .95)), "null_q975": float(np.quantile(null, .975)),
        "effect_vs_null_mean": float(observed - null.mean()), "p_one_sided": p,
        "monte_carlo_se": float(np.sqrt(p * (1 - p) / (len(null) + 1))),
        "extreme_draws": extreme, "permutations": len(null), "n_cells": keys.get("n_cells", 1000),
        "embedding_seed": 0, "permutation_seed": SEED, "k": K,
    }


def bh(frame):
    frame["q_bh"] = np.nan
    for _, indices in frame.groupby("null_family").groups.items():
        p = frame.loc[indices, "p_one_sided"].to_numpy()
        order = np.argsort(p)
        adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        q = np.empty(len(p)); q[order] = np.minimum(adjusted, 1)
        frame.loc[indices, "q_bh"] = q
    frame["null_rejected_q05"] = frame.q_bh.lt(.05)
    return frame


def entropy(donors, nn):
    ncat = int(donors.max()) + 1
    counts = np.stack([(donors[nn] == x).sum(axis=1) for x in range(ncat)], axis=1)
    probabilities = counts / K
    logs = np.zeros_like(probabilities)
    np.log(probabilities, out=logs, where=probabilities > 0)
    return float(np.mean(-np.sum(probabilities * logs, axis=1) / np.log(ncat)))


def embedding(dataset, method, dimension=2, seed=0, scvi_dir=None):
    if method == "scVI":
        return scvi_dir / f"{dataset}_scvi_dim{dimension}_seed{seed}.npy"
    stem = method.lower().replace("-", "").replace(" ", "_")
    return ANCHOR_EMBEDDINGS_DIR / f"{dataset}_{stem}_seed{seed}.npy"


def run(b, scvi_dir, out, export_plot_data=False):
    out.mkdir(parents=True, exist_ok=True)
    design = json.loads((REPO_ROOT / "analysis/calibration_design.json").read_text())
    design["permutations"] = b
    (out / "analysis_design.json").write_text(json.dumps(design, indent=2))
    rows, donor_rows, continuum_rows, scvi_full, reference_rows, stability_rows = [], [], [], [], [], []
    subsets, provenance, null_store = [], [], []
    draw_index = []
    def store_draws(values, dataset, method, family, metrics, root=""):
        name = f"draws_{len(null_store):03d}"
        null_store.append(np.asarray(values, dtype=np.float32))
        for column, metric in enumerate(metrics):
            draw_index.append({"array": name, "column": column, "dataset_id": dataset, "method": method, "metric": metric, "null_family": family, "root": root, "n_permutations": b})
    rng = np.random.default_rng(SEED)

    for ds in DATASETS:
        obj_path = ANALYSIS_OBJECTS_DIR / f"{ds}_proc.h5ad"
        ad = sc.read_h5ad(obj_path)
        field = str(ad.uns.get("ijbs_label_field", ""))
        if field not in ad.obs:
            raise ValueError(f"Missing annotation field for {ds}")
        labels_full = ad.obs[field].astype(str).to_numpy()
        selection = select_cells(labels_full)
        labels = labels_full[selection]
        n = len(selection)
        high = np.asarray(ad.obsm["X_pca_ref"], float)[:, :50]
        if not np.isfinite(high).all():
            raise ValueError(f"Non-finite reference for {ds}")
        ref = high[selection]
        i, j = pair_indices(n)
        high_distance = cdist(ref, ref)
        ref_nn = neighbours(ref)
        ref_adj = np.zeros((n, n), bool); ref_adj[np.arange(n)[:, None], ref_nn] = True
        high_sort_distance = high_distance.copy(); np.fill_diagonal(high_sort_distance, -np.inf)
        ranks = np.argsort(np.argsort(high_sort_distance, axis=1, kind="stable"), axis=1, kind="stable")
        high_pair_rank = ranked(high_distance[i, j])
        label_groups = groups_for(labels)
        low_nn, low_distance, coords_all = {}, {}, {}
        donors = None
        if ds == DATASETS[2]:
            donors = pd.Categorical(ad.obs["donor"].astype(str).to_numpy()[selection]).codes
        provenance.append({"input": f"analysis_objects/{obj_path.name}", "sha256": sha(obj_path), "n_cells": ad.n_obs})
        subsets.extend({"dataset_id": ds, "cell_index": int(ix), "cell_id": str(ad.obs_names[ix]), "annotation": str(labels_full[ix])} for ix in selection)

        for mi, method in enumerate(METHODS):
            path = embedding(ds, method, scvi_dir=scvi_dir)
            coords = np.load(path).astype(float)
            if coords.shape != (ad.n_obs, 2) or not np.isfinite(coords).all():
                raise ValueError(f"Invalid embedding {path.name}: {coords.shape}")
            provenance.append({"input": f"{'scvi_outputs' if method == 'scVI' else 'anchor_embeddings'}/{path.name}", "sha256": sha(path), "n_cells": ad.n_obs})
            coords_all[method] = coords
            low = coords[selection]
            low_nn[method] = neighbours(low)
            low_distance[method] = cdist(low, low)
            nn = low_nn[method]
            local = float(ref_adj[np.arange(n)[:, None], nn].mean())
            trust = 1 - 2 * np.maximum(ranks[np.arange(n)[:, None], nn] - K, 0).sum() / (n * K * (2 * n - 3 * K - 1))
            check = trustworthiness(ref, low, n_neighbors=K)
            if not np.isclose(trust, check, atol=1e-12):
                raise AssertionError(f"Trustworthiness implementation mismatch {trust} vs {check}")
            observed = [local, trust, float(high_pair_rank @ ranked(low_distance[method][i, j])), float((labels[nn] == labels[:, None]).mean())]
            for null_name, groups, metric_indices in [
                ("geometry_unrestricted", None, [0, 1, 2]),
                ("geometry_within_annotation", label_groups, [0, 1, 2]),
            ]:
                null = np.empty((b, 3))
                for bi in range(b):
                    p = permutation(rng, n, groups)
                    inv = np.empty(n, int); inv[p] = np.arange(n)
                    perm_nn = inv[nn[p]]
                    null[bi, 0] = ref_adj[np.arange(n)[:, None], perm_nn].mean()
                    null[bi, 1] = 1 - 2 * np.maximum(ranks[np.arange(n)[:, None], perm_nn] - K, 0).sum() / (n * K * (2 * n - 3 * K - 1))
                    null[bi, 2] = high_pair_rank @ ranked(low_distance[method][p[i], p[j]])
                for index in metric_indices:
                    rows.append(null_summary(observed[index], null[:, index], CUTOFF[METRICS[index]], null_name, dataset_id=ds, method=method, metric=METRICS[index], n_cells=n, sampled_pairs=len(i)))
                store_draws(null, ds, method, null_name, METRICS[:3])
            for family, groups in [("annotation_unrestricted", None)] + ([("annotation_within_donor", groups_for(donors))] if donors is not None else []):
                null = np.array([float((labels[(p := permutation(rng, n, groups))][nn] == labels[p][:, None]).mean()) for _ in range(b)])
                rows.append(null_summary(observed[3], null, CUTOFF[METRICS[3]], family, dataset_id=ds, method=method, metric=METRICS[3], n_cells=n, sampled_pairs=0))
                store_draws(null, ds, method, family, [METRICS[3]])
            if donors is not None:
                null = np.array([entropy(donors[permutation(rng, n, label_groups)], nn) for _ in range(b)])
                donor_rows.append(null_summary(entropy(donors, nn), null, .50, "donor_within_cell_type", tail="lower", dataset_id=ds, method=method, metric="donor_entropy", n_cells=n, n_donors=len(np.unique(donors))))
                store_draws(null, ds, method, "donor_within_cell_type", ["donor_entropy"])
            print(f"Calibration {ds} {method} complete", flush=True)

        # scVI metrics are recomputed directly from each full-object latent.
        full_i, full_j = pair_indices(ad.n_obs)
        full_high_rank = ranked(np.linalg.norm(high[full_i] - high[full_j], axis=1))
        full_ref_nn = neighbours(high)
        scvi_nn, scvi_pair_ranks = {}, {}
        for dim in [2, 10]:
            for seed in range(5):
                path = embedding(ds, "scVI", dim, seed, scvi_dir)
                z = np.load(path).astype(float)
                if z.shape != (ad.n_obs, dim) or not np.isfinite(z).all():
                    raise ValueError(f"Invalid scVI latent {path.name}")
                nn = neighbours(z)
                distance_rank = ranked(np.linalg.norm(z[full_i] - z[full_j], axis=1))
                scvi_nn[dim, seed] = nn; scvi_pair_ranks[dim, seed] = distance_rank
                scores = [float(np.mean([len(set(a) & set(c)) / K for a, c in zip(full_ref_nn, nn)])), float(trustworthiness(high, z, n_neighbors=K)), float(full_high_rank @ distance_rank), float((labels_full[nn] == labels_full[:, None]).mean())]
                scvi_full.extend({"dataset_id": ds, "method": "scVI", "output_dimension": dim, "seed": seed, "metric": metric, "value": score, "n_cells": ad.n_obs, "k": K, "reference_dimension": 50, "sampled_pairs": len(full_i)} for metric, score in zip(METRICS, scores))
                provenance.append({"input": f"scvi_outputs/{path.name}", "sha256": sha(path), "n_cells": ad.n_obs})
                for ref_dim in [2, 5, 10, 20, 50]:
                    rnn = neighbours(ref[:, :ref_dim]); znn = neighbours(z[selection])
                    value = float(np.mean([len(set(a) & set(c)) / K for a, c in zip(rnn, znn)]))
                    reference_rows.append({"dataset_id": ds, "output_dimension": dim, "seed": seed, "reference_dimension": ref_dim, "metric": "local_retention", "value": value, "n_cells": n, "k": K})
            for first, second in combinations(range(5), 2):
                overlap = float(np.mean([len(set(a) & set(c)) / K for a, c in zip(scvi_nn[dim, first], scvi_nn[dim, second])]))
                for metric, value in [("neighbour_overlap", overlap), ("distance_rank_stability", float(scvi_pair_ranks[dim, first] @ scvi_pair_ranks[dim, second]))]:
                    stability_rows.append({"dataset_id": ds, "output_dimension": dim, "seed_a": first, "seed_b": second, "metric": metric, "value": value, "n_cells": ad.n_obs, "k": K, "sampled_pairs": len(full_i)})
        for method in METHODS[:-1]:
            nn = neighbours(coords_all[method])
            scores = [float(np.mean([len(set(a) & set(c)) / K for a, c in zip(full_ref_nn, nn)])), float(trustworthiness(high, coords_all[method], n_neighbors=K)), float(full_high_rank @ ranked(np.linalg.norm(coords_all[method][full_i] - coords_all[method][full_j], axis=1))), float((labels_full[nn] == labels_full[:, None]).mean())]
            scvi_full.extend({"dataset_id": ds, "method": method, "output_dimension": 2, "seed": 0, "metric": metric, "value": score, "n_cells": ad.n_obs, "k": K, "reference_dimension": 50, "sampled_pairs": len(full_i)} for metric, score in zip(METRICS, scores))

        if ds == "paul15":
            work = ad.copy()
            sc.pp.neighbors(work, n_neighbors=K, use_rep="X_pca_ref", random_state=0)
            sc.tl.diffmap(work, n_comps=15)
            roots = {"stored_8Mk": int(ad.uns["iroot"])}
            for root_label in ["7MEP", "9GMP", "1Ery"]:
                ix = np.flatnonzero(labels_full == root_label)
                root_ref = high[ix, :20]
                roots[root_label] = int(ix[np.argmin(np.linalg.norm(root_ref - root_ref.mean(axis=0), axis=1))])
            lineage = np.array(["erythroid" if x in ["7MEP", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"] else "megakaryocyte" if x == "8Mk" else "basophil" if x in ["12Baso", "13Baso"] else "myeloid" if x in ["9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"] else "other" for x in labels])
            for root_name, root in roots.items():
                work.uns["iroot"] = root
                sc.tl.dpt(work, n_dcs=10)
                time = work.obs.dpt_pseudotime.to_numpy(dtype=float)[selection]
                if not np.isfinite(time).all():
                    raise ValueError("Non-finite DPT on the shared evaluation subset; explicit exclusion is required.")
                random_delta = np.abs(time[i] - time[j]).mean()
                for method in METHODS:
                    nn, distance = low_nn[method], low_distance[method]
                    distance_rank = ranked(distance[i, j])
                    obs = [float(distance_rank @ ranked(np.abs(time[i] - time[j]))), 1 - float(np.abs(time[nn] - time[:, None]).mean()) / random_delta]
                    for family, groups in [("continuum_unrestricted", None), ("continuum_within_lineage", groups_for(lineage))]:
                        null = np.empty((b, 2))
                        for bi in range(b):
                            t = time[permutation(rng, n, groups)]
                            null[bi, 0] = distance_rank @ ranked(np.abs(t[i] - t[j]))
                            null[bi, 1] = 1 - float(np.abs(t[nn] - t[:, None]).mean()) / float(np.abs(t[i] - t[j]).mean())
                        for m, metric in enumerate(["pseudotime_distance_correlation", "local_pseudotime_retention"]):
                            continuum_rows.append(null_summary(obs[m], null[:, m], [.45, .50][m], family, dataset_id=ds, method=method, metric=metric, root=root_name, root_cell_id=str(ad.obs_names[root]), n_cells=n, excluded_cells=0, sampled_pairs=len(i)))
                        store_draws(null, ds, method, family, ["pseudotime_distance_correlation", "local_pseudotime_retention"], root_name)
                print(f"Continuum calibration {root_name} complete", flush=True)

    frames = {"calibration_geometry.csv": bh(pd.DataFrame(rows)), "calibration_continuum.csv": bh(pd.DataFrame(continuum_rows)), "calibration_donor.csv": bh(pd.DataFrame(donor_rows)), "scvi_geometry_full.csv": pd.DataFrame(scvi_full), "scvi_reference_sensitivity.csv": pd.DataFrame(reference_rows), "scvi_seed_stability.csv": pd.DataFrame(stability_rows), "calibration_evaluation_cells.csv": pd.DataFrame(subsets)}
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False, float_format="%.12g")
    np.savez_compressed(out / "permutation_draws.npz", **{f"draws_{i:03d}": x for i, x in enumerate(null_store)})
    pd.DataFrame(draw_index).to_csv(out / "permutation_draw_index.csv", index=False)
    metadata = {"design": design, "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__, "scanpy": version("scanpy"), "anndata": version("anndata")}, "inputs": provenance, "draw_array_order": "Dataset, method, unrestricted geometry, annotation-conditioned geometry, annotation null, optional within-donor annotation, optional donor null; Paul15 root/method/unrestricted/lineage-conditioned continuum follows Paul15 geometry, before heart geometry. Arrays are stored sequentially in this loop order.", "missing_values": 0, "interpretation": "No biological replication or universal cutoff validation is inferred from computational permutation or seed repeats."}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    if export_plot_data:
        destination = REPO_ROOT / "data/source_data/calibration"
        destination.mkdir(parents=True, exist_ok=True)
        for path in out.iterdir():
            if path.is_file():
                shutil.copy2(path, destination / path.name)
    print(json.dumps({name: len(frame) for name, frame in frames.items()}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--scvi-dir", type=Path, default=RESULTS_DIR / "embeddings")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "calibration")
    parser.add_argument("--export-plot-data", action="store_true")
    args = parser.parse_args()
    run(args.permutations, args.scvi_dir, args.output, args.export_plot_data)
