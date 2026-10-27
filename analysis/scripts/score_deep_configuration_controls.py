"""Score matched saved deep representations and explicit PCA projection controls."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import re

from analysis import paths
from analysis.scripts.run_deep_configuration_controls import DATASETS, sha256, write_new_json


RESOLUTIONS = [.2, .4, .6, .8, 1., 1.2, 1.5, 2.]
LABEL_FIELDS = {"pbmc3k": "louvain", "paul15": "paul15_clusters",
                "heart_cell_atlas_subsampled": "cell_type"}


def neighbors(X, k=15):
    from sklearn.neighbors import NearestNeighbors

    return NearestNeighbors(n_neighbors=k, n_jobs=2).fit(X).kneighbors(X=None, return_distance=False)


def original_path(directory: Path, pattern: str, dataset: str, method: str, seed: int) -> Path:
    name = pattern.format(dataset=dataset, method=method, seed=seed)
    if Path(name).name != name or "/" in name or "\\" in name:
        raise ValueError("Original array patterns must produce a filename, not a directory path.")
    return directory / name


def candidates(args: argparse.Namespace, output: Path) -> tuple[list[dict], list[tuple[Path, dict]]]:
    result = []
    if not args.without_originals:
        for dataset in DATASETS:
            for seed in range(5):
                latent = original_path(args.original_latent_dir, args.original_latent_pattern,
                                       dataset, "scscope", seed)
                projected = original_path(args.original_embedding_dir, args.original_embedding_pattern,
                                          dataset, "scscope", seed)
                for configuration, path, projection in (
                    ("R1_latent", latent, None), ("R1_tSNE2", projected, None),
                    ("R1_PCA2", output / "projection_controls" / f"{dataset}_scscope_PCA2_seed{seed}.npy", latent),
                ):
                    result.append({"dataset": dataset, "method": "scScope", "configuration": configuration,
                                   "seed": seed, "stage": "original", "path": path,
                                   "projection_source": projection})
            for seed in range(3):
                original = original_path(args.original_embedding_dir, args.original_embedding_pattern,
                                         dataset, "saucie", seed)
                result.append({"dataset": dataset, "method": "SAUCIE",
                               "configuration": "R1_reconstruction_300", "seed": seed,
                               "stage": "original", "path": original, "projection_source": None})
    controls = []
    identities = set()
    for meta_path in sorted(args.controls_dir.glob("*/run_metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "complete":
            continue
        if args.attempt and meta.get("attempt", "original") not in args.attempt:
            continue
        if meta.get("method") not in ("SAUCIE", "scScope") or meta.get("dataset") not in DATASETS:
            raise ValueError(f"Unknown registered control: {meta_path}")
        extension_suffix = "_extended" if meta.get("maximum_budget_extension") else ""
        configuration = meta["configuration"] + ("_stabilized" if meta.get("saucie_distance_stabilization") else "") + extension_suffix
        identity = (meta["dataset"], meta["method"], configuration, meta["seed"])
        if identity in identities:
            raise ValueError("Multiple completed attempts share a configuration/seed; select one with --attempt (original selects the unlabelled run).")
        identities.add(identity)
        controls.append((meta_path, meta))
        folder = meta_path.parent
        if meta["method"] == "SAUCIE":
            for suffix, stage in (("embedding_step300.npy", "300_steps"), ("embedding_final.npy", "final")):
                result.append({"dataset": meta["dataset"], "method": "SAUCIE", "configuration": configuration,
                               "seed": meta["seed"], "stage": stage, "path": folder / suffix,
                               "projection_source": None})
        else:
            for suffix, stage in (("latent_epoch300.npy", "300_epochs"), ("latent_final.npy", "final")):
                latent = folder / suffix
                result.append({"dataset": meta["dataset"], "method": "scScope", "configuration": "retrained_latent" + extension_suffix,
                               "seed": meta["seed"], "stage": stage, "path": latent, "projection_source": None})
                projected = output / "projection_controls" / folder.name / f"{latent.stem}_PCA2.npy"
                result.append({"dataset": meta["dataset"], "method": "scScope", "configuration": "retrained_PCA2" + extension_suffix,
                               "seed": meta["seed"], "stage": stage, "path": projected, "projection_source": latent})
    return result, controls


def validate_array(path: Path, n_cells: int, dimension: int) -> None:
    import numpy as np

    if not path.is_file():
        raise FileNotFoundError(f"Required array missing: {path}. Supply the saved arrays or explicitly use --without-originals; no fallback is provided.")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (n_cells, dimension) or not np.isfinite(array).all():
        raise ValueError(f"Expected finite {(n_cells, dimension)} array, got {array.shape}: {path}")


def validate_control_input_hash(path: Path, meta: dict) -> str:
    """Keep the historical training hash distinct from a portable serialization."""
    original = meta["input_file_sha256"]
    portable = meta.get("portable_input_file_sha256", original)
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (original, portable)):
        raise ValueError("Expected actual original/portable input SHA256 values.")
    if "portable_input_file_sha256" in meta:
        record = meta.get("input_serialization")
        if not isinstance(record, dict) or any(record.get(key) != value for key, value in (
            ("source_npz_sha256", original), ("portable_npz_sha256", portable),
            ("counts_hvg_sha256", meta["original_counts_file_sha256"]),
            ("numeric_X_npy_byte_identical", True), ("object_id_payloads_deserialized", False),
            ("training_inputs_modified", False), ("allow_pickle", False))):
            raise ValueError("Portable input serialization provenance is missing or inconsistent.")
    if sha256(path) != portable:
        raise ValueError(f"Control input hash differs: {path.name}")
    return portable


def project(item: dict) -> None:
    import numpy as np
    from sklearn.decomposition import PCA

    source = item["projection_source"]
    if source is None:
        return
    item["path"].parent.mkdir(parents=True, exist_ok=True)
    # A declared dimension-matched control, never a substitute for a failed model.
    value = PCA(n_components=2, random_state=0).fit_transform(np.load(source)).astype(np.float32)
    with item["path"].open("xb") as stream:
        np.save(stream, value)


def score_one(item, proc, reference_neighbors, labels):
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.stats import spearmanr
    from sklearn.manifold import trustworthiness
    from sklearn.metrics import adjusted_rand_score, f1_score

    Z = np.load(item['path'])
    if Z.shape[0] != proc.n_obs or not np.isfinite(Z).all():
        raise ValueError(f"Invalid saved representation: {item['path']}")
    lo = neighbors(Z)
    local = np.mean([len(set(a) & set(b)) / 15 for a, b in zip(reference_neighbors, lo)])
    label = float(np.mean(labels[lo] == labels[:, None]))
    X = np.asarray(proc.obsm['X_pca_ref'])[:, :50]
    rng = np.random.default_rng(20261027)
    subset = np.sort(rng.choice(proc.n_obs, min(1000, proc.n_obs), replace=False))
    trust = trustworthiness(X[subset], Z[subset], n_neighbors=15)
    i = rng.integers(0, proc.n_obs, 6000)
    j = rng.integers(0, proc.n_obs, 6000)
    valid = i != j
    i, j = i[valid][:5000], j[valid][:5000]
    rank = float(spearmanr(np.linalg.norm(X[i] - X[j], axis=1), np.linalg.norm(Z[i] - Z[j], axis=1)).statistic)
    work = ad.AnnData(np.zeros((proc.n_obs, 1), dtype=np.float32))
    work.obsm['Z'] = Z
    sc.pp.neighbors(work, n_neighbors=15, use_rep='Z', metric='euclidean', random_state=0)
    partitions = []
    per_labels = []
    for resolution in RESOLUTIONS:
        sc.tl.leiden(work, resolution=resolution, random_state=0, flavor='igraph', directed=False, n_iterations=-1)
        clusters = work.obs['leiden'].astype(str).to_numpy()
        table = pd.crosstab(pd.Series(clusters), pd.Series(labels))
        mapping = table.idxmax(axis=1).to_dict()
        predicted = np.array([mapping[c] for c in clusters])
        partitions.append({'resolution': resolution, 'ari': adjusted_rand_score(labels, clusters),
                           'macro_f1': f1_score(labels, predicted, average='macro', zero_division=0)})
        for name in sorted(np.unique(labels)):
            truth, estimate = labels == name, predicted == name
            per_labels.append({'resolution': resolution, 'label': name,
                               'f1': f1_score(truth, estimate, zero_division=0), 'n_label': int(truth.sum())})
    frame = pd.DataFrame(partitions)
    auc = lambda name: float(np.trapezoid(frame[name], frame['resolution']) / 1.8)
    row = {k: item[k] for k in ('dataset', 'method', 'configuration', 'seed', 'stage')}
    row.update({'n_cells': proc.n_obs, 'representation_dimension': Z.shape[1], 'local_retention': local,
                'same_label_neighbor_fraction': label, 'trustworthiness_fixed_subset': float(trust),
                'trust_subset_n': len(subset), 'global_rank_correlation': rank, 'n_distance_pairs': len(i),
                'resolution_integrated_ari': auc('ari'), 'resolution_integrated_macro_f1': auc('macro_f1'),
                'array_sha256': sha256(item['path']), 'seed_is_biological_replicate': False})
    pop = pd.DataFrame(per_labels)
    population = []
    for name, values in pop.groupby('label'):
        v = values.sort_values('resolution')
        population.append({**{k: row[k] for k in ('dataset', 'method', 'configuration', 'seed', 'stage')},
                           'label': name, 'n_label': int(v.iloc[0].n_label),
                           'resolution_integrated_f1': float(np.trapezoid(v.f1, v.resolution) / 1.8)})
    return row, population


def checkpoint_stability(controls: list[tuple[Path, dict]]) -> list[dict]:
    import numpy as np
    from scipy.stats import spearmanr

    stability = []
    for meta_path, meta in controls:
        if meta['method'] != 'scScope':
            continue
        folder = meta_path.parent
        end = int(meta['final_epoch'])
        starts = sorted(int(p.stem.replace('latent_epoch', '')) for p in folder.glob('latent_epoch[0-9]*.npy')
                        if '_PCA2' not in p.stem and int(p.stem.replace('latent_epoch', '')) < end)
        if not starts:
            raise ValueError(f"No checkpoint strictly preceding the final epoch: {folder}")
        start = starts[-1]
        A = np.load(folder / ('latent_epoch%d.npy' % start))
        B = np.load(folder / 'latent_final.npy')
        if A.shape != B.shape or not np.isfinite(A).all() or not np.isfinite(B).all():
            raise ValueError(f"Invalid checkpoint pair: {folder}")
        left, right = neighbors(A), neighbors(B)
        overlap = float(np.mean([len(set(a) & set(b)) / 15 for a, b in zip(left, right)]))
        rng = np.random.default_rng(20261027)
        i, j = rng.integers(0, len(A), (2, 6000))
        keep = i != j
        i, j = i[keep][:5000], j[keep][:5000]
        rho = float(spearmanr(np.linalg.norm(A[i]-A[j], axis=1), np.linalg.norm(B[i]-B[j], axis=1)).statistic)
        stability.append({'dataset': meta['dataset'], 'method': 'scScope', 'seed': meta['seed'],
                          'start_epoch': start, 'final_epoch': end, 'n_cells': len(A), 'n_pairs': len(i),
                          'pairwise_neighbor_overlap': overlap, 'pairwise_distance_rank_rho': rho,
                          'budget_extension': bool(meta.get('maximum_budget_extension')),
                          'n_biological_replicates': 0})
    return stability


def uncompleted_jobs(directory: Path) -> list[dict]:
    unavailable = []
    registration = directory / "analysis_manifest.json"
    plans = ([registration] if registration.is_file() else []) + sorted(directory.glob("attempt_*.json"))
    for plan in plans:
        payload = json.loads(plan.read_text(encoding="utf-8"))
        for job in payload["jobs"] if isinstance(payload, dict) else payload:
            folder = directory / Path(job["out_dir"]).name
            if not (folder / "run_metadata.json").is_file():
                failed = (folder / "failure_metadata.json").is_file() or (folder / "failure.json").is_file()
                unavailable.append({"job": folder.name, "status": "failed" if failed else "not_completed",
                                    "budget_extension": bool(job.get("maximum_budget_extension")),
                                    "excluded_from_endpoint_scores": True})
    return unavailable


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects-dir", type=Path,
                        default=Path(os.environ.get("SCDR_CONTROL_OBJECTS_DIR", paths.ANALYSIS_OBJECTS_DIR)))
    parser.add_argument("--controls-dir", type=Path,
                        default=Path(os.environ.get("SCDR_DEEP_CONTROLS_DIR", paths.RESULTS_DIR / "deep_configuration_controls")))
    parser.add_argument("--original-latent-dir", type=Path,
                        default=Path(os.environ.get("SCDR_ORIGINAL_LATENT_DIR", paths.LEGACY_CACHE_DIR)))
    parser.add_argument("--original-embedding-dir", type=Path,
                        default=Path(os.environ.get("SCDR_ORIGINAL_EMBEDDING_DIR", paths.ANCHOR_EMBEDDINGS_DIR)))
    parser.add_argument("--output-dir", type=Path, default=None, help="Fresh score directory; defaults to <controls-dir>/scores.")
    parser.add_argument("--original-latent-pattern", default="scscope_seed{seed}_analysis_{dataset}_recommended_v1.npy")
    parser.add_argument("--original-embedding-pattern", default="{dataset}_{method}_seed{seed}.npy")
    parser.add_argument("--without-originals", action="store_true", help="Explicitly score only completed new controls; does not supply missing original latents.")
    parser.add_argument("--attempt", action="append", default=[], help="Filter completed control attempts; original means the unlabelled jobs. Repeat to select distinct configurations.")
    parser.add_argument("--limit", type=int, help="Limit endpoint candidates only; checkpoint stability still covers all selected completed scScope fits.")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    return args


def main() -> None:
    args = parse_args()
    import anndata as ad
    import numpy as np
    import pandas as pd

    for name in ("objects_dir", "controls_dir", "original_latent_dir", "original_embedding_dir"):
        setattr(args, name, getattr(args, name).resolve())
    output = (args.output_dir or args.controls_dir / "scores").resolve()
    for readonly in (args.objects_dir, args.original_latent_dir, args.original_embedding_dir):
        if output == readonly or output in readonly.parents or readonly in output.parents:
            raise ValueError("Score output must not overlap original inputs.")
    if output == args.controls_dir or output in args.controls_dir.parents:
        raise ValueError("Use a separate fresh score directory, not the control root or its parent.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Scores already exist; choose a fresh --output-dir rather than appending stale results.")
    items, controls = candidates(args, output)
    if args.limit:
        items = items[:args.limit]
    if not items:
        raise ValueError("No saved representations selected; no scores were generated.")
    objects, count_gene_ids = {}, {}
    object_records = []
    for dataset in sorted({item["dataset"] for item in items} | {meta["dataset"] for _, meta in controls}):
        proc_path = args.objects_dir / f"{dataset}_proc.h5ad"
        count_path = args.objects_dir / f"{dataset}_counts_hvg.h5ad"
        proc = ad.read_h5ad(proc_path)
        counts = ad.read_h5ad(count_path, backed="r")
        try:
            if not np.array_equal(np.asarray(counts.obs_names), np.asarray(proc.obs_names)):
                raise ValueError(f"Count/processed cell order differs: {dataset}")
            count_gene_ids[dataset] = np.asarray(counts.var_names.astype(str), dtype=str)
        finally:
            counts.file.close()
        reference = np.asarray(proc.obsm["X_pca_ref"])
        if reference.shape[1] < 50 or not np.isfinite(reference[:, :50]).all() or proc.n_obs <= 30:
            raise ValueError(f"Expected finite PCA50 reference and more than 30 cells: {dataset}")
        objects[dataset] = proc
        object_records.append({"dataset": dataset, "n_cells": proc.n_obs,
                               "proc_sha256": sha256(proc_path), "counts_sha256": sha256(count_path)})
    for meta_path, meta in controls:
        inp = args.controls_dir / Path(meta["input"]).name
        validate_control_input_hash(inp, meta)
        if sha256(args.objects_dir / f"{meta['dataset']}_counts_hvg.h5ad") != meta["original_counts_file_sha256"]:
            raise ValueError(f"Control/count object hash differs: {meta_path}")
        with np.load(inp, allow_pickle=False) as data:
            if not np.array_equal(data["cell_ids"], np.asarray(objects[meta["dataset"]].obs_names.astype(str))):
                raise ValueError(f"Control input/processed cell order differs: {meta_path}")
            if not np.array_equal(data["gene_ids"], count_gene_ids[meta["dataset"]]):
                raise ValueError(f"Control input/count gene order differs: {meta_path}")
    for item in items:
        dimension = 100 if item["configuration"] in ("R1_latent", "retrained_latent", "retrained_latent_extended") else 2
        source = item["projection_source"]
        validate_array(source or item["path"], objects[item["dataset"]].n_obs, 100 if source else dimension)
    stability = checkpoint_stability(controls)
    output.mkdir(parents=True, exist_ok=True)
    write_new_json(output / "endpoint_design.json", {
        "k": 15, "reference": "full-cell PCA50", "trustworthiness_only_subset_n": 1000,
        "subset_and_pair_seed": 20261027, "resolution_grid": RESOLUTIONS,
        "clustering": "fixed eight-resolution grid and Leiden seed 0; majority mapping after clustering",
        "cell_ids": "identical ordered count-HVG and processed-object cell IDs; original arrays must use that same order",
        "ranges": "computational seed ranges, not biological confidence intervals",
        "score_script_sha256": sha256(Path(__file__)), "objects": object_records,
        "software_versions": {name: version(name) for name in
                              ("anndata", "numpy", "pandas", "scanpy", "scipy", "scikit-learn")},
        "without_originals": args.without_originals, "attempt_filters": args.attempt,
        "endpoint_limit": args.limit, "checkpoint_stability_rows": len(stability),
        "uncompleted_registered_jobs": uncompleted_jobs(args.controls_dir),
        "candidates": [{key: str(value) if isinstance(value, Path) else value for key, value in item.items()}
                       for item in items],
        "completed_controls": [{"metadata": str(path), "sha256": sha256(path),
                                "attempt": meta.get("attempt", "original"),
                                "maximum_budget_extension": bool(meta.get("maximum_budget_extension")),
                                "max_epochs": meta.get("max_epochs"), "max_steps": meta.get("max_steps"),
                                "input_file_sha256": meta["input_file_sha256"],
                                "portable_input_file_sha256": meta.get("portable_input_file_sha256"),
                                "input_serialization": meta.get("input_serialization"),
                                "checkpoint_stability_rule_met": meta.get("checkpoint_stability_rule_met")}
                               for path, meta in controls],
    })
    rows, populations = [], []
    for dataset, proc in objects.items():
        field = proc.uns.get("analysis_label_field", LABEL_FIELDS[dataset])
        labels = proc.obs[field].astype(str).to_numpy()
        reference_neighbors = neighbors(np.asarray(proc.obsm["X_pca_ref"])[:, :50])
        for item in (item for item in items if item["dataset"] == dataset):
            project(item)
            row, population = score_one(item, proc, reference_neighbors, labels)
            rows.append(row)
            populations.extend(population)
            pd.DataFrame(rows).to_csv(output / "matched_endpoint_scores.csv", index=False)
            pd.DataFrame(populations).to_csv(output / "matched_population_scores.csv", index=False)
            print(dataset, item["configuration"], item["seed"], item["stage"],
                  "macroF1", row["resolution_integrated_macro_f1"], flush=True)
    stability_fields = ["dataset", "method", "seed", "start_epoch", "final_epoch", "n_cells", "n_pairs",
                        "pairwise_neighbor_overlap", "pairwise_distance_rank_rho", "budget_extension", "n_biological_replicates"]
    pd.DataFrame(stability, columns=stability_fields).to_csv(output / "checkpoint_representation_stability.csv", index=False)
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["dataset", "method", "configuration", "stage"])[[
        "local_retention", "global_rank_correlation", "same_label_neighbor_fraction",
        "resolution_integrated_macro_f1"]].agg(["median", "min", "max"])
    summary.to_csv(output / "endpoint_seed_summary.csv")
    write_new_json(output / "scoring_complete.json", {"status": "complete", "endpoint_rows": len(rows),
                                                     "population_rows": len(populations),
                                                     "checkpoint_stability_rows": len(stability)})


if __name__ == "__main__":
    main()
