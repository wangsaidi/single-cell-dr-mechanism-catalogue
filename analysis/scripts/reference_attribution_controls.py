"""Reference/root attribution for saved Paul15 representations.

Run from the repository root with python -B -m analysis.scripts.reference_attribution_controls.
Inputs follow analysis.paths; explicit CLI directories override the corresponding
environment settings before the trajectory module is imported. No model is refitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version
from itertools import combinations
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

OUT = DATA = SCVI_DIR = PUBLIC_SOURCE = None
OPTIONS = None
public = ad = sc = sparse = connected_components = NearestNeighbors = None

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
ROOT_NAMES = ["7MEP_centroid", "9GMP_centroid", "1Ery_centroid", "stored_8Mk_root"]
SCOPES = {
    "pooled": None,
    "erythroid": ["7MEP", "1Ery", "2Ery", "3Ery", "4Ery", "5Ery", "6Ery"],
    "megakaryocytic": ["7MEP", "8Mk"],
    "basophil": ["7MEP", "12Baso", "13Baso"],
    "myeloid": ["9GMP", "10GMP", "11DC", "14Mo", "15Mo", "16Neu", "17Neu", "18Eos"],
}
PAIR_SEED = 20260714
DESIGN = {
    "dataset": "Paul15", "methods": METHODS, "embedding_seeds": list(range(5)),
    "expression_reference": "existing X_pca_ref, all 50 columns; no reprocessing",
    "representation_dimension": 2, "graph_k": 15, "graph_metric": "euclidean",
    "graph_method": "umap", "graph_random_state": 0, "diffmap_components": 15,
    "diffmap_random_state": 0, "dpt_components": 10, "dpt_n_branchings": 0,
    "root_selection": "public root_definitions: centroid in expression PCA20; stored iroot",
    "root_names": ROOT_NAMES, "lineage_clusters": {k: v for k, v in SCOPES.items() if k != "pooled"},
    "pair_seed": PAIR_SEED, "pooled_nonself_pair_draws": 5000,
    "lineage_pair_draws": "min(3000, 5*n_lineage); seed=20260714+len(lineage)",
    "pairs": "fixed full-object draws before finite-mask restriction; duplicates retained as in S11",
    "primary_mask": "intersection of finite values over all roots, all nine methods and all five seeds",
    "secondary_masks": ["within-seed all-root nine-method intersection", "method/root-pair finite intersection"],
    "primary_statistics": ["signed Spearman", "Kendall tau-b", "absolute Spearman (orientation-free order)",
                           "Spearman of pairwise absolute pseudotime differences", "rank MAE before/after reversal"],
    "secondary_cutoffs": {"legacy_expression_distance_vs_embedding_distance": 0.45,
                          "legacy_local_pseudotime_retention": 0.50},
    "cutoff_scope": "legacy geometry diagnostics only, not thresholds for crossed DPT or biological reliability",
    "missing": "all disconnected or other nonfinite DPT values retained as NaN; never terminal imputation",
    "biological_replication": "one existing dataset; cells, pairs, roots and computational seeds are not biological replicates",
    "protocol_extension": "full 4x4 crossed DPT, global five-seed finite mask, explicit rank reversal residuals; descriptive only",
}



def configure_paths(arguments):
    global OUT, DATA, SCVI_DIR, PUBLIC_SOURCE, OPTIONS
    for name, value in [
        ("SCDR_ANALYSIS_DATA", arguments.analysis_data),
        ("SCDR_SCVI_EMBEDDINGS", arguments.scvi_embeddings),
    ]:
        if value is not None:
            os.environ[name] = str(value.expanduser().resolve())
    from analysis import paths
    DATA = paths.DATA_DIR
    SCVI_DIR = paths.SCVI_EMBEDDINGS_DIR
    PUBLIC_SOURCE = paths.PUBLIC_SOURCE_DATA_DIR
    OUT = (arguments.output_dir or paths.RESULTS_DIR / "reference_attribution").expanduser().resolve()
    OPTIONS = arguments
    for field in ["coordinate_table", "legacy_reproduction_table", "historical_assignments"]:
        value = getattr(arguments, field)
        if value is not None:
            setattr(arguments, field, value.expanduser().resolve())
    if DATA == OUT or DATA.is_relative_to(OUT) or SCVI_DIR == OUT or SCVI_DIR.is_relative_to(OUT):
        raise ValueError("Output directory must not contain or equal an input directory")
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
        os.environ.setdefault(name, "2")
    os.environ.setdefault("NUMBA_CACHE_DIR", str(OUT / "cache/numba"))
    os.environ.setdefault("MPLCONFIGDIR", str(OUT / "cache/matplotlib"))


def load_analysis_runtime():
    global public, ad, sc, sparse, connected_components, NearestNeighbors
    import anndata as ad
    import scanpy as sc
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    from sklearn.neighbors import NearestNeighbors
    from analysis.scripts import compute_trajectory as public
    if METHODS != public.METHODS or {k: v for k, v in SCOPES.items() if k != "pooled"} != public.LINEAGE_CLUSTERS:
        raise ValueError("Public trajectory method/lineage definitions changed; review the control design")


def input_label(path: Path) -> str:
    path = path.resolve()
    for prefix, base in [("scvi-embeddings", SCVI_DIR), ("analysis-data", DATA), ("code", CODE_ROOT)]:
        if base is not None and path.is_relative_to(base.resolve()):
            return str(Path(prefix) / path.relative_to(base.resolve())).replace("\\", "/")
    return "optional-audit/" + path.name


def validate_scvi_metadata(metadata: dict, latent: np.ndarray, count_file_hash: str) -> None:
    if metadata.get("input_sha256") != count_file_hash:
        raise ValueError("scVI metadata does not match the supplied count-input file")
    if metadata.get("latent_sha256") != scvi_array_sha(latent):
        raise ValueError("scVI metadata does not match the latent shape/dtype/content hash")


def coordinate_evidence(frame, method, seed, ids, values):
    result = {"method": method, "seed": seed,
              "array_row_contract": "supplied processed-object row order",
              "coordinate_table_status": "not_available"}
    if frame is None or seed != 0:
        return result
    required = {"method", "seed", "cell_id", "x", "y"}
    if not required.issubset(frame.columns):
        raise ValueError("Coordinate audit needs method, seed, cell_id, x and y")
    linked = frame.loc[frame.method.eq(method) & frame.seed.eq(seed)]
    if linked.empty:
        result["coordinate_table_status"] = "method_not_in_table"
        return result
    ids_match = linked.cell_id.is_unique and set(linked.cell_id.astype(str)) == set(ids)
    result["coordinate_cell_id_set_exact"] = bool(ids_match)
    result["coordinate_table_status"] = "compared" if ids_match else "cell_id_mismatch"
    result["coordinate_values_match"] = False
    if ids_match:
        linked = linked.assign(cell_id=linked.cell_id.astype(str)).set_index("cell_id").loc[ids, ["x", "y"]].to_numpy()
        result["coordinate_values_match"] = bool(np.allclose(linked, values, atol=2e-5, rtol=2e-6))
    if OPTIONS.strict_audits and not result["coordinate_values_match"]:
        raise ValueError(f"Coordinate audit differs for {method}; omit strict audits for fresh fits")
    return result


def audit_legacy_table(legacy, path):
    if path is None or not path.is_file():
        return []
    old = pd.read_csv(path)
    required = {"method", "root_definition", "metric", "value"}
    if not required.issubset(old.columns):
        raise ValueError("Legacy table needs method, root_definition, metric and value")
    audit = []
    for row in old.to_dict("records"):
        new = legacy.loc[legacy.seed.eq(0) & legacy.scope.eq("pooled") & legacy.method.eq(row["method"])
                         & legacy.expression_root.eq(row["root_definition"])]
        if len(new) != 1 or row["metric"] not in new.columns:
            raise ValueError("Legacy audit entry has no unique matching computed endpoint")
        value = float(new.iloc[0][row["metric"]])
        audit.append({"method": row["method"], "root_definition": row["root_definition"],
                      "metric": row["metric"], "historical_value": row["value"],
                      "recomputed_value": value, "absolute_difference": abs(value - row["value"])})
    if OPTIONS.strict_audits and audit and max(r["absolute_difference"] for r in audit) >= 1e-6:
        raise ValueError("Legacy values differ; this optional equality audit is not a fresh-fit requirement")
    return audit


def provenance_self_checks():
    latent = np.arange(12, dtype=np.float32).reshape(6, 2)
    metadata = {"input_sha256": "count-input-fixture", "latent_sha256": scvi_array_sha(latent)}
    validate_scvi_metadata(metadata, latent, "count-input-fixture")
    for changed_metadata, values, expected in [
        ({**metadata, "input_sha256": "different"}, latent, "count-input-fixture"),
        ({**metadata, "latent_sha256": array_sha(latent)}, latent, "count-input-fixture"),
        (metadata, latent[::-1].copy(), "count-input-fixture"),
    ]:
        try:
            validate_scvi_metadata(changed_metadata, values, expected)
        except ValueError:
            pass
        else:
            raise AssertionError("Semantic provenance mismatch was accepted")


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str(array.dtype).encode() + str(array.shape).encode() + array.tobytes()).hexdigest()


def scvi_array_sha(values: np.ndarray) -> str:
    """Match the public fit_scvi.py shape/dtype/content hash, not file SHA."""
    array = np.ascontiguousarray(values)
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def csv(name: str, rows) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, index=False, float_format="%.12g")
    return frame


def corr(x, y, kind="spearman") -> float:
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.unique(x[keep]).size < 2 or np.unique(y[keep]).size < 2:
        return float("nan")
    function = spearmanr if kind == "spearman" else kendalltau
    return float(function(x[keep], y[keep]).statistic)


def pairs(indices: np.ndarray, count: int, seed: int):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(indices), count * 2)
    j = rng.integers(0, len(indices), count * 2)
    keep = i != j
    return indices[i[keep][:count]], indices[j[keep][:count]]


def agreement(x, y, mask, pair_indices) -> dict:
    keep = mask & np.isfinite(x) & np.isfinite(y)
    i, j = pair_indices
    pair_keep = keep[i] & keep[j]
    i, j = i[pair_keep], j[pair_keep]
    xx, yy = x[keep], y[keep]
    rho = corr(xx, yy)
    if len(xx) >= 3:
        rx = (rankdata(xx) - 1) / (len(xx) - 1)
        ry = (rankdata(yy) - 1) / (len(yy) - 1)
        direct = float(np.abs(rx - ry).mean())
        reverse = float(np.abs(rx - (1 - ry)).mean())
    else:
        direct = reverse = float("nan")
    return {
        "n_cells": int(keep.sum()), "n_pairs": len(i), "signed_spearman": rho,
        "kendall_tau_b": corr(xx, yy, "kendall"), "absolute_spearman": abs(rho),
        "absolute_distance_spearman": corr(np.abs(x[i] - x[j]), np.abs(y[i] - y[j])),
        "rank_mae_direct": direct, "rank_mae_reversed": reverse,
        "rank_mae_best_orientation": min(direct, reverse),
        "best_orientation": "reversed" if reverse < direct else "direct",
        "rho_negative": bool(rho < 0) if np.isfinite(rho) else None,
    }


def fit_graph(coords: np.ndarray, roots: dict, graph_id: str) -> dict:
    """Cache graph and eigensystem once; reroot only the same diffusion geometry."""
    folder = OUT / "cache/graphs"
    folder.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256((array_sha(coords) + json.dumps(DESIGN, sort_keys=True)
                          + json.dumps(roots, sort_keys=True) + version("scanpy")
                          + version("scipy")).encode()).hexdigest()
    values_path, meta_path = folder / f"{graph_id}.npz", folder / f"{graph_id}.json"
    if values_path.exists() and meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if metadata["cache_key"] == key:
            with np.load(values_path, allow_pickle=False) as saved:
                result = {name: saved[name].copy() for name in saved.files}
            print(f"cache reused {graph_id}", flush=True)
            return result
    if coords.shape[0] != 2730 or not np.isfinite(coords).all():
        raise ValueError(f"Invalid graph input {graph_id}: {coords.shape}")
    work = ad.AnnData(X=np.zeros((len(coords), 1), dtype=np.float32))
    work.obsm["X_representation"] = np.asarray(coords, dtype=np.float32)
    sc.pp.neighbors(work, n_neighbors=15, use_rep="X_representation", metric="euclidean",
                    method="umap", random_state=0, n_pcs=None)
    sc.tl.diffmap(work, n_comps=15, random_state=0)
    connectivity = work.obsp["connectivities"].tocsr()
    n_components, components = connected_components(connectivity, directed=False)
    values, raw_nonfinite = [], []
    for name in ROOT_NAMES:
        root = roots[name]
        work.uns["iroot"] = root
        sc.tl.dpt(work, n_dcs=10, n_branchings=0)
        raw = work.obs["dpt_pseudotime"].to_numpy(dtype=float).copy()
        raw_nonfinite.append(~np.isfinite(raw))
        pt = public.preserve_missing_pseudotime(raw)
        pt[components != components[root]] = np.nan
        if not np.isfinite(pt[root]) or abs(pt[root]) > 1e-6:
            raise ValueError(f"Invalid root pseudotime {graph_id} {name}")
        values.append(pt)
    result = {"pseudotime": np.asarray(values), "component": components,
              "n_components": np.asarray(n_components), "raw_nonfinite": np.asarray(raw_nonfinite),
              "diffusion_eigenvalues": work.uns["diffmap_evals"],
              "diffusion_coordinates": work.obsm["X_diffmap"]}
    np.savez_compressed(values_path, **result)
    sparse.save_npz(folder / f"{graph_id}_connectivities.npz", connectivity)
    sparse.save_npz(folder / f"{graph_id}_distances.npz", work.obsp["distances"])
    meta_path.write_text(json.dumps({"cache_key": key, "graph_id": graph_id,
                                    "coordinate_array_sha256": array_sha(coords), "roots": roots,
                                    "n_components": int(n_components)}, indent=2), encoding="utf-8")
    print(f"computed {graph_id}: components={n_components}, finite={[int(np.isfinite(t).sum()) for t in values]}", flush=True)
    return result


def validate_array(path: Path, shape: tuple) -> np.ndarray:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"Not a NumPy file: {path}")
    x = np.load(path, allow_pickle=False)
    if x.shape != shape or not np.isfinite(x).all():
        raise ValueError(f"Invalid embedding {path}: shape={x.shape}")
    return x


def self_checks():
    x = np.arange(10, dtype=float)
    p = pairs(np.arange(10), 30, PAIR_SEED)
    r = agreement(x, 9 - x, np.ones(10, bool), p)
    assert np.isclose(r["signed_spearman"], -1) and np.isclose(r["absolute_distance_spearman"], 1)
    assert np.isclose(r["rank_mae_best_orientation"], 0)
    y = x.copy()
    y[0] = np.nan
    assert agreement(x, y, np.ones(10, bool), p)["n_cells"] == 9
    assert np.isnan(corr(np.ones(10), x))


def run(seeds: list[int]):
    start = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    load_analysis_runtime()
    sc.settings.n_jobs = 2
    self_checks()
    design = {**DESIGN, "embedding_seeds": seeds, "registered_utc": datetime.now(timezone.utc).isoformat()}
    if (OUT / "analysis_design.json").exists():
        previous = json.loads((OUT / "analysis_design.json").read_text(encoding="utf-8"))
        if {k: v for k, v in previous.items() if k != "registered_utc"} == {k: v for k, v in design.items() if k != "registered_utc"}:
            design["registered_utc"] = previous["registered_utc"]
    write_json("analysis_design.json", design)
    write_json("run_status.json", {"status": "running", "seeds": seeds})
    proc_path = DATA / "analysis_objects/paul15_proc.h5ad"
    proc = sc.read_h5ad(proc_path)
    count_object = sc.read_h5ad(DATA / "analysis_objects/paul15_counts_hvg.h5ad", backed="r")
    assert np.array_equal(proc.obs_names, count_object.obs_names)
    count_object.file.close()
    ids = proc.obs_names.astype(str).to_numpy()
    assert len(ids) == 2730 and len(set(ids)) == 2730
    labels = proc.obs["paul15_clusters"].astype(str).to_numpy()
    roots = public.root_definitions(proc)
    assert list(roots) == ROOT_NAMES
    scopes = {name: np.ones(len(ids), bool) if clusters is None else np.isin(labels, clusters)
              for name, clusters in SCOPES.items()}
    pair_sets = {name: pairs(np.flatnonzero(mask), 5000 if name == "pooled" else min(3000, int(mask.sum()) * 5),
                             PAIR_SEED if name == "pooled" else PAIR_SEED + len(name))
                 for name, mask in scopes.items()}
    csv("root_definitions.csv", [{"root_definition": name, "root_index": roots[name],
                                 "root_cell_id": ids[roots[name]], "root_cluster": labels[roots[name]]}
                                for name in ROOT_NAMES])
    csv("cell_metadata.csv", {"cell_id": ids, "cell_index": np.arange(len(ids)), "cluster": labels,
                              **{f"in_{name}": mask for name, mask in scopes.items() if name != "pooled"}})
    csv("sampled_pairs.csv.gz", pd.concat([pd.DataFrame({"scope": name, "pair_index": np.arange(len(i)),
                                                         "cell_index_a": i, "cell_index_b": j,
                                                         "cell_id_a": ids[i], "cell_id_b": ids[j]})
                                            for name, (i, j) in pair_sets.items()], ignore_index=True))
    csv("lineage_definitions.csv", [{"scope": name, "clusters": ";".join(clusters),
                                     "n_cells": int(scopes[name].sum())} for name, clusters in public.LINEAGE_CLUSTERS.items()])
    inputs, input_paths = [], []

    def record(path, role, **extra):
        path = Path(path).resolve()
        if path.is_relative_to(OUT):
            raise ValueError("Choose an output directory separate from every supplied input file")
        inputs.append({"path": input_label(path), "role": role, "bytes": path.stat().st_size,
                       "sha256": sha(path), **extra})
        input_paths.append(path)

    for path, role in [(proc_path, "processed expression/PCA/cell order"),
                       (DATA / "expression_objects/paul15_expression.h5ad", "marker expression"),
                       (DATA / "analysis_objects/paul15_counts_hvg.h5ad", "scVI count-input verification"),
                       (Path(public.__file__), "public trajectory definitions")]:
        record(path, role)
    manifest = DATA / "analysis_object_manifest.csv"
    if manifest.is_file():
        record(manifest, "release or fresh-fit processed-object manifest")
    legacy_path = OPTIONS.legacy_reproduction_table
    if legacy_path is None:
        candidate = PUBLIC_SOURCE / "generated/SuppS11_root_sensitive_continuum_metrics.csv"
        legacy_path = candidate if candidate.is_file() else None
    elif not legacy_path.is_file():
        raise FileNotFoundError(legacy_path)
    old_paths = [p for p in [
        PUBLIC_SOURCE / "generated/SuppS11_dpt_root_definitions.csv",
        PUBLIC_SOURCE / "generated/SuppS11_dpt_root_concordance.csv",
        legacy_path,
        PUBLIC_SOURCE / "generated/SuppS11_lineage_restricted_metrics.csv",
        PUBLIC_SOURCE / "calibration/calibration_continuum.csv",
        PUBLIC_SOURCE / "fig4_paul15_continuum_metrics.csv",
    ] if p is not None]
    source_inventory = []
    for path in old_paths:
        if path.exists():
            record(path, "read-only previous source table")
            frame = pd.read_csv(path)
            root_column = next((name for name in ["root_definition", "root", "root_a"] if name in frame), None)
            cell_column = next((name for name in ["n_cells", "n_lineage_cells"] if name in frame), None)
            source_inventory.append({"path": input_label(path), "n_rows": len(frame),
                "n_methods": int(frame.method.nunique()) if "method" in frame else None,
                "roots": ";".join(sorted(frame[root_column].unique())) if root_column else "stored_8Mk only; identified in reference field",
                "cell_denominators": ";".join(map(str, sorted(frame[cell_column].unique()))) if cell_column else "not in table",
                "contains_per_cell_pseudotime": False,
                "role": "S15 conditional/unrestricted permutation summaries on 1000 cells" if "calibration_continuum" in path.name
                        else "S14 stored-root geometry diagnostics" if "fig4_paul15" in path.name
                        else "S11 root/lineage expression-reference calibration"})
    csv("source_table_inventory.csv", pd.DataFrame(source_inventory, columns=[
        "path", "n_rows", "n_methods", "roots", "cell_denominators",
        "contains_per_cell_pseudotime", "role"]))
    saved = None
    if OPTIONS.historical_assignments is not None:
        saved_path = OPTIONS.historical_assignments
        saved = pd.read_csv(saved_path)
        record(saved_path, "optional single-root historical per-cell audit")
    coords, graphs = {}, {}
    reference = fit_graph(np.asarray(proc.obsm["X_pca_ref"], dtype=np.float32), roots, "expression_PCA50")
    coordinate_path = OPTIONS.coordinate_table or PUBLIC_SOURCE / "fig3_paul15_embedding_coordinates.csv"
    coordinate_audit = None
    if coordinate_path.is_file():
        coordinate_audit = pd.read_csv(coordinate_path)
        record(coordinate_path, "optional cell-ID-linked seed-zero coordinate audit")
    elif OPTIONS.coordinate_table is not None:
        raise FileNotFoundError(coordinate_path)
    alignment_rows = []
    for seed in seeds:
        for method in METHODS:
            if method == "scVI":
                path = SCVI_DIR / f"paul15_scvi_dim2_seed{seed}.npy"
                meta_path = path.with_suffix(".json")
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                validate_scvi_metadata(metadata, validate_array(path, (len(ids), 2)),
                                       sha(DATA / "analysis_objects/paul15_counts_hvg.h5ad"))
                record(meta_path, "scVI fitted-array provenance", method=method, seed=seed)
            else:
                path = DATA / f"anchor_embeddings/paul15_{public.stem(method)}_seed{seed}.npy"
            x = validate_array(path, (len(ids), 2))
            record(path, "fixed evaluated representation", method=method, seed=seed, shape=list(x.shape),
                   cell_order="supplied processed Paul15 row order; exact count-object IDs checked")
            if method != "scVI":
                alignment_rows.append(coordinate_evidence(coordinate_audit, method, seed, ids, x))
            coords[method, seed] = x
            graphs[method, seed] = fit_graph(x, roots, f"{public.stem(method)}_seed{seed}")
    csv("alignment_checks.csv", alignment_rows)
    csv("input_manifest.csv", inputs)
    ref_times = reference["pseudotime"]
    reference_common = np.isfinite(ref_times).all(axis=0)
    seed_common = {seed: reference_common & np.logical_and.reduce([
        np.isfinite(graphs[method, seed]["pseudotime"]).all(axis=0) for method in METHODS]) for seed in seeds}
    global_common = np.logical_and.reduce(list(seed_common.values()))
    csv("common_cell_membership.csv.gz", pd.concat([pd.DataFrame({"cell_id": ids, "seed": seed,
          "reference_all_roots_finite": reference_common, "seed_all_roots_all_methods_finite": seed_common[seed],
          "global_all_roots_all_methods_all_seeds_finite": global_common}) for seed in seeds], ignore_index=True))
    per_cell, coverage = [], []
    for (method, seed), graph in [(('expression_PCA50', -1), reference), *graphs.items()]:
        for ri, root_name in enumerate(ROOT_NAMES):
            pt = graph["pseudotime"][ri]
            disconnected = graph["component"] != graph["component"][roots[root_name]]
            per_cell.append(pd.DataFrame({"cell_id": ids, "method": method, "seed": seed,
                                          "root_definition": root_name, "pseudotime": pt,
                                          "finite": np.isfinite(pt), "disconnected": disconnected,
                                          "component_id": graph["component"],
                                          "nonfinite_other": ~np.isfinite(pt) & ~disconnected}))
            for scope, mask in scopes.items():
                coverage.append({"method": method, "seed": seed, "root_definition": root_name, "scope": scope,
                                 "n_total": int(mask.sum()), "n_finite": int((mask & np.isfinite(pt)).sum()),
                                 "n_disconnected": int((mask & disconnected).sum()),
                                 "n_nonfinite_other": int((mask & ~np.isfinite(pt) & ~disconnected).sum()),
                                 "n_components": int(graph["n_components"]),
                                 "n_in_global_common": int((mask & global_common).sum()),
                                 "n_in_seed_common": int((mask & (reference_common if seed == -1 else seed_common[seed])).sum())})
    per_cell_frame = csv("pseudotime_per_cell.csv.gz", pd.concat(per_cell, ignore_index=True))
    coverage_frame = csv("coverage.csv", coverage)
    audit_rows = []
    if 0 in seeds and saved is not None:
        for method in METHODS:
            old = saved.loc[saved.method.eq(method)].set_index("cell_id").loc[ids]
            assert (old.root_definition == "7MEP_centroid").all()
            current = graphs[method, 0]["pseudotime"][0]
            old_values = old.method_dpt_pseudotime.to_numpy(float)
            both = np.isfinite(current) & np.isfinite(old_values)
            audit_rows.append({"method": method, "n_aligned_cells": len(ids), "old_n_finite": int(np.isfinite(old_values).sum()),
                               "new_n_finite": int(np.isfinite(current).sum()),
                               "old_finite_now_missing": int((np.isfinite(old_values) & ~np.isfinite(current)).sum()),
                               "old_terminal_now_missing": int(((old_values == 1) & ~np.isfinite(current)).sum()),
                               "signed_spearman_on_joint_finite": corr(old_values[both], current[both]),
                               "max_absolute_difference_joint_finite": float(np.max(np.abs(old_values[both] - current[both]))),
                               "reference_max_absolute_difference": float(np.max(np.abs(old.reference_dpt_pseudotime - ref_times[0]))),
                               "reuse_decision": "fresh coherent four-root graph; old single-root values audited because disconnected values were terminal-coded"})
    csv("historical_array_audit.csv", pd.DataFrame(audit_rows, columns=[
        "method", "n_aligned_cells", "old_n_finite", "new_n_finite",
        "old_finite_now_missing", "old_terminal_now_missing",
        "signed_spearman_on_joint_finite", "max_absolute_difference_joint_finite",
        "reference_max_absolute_difference", "reuse_decision"]))
    reference_rows, representation_rows, crossed_rows = [], [], []
    for scope, scope_mask in scopes.items():
        for ai, first in enumerate(ROOT_NAMES):
            for bi, second in enumerate(ROOT_NAMES):
                for mask_name, mask in [("reference_all_roots_common", reference_common),
                                        ("global_common", global_common), ("pair_finite", np.ones(len(ids), bool))]:
                    reference_rows.append({"expression_root_a": first, "expression_root_b": second,
                        "scope": scope, "mask": mask_name, "n_scope": int(scope_mask.sum()),
                        "n_coverage_changed": int((scope_mask & (np.isfinite(ref_times[ai]) != np.isfinite(ref_times[bi]))).sum()),
                        **agreement(ref_times[ai], ref_times[bi], scope_mask & mask, pair_sets[scope])})
    for (method, seed), graph in graphs.items():
        for scope, scope_mask in scopes.items():
            for ai, first in enumerate(ROOT_NAMES):
                for bi, second in enumerate(ROOT_NAMES):
                    for mask_name, mask in [("global_common", global_common), ("seed_common", seed_common[seed]),
                                            ("pair_finite", np.ones(len(ids), bool))]:
                        crossed_rows.append({"method": method, "seed": seed, "expression_root": first,
                            "representation_root": second, "scope": scope, "mask": mask_name,
                            "n_scope": int(scope_mask.sum()), "same_root": first == second,
                            **agreement(ref_times[ai], graph["pseudotime"][bi], scope_mask & mask, pair_sets[scope])})
                    if ai < bi:
                        representation_rows.append({"method": method, "seed": seed, "representation_root_a": first,
                            "representation_root_b": second, "scope": scope, "mask": "global_common",
                            "n_coverage_changed": int((scope_mask & (np.isfinite(graph["pseudotime"][ai]) != np.isfinite(graph["pseudotime"][bi]))).sum()),
                            **agreement(graph["pseudotime"][ai], graph["pseudotime"][bi], scope_mask & global_common, pair_sets[scope])})
        print(f"crossed metrics {method} seed={seed}", flush=True)
    reference_frame = csv("reference_root_concordance.csv", reference_rows)
    csv("representation_root_concordance.csv", representation_rows)
    crossed_frame = csv("crossed_root_agreement.csv", crossed_rows)
    marker_scores, marker_inputs = public.marker_scores(ids)
    csv("marker_programs.csv", marker_inputs)
    csv("marker_scores_per_cell.csv.gz", marker_scores.reset_index(names="cell_id"))
    marker_rows = []
    for (method, seed), graph in [(('expression_PCA50', -1), reference), *graphs.items()]:
        for ri, root_name in enumerate(ROOT_NAMES):
            for lineage in public.LINEAGE_CLUSTERS:
                for mask_name, base in [("global_common", global_common), ("finite_only", np.ones(len(ids), bool))]:
                    mask = scopes[lineage] & base & np.isfinite(graph["pseudotime"][ri])
                    t = graph["pseudotime"][ri][mask]
                    marker_rows.append({"method": method, "seed": seed, "root_definition": root_name,
                         "lineage": lineage, "mask": mask_name, "n_cells": int(mask.sum()),
                         "lineage_program_spearman": corr(t, marker_scores[lineage].to_numpy()[mask]),
                         "progenitor_program_spearman": corr(t, marker_scores["progenitor"].to_numpy()[mask])})
    csv("marker_consistency.csv", marker_rows)
    # Reproduce the legacy continuum endpoint separately from crossed DPT agreement.
    legacy_rows = []
    for (method, seed), x in coords.items():
        nn = NearestNeighbors(n_neighbors=15).fit(x).kneighbors(X=None, return_distance=False)
        for ri, root_name in enumerate(ROOT_NAMES):
            t = ref_times[ri]
            for scope, scope_mask in scopes.items():
                i, j = pair_sets[scope]
                keep = np.isfinite(t[i]) & np.isfinite(t[j])
                i, j = i[keep], j[keep]
                delta = np.abs(t[i] - t[j])
                rho = corr(delta, np.linalg.norm(x[i] - x[j], axis=1))
                local = float("nan")
                if scope == "pooled":
                    edges = np.isfinite(t[:, None]) & np.isfinite(t[nn])
                    local = 1 - float(np.abs(t[:, None] - t[nn])[edges].mean() / delta.mean())
                legacy_rows.append({"method": method, "seed": seed, "expression_root": root_name,
                                    "scope": scope, "n_cells": int((scope_mask & np.isfinite(t)).sum()), "n_pairs": len(i),
                                    "pseudotime_distance_correlation": rho, "local_pseudotime_retention": local,
                                    "distance_cutoff_secondary": .45, "distance_cutoff_met": rho >= .45,
                                    "local_cutoff_secondary": .50, "local_cutoff_met": local >= .50 if np.isfinite(local) else None})
    legacy = csv("legacy_geometry_continuum.csv", legacy_rows)
    audit = audit_legacy_table(legacy, legacy_path) if 0 in seeds else []
    csv("legacy_reproduction_audit.csv", pd.DataFrame(audit, columns=[
        "method", "root_definition", "metric", "historical_value", "recomputed_value", "absolute_difference"]))
    count_rows = []
    for seed in seeds:
        for root in ROOT_NAMES:
            for group, methods in [("original_eight", METHODS[:-1]), ("all_nine", METHODS)]:
                frame = legacy.loc[legacy.seed.eq(seed) & legacy.scope.eq("pooled")
                                   & legacy.expression_root.eq(root) & legacy.method.isin(methods)]
                count_rows.append({"seed": seed, "expression_root": root, "method_panel": group,
                                   "n_methods": len(frame), "n_distance_cutoff_met": int(frame.distance_cutoff_met.sum()),
                                   "n_both_cutoffs_met": int((frame.distance_cutoff_met & frame.local_cutoff_met.eq(True)).sum()),
                                   "distance_rho_min": float(frame.pseudotime_distance_correlation.min()),
                                   "distance_rho_max": float(frame.pseudotime_distance_correlation.max())})
    csv("secondary_cutoff_counts.csv", count_rows)
    contrast_rows = []
    primary = crossed_frame.loc[crossed_frame["mask"].eq("global_common")]
    for (method, seed, scope), frame in primary.groupby(["method", "seed", "scope"], sort=False):
        for metric in ["signed_spearman", "absolute_distance_spearman"]:
            matrix = frame.pivot(index="expression_root", columns="representation_root", values=metric).reindex(index=ROOT_NAMES, columns=ROOT_NAMES)
            for fixed in ROOT_NAMES:
                for first, second in combinations(ROOT_NAMES, 2):
                    contrast_rows.extend([
                        {"method": method, "seed": seed, "scope": scope, "metric": metric,
                         "varied_side": "expression", "fixed_root": fixed, "root_a": first, "root_b": second,
                         "value_a": matrix.loc[first, fixed], "value_b": matrix.loc[second, fixed],
                         "delta_b_minus_a": matrix.loc[second, fixed] - matrix.loc[first, fixed]},
                        {"method": method, "seed": seed, "scope": scope, "metric": metric,
                         "varied_side": "representation", "fixed_root": fixed, "root_a": first, "root_b": second,
                         "value_a": matrix.loc[fixed, first], "value_b": matrix.loc[fixed, second],
                         "delta_b_minus_a": matrix.loc[fixed, second] - matrix.loc[fixed, first]}])
    csv("fixed_side_contrasts.csv", contrast_rows)
    same = primary.loc[primary.same_root]
    seed_summary = same.groupby(["method", "scope", "expression_root"], as_index=False).agg(
        n_seeds=("seed", "nunique"), n_cells=("n_cells", "min"),
        signed_rho_median=("signed_spearman", "median"), signed_rho_min=("signed_spearman", "min"),
        signed_rho_max=("signed_spearman", "max"), distance_rho_median=("absolute_distance_spearman", "median"),
        distance_rho_min=("absolute_distance_spearman", "min"), distance_rho_max=("absolute_distance_spearman", "max"))
    csv("matched_root_seed_summary.csv", seed_summary)
    validation = {
        "n_per_cell_rows": len(per_cell_frame),
        "expected_per_cell_rows": (len(graphs) + 1) * 4 * len(ids),
        "n_crossed_rows": len(crossed_frame),
        "expected_crossed_rows": len(graphs) * 4 * 4 * 5 * 3,
        "n_legacy_source_rows_checked": len(audit),
        "max_legacy_source_absolute_difference": max((r["absolute_difference"] for r in audit), default=None),
        "max_historical_joint_finite_absolute_difference": max((r["max_absolute_difference_joint_finite"] for r in audit_rows), default=None),
        "all_inputs_hash_unchanged": all(sha(path) == entry["sha256"] for path, entry in zip(input_paths, inputs)),
        "historical_audit_status": "performed" if saved is not None else "skipped_optional",
        "legacy_audit_status": "performed" if audit else "skipped_optional",
        "all_primary_cell_counts_fixed_by_scope": all(
            set(primary.loc[primary.scope.eq(scope), "n_cells"]) == {int((mask & global_common).sum())}
            for scope, mask in scopes.items()),
        "all_missing_values_accounted_for": bool((coverage_frame.n_total == coverage_frame.n_finite
                                                  + coverage_frame.n_disconnected + coverage_frame.n_nonfinite_other).all()),
        "all_finite_flags_correct": bool(np.array_equal(per_cell_frame.finite, np.isfinite(per_cell_frame.pseudotime))),
        "all_disconnected_values_missing": bool(per_cell_frame.loc[per_cell_frame.disconnected, "pseudotime"].isna().all()),
        "scvi_count_input_cell_ids_exact": True,
        "self_checks": "pure-reversal distance invariance; NaN exclusions; constants remain undefined",
    }
    assert validation["n_per_cell_rows"] == validation["expected_per_cell_rows"]
    assert validation["n_crossed_rows"] == validation["expected_crossed_rows"]
    if OPTIONS.strict_audits and audit_rows:
        assert validation["max_historical_joint_finite_absolute_difference"] < 1e-12
    assert all(value for key, value in validation.items() if key.startswith("all_"))
    write_json("validation.json", validation)
    summary = {
        "status": "completed", "dataset_n_cells": len(ids), "n_methods": len(METHODS), "seeds": seeds,
        "n_fixed_representations": len(graphs), "n_graphs_including_expression": len(graphs) + 1,
        "n_roots": 4, "n_crossed_rows": len(crossed_frame), "n_global_common": int(global_common.sum()),
        "n_global_excluded": int((~global_common).sum()),
        "n_seed_common": {str(seed): int(mask.sum()) for seed, mask in seed_common.items()},
        "unique_coordinate_hashes_by_method": {method: len({array_sha(coords[method, seed]) for seed in seeds}) for method in METHODS},
        "lineage_n": {name: int(mask.sum()) for name, mask in scopes.items()},
        "global_common_lineage_n": {name: int((mask & global_common).sum()) for name, mask in scopes.items()},
        "reference_n_components": int(reference["n_components"]),
        "reference_finite_by_root": {root: int(np.isfinite(ref_times[i]).sum()) for i, root in enumerate(ROOT_NAMES)},
        "connectivity_components_by_method_seed": {f"{method}:{seed}": int(graph["n_components"]) for (method, seed), graph in graphs.items()},
        "elapsed_seconds": round(time.monotonic() - start, 3),
        "self_checks": "pure sign reversal, distance invariance, NaN exclusion and constant-vector missingness passed",
        "interpretation": "reference dependence and diagnostic sensitivity, not independent biology or definitive embedding causal failure",
    }
    write_json("summary.json", summary)
    provenance = {"design": design, "inputs": inputs, "script_sha256": sha(Path(__file__)),
                  "python": sys.version, "platform": platform.platform(),
                  "versions": {name: version(name) for name in ["numpy", "pandas", "scipy", "scanpy", "anndata", "scikit-learn", "umap-learn", "numba"]},
                  "historical_arrays": validation["historical_audit_status"],
                  "array_alignment": "Processed/count object IDs checked in exact order; scVI count-file and latent content hashes checked. Anchor arrays follow supplied processed-object row order. Optional coordinate comparisons are logged without requiring equality for fresh fits.",
                  "audit_policy": "Historical/source-table audits are optional and do not supply pseudotime to the main calculations; strict equality applies only when --strict-audits is requested.",
                  "warnings": ["No DPT branch assignment inference (n_branchings=0); root effects assess scalar ordering/distances and connectivity, not new topology.",
                               "The all-root all-method all-seed complete case population can be selective; per-cell membership and finite-only sensitivity are retained.",
                               "Original lineage masks overlap in 7MEP; marker consistency uses the same expression data and is not independent validation.",
                               "No bootstrap/p-values over cells, pair draws, seeds or roots; no biological n inferred."]}
    write_json("provenance.json", provenance)
    write_data_dictionary()
    write_json("output_manifest.json", [{"path": str(path.relative_to(OUT)), "bytes": path.stat().st_size,
                                         "sha256": sha(path)} for path in sorted(OUT.rglob("*"))
                                        if path.is_file() and path.name not in ["output_manifest.json", "run_status.json"]])
    write_json("run_status.json", {"status": "completed", "elapsed_seconds": round(time.monotonic() - start, 3)})
    print(json.dumps(summary, indent=2), flush=True)


def write_data_dictionary():
    text = """# Reference controls outputs

Saved Paul15 representations are evaluated without refitting. Correlations and
computational repeats do not establish independent biological validation.

- analysis_design.json records the declared graph, root, pair and metric definitions.
- input_manifest.csv and provenance.json record semantic input roles and file hashes.
  Paths use analysis-data, scvi-embeddings, code or optional-audit labels, not a
  private project checkout. scVI metadata uses its distinct shape/dtype/content
  digest; processed/count cell IDs must match in exact order.
- pseudotime_per_cell.csv.gz preserves every nonfinite or disconnected value as
  missing, not terminal time. Component identifiers belong to individual graphs.
- common_cell_membership.csv.gz and coverage.csv describe global, seed-specific
  and pair-finite populations. Original lineage masks overlap.
- sampled_pairs.csv.gz preserves original nonself draws, including duplicates.
- reference_root_concordance.csv describes reference-only root changes.
- crossed_root_agreement.csv records all expression-root/representation-root
  combinations under global_common, seed_common and pair_finite masks.
- signed_spearman and kendall_tau_b are orientation-sensitive. absolute_spearman
  removes rank sign only; absolute_distance_spearman compares magnitudes of
  pseudotime differences. Direct/reversed/best rank MAE uses average-tie ranks
  scaled to [0,1]. A negative correlation is not necessarily pure reversal.
- representation_root_concordance.csv changes roots within a fixed graph.
- fixed_side_contrasts.csv varies only one side of a matched comparison.
- matched_root_seed_summary.csv reports medians/minima/maxima of indexed
  computational repeats, not biological confidence intervals.
- marker_consistency.csv uses mean-log1p inherited programmes from the same data.
- legacy_geometry_continuum.csv is a separate reference-pseudotime-distance
  versus Euclidean-embedding-distance diagnostic, not crossed DPT.
- secondary_cutoff_counts.csv applies legacy screening rules only to that
  separate endpoint, never to crossed DPT or biological truth.
- historical_array_audit.csv and legacy_reproduction_audit.csv are optional
  comparisons. An absent audit is reported as skipped, not successful.
- alignment_checks.csv records optional public coordinate-table comparisons.
  Different fresh fitted coordinates need not equal the published arrays.
- validation.json, summary.json and output_manifest.json describe the actual
  completed run. A fresh run can have different finite populations and scores.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-data", type=Path, help="analysis.paths DATA_DIR override")
    parser.add_argument("--scvi-embeddings", type=Path, help="analysis.paths SCVI_EMBEDDINGS_DIR override")
    parser.add_argument("--output-dir", type=Path, help="default: analysis.paths RESULTS_DIR/reference_attribution")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--coordinate-table", type=Path, help="optional cell-ID-linked public seed-zero CSV")
    parser.add_argument("--legacy-reproduction-table", type=Path, help="optional previous S11 metric CSV")
    parser.add_argument("--historical-assignments", type=Path, help="optional single-root per-cell CSV audit")
    parser.add_argument("--strict-audits", action="store_true", help="require optional historical/coordinate equality; not a fresh-fit requirement")
    parser.add_argument("--self-test", action="store_true", help="small metric/provenance fixtures only; no data, fitting, plots or writes")
    args = parser.parse_args()
    if sorted(set(args.seeds)) != args.seeds or 0 not in args.seeds or any(seed not in range(5) for seed in args.seeds):
        parser.error("Use sorted distinct seeds from 0..4 including seed zero")
    if args.self_test:
        self_checks()
        provenance_self_checks()
        print(json.dumps({"status": "passed", "test": "metric and semantic provenance fixtures",
                          "model_or_graph_fits": 0, "files_written": 0}))
        return
    configure_paths(args)
    try:
        run(args.seeds)
    except Exception as exc:
        if OUT.exists():
            write_json("run_status.json", {"status": "failed", "exception": repr(exc)})
        raise


if __name__ == "__main__":
    main()
