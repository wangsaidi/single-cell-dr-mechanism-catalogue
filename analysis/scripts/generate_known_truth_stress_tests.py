"""Generate the controlled stress tests with known generating structure.

The simulation suite is intentionally claim-specific rather than benchmark-like:
each scenario has known ground truth and answers one mechanism question.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import spearmanr
from sklearn.manifold import TSNE

from analysis import config, diagnostics
from analysis.paths import (
    ANALYSIS_ROOT,
    REPO_ROOT,
    SIMULATION_CACHE_DIR,
    SOURCE_RESULTS_DIR,
    ensure_analysis_dirs,
)
from analysis.scripts.generate_robustness_embeddings import (
    _embed_from_reference,
    _preprocess_counts_to_pca,
)


ROOT = REPO_ROOT
SOURCE_DIR = SOURCE_RESULTS_DIR
SIM_DIR = SIMULATION_CACHE_DIR

SCENARIOS = [
    "linear_low_rank",
    "nonlinear_manifold",
    "branching_trajectory",
    "dropout_stress",
    "batch_shift",
    "rare_population",
]

METHODS = config.ANCHOR_METHODS
LEGACY_METHODS = {"scScope", "SAUCIE"}
# Independent generated datasets, not repeated embeddings of one simulated matrix.
SIM_REPLICATES = list(range(5))

METRIC_THRESHOLDS = {
    "truth_local_retention": 0.30,
    "truth_trustworthiness": 0.90,
    "latent_distance_corr": 0.45,
    "label_neighbor_recall": 0.55,
    "pseudotime_distance_corr": 0.45,
    "batch_entropy_norm": 0.65,
    "rare_label_recall": 0.50,
}

SCENARIO_DEFINITIONS = {
    "linear_low_rank": "Known low-rank count model with discrete biological states and no batch shift.",
    "nonlinear_manifold": "Known curved continuum with pseudotime and smooth nonlinear latent geometry.",
    "branching_trajectory": "Known Y-shaped branching process with branch labels and pseudotime.",
    "dropout_stress": "Linear biological states with elevated zero inflation applied after count generation.",
    "batch_shift": "Discrete biological states with a technical batch shift orthogonal to biology.",
    "rare_population": "Discrete biological states with one rare population below 5% abundance.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _dense(x) -> np.ndarray:
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def _method_stem(method: str) -> str:
    return method.lower().replace("-", "").replace(" ", "_")


def _array_hash(arr: np.ndarray) -> str:
    arr = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(arr.view(np.uint8))
    return h.hexdigest()


def _label_recall(Z: np.ndarray, labels, k: int = config.N_NEIGHBORS) -> float:
    idx = diagnostics.knn_idx(Z, k)
    labels = np.asarray(labels).astype(str)
    return float(np.mean([(labels[idx[i]] == labels[i]).mean() for i in range(labels.shape[0])]))


def _target_label_recall(Z: np.ndarray, labels, target: str, k: int = config.N_NEIGHBORS) -> float:
    idx = diagnostics.knn_idx(Z, k)
    labels = np.asarray(labels).astype(str)
    mask = labels == target
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean([(labels[idx[i]] == target).mean() for i in np.where(mask)[0]]))


def _batch_entropy_norm(Z: np.ndarray, batches, k: int = config.N_NEIGHBORS) -> float:
    idx = diagnostics.knn_idx(Z, k)
    batches = np.asarray(batches).astype(str)
    levels = np.unique(batches)
    max_entropy = np.log(len(levels))
    if max_entropy == 0:
        return float("nan")
    values = []
    for i in range(batches.shape[0]):
        counts = np.array([(batches[idx[i]] == level).mean() for level in levels])
        counts = counts[counts > 0]
        values.append(float(-(counts * np.log(counts)).sum() / max_entropy))
    return float(np.mean(values))


def _distance_corr(x: np.ndarray, z: np.ndarray, seed: int) -> float:
    value = diagnostics.global_rank_corr(np.asarray(x, dtype=np.float32), np.asarray(z, dtype=np.float32), seed=seed)
    return float(max(-1.0, min(1.0, value)))


def _rank_corr_vector(x: np.ndarray, y: np.ndarray) -> float:
    value = spearmanr(np.asarray(x).ravel(), np.asarray(y).ravel()).correlation
    if value is None or np.isnan(value):
        return 0.0
    return float(value)


def _latent_to_counts(
    *,
    latent: np.ndarray,
    labels: np.ndarray,
    batch: np.ndarray,
    seed: int,
    n_genes: int = 320,
    nonlinear: bool = False,
    batch_effect_scale: float = 0.0,
    dropout_scale: float = 0.30,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.asarray(latent, dtype=np.float32)
    if nonlinear:
        features = np.column_stack(
            [
                z[:, 0],
                z[:, 1],
                z[:, 0] * z[:, 1],
                z[:, 0] ** 2,
                z[:, 1] ** 2,
                np.sin(2 * np.pi * z[:, 0]),
            ]
        )
    else:
        features = z
    loadings = rng.normal(0, 0.32, size=(features.shape[1], n_genes))
    base = rng.normal(-0.85, 0.35, size=n_genes)
    eta = base + features @ loadings

    unique_labels = [x for x in pd.unique(labels)]
    marker_width = max(8, min(18, n_genes // (len(unique_labels) * 5)))
    for i, label in enumerate(unique_labels):
        start = (i * marker_width) % n_genes
        stop = min(n_genes, start + marker_width)
        eta[np.asarray(labels) == label, start:stop] += 0.85

    batch_levels = [x for x in pd.unique(batch)]
    if batch_effect_scale > 0 and len(batch_levels) > 1:
        effect_genes = rng.choice(n_genes, size=n_genes // 3, replace=False)
        batch_sign = np.where(batch == batch_levels[0], -1.0, 1.0)[:, None]
        effect = rng.normal(batch_effect_scale, 0.05, size=(1, effect_genes.shape[0]))
        eta[:, effect_genes] += batch_sign * effect

    mu = np.exp(np.clip(eta, -5, 4))
    counts = rng.poisson(mu).astype(np.float32)
    dropout_prob = 1 / (1 + np.exp(np.log1p(mu) - 1.0))
    counts[rng.random(counts.shape) < dropout_prob * dropout_scale] = 0
    return counts.astype(np.float32)


def _simulate_scenario(scenario: str, seed: int) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    n = 720
    batch = np.repeat(["batch_1"], n).astype(object)
    pseudotime = np.full(n, np.nan, dtype=np.float32)

    if scenario in {"linear_low_rank", "dropout_stress", "batch_shift"}:
        labels = np.repeat(["state_A", "state_B", "state_C"], n // 3)
        centers = {
            "state_A": np.array([-1.8, -0.4]),
            "state_B": np.array([0.1, 1.6]),
            "state_C": np.array([1.7, -0.5]),
        }
        latent = np.vstack([centers[label] + rng.normal(0, 0.38, size=2) for label in labels]).astype(np.float32)
        if scenario == "batch_shift":
            batch = rng.choice(["batch_1", "batch_2"], size=labels.shape[0], p=[0.5, 0.5]).astype(object)
        counts = _latent_to_counts(
            latent=latent,
            labels=labels,
            batch=batch,
            seed=seed,
            batch_effect_scale=0.70 if scenario == "batch_shift" else 0.0,
            dropout_scale=0.65 if scenario == "dropout_stress" else 0.30,
        )

    elif scenario == "nonlinear_manifold":
        pseudotime = np.linspace(0, 1, n, dtype=np.float32)
        rng.shuffle(pseudotime)
        latent = np.column_stack(
            [
                np.cos(1.75 * np.pi * pseudotime),
                np.sin(1.75 * np.pi * pseudotime) + 0.45 * pseudotime,
            ]
        )
        latent += rng.normal(0, 0.045, size=latent.shape)
        labels = np.asarray(
            pd.cut(pseudotime, bins=[-0.01, 0.33, 0.66, 1.01], labels=["early", "middle", "late"]).astype(str)
        )
        counts = _latent_to_counts(latent=latent, labels=labels, batch=batch, seed=seed, nonlinear=True, dropout_scale=0.35)

    elif scenario == "branching_trajectory":
        trunk_n = 240
        branch_n = (n - trunk_n) // 2
        trunk_t = rng.uniform(0, 0.55, trunk_n)
        branch_t1 = rng.uniform(0.55, 1.0, branch_n)
        branch_t2 = rng.uniform(0.55, 1.0, n - trunk_n - branch_n)
        trunk = np.column_stack([trunk_t * 2 - 1.0, np.zeros_like(trunk_t)])
        branch1 = np.column_stack([branch_t1 * 2 - 1.0, (branch_t1 - 0.55) * 2.2])
        branch2 = np.column_stack([branch_t2 * 2 - 1.0, -(branch_t2 - 0.55) * 2.2])
        latent = np.vstack([trunk, branch1, branch2]).astype(np.float32)
        latent += rng.normal(0, 0.055, size=latent.shape)
        pseudotime = np.concatenate([trunk_t, branch_t1, branch_t2]).astype(np.float32)
        labels = np.array(["trunk"] * trunk_n + ["branch_A"] * branch_n + ["branch_B"] * (n - trunk_n - branch_n), dtype=object)
        counts = _latent_to_counts(latent=latent, labels=labels, batch=batch, seed=seed, nonlinear=True, dropout_scale=0.35)

    elif scenario == "rare_population":
        label_counts = [260, 230, 195, 35]
        labels = np.repeat(["state_A", "state_B", "state_C", "rare_state"], label_counts)
        centers = {
            "state_A": np.array([-1.8, -0.4]),
            "state_B": np.array([0.0, 1.5]),
            "state_C": np.array([1.8, -0.4]),
            "rare_state": np.array([0.2, -2.2]),
        }
        latent = np.vstack([centers[label] + rng.normal(0, 0.34 if label != "rare_state" else 0.22, size=2) for label in labels]).astype(np.float32)
        counts = _latent_to_counts(latent=latent, labels=labels, batch=batch[: labels.shape[0]], seed=seed, dropout_scale=0.38)
        batch = batch[: labels.shape[0]]
        pseudotime = pseudotime[: labels.shape[0]]

    else:
        raise KeyError(scenario)

    return {
        "counts": counts,
        "latent": np.asarray(latent, dtype=np.float32),
        "labels": np.asarray(labels).astype(str),
        "batch": np.asarray(batch).astype(str),
        "pseudotime": np.asarray(pseudotime, dtype=np.float32),
    }


def _glmpca_embedding(counts: np.ndarray, seed: int) -> np.ndarray:
    from glmpca.glmpca import glmpca

    config.set_seeds(seed)
    result = glmpca(
        np.asarray(counts, dtype=np.float32).T,
        2,
        fam="poi",
        ctl={"maxIter": 80, "eps": 1e-4},
        verbose=False,
    )
    return np.asarray(result["factors"], dtype=np.float32)


def _legacy_input_path(scenario: str, replicate: int, method: str) -> Path:
    if method == "scScope":
        return SIM_DIR / f"{scenario}_rep{replicate}_scscope_library_normalised_input.npz"
    return SIM_DIR / f"{scenario}_rep{replicate}_log1p_input.npz"


def _legacy_output_path(method: str, seed: int, tag: str) -> Path:
    return SIM_DIR / f"{_method_stem(method)}_seed{seed}_{tag}.npy"


def _load_finite_cache(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    arr = np.load(path).astype(np.float32)
    if np.isfinite(arr).all():
        return arr
    path.unlink()
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        meta_path.unlink()
    return None


def _prepare_legacy_input(
    counts: np.ndarray,
    labels: np.ndarray,
    scenario: str,
    replicate: int,
    method: str,
) -> Path:
    raw_counts = np.asarray(counts, dtype=np.float32)
    if method == "scScope":
        library_size = raw_counts.sum(axis=1)
        positive = library_size > 0
        target_library_size = float(np.median(library_size[positive]))
        X = np.zeros_like(raw_counts, dtype=np.float32)
        X[positive] = raw_counts[positive] * (target_library_size / library_size[positive, None])
        transform = (
            "per-cell library-size normalisation to the median positive library size; "
            "no log transform, following the official scScope demonstration"
        )
    else:
        X = np.log1p(raw_counts).astype(np.float32)
        target_library_size = None
        transform = "log1p(simulated count matrix)"
    cells = np.asarray([f"{scenario}_rep{replicate}_cell{i}" for i in range(X.shape[0])])
    genes = np.asarray([f"sim_gene_{j}" for j in range(X.shape[1])])
    labels = np.asarray(labels).astype(str)
    path = _legacy_input_path(scenario, replicate, method)
    meta_path = path.with_suffix(".json")
    digest = _array_hash(X)
    if path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("matrix_sha256") == digest and meta.get("shape") == list(X.shape):
                return path
        except json.JSONDecodeError:
            pass
    np.savez_compressed(path, X=X, cells=cells, genes=genes, labels=labels)
    meta_path.write_text(
        json.dumps(
            {
                "scenario": scenario,
                "replicate": replicate,
                "shape": list(X.shape),
                "matrix_sha256": digest,
                "method": method,
                "transform": transform,
                "target_library_size": target_library_size,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


def _project_scscope_latent(
    latent: np.ndarray,
    seed: int,
    scenario: str,
    replicate: int,
) -> np.ndarray:
    output = SIM_DIR / f"{scenario}_rep{replicate}_scscope_recommended_v1_tsne2.npy"
    cached = _load_finite_cache(output)
    if cached is not None:
        return cached
    coords = TSNE(
        n_components=2,
        perplexity=config.TSNE_PERPLEXITY,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=seed,
    ).fit_transform(latent).astype(np.float32)
    np.save(output, coords)
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "method": "scScope",
                "scenario": scenario,
                "replicate": replicate,
                "representation": "official-style latent representation projected to two dimensions with t-SNE",
                "latent_shape": list(latent.shape),
                "embedding_shape": list(coords.shape),
                "latent_sha256": _array_hash(latent),
                "seed": seed,
                "tsne_perplexity": config.TSNE_PERPLEXITY,
                "tsne_init": "pca",
                "tsne_learning_rate": "auto",
                "tsne_max_iter": 1000,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return coords


def _run_legacy_method(counts: np.ndarray, labels: np.ndarray, method: str, seed: int, scenario: str, replicate: int) -> np.ndarray:
    tag = f"mechsim_{scenario}_rep{replicate}_recommended_v1" if method == "scScope" else f"mechsim_{scenario}_rep{replicate}"
    out = _legacy_output_path(method, seed, tag)
    latent = _load_finite_cache(out)
    if latent is not None:
        return _project_scscope_latent(latent, seed, scenario, replicate) if method == "scScope" else latent

    configured_python = os.environ.get("SCDR_LEGACY_PYTHON", "")
    python_exe = Path(configured_python) if configured_python else Path()
    if not python_exe.exists():
        raise FileNotFoundError(
            "Set SCDR_LEGACY_PYTHON to the Python executable in the recorded Python 3.7 environment."
        )

    input_path = _prepare_legacy_input(counts, labels, scenario, replicate, method)
    runner = ANALYSIS_ROOT / "legacy_deep_runner.py"
    saucie_parent = Path(
        os.environ.get("SCDR_SAUCIE_SOURCE", ANALYSIS_ROOT / "external" / "SAUCIE")
    )
    cmd = [
        str(python_exe),
        str(runner),
        "--input",
        str(input_path),
        "--out-dir",
        str(SIM_DIR),
        "--methods",
        method,
        "--seed",
        str(seed),
        "--tag",
        tag,
        "--batch-size",
        "64",
        "--latent-dim",
        "50" if method == "scScope" else "2",
        "--scscope-epochs",
        "300" if method == "scScope" else "20",
        "--scscope-recurrence",
        "2" if method == "scScope" else "1",
        "--saucie-steps",
        "300",
        "--saucie-parent",
        str(saucie_parent),
    ]
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [
            str(python_exe.parent),
            str(python_exe.parent / "Library" / "bin"),
            str(python_exe.parent / "Library" / "mingw-w64" / "bin"),
            str(python_exe.parent / "DLLs"),
            str(python_exe.parent / "Scripts"),
            env.get("PATH", ""),
        ]
    )
    env["PYTHONPATH"] = str(saucie_parent)
    env.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    subprocess.run(cmd, check=True, env=env)
    latent = _load_finite_cache(out)
    if latent is None:
        raise FileNotFoundError(f"Legacy runner did not create {out}")
    return _project_scscope_latent(latent, seed, scenario, replicate) if method == "scScope" else latent


def _embed(counts: np.ndarray, X_ref: np.ndarray, labels: np.ndarray, method: str, seed: int, scenario: str, replicate: int) -> np.ndarray:
    if method == "GLM-PCA":
        return _glmpca_embedding(counts, seed)
    if method in LEGACY_METHODS:
        return _run_legacy_method(counts, labels, method, seed, scenario, replicate)
    return _embed_from_reference(X_ref, method, 2, seed)


def _metric_rows(
    *,
    scenario: str,
    replicate: int,
    method: str,
    seed: int,
    sim: dict[str, object],
    Z: np.ndarray,
) -> list[dict[str, object]]:
    latent = np.asarray(sim["latent"], dtype=np.float32)
    labels = np.asarray(sim["labels"]).astype(str)
    batch = np.asarray(sim["batch"]).astype(str)
    pseudotime = np.asarray(sim["pseudotime"], dtype=np.float32)
    local_mean, _ = diagnostics.local_retention(latent, Z, k=config.N_NEIGHBORS)
    base_rows = [
        ("truth_local_retention", local_mean, "truth_geometry"),
        ("truth_trustworthiness", diagnostics.trust(latent, Z, k=config.N_NEIGHBORS), "truth_geometry"),
        ("latent_distance_corr", _distance_corr(latent, Z, seed), "truth_geometry"),
        ("label_neighbor_recall", _label_recall(Z, labels), "cell_state_support"),
    ]
    if np.isfinite(pseudotime).all():
        base_rows.append(("pseudotime_distance_corr", _distance_corr(pseudotime[:, None], Z, seed), "continuum_support"))
    if len(np.unique(batch)) > 1:
        base_rows.append(("batch_entropy_norm", _batch_entropy_norm(Z, batch), "batch_biology_tradeoff"))
    if "rare_state" in set(labels):
        base_rows.append(("rare_label_recall", _target_label_recall(Z, labels, "rare_state"), "rare_state_support"))

    rows = []
    for metric, value, claim_axis in base_rows:
        threshold = METRIC_THRESHOLDS[metric]
        rows.append(
            {
                "simulation_suite": "known_truth_mechanism_stress_test",
                "scenario": scenario,
                "scenario_definition": SCENARIO_DEFINITIONS[scenario],
                "replicate": replicate,
                "method": method,
                "family": config.METHOD_FAMILY.get(method, ""),
                "metric": metric,
                "claim_axis": claim_axis,
                "value": float(value),
                "threshold": threshold,
                "support": "pass" if float(value) >= threshold else "below_threshold",
                "n_cells": int(latent.shape[0]),
                "n_unit": "one method-scenario-replicate run; cells are generated simulation units, not biological replicates",
                "ground_truth": "latent coordinates, labels, batch assignments and pseudotime are known by construction",
                "seed": seed,
            }
        )
    return rows


def _not_run_rows(scenario: str, replicate: int, method: str, seed: int, n_cells: int, reason: str) -> list[dict[str, object]]:
    return [
        {
            "simulation_suite": "known_truth_mechanism_stress_test",
            "scenario": scenario,
            "scenario_definition": SCENARIO_DEFINITIONS[scenario],
            "replicate": replicate,
            "method": method,
            "family": config.METHOD_FAMILY.get(method, ""),
            "metric": "run_status",
            "claim_axis": "method_coverage",
            "value": np.nan,
            "threshold": np.nan,
            "support": "not_run",
            "n_cells": int(n_cells),
            "n_unit": "one method-scenario-replicate run",
            "ground_truth": "latent coordinates, labels, batch assignments and pseudotime are known by construction",
            "seed": seed,
            "error": reason,
        }
    ]


def run_suite() -> tuple[Path, Path, Path, Path]:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    design_rows: list[dict[str, object]] = []

    for scenario in SCENARIOS:
        for replicate in SIM_REPLICATES:
            seed = config.SEED + replicate + 1000 * (SCENARIOS.index(scenario) + 1)
            sim = _simulate_scenario(scenario, seed)
            counts = np.asarray(sim["counts"], dtype=np.float32).copy()
            labels = np.asarray(sim["labels"]).astype(str)
            batch = np.asarray(sim["batch"]).astype(str)
            pseudotime = np.asarray(sim["pseudotime"], dtype=np.float32)
            X_ref = _preprocess_counts_to_pca(counts.copy())
            design_rows.append(
                {
                    "scenario": scenario,
                    "scenario_definition": SCENARIO_DEFINITIONS[scenario],
                    "replicate": replicate,
                    "seed": seed,
                    "n_cells": int(counts.shape[0]),
                    "n_genes": int(counts.shape[1]),
                    "n_labels": int(pd.Series(labels).nunique()),
                    "label_counts": json.dumps(pd.Series(labels).value_counts().to_dict(), sort_keys=True),
                    "batch_counts": json.dumps(pd.Series(batch).value_counts().to_dict(), sort_keys=True),
                    "has_pseudotime": bool(np.isfinite(pseudotime).all()),
                }
            )
            for method in METHODS:
                try:
                    cache_suffix = "_recommended_v1" if method == "scScope" else ""
                    cache = SIM_DIR / f"{scenario}_rep{replicate}_{method.lower().replace('-', '')}{cache_suffix}.npy"
                    Z = _load_finite_cache(cache)
                    if Z is None:
                        Z = _embed(counts, X_ref, labels, method, seed, scenario, replicate)
                        np.save(cache, np.asarray(Z, dtype=np.float32))
                    rows.extend(_metric_rows(scenario=scenario, replicate=replicate, method=method, seed=seed, sim=sim, Z=Z))
                except Exception as exc:
                    rows.extend(_not_run_rows(scenario, replicate, method, seed, counts.shape[0], f"{type(exc).__name__}: {exc}"))

    out = SOURCE_DIR / "fig6_mechanism_simulation_suite.csv"
    pd.DataFrame(rows).to_csv(out, index=False)

    design_out = SOURCE_DIR / "fig6_mechanism_simulation_design.csv"
    pd.DataFrame(design_rows).to_csv(design_out, index=False)

    metric_df = pd.DataFrame(rows)
    valid = metric_df[metric_df["support"].isin(["pass", "below_threshold"])].copy()
    summary = (
        valid.groupby(["scenario", "family", "claim_axis"], as_index=False)
        .agg(
            n_metric_rows=("support", "size"),
            fraction_pass=("support", lambda s: float((s == "pass").mean())),
            median_score=("value", "median"),
        )
    )
    summary_out = SOURCE_DIR / "fig6_mechanism_simulation_scenario_summary.csv"
    summary.to_csv(summary_out, index=False)

    coverage = (
        metric_df.groupby(["method", "family"], as_index=False)
        .agg(
            n_completed_rows=("support", lambda s: int((s != "not_run").sum())),
            n_unavailable_rows=("support", lambda s: int((s == "not_run").sum())),
            n_scenarios_completed=("scenario", lambda s: int(metric_df.loc[s.index][metric_df.loc[s.index, "support"] != "not_run"]["scenario"].nunique())),
        )
    )
    coverage["coverage_note"] = "Completed rows are explicit; unavailable rows are excluded from quantitative plotting and no values are imputed"
    coverage_out = SOURCE_DIR / "fig6_mechanism_simulation_coverage.csv"
    coverage.to_csv(coverage_out, index=False)

    meta_out = SOURCE_DIR / "fig6_mechanism_simulation_design.json"
    meta_out.write_text(
        json.dumps(
            {
                "suite": "known_truth_mechanism_stress_test",
                "purpose": "compact mechanism stress testing for review-paper claims, not method ranking",
                "scenarios": SCENARIO_DEFINITIONS,
                "methods_requested": METHODS,
                "methods_recomputed": METHODS,
                "legacy_methods": sorted(LEGACY_METHODS),
                "replicates": SIM_REPLICATES,
                "metric_thresholds": METRIC_THRESHOLDS,
                "random_seed_base": config.SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return out, design_out, summary_out, coverage_out


def main() -> None:
    ensure_analysis_dirs()
    outputs = list(run_suite())
    outputs.append(SOURCE_DIR / "fig6_mechanism_simulation_design.json")
    hash_out = SOURCE_DIR / "fig6_mechanism_simulation_suite.sha256.txt"
    hash_out.write_text("\n".join(f"{_sha256(path)}  {path.relative_to(ROOT)}" for path in outputs) + "\n", encoding="utf-8")
    for path in outputs:
        print(f"wrote {path}")
    print(f"wrote {hash_out}")


if __name__ == "__main__":
    main()
