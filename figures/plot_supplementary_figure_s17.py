"""Post hoc nine-method specification/empirical-profile sensitivity.

The original eight-method analyses and simulations are not overwritten.
Whole method labels, never dependent method-pair dots, are permuted.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import platform
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import sklearn
import matplotlib
from matplotlib.lines import Line2D
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from .figure_style import DATASET_COLORS, INK, TEXT_MUTED, apply_final_style, clean_axis
from .plot_figure_2 import (
    DATASET_ORDER, DATASET_LABELS, METRIC_ORDER, METHOD_ORDER,
    EXECUTION_SIGNATURE_OVERRIDES, _bh_adjust, _complete_empirical_matrix,
    _objective_signature_matrix,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_data/generated"
OUTPUT = ROOT / "outputs/supplementary_figures"
METADATA = ROOT / "metadata"
METHODS = list(METHOD_ORDER) + ["scVI"]
FIELDS = ["observation_model", "latent_parameterization", "explicit_objective_terms"]
STEM = "Supplementary_Figure_S17_specification_extension"


def encode_specifications(path: Path, methods: list[str]) -> pd.DataFrame:
    records = pd.read_csv(path).set_index("method").loc[methods]
    if records[FIELDS].isna().any().any():
        raise ValueError("Missing specification field")
    tokens = {}
    for method, row in records.iterrows():
        tokens[method] = {
            f"{field}:{token.strip()}"
            for field in FIELDS
            for token in str(EXECUTION_SIGNATURE_OVERRIDES.get(method, {}).get(field, row[field])).split(";")
            if token.strip()
        }
    features = sorted(set().union(*tokens.values()))
    return pd.DataFrame([[int(f in tokens[m]) for f in features] for m in methods], index=methods, columns=features)


def profile_matrix(frame: pd.DataFrame, methods: list[str]) -> pd.DataFrame:
    key = ["method", "dataset_id", "metric"]
    if frame.duplicated(key).any():
        raise ValueError("Duplicate method/context/diagnostic")
    expected = pd.MultiIndex.from_product([DATASET_ORDER, METRIC_ORDER], names=["dataset_id", "metric"])
    matrix = frame.pivot(index="method", columns=["dataset_id", "metric"], values="value").reindex(index=methods, columns=expected)
    if not np.isfinite(matrix.to_numpy(dtype=float)).all():
        raise ValueError("Incomplete or non-finite empirical profile; no imputation permitted")
    return matrix


def profile_distance(matrix: pd.DataFrame, already_scaled: bool = False) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    if not already_scaled:
        values = StandardScaler().fit_transform(values)
    return squareform(pdist(values) / np.sqrt(values.shape[1]))


def specification_distance(matrix: pd.DataFrame, metric: str = "jaccard") -> np.ndarray:
    return squareform(pdist(matrix.to_numpy(dtype=bool if metric == "jaccard" else float), metric=metric))


@lru_cache(maxsize=3)
def permutations(n: int) -> np.ndarray:
    return np.asarray(list(itertools.permutations(range(n))), dtype=np.int8)


def exact_mantel(x: np.ndarray, y: np.ndarray, batch_size: int = 16384) -> tuple[dict, np.ndarray]:
    if x.shape != y.shape or x.shape[0] != x.shape[1]:
        raise ValueError("Distance matrices must be matched and square")
    for matrix in (x, y):
        if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T) or not np.allclose(np.diag(matrix), 0):
            raise ValueError("Invalid distance matrix")
    n = len(x)
    upper = np.triu_indices(n, 1)
    xr = rankdata(x[upper], method="average")
    yr = rankdata(y[upper], method="average")
    xc, yc = xr - xr.mean(), yr - yr.mean()
    denominator = np.linalg.norm(xc) * np.linalg.norm(yc)
    if denominator == 0:
        raise ValueError("Constant rank-distance vector")
    observed = float(xc @ yc / denominator)
    ranks = np.zeros_like(y)
    ranks[upper], ranks[(upper[1], upper[0])] = yr, yr
    assignments = permutations(n)
    null = np.empty(len(assignments))
    # All assignments preserve the upper-triangle rank multiset and its mean.
    for start in range(0, len(assignments), batch_size):
        p = assignments[start:start + batch_size]
        samples = ranks[p[:, upper[0]], p[:, upper[1]]] - yr.mean()
        null[start:start + len(p)] = samples @ xc / denominator
    tail = int(np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12))
    lower, higher = np.quantile(null, [0.025, 0.975], method="linear")
    return {
        "rho": observed, "p_exact": tail / len(null), "tail_count": tail,
        "n_permutations": len(null), "n_methods": n, "n_dependent_pairs": math.comb(n, 2),
        "null_lower_95": float(lower), "null_upper_95": float(higher),
    }, null


def build_tables() -> dict[str, pd.DataFrame]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalogue = METADATA / "method_mathematical_specifications.csv"
    matched_path = ROOT / "data/source_data/calibration/scvi_geometry_full.csv"
    original_path = ROOT / "data/source_data/fig3_family_local_label_metrics.csv"
    # Prospective panel responsibilities are written before plotting or testing.
    questions = [
        "Which pair distances change when scVI is included?",
        "Does rescoring or method inclusion alter the pooled association?",
        "Does the association differ across contexts?",
        "Does the association depend on coding?",
        "Does a single method or rescaling drive the association?",
        "Which measured endpoint blocks affect the association?",
    ]
    components = ["pairs", "controls", "contexts", "encodings", "method_influence", "block_influence"]
    pd.DataFrame([{
        "figure": "S17", "panel": chr(97+i), "main_claim": "Empirical specification/profile association is conditional on method inclusion and analysis definition",
        "scientific_question": q, "evidence_level": "post hoc empirical sensitivity",
        "source_data": f"data/source_data/generated/s17_{components[i]}.csv",
        "analysis_code": "figures/plot_supplementary_figure_s17.py",
        "statistical_unit": "whole method labels; pair distances are dependent",
        "reviewer_risk": "Nine configurations in three fixed contexts; no causal or population-wide inference",
    } for i, q in enumerate(questions)]).to_csv(METADATA / "supplementary_figure_s17_claim_to_evidence_matrix.csv", index=False)
    full = pd.read_csv(matched_path)
    matched = full.loc[full.output_dimension.eq(2) & full.seed.eq(0)].copy()
    if len(matched) != 108 or set(matched.method) != set(METHODS):
        raise ValueError("Expected 108 seed-zero two-dimensional scores for nine methods")
    for column, expected in [("k", 15), ("reference_dimension", 50), ("sampled_pairs", 5000)]:
        if not matched[column].eq(expected).all():
            raise ValueError(f"Unmatched {column} in full-object diagnostic input")
    counts = {"pbmc3k": 2638, "paul15": 2730, "heart_cell_atlas_subsampled": 8000}
    if not matched.n_cells.eq(matched.dataset_id.map(counts)).all():
        raise ValueError("Diagnostic input does not use the complete stated objects")
    profiles = profile_matrix(matched, METHODS)
    original = _complete_empirical_matrix(pd.read_csv(original_path))
    encoding = encode_specifications(catalogue, METHODS)
    x = specification_distance(encoding)
    old_x = specification_distance(_objective_signature_matrix())
    if not np.array_equal(x[:8, :8], old_x):
        raise AssertionError("Adding scVI changes the original eight-method Jaccard distances")
    y = profile_distance(profiles)
    nulls = {}
    controls = []
    for name, xx, yy in [
        ("Original eight", old_x, profile_distance(original)),
        ("Rescored eight", x[:8, :8], profile_distance(profiles.loc[METHOD_ORDER])),
        ("Rescored nine", x, y),
    ]:
        test, null = exact_mantel(xx, yy)
        controls.append(dict(comparison=name, **test))
        nulls[name.replace(" ", "_")] = null
    contexts = [dict(context="Pooled", **controls[-1])]
    for dataset in DATASET_ORDER:
        test, null = exact_mantel(x, profile_distance(profiles.loc[:, dataset]))
        contexts.append(dict(context=DATASET_LABELS[dataset], **test))
        nulls[dataset] = null
    context_frame = pd.DataFrame(contexts).drop(columns="comparison")
    context_frame["q_bh"] = _bh_adjust(context_frame.p_exact.to_numpy())
    context_frame["multiplicity_family"] = "pooled plus three contexts (four tests)"
    variants = [
        ("Hamming", "hamming", list(encoding.columns)),
        ("Objective terms", "jaccard", [c for c in encoding if c.startswith("explicit_objective_terms:")]),
        ("Observation model", "jaccard", [c for c in encoding if c.startswith("observation_model:")]),
        ("Latent model", "jaccard", [c for c in encoding if c.startswith("latent_parameterization:")]),
        ("Observation + latent", "jaccard", [c for c in encoding if not c.startswith("explicit_objective_terms:")]),
        ("Unspecified flags omitted", "jaccard", [c for c in encoding if not c.endswith(":not_explicit")]),
    ]
    alternatives = []
    for label, metric, features in variants:
        test, _ = exact_mantel(specification_distance(encoding[features], metric), y)
        alternatives.append(dict(encoding=label, distance=metric, n_features=len(features), **test))
    encoding_frame = pd.DataFrame(alternatives)
    encoding_frame["q_bh"] = _bh_adjust(encoding_frame.p_exact.to_numpy())
    encoding_frame["multiplicity_family"] = "six alternative encodings (exploratory)"
    scaled = pd.DataFrame(StandardScaler().fit_transform(profiles), index=profiles.index, columns=profiles.columns)
    influence = []
    for i, method in enumerate(METHODS):
        keep = [j for j in range(9) if j != i]
        for label, distance in [
            ("Fixed nine-method scaling", profile_distance(scaled.drop(index=method), True)),
            ("Rescaled eight-method subset", profile_distance(profiles.drop(index=method))),
        ]:
            test, _ = exact_mantel(x[np.ix_(keep, keep)], distance)
            influence.append(dict(omitted_method=method, scaling=label, **test))
    blocks = []
    for field, values, labels in [
        (1, METRIC_ORDER, ["Local retention", "Trustworthiness", "Global ranks", "Same-label neighbours"]),
        (0, DATASET_ORDER, [DATASET_LABELS[d] for d in DATASET_ORDER]),
    ]:
        for value, label in zip(values, labels):
            keep = profiles.columns.get_level_values(field) != value
            test, _ = exact_mantel(x, profile_distance(profiles.loc[:, keep]))
            blocks.append(dict(omitted_block=label, block_type="Diagnostic" if field == 1 else "Dataset", delta_rho=test["rho"]-controls[-1]["rho"], **test))
    upper = np.triu_indices(9, 1)
    pairs = pd.DataFrame([{
        "method_1": METHODS[i], "method_2": METHODS[j],
        "includes_scVI": "scVI" in (METHODS[i], METHODS[j]),
        "specification_distance": x[i, j], "empirical_profile_distance": y[i, j],
    } for i, j in zip(*upper)])
    tables = dict(pairs=pairs, controls=pd.DataFrame(controls), contexts=context_frame,
                  encodings=encoding_frame, method_influence=pd.DataFrame(influence),
                  block_influence=pd.DataFrame(blocks), measurements=matched,
                  binary_encoding=encoding.rename_axis("method").reset_index())
    for name, frame in tables.items():
        frame.to_csv(SOURCE / f"s17_{name}.csv", index=False, float_format="%.15g")
    np.savez_compressed(SOURCE / "s17_exact_permutation_nulls.npz", **nulls)
    metadata = {
        "status": "post hoc revision sensitivity", "methods": METHODS,
        "datasets": DATASET_ORDER, "metrics": METRIC_ORDER, "n_features": encoding.shape[1],
        "original_features": 17, "new_features": [c for c in encoding if c not in _objective_signature_matrix()],
        "eight_method_jaccard_submatrix_unchanged": True,
        "profile_scaling": "StandardScaler across evaluated methods, Euclidean distance divided by sqrt(number of profile variables)",
        "matching": "full objects, output dimension 2, seed 0, k=15, PCA50, 5000 fixed valid pairs with seed 20261005",
        "exact_test": "two-sided Mantel Spearman; average ties; all method-label assignments including identity; abs tail tolerance 1e-12; no plus-one correction",
        "intervals": "central 95% exact permutation reference intervals, not confidence intervals",
        "multiplicity": ["four pooled/context tests", "six alternative encoding tests"],
        "influence_scope": "descriptive overlapping method and endpoint omissions; raw probabilities not independent confirmatory tests",
        "unchanged_scope": "original main figures, original eight-method specification analysis and six-scenario simulations",
        "hash_basis": "input_sha256 and code_sha256 use LF-normalised UTF-8 text matching canonical Git blobs; executed_input_sha256 retains actual input bytes",
        "input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in (catalogue, matched_path, original_path)},
        "executed_input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in (catalogue, matched_path, original_path)},
        "code_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in (Path(__file__), Path(__file__).with_name("plot_figure_2.py"))},
        "conditional_null": "Whole empirical profiles are exchangeably reassigned to fixed specification records of the evaluated methods; this is not a randomly sampled method population",
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (METADATA / "s17_analysis_design.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(context_frame.to_string(index=False), flush=True)
    print(encoding_frame.to_string(index=False), flush=True)
    print(f"Expanded vocabulary: {encoding.shape[1]} features; original eight distances unchanged", flush=True)
    return tables


def plot(tables: dict[str, pd.DataFrame]) -> None:
    apply_final_style()
    plt.rcParams.update({"font.size": 7.2, "axes.titlesize": 8.3, "axes.labelsize": 7.2, "xtick.labelsize": 6.6, "ytick.labelsize": 6.6})
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 8.3))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.07, top=0.955, hspace=0.76, wspace=0.66)
    primary = tables["controls"].iloc[-1].rho
    titles = ["Pairwise correspondence", "Matched extension", "Context dependence", "Encoding dependence", "Method influence", "Diagnostic dependence"]
    for letter, ax, title in zip("abcdef", axes.flat, titles):
        clean_axis(ax)
        ax.set_title(title, loc="left", pad=10)
        ax.text(-0.20, 1.09, letter, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")
    a, b, c, d, e, f = axes.flat
    pairs = tables["pairs"]
    for flag, color, marker, label in [(False, "#7C8794", "o", "Anchor pairs"), (True, "#0072B2", "D", "scVI pairs")]:
        sub = pairs[pairs.includes_scVI.eq(flag)]
        a.scatter(sub.specification_distance, sub.empirical_profile_distance, s=21, c=color, marker=marker, alpha=0.85, edgecolor="white", linewidth=0.35, label=label)
    a.set(xlabel="Specification Jaccard distance", ylabel="Empirical profile distance (RMS z)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=2, frameon=False, fontsize=6.4, handletextpad=0.4, columnspacing=0.8)
    contrasts = tables["controls"]
    for i, row in contrasts.iterrows():
        b.plot([row.null_lower_95, row.null_upper_95], [i, i], color="#B4BEC7", lw=5, solid_capstyle="butt")
        b.scatter(row.rho, i, color="#0072B2" if i == 2 else INK, s=26, zorder=3)
        b.text(0.98, i+0.22, f"P = {row.p_exact:.3f}", transform=b.get_yaxis_transform(), ha="right", fontsize=6.3)
    b.set(yticks=range(3), yticklabels=contrasts.comparison, ylim=(2.55, -0.5), xlim=(-0.72, 1.02), xlabel="Mantel Spearman correlation")
    b.axvline(0, color="#B4BEC7", lw=0.7)
    b.text(0.5, -0.32, "Grey bars: 95% permutation reference", transform=b.transAxes, ha="center", fontsize=6.3, color=TEXT_MUTED)
    context = tables["contexts"].iloc[1:]
    for i, (_, row) in enumerate(context.iterrows()):
        c.scatter(row.rho, i, color=list(DATASET_COLORS.values())[i], s=27)
        c.text(0.99, i+0.24, f"P = {row.p_exact:.3f}; q = {row.q_bh:.3f}", transform=c.get_yaxis_transform(), ha="right", fontsize=6.2)
    c.set(yticks=range(3), yticklabels=context.context, ylim=(2.55, -0.45), xlim=(-0.5, 1.0), xlabel="Mantel Spearman correlation")
    c.axvline(primary, color=INK, ls="--", lw=0.8)
    variants = tables["encodings"]
    for i, row in variants.iterrows():
        d.scatter(row.rho, i, color="#009E73", s=21)
        d.text(1.02, i, f"{row.q_bh:.3f}", transform=d.get_yaxis_transform(), va="center", fontsize=6.3)
    d.text(1.02, 1.04, "q", transform=d.transAxes, fontsize=6.6)
    d.set(yticks=range(6), yticklabels=["Hamming", "Objective terms", "Observation", "Latent model", "Observation + latent", "Flags omitted"], ylim=(5.65, -0.5), xlim=(-0.5, 0.95), xlabel="Mantel Spearman correlation")
    d.axvline(primary, color=INK, ls="--", lw=0.8)
    for label, color, marker, short in [("Fixed nine-method scaling", "#0072B2", "o", "Fixed scaling"), ("Rescaled eight-method subset", "#D55E00", "s", "Rescaled subset")]:
        sub = tables["method_influence"].query("scaling == @label").set_index("omitted_method").loc[METHODS]
        e.scatter(sub.rho, np.arange(9), c=color, marker=marker, s=19, label=short, zorder=3)
    for i, method in enumerate(METHODS):
        vals = tables["method_influence"].query("omitted_method == @method").rho.to_numpy()
        e.plot(vals, [i, i], color="#B4BEC7", lw=0.8)
    e.set(yticks=range(9), yticklabels=METHODS, ylim=(8.6, -0.5), xlim=(-0.45, 1.0), xlabel="Leave-one-method-out correlation", ylabel="Omitted method")
    e.axvline(primary, color=INK, ls="--", lw=0.8)
    e.legend(loc="upper center", bbox_to_anchor=(0.5, -0.29), ncol=2, frameon=False, fontsize=6.3, columnspacing=0.7, handletextpad=0.4)
    blocks = tables["block_influence"]
    for i, row in blocks.iterrows():
        f.plot([0, row.delta_rho], [i, i], color="#B4BEC7", lw=1.4)
        f.scatter(row.delta_rho, i, s=22, color="#009E73" if row.block_type == "Diagnostic" else "#CC79A7")
    f.set(yticks=range(7), yticklabels=["Local", "Trust", "Global", "Same-label", "PBMC3k", "Paul15", "Heart atlas"], ylim=(6.65, -0.5), xlabel="Change in Mantel correlation", ylabel="Omitted profile block")
    f.axvline(0, color=INK, lw=0.7)
    for extension in ("pdf", "svg", "png", "jpg"):
        kwargs = dict(facecolor="white", bbox_inches="tight")
        if extension in ("png", "jpg"):
            kwargs["dpi"] = 600
        fig.savefig(OUTPUT / f"{STEM}.{extension}", **kwargs)
    plt.close(fig)


def main() -> None:
    plot(build_tables())


if __name__ == "__main__":
    main()
