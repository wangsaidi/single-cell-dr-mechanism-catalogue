"""Stress-test mathematical-specification to behaviour concordance.

This analysis evaluates whether the Fig. 2e Mantel result depends on feature
blocks, binary-distance definitions, the inclusion of the scScope projection
stage or any single method. It writes complete source tables and a publication-
ready supplementary figure without replacing the primary Fig. 2 source data.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures import config  # noqa: E402
from figures.figure_style import (  # noqa: E402
    BORDER,
    FAIL_COLOR,
    GRID,
    INK,
    PASS_COLOR,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    panel_label,
)
from figures.plot_figure_2 import (  # noqa: E402
    DATASET_ORDER,
    EXECUTION_SIGNATURE_OVERRIDES,
    METRIC_ORDER,
    METHOD_ORDER,
    SCENARIO_ORDER,
    SIM_METRIC_ORDER,
    _complete_empirical_matrix,
    _exact_mantel,
    _objective_signature_matrix,
    _standardised_profile_distances,
)


SOURCE = ROOT / "data" / "source_data" / "generated"
FIGURES = ROOT / "outputs" / "supplementary_figures"
METADATA = ROOT / "metadata"
BASE_SOURCE = ROOT / "data" / "source_data"
SIGNATURE_FILE = ROOT / "metadata" / "method_mathematical_specifications.csv"

FEATURE_COLUMNS = ["observation_model", "latent_parameterization", "explicit_objective_terms"]
PROFILE_LABELS = {"empirical": "Empirical profile", "simulation": "Known-truth profile"}


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = SOURCE / name
    frame.to_csv(path, index=False, float_format="%.10g")
    return path


def _encoded_matrix(overrides: dict[str, dict[str, str]] | None = None) -> pd.DataFrame:
    signatures = pd.read_csv(SIGNATURE_FILE).set_index("method").reindex(METHOD_ORDER)
    effective_overrides = overrides or EXECUTION_SIGNATURE_OVERRIDES
    method_features: dict[str, set[str]] = {}
    for method, row in signatures.iterrows():
        encoded: set[str] = set()
        for column in FEATURE_COLUMNS:
            raw = effective_overrides.get(method, {}).get(column, row[column])
            values = [value.strip() for value in str(raw).split(";") if value.strip()]
            encoded.update(f"{column}:{value}" for value in values)
        method_features[method] = encoded
    features = sorted(set().union(*method_features.values()))
    return pd.DataFrame(
        [[int(feature in method_features[method]) for feature in features] for method in METHOD_ORDER],
        index=METHOD_ORDER,
        columns=features,
        dtype=int,
    )


def _one_hot(values: list[str], prefix: str) -> pd.DataFrame:
    levels = sorted(set(values))
    return pd.DataFrame(
        [[int(value == level) for level in levels] for value in values],
        index=METHOD_ORDER,
        columns=[f"{prefix}:{level}" for level in levels],
        dtype=int,
    )


def _distance(matrix: pd.DataFrame, metric: str) -> np.ndarray:
    values = matrix.to_numpy(dtype=bool if metric == "jaccard" else float)
    return squareform(pdist(values, metric=metric))


def _profile_distances() -> dict[str, np.ndarray]:
    empirical = pd.read_csv(BASE_SOURCE / "fig3_family_local_label_metrics.csv")
    empirical_matrix = _complete_empirical_matrix(empirical)

    simulation = pd.read_csv(BASE_SOURCE / "fig6_mechanism_simulation_suite.csv")
    simulation = simulation[simulation["support"].isin(["pass", "below_threshold"])].copy()
    simulation_matrix = simulation.pivot(index="method", columns=["scenario", "replicate", "metric"], values="value")
    columns = [
        (scenario, replicate, metric)
        for scenario in SCENARIO_ORDER
        for replicate in sorted(simulation["replicate"].unique())
        for metric in SIM_METRIC_ORDER
        if (scenario, replicate, metric) in simulation_matrix.columns
    ]
    simulation_matrix = simulation_matrix.reindex(
        index=METHOD_ORDER,
        columns=pd.MultiIndex.from_tuples(columns),
    )
    if empirical_matrix.shape != (8, len(DATASET_ORDER) * len(METRIC_ORDER)):
        raise RuntimeError(f"Unexpected empirical profile shape: {empirical_matrix.shape}")
    if simulation_matrix.isna().any().any():
        raise RuntimeError("Known-truth profile contains missing values")
    return {
        "empirical": _standardised_profile_distances(empirical_matrix),
        "simulation": _standardised_profile_distances(simulation_matrix),
    }


def build_tables() -> dict[str, Path]:
    SOURCE.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)

    full = _objective_signature_matrix()
    if not full.equals(_encoded_matrix()):
        raise AssertionError("Primary mathematical-specification matrix and sensitivity encoding disagree")
    legacy_overrides = {
        "SAUCIE": {"explicit_objective_terms": "reconstruction"},
    }
    legacy_scscope = _encoded_matrix(legacy_overrides)
    signature_source = pd.read_csv(SIGNATURE_FILE).set_index("method").reindex(METHOD_ORDER)
    navigation = _one_hot(signature_source["navigation_group"].astype(str).tolist(), "navigation_group")
    execution_inputs = _one_hot(
        [
            "scaled_log_expression",
            "integer_counts",
            "library_normalised_counts",
            "log_counts",
            "pca50",
            "pca50",
            "pca50",
            "pca50",
        ],
        "fitting_input",
    )

    objective_columns = [column for column in full.columns if column.startswith("explicit_objective_terms:")]
    observation_columns = [column for column in full.columns if column.startswith("observation_model:")]
    parameter_columns = [column for column in full.columns if column.startswith("latent_parameterization:")]
    no_not_explicit_columns = [column for column in full.columns if not column.endswith(":not_explicit")]
    variants: list[tuple[str, str, pd.DataFrame]] = [
        ("full_pipeline_jaccard", "jaccard", full),
        ("full_pipeline_hamming", "hamming", full),
        ("objective_terms_only", "jaccard", full[objective_columns]),
        ("observation_model_only", "jaccard", full[observation_columns]),
        ("latent_parameterisation_only", "jaccard", full[parameter_columns]),
        ("observation_plus_parameterisation", "jaccard", full[observation_columns + parameter_columns]),
        ("without_not_explicit", "jaccard", full[no_not_explicit_columns]),
        ("legacy_scscope_stage_omitted", "jaccard", legacy_scscope),
        ("execution_input_baseline", "jaccard", execution_inputs),
        ("navigation_group_baseline", "jaccard", navigation),
    ]

    profiles = _profile_distances()
    permutations = np.asarray(list(itertools.permutations(range(len(METHOD_ORDER)))), dtype=np.int16)
    sensitivity_rows: list[dict[str, object]] = []
    distance_cache: dict[str, np.ndarray] = {}
    for variant, metric, matrix in variants:
        distance = _distance(matrix, metric)
        distance_cache[variant] = distance
        unique_signatures = int(np.unique(matrix.to_numpy(dtype=int), axis=0).shape[0])
        triangle = distance[np.triu_indices(len(METHOD_ORDER), 1)]
        saturation = float(np.mean(np.isclose(triangle, triangle.max())))
        for profile, profile_distance in profiles.items():
            rho, p_value = _exact_mantel(distance, profile_distance, permutations)
            sensitivity_rows.append(
                {
                    "signature_definition": variant,
                    "binary_distance": metric,
                    "profile": profile,
                    "mantel_spearman_rho": rho,
                    "p_exact_two_sided": p_value,
                    "n_methods": len(METHOD_ORDER),
                    "n_method_pairs": 28,
                    "n_label_permutations": len(permutations),
                    "n_binary_features": int(matrix.shape[1]),
                    "n_unique_signatures": unique_signatures,
                    "fraction_pairs_at_maximum_distance": saturation,
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    loo_rows: list[dict[str, object]] = []
    full_distance = distance_cache["full_pipeline_jaccard"]
    for omitted_index, omitted_method in enumerate(METHOD_ORDER):
        keep = np.array([index for index in range(len(METHOD_ORDER)) if index != omitted_index])
        subset_permutations = np.asarray(list(itertools.permutations(range(len(keep)))), dtype=np.int16)
        signature_distance = full_distance[np.ix_(keep, keep)]
        for profile, profile_distance in profiles.items():
            rho, p_value = _exact_mantel(
                signature_distance,
                profile_distance[np.ix_(keep, keep)],
                subset_permutations,
            )
            loo_rows.append(
                {
                    "omitted_method": omitted_method,
                    "profile": profile,
                    "mantel_spearman_rho": rho,
                    "p_exact_two_sided": p_value,
                    "n_methods": len(keep),
                    "n_method_pairs": len(keep) * (len(keep) - 1) // 2,
                    "n_label_permutations": len(subset_permutations),
                }
            )
    loo = pd.DataFrame(loo_rows)

    deletion_rows: list[dict[str, object]] = []
    baseline = sensitivity[sensitivity["signature_definition"].eq("full_pipeline_jaccard")].set_index("profile")
    for feature in full.columns:
        reduced = full.drop(columns=feature)
        distance = _distance(reduced, "jaccard")
        for profile, profile_distance in profiles.items():
            rho, p_value = _exact_mantel(distance, profile_distance, permutations)
            deletion_rows.append(
                {
                    "deleted_feature": feature,
                    "feature_group": feature.split(":", 1)[0],
                    "profile": profile,
                    "mantel_spearman_rho": rho,
                    "delta_rho_from_full": rho - float(baseline.loc[profile, "mantel_spearman_rho"]),
                    "p_exact_two_sided": p_value,
                    "n_remaining_features": int(reduced.shape[1]),
                }
            )
    deletion = pd.DataFrame(deletion_rows)

    triangle = np.triu_indices(len(METHOD_ORDER), 1)
    pair_rows = []
    for i, j in zip(*triangle):
        pair_rows.append(
            {
                "method_a": METHOD_ORDER[i],
                "method_b": METHOD_ORDER[j],
                "jaccard_distance": distance_cache["full_pipeline_jaccard"][i, j],
                "hamming_distance": distance_cache["full_pipeline_hamming"][i, j],
                "empirical_profile_distance": profiles["empirical"][i, j],
                "simulation_profile_distance": profiles["simulation"][i, j],
            }
        )
    pairwise = pd.DataFrame(pair_rows)

    paths = {
        "sensitivity": _write_csv(sensitivity, "SuppS9_signature_definition_sensitivity.csv"),
        "loo": _write_csv(loo, "SuppS9_leave_one_method_out.csv"),
        "deletion": _write_csv(deletion, "SuppS9_leave_one_feature_out.csv"),
        "pairwise": _write_csv(pairwise, "SuppS9_pairwise_distance_sensitivity.csv"),
    }
    return paths


def _save(fig: plt.Figure) -> None:
    stem = FIGURES / "Supplementary_Figure_S9_signature_sensitivity"
    for extension in ("pdf", "svg", "png", "jpg"):
        kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
        if extension in {"png", "jpg"}:
            kwargs["dpi"] = 600
        if extension == "jpg":
            kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
        fig.savefig(stem.with_suffix(f".{extension}"), **kwargs)
    plt.close(fig)


def build_figure(paths: dict[str, Path]) -> None:
    apply_final_style()
    sensitivity = pd.read_csv(paths["sensitivity"])
    loo = pd.read_csv(paths["loo"])
    deletion = pd.read_csv(paths["deletion"])
    pairwise = pd.read_csv(paths["pairwise"])

    variant_order = [
        "full_pipeline_jaccard",
        "full_pipeline_hamming",
        "objective_terms_only",
        "observation_model_only",
        "latent_parameterisation_only",
        "observation_plus_parameterisation",
        "without_not_explicit",
        "legacy_scscope_stage_omitted",
        "execution_input_baseline",
        "navigation_group_baseline",
    ]
    variant_labels = {
        "full_pipeline_jaccard": "Full pipeline\nJaccard",
        "full_pipeline_hamming": "Full pipeline\nHamming",
        "objective_terms_only": "Objective\nterms",
        "observation_model_only": "Observation\nmodel",
        "latent_parameterisation_only": "Latent\nparameterisation",
        "observation_plus_parameterisation": "Observation +\nparameterisation",
        "without_not_explicit": "Without\nnot-explicit",
        "legacy_scscope_stage_omitted": "scScope projection\nomitted",
        "execution_input_baseline": "Input\nbaseline",
        "navigation_group_baseline": "Navigation-group\nbaseline",
    }
    profile_colors = {"empirical": "#2C7FB8", "simulation": "#D95F0E"}

    fig = plt.figure(figsize=(7.3, 8.2))
    grid = fig.add_gridspec(3, 2, hspace=0.72, wspace=0.42)

    ax = fig.add_subplot(grid[0, 0])
    for profile, offset in [("empirical", -0.10), ("simulation", 0.10)]:
        sub = sensitivity[sensitivity["profile"].eq(profile)].set_index("signature_definition").reindex(variant_order)
        x = np.arange(len(variant_order)) + offset
        ax.scatter(x, sub["mantel_spearman_rho"], s=24, color=profile_colors[profile], label=PROFILE_LABELS[profile], zorder=3)
    ax.axhline(0, color=GRID, lw=0.8)
    ax.set_xticks(np.arange(len(variant_order)))
    ax.set_xticklabels([variant_labels[value] for value in variant_order], rotation=55, ha="right")
    ax.set_ylabel("Mantel Spearman rho")
    ax.set_title("Concordance depends on specification encoding", loc="left")
    ax.legend(frameon=False, fontsize=5.4, loc="lower left")
    clean_axis(ax, grid=True)
    panel_label(ax, "a")

    ax = fig.add_subplot(grid[0, 1])
    for profile, offset in [("empirical", -0.10), ("simulation", 0.10)]:
        sub = sensitivity[sensitivity["profile"].eq(profile)].set_index("signature_definition").reindex(variant_order)
        values = -np.log10(np.maximum(sub["p_exact_two_sided"].to_numpy(dtype=float), 1 / 40320))
        ax.scatter(np.arange(len(variant_order)) + offset, values, s=24, color=profile_colors[profile], zorder=3)
    ax.axhline(-np.log10(0.05), color=FAIL_COLOR, lw=0.8, ls="--", label="P = 0.05")
    ax.set_xticks(np.arange(len(variant_order)))
    ax.set_xticklabels([variant_labels[value] for value in variant_order], rotation=55, ha="right")
    ax.set_ylabel("-log10 exact P")
    ax.set_title("Exact evidence is encoding-sensitive", loc="left")
    ax.legend(frameon=False, fontsize=5.4, loc="upper right")
    clean_axis(ax, grid=True)
    panel_label(ax, "b")

    for column, profile in enumerate(["empirical", "simulation"]):
        ax = fig.add_subplot(grid[1, column])
        sub = loo[loo["profile"].eq(profile)].set_index("omitted_method").reindex(METHOD_ORDER)
        colors = [mpl.colors.to_hex(mpl.colormaps["tab10"](i)) for i in range(len(METHOD_ORDER))]
        ax.scatter(sub["mantel_spearman_rho"], np.arange(len(METHOD_ORDER)), c=colors, s=27, zorder=3)
        ax.axvline(0, color=GRID, lw=0.8)
        ax.set_yticks(np.arange(len(METHOD_ORDER)))
        ax.set_yticklabels(METHOD_ORDER)
        ax.invert_yaxis()
        ax.set_xlabel("Leave-one-method-out rho")
        ax.set_title(f"{PROFILE_LABELS[profile]} sensitivity", loc="left")
        clean_axis(ax, grid=True)
        panel_label(ax, "c" if column == 0 else "d")

    ax = fig.add_subplot(grid[2, 0])
    ordered_features = (
        deletion.groupby("deleted_feature")["delta_rho_from_full"].apply(lambda s: float(np.max(np.abs(s))))
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )
    for profile, offset in [("empirical", -0.10), ("simulation", 0.10)]:
        sub = deletion[deletion["profile"].eq(profile)].set_index("deleted_feature").reindex(ordered_features)
        ax.scatter(sub["delta_rho_from_full"], np.arange(len(ordered_features)) + offset, s=22, color=profile_colors[profile])
    ax.axvline(0, color=GRID, lw=0.8)
    ax.set_yticks(np.arange(len(ordered_features)))
    ax.set_yticklabels([feature.split(":", 1)[1].replace("_", " ") for feature in ordered_features])
    ax.invert_yaxis()
    ax.set_xlabel("Change in Mantel rho after deletion")
    ax.set_title("Influential encoded features", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "e")

    ax = fig.add_subplot(grid[2, 1])
    ax.scatter(pairwise["jaccard_distance"], pairwise["hamming_distance"], s=24, color="#4C78A8", alpha=0.85, edgecolor="white", linewidth=0.35)
    rho = pd.Series(pairwise["jaccard_distance"]).corr(pd.Series(pairwise["hamming_distance"]), method="spearman")
    ax.text(0.04, 0.94, f"pairwise rho = {rho:.2f}", transform=ax.transAxes, va="top", fontsize=5.8, color=INK)
    ax.set_xlabel("Jaccard specification distance")
    ax.set_ylabel("Hamming specification distance")
    ax.set_title("Binary distances weight sparsity differently", loc="left")
    clean_axis(ax, grid=True)
    panel_label(ax, "f")

    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.08)
    _save(fig)


def main() -> None:
    paths = {
        "sensitivity": SOURCE / "SuppS9_signature_definition_sensitivity.csv",
        "loo": SOURCE / "SuppS9_leave_one_method_out.csv",
        "deletion": SOURCE / "SuppS9_leave_one_feature_out.csv",
        "pairwise": SOURCE / "SuppS9_pairwise_distance_sensitivity.csv",
    }
    build_figure(paths)

if __name__ == "__main__":
    main()
