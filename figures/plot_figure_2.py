"""Build Figure 2 from encoded method specifications and continuous measurements.

The figure deliberately avoids support-count heatmaps. Each panel answers a
distinct question with run-level measurements, multivariate structure, exact
rank tests, variance decomposition, or controlled known-truth evidence.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config, style


ROOT = Path(__file__).resolve().parents[1]
INPUT_SOURCE_DIR = ROOT / "data" / "source_data"
SIGNATURE_FILE = ROOT / "metadata" / "method_mathematical_specifications.csv"
SOURCE_DIR = INPUT_SOURCE_DIR / "generated"
OUTPUT_DIR = ROOT / "outputs" / "main_figures"
METADATA_DIR = ROOT / "metadata"

METHOD_ORDER = config.ANCHOR_METHODS
EXECUTION_SIGNATURE_OVERRIDES = {
    # The reported scScope map is a two-stage pipeline. The recurrent
    # autoencoder produces a 50-dimensional latent code, which is then
    # projected with t-SNE. Both stages influence the displayed coordinates.
    "scScope": {
        "latent_parameterization": "recurrent_neural_encoder_decoder;directly_optimized_coordinates",
        "explicit_objective_terms": "reconstruction;zero_model_or_imputation;pairwise_probability",
    },
    # The evaluated SAUCIE constructor used its zero-valued defaults for the
    # optional MMD, information-dimension and intracluster penalties.  Retain
    # the published full model in the 26-method catalogue, but compare observed
    # behaviour with the objective terms that were active in this execution.
    "SAUCIE": {"explicit_objective_terms": "reconstruction"},
}
DATASET_ORDER = ["pbmc3k", "paul15", "heart_cell_atlas_subsampled"]
DATASET_LABELS = {
    "pbmc3k": "PBMC3k",
    "paul15": "Paul15",
    "heart_cell_atlas_subsampled": "Heart atlas",
}
METRIC_ORDER = ["local_retention", "trustworthiness", "global_rank_corr", "label_neighbor_recall"]
METRIC_LABELS = {
    "local_retention": "Local",
    "trustworthiness": "Trust",
    "global_rank_corr": "Global",
    "label_neighbor_recall": "Same-label",
}
METRIC_THRESHOLDS = {
    "local_retention": config.SUPPORT_THRESHOLDS["local_retention"],
    "trustworthiness": config.SUPPORT_THRESHOLDS["trustworthiness"],
    "global_rank_corr": config.SUPPORT_THRESHOLDS["global_rank_corr"],
    "label_neighbor_recall": config.SUPPORT_THRESHOLDS["label_recall"],
}
SCENARIO_ORDER = [
    "linear_low_rank",
    "nonlinear_manifold",
    "branching_trajectory",
    "dropout_stress",
    "batch_shift",
    "rare_population",
]
SCENARIO_LABELS = {
    "linear_low_rank": "Linear",
    "nonlinear_manifold": "Nonlinear",
    "branching_trajectory": "Branch",
    "dropout_stress": "Dropout",
    "batch_shift": "Batch",
    "rare_population": "Rare",
}
SIM_METRIC_ORDER = [
    "truth_local_retention",
    "truth_trustworthiness",
    "latent_distance_corr",
    "label_neighbor_recall",
    "pseudotime_distance_corr",
    "batch_entropy_norm",
    "rare_label_recall",
]
SIM_METRIC_LABELS = {
    "truth_local_retention": "Local",
    "truth_trustworthiness": "Trust",
    "latent_distance_corr": "Geometry",
    "label_neighbor_recall": "Same-label",
    "pseudotime_distance_corr": "Pseudotime",
    "batch_entropy_norm": "Batch mix",
    "rare_label_recall": "Rare-state",
}
SCENARIO_MARKERS = {
    "linear_low_rank": "o",
    "nonlinear_manifold": "s",
    "branching_trajectory": "^",
    "dropout_stress": "D",
    "batch_shift": "P",
    "rare_population": "X",
}


def _apply_style() -> None:
    style.apply_style()
    mpl.rcParams.update(
        {
            "font.size": 6.8,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.6,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.7,
            "legend.fontsize": 5.1,
            "axes.titlepad": 4,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _panel_label(ax: plt.Axes, letter: str, x: float = -0.08, y: float = 1.10) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, ha="left", va="top", fontsize=9, fontweight="bold")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    path = SOURCE_DIR / name
    df.to_csv(path, index=False, float_format="%.10g")
    return path


def _write_claim_matrix() -> Path:
    main_claim = (
        "The 17-feature pipeline record is associated with known-truth, but not empirical, profile distance "
        "under the primary coding."
    )
    claim_type = "empirical and known-truth quantitative analysis"
    code = "figures/plot_figure_2.py"
    columns = [
        "figure_id",
        "figure_main_claim",
        "figure_claim_type",
        "panel_id",
        "panel_scientific_question",
        "data_source_class",
        "evidence_role",
        "counts_toward_evidence_panel_target",
        "source_data_file",
        "plotting_code_path",
        "statistical_test_or_model",
        "n_definition",
        "reviewer_risk",
    ]

    def row(
        panel: str,
        question: str,
        data_class: str,
        role: str,
        source: str,
        analysis: str,
        n_definition: str,
        risk: str,
    ) -> list[object]:
        return ["Fig2", main_claim, claim_type, panel, question, data_class, role, True, source, code, analysis, n_definition, risk]

    rows = [
        row(
            "a",
            "Which empirical structures are preserved by each method in each biological context, and by what margin relative to the fixed operational boundary?",
            "empirical diagnostic measurements",
            "primary continuous evidence",
            "data/source_data/generated/Fig2a_empirical_threshold_margins.csv",
            "Signed relative boundary margin, (observed value minus boundary) divided by boundary; no inferential test.",
            "n = 24 method-context runs, each contributing four diagnostics; cells contribute to each run-level metric but are not biological replicates.",
            "Boundaries are study-level diagnostic rules rather than universal standards; raw values and boundaries remain available in source data.",
        ),
        row(
            "b",
            "Which combinations of empirical endpoints separate the eight method profiles?",
            "multivariate empirical profile analysis",
            "orthogonal multivariate synthesis",
            "data/source_data/generated/Fig2b_empirical_profile_pca_scores.csv; data/source_data/generated/Fig2b_empirical_profile_pca_loadings.csv",
            "Principal-component analysis of 12 z-standardised context-by-diagnostic variables; descriptive only.",
            "n = 8 method profiles, each defined by 12 empirical context-by-diagnostic values.",
            "The focused method panel permits descriptive ordination but not population-level inference or universal method ranking.",
        ),
        row(
            "c",
            "Are method rankings for each preservation endpoint stable across biological contexts?",
            "empirical rank analysis",
            "context-transfer test",
            "data/source_data/generated/Fig2c_context_rank_concordance.csv",
            "Spearman correlation across eight methods; exact two-sided P values over 8! permutations with Benjamini-Hochberg correction across 12 tests.",
            "n = 8 paired method values for each of 12 endpoint-by-context-pair comparisons.",
            "Power is limited by eight methods; effect sizes are primary and adjusted P values are secondary.",
        ),
        row(
            "d",
            "How much observed variation in each endpoint is attributable to method, biological context or unresolved method-by-context structure?",
            "descriptive variance decomposition",
            "context-dependence quantification",
            "data/source_data/generated/Fig2d_empirical_variance_partition.csv",
            "Two-way additive sums-of-squares decomposition into method, context and residual fractions; no inferential P value.",
            "n = 24 method-context values per diagnostic endpoint.",
            "The residual combines method-by-context interaction and run-level variation because there is one result per method-context cell.",
        ),
        row(
            "e",
            "Are distances between 17-feature pipeline records concordant with empirical and known-truth profile distances?",
            "specification, empirical and simulation distance analysis",
            "exact distance-matrix concordance test",
            "data/source_data/generated/Fig2e_objective_signature_matrix.csv; data/source_data/generated/Fig2e_profile_distance_concordance.csv; data/source_data/generated/Fig2e_mantel_summary.csv",
            "Jaccard distance for 17 binary pipeline features and standardised Euclidean profile distances; exact two-sided Mantel tests over all 8! method-label permutations. The two specification-profile tests are Benjamini-Hochberg adjusted.",
            "n = 8 methods and 28 unique method pairs; inference uses 40,320 permutations of each complete distance matrix.",
            "Power is limited by eight methods, and the signatures require independent primary-source coding review; pairwise distances are not treated as independent observations.",
        ),
        row(
            "f",
            "Which known structures are retained by each method across six controlled simulation scenarios?",
            "controlled known-truth simulation",
            "replicated known-truth stress test",
            "data/source_data/generated/Fig2f_simulation_boundary_margins.csv",
            "Mean signed relative boundary margin across five independently generated datasets for each valid method-scenario-metric combination; no values are imputed.",
            "n = 224 method-scenario-metric summaries from 240 complete method-scenario-generated-dataset runs and 1,120 metric rows.",
            "Generated datasets are controlled simulation replicates rather than biological replicates; conclusions are restricted to the six stated generative scenarios.",
        ),
        row(
            "g",
            "Can same-label recovery remain high while known local geometry is poorly retained?",
            "controlled known-truth simulation",
            "geometry-identity dissociation test",
            "data/source_data/generated/Fig2g_geometry_identity_decoupling.csv",
            "Truth-local-retention margin versus same-label-neighbour margin; Spearman rho is descriptive because rows share methods and scenarios.",
            "n = 240 method-scenario-generated-dataset runs, comprising eight methods, six scenarios and five independently generated datasets per scenario.",
            "Rows share methods and generative scenarios; no naive point-level P value is reported.",
        ),
    ]
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    path = METADATA_DIR / "claim_to_evidence_matrix_Fig2.tsv"
    pd.DataFrame(rows, columns=columns).to_csv(path, sep="\t", index=False)
    return path


def _save_figure(fig: plt.Figure) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for stem in {"Figure_2", "Figure_2"}:
        for ext in ["pdf", "svg", "png", "jpg"]:
            kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
            if ext in {"png", "jpg"}:
                kwargs["dpi"] = 600
            fig.savefig(OUTPUT_DIR / f"{stem}.{ext}", **kwargs)
    plt.close(fig)


def _relative_margin(value: pd.Series, threshold: pd.Series) -> pd.Series:
    return (value.astype(float) - threshold.astype(float)) / threshold.astype(float)


def _complete_empirical_matrix(metrics: pd.DataFrame) -> pd.DataFrame:
    matrix = metrics.pivot(index="method", columns=["dataset_id", "metric"], values="value")
    expected = pd.MultiIndex.from_product([DATASET_ORDER, METRIC_ORDER], names=["dataset_id", "metric"])
    matrix = matrix.reindex(index=METHOD_ORDER, columns=expected)
    if matrix.isna().any().any():
        raise ValueError("Empirical profile matrix is incomplete; Figure 2 cannot be generated.")
    return matrix.astype(float)


def _orient_pca(scores: np.ndarray, components: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = scores.copy()
    components = components.copy()
    for component in range(components.shape[0]):
        anchor = int(np.argmax(np.abs(components[component])))
        if components[component, anchor] < 0:
            scores[:, component] *= -1
            components[component] *= -1
    return scores, components


def _bh_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 1.0
    m = len(p_values)
    for reverse_rank in range(m - 1, -1, -1):
        idx = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, p_values[idx] * m / rank)
        adjusted[idx] = min(1.0, running)
    return adjusted


def _exact_spearman(x: np.ndarray, y: np.ndarray, permutations: np.ndarray) -> tuple[float, float]:
    x_rank = rankdata(np.asarray(x, dtype=float))
    y_rank = rankdata(np.asarray(y, dtype=float))
    x_center = x_rank - x_rank.mean()
    y_center = y_rank - y_rank.mean()
    denominator = float(np.sqrt(np.sum(x_center**2) * np.sum(y_center**2)))
    if denominator == 0:
        return float("nan"), float("nan")
    observed = float(np.sum(x_center * y_center) / denominator)
    permuted_y = y_rank[permutations]
    permuted_y = permuted_y - permuted_y.mean(axis=1, keepdims=True)
    null = (permuted_y @ x_center) / denominator
    p_exact = float(np.mean(np.abs(null) >= abs(observed) - 1e-12))
    return observed, p_exact


def _variance_partition(values: pd.DataFrame, metric: str) -> list[dict[str, object]]:
    sub = values.loc[values["metric"] == metric, ["method", "dataset_id", "value"]].copy()
    matrix = sub.pivot(index="method", columns="dataset_id", values="value").reindex(
        index=METHOD_ORDER, columns=DATASET_ORDER
    )
    if matrix.isna().any().any():
        raise ValueError(f"Incomplete variance-partition matrix for {metric}")
    arr = matrix.to_numpy(dtype=float)
    grand = float(arr.mean())
    ss_total = float(np.sum((arr - grand) ** 2))
    ss_method = float(arr.shape[1] * np.sum((arr.mean(axis=1) - grand) ** 2))
    ss_context = float(arr.shape[0] * np.sum((arr.mean(axis=0) - grand) ** 2))
    ss_residual = max(0.0, ss_total - ss_method - ss_context)
    return [
        {
            "metric": metric,
            "source": source,
            "sum_squares": ss,
            "fraction_total_variation": ss / ss_total if ss_total > 0 else np.nan,
            "n_method_context_values": int(arr.size),
            "residual_definition": "method-by-context structure plus run-level residual; one run per method-context cell",
        }
        for source, ss in [("method", ss_method), ("biological_context", ss_context), ("residual", ss_residual)]
    ]


def _standardised_profile_distances(matrix: pd.DataFrame) -> np.ndarray:
    standardised = StandardScaler().fit_transform(matrix.to_numpy(dtype=float))
    return squareform(pdist(standardised, metric="euclidean") / np.sqrt(standardised.shape[1]))


def _objective_signature_matrix() -> pd.DataFrame:
    signatures = pd.read_csv(SIGNATURE_FILE).set_index("method").reindex(METHOD_ORDER)
    if signatures.isna().all(axis=1).any():
        missing = signatures.index[signatures.isna().all(axis=1)].tolist()
        raise ValueError(f"Objective signatures are missing for: {missing}")
    feature_columns = [
        "observation_model",
        "latent_parameterization",
        "explicit_objective_terms",
    ]
    method_features: dict[str, set[str]] = {}
    for method, row in signatures.iterrows():
        encoded: set[str] = set()
        for column in feature_columns:
            raw_value = EXECUTION_SIGNATURE_OVERRIDES.get(method, {}).get(column, row[column])
            values = [value.strip() for value in str(raw_value).split(";") if value.strip()]
            encoded.update(f"{column}:{value}" for value in values)
        method_features[method] = encoded
    feature_order = sorted(set().union(*method_features.values()))
    matrix = pd.DataFrame(
        [[int(feature in method_features[method]) for feature in feature_order] for method in METHOD_ORDER],
        index=METHOD_ORDER,
        columns=feature_order,
        dtype=int,
    )
    if matrix.shape[1] != 17:
        raise ValueError(f"Expected 17 binary pipeline features, found {matrix.shape[1]}")
    return matrix


def _exact_mantel(x_dist: np.ndarray, y_dist: np.ndarray, permutations: np.ndarray) -> tuple[float, float]:
    triangle = np.triu_indices(x_dist.shape[0], 1)
    x_rank = rankdata(x_dist[triangle])
    y_rank_values = rankdata(y_dist[triangle])
    y_rank_matrix = np.zeros_like(y_dist, dtype=float)
    y_rank_matrix[triangle] = y_rank_values
    y_rank_matrix[(triangle[1], triangle[0])] = y_rank_values
    x_center = x_rank - x_rank.mean()
    denominator_x = float(np.sqrt(np.sum(x_center**2)))
    observed_y = y_rank_values - y_rank_values.mean()
    observed = float(np.sum(x_center * observed_y) / (denominator_x * np.sqrt(np.sum(observed_y**2))))
    exceed = 0
    for permutation in permutations:
        permuted = y_rank_matrix[np.ix_(permutation, permutation)][triangle]
        permuted_center = permuted - permuted.mean()
        denominator = denominator_x * float(np.sqrt(np.sum(permuted_center**2)))
        correlation = float(np.sum(x_center * permuted_center) / denominator)
        exceed += abs(correlation) >= abs(observed) - 1e-12
    return observed, float(exceed / len(permutations))


def build_source_tables() -> dict[str, Path]:
    empirical = pd.read_csv(INPUT_SOURCE_DIR / "fig3_family_local_label_metrics.csv")
    simulation = pd.read_csv(INPUT_SOURCE_DIR / "fig6_mechanism_simulation_suite.csv")
    simulation = simulation.loc[simulation["support"].isin(["pass", "below_threshold"])].copy()

    empirical["threshold"] = empirical["metric"].map(METRIC_THRESHOLDS)
    if empirical["threshold"].isna().any():
        raise ValueError("One or more empirical metrics have no diagnostic boundary.")
    empirical["relative_threshold_margin"] = _relative_margin(empirical["value"], empirical["threshold"])
    empirical["dataset_label"] = empirical["dataset_id"].map(DATASET_LABELS)
    empirical["metric_label"] = empirical["metric"].map(METRIC_LABELS)
    empirical["column_label"] = empirical["dataset_label"] + " | " + empirical["metric_label"]
    empirical["value_definition"] = "run-level continuous diagnostic value; no binary aggregation"
    panel_a = empirical.sort_values(
        ["method", "dataset_id", "metric"],
        key=lambda s: s.map({**{x: i for i, x in enumerate(METHOD_ORDER)}, **{x: i for i, x in enumerate(DATASET_ORDER)}, **{x: i for i, x in enumerate(METRIC_ORDER)}}).fillna(999),
    )

    empirical_matrix = _complete_empirical_matrix(empirical)
    standardised = StandardScaler().fit_transform(empirical_matrix.to_numpy(dtype=float))
    pca = PCA(n_components=2, svd_solver="full")
    scores = pca.fit_transform(standardised)
    scores, components = _orient_pca(scores, pca.components_)
    panel_b_scores = pd.DataFrame(
        {
            "method": METHOD_ORDER,
            "family": [config.METHOD_FAMILY[m] for m in METHOD_ORDER],
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "PC1_variance_fraction": pca.explained_variance_ratio_[0],
            "PC2_variance_fraction": pca.explained_variance_ratio_[1],
            "n_profile_variables": empirical_matrix.shape[1],
        }
    )
    loading_rows = []
    for component_idx, component_name in enumerate(["PC1", "PC2"]):
        for column_idx, (dataset_id, metric) in enumerate(empirical_matrix.columns):
            loading_rows.append(
                {
                    "component": component_name,
                    "dataset_id": dataset_id,
                    "dataset_label": DATASET_LABELS[dataset_id],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "loading": components[component_idx, column_idx],
                    "variance_fraction": pca.explained_variance_ratio_[component_idx],
                }
            )
    panel_b_loadings = pd.DataFrame(loading_rows)

    permutations = np.asarray(list(itertools.permutations(range(len(METHOD_ORDER)))), dtype=np.int16)
    concordance_rows = []
    context_pairs = list(itertools.combinations(DATASET_ORDER, 2))
    for metric in METRIC_ORDER:
        metric_matrix = empirical.loc[empirical["metric"] == metric].pivot(
            index="method", columns="dataset_id", values="value"
        ).reindex(index=METHOD_ORDER, columns=DATASET_ORDER)
        for context_a, context_b in context_pairs:
            rho, p_exact = _exact_spearman(
                metric_matrix[context_a].to_numpy(dtype=float),
                metric_matrix[context_b].to_numpy(dtype=float),
                permutations,
            )
            concordance_rows.append(
                {
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "context_a": context_a,
                    "context_b": context_b,
                    "context_pair": f"{DATASET_LABELS[context_a]} vs {DATASET_LABELS[context_b]}",
                    "spearman_rho": rho,
                    "p_exact_two_sided": p_exact,
                    "n_methods": len(METHOD_ORDER),
                    "n_label_permutations": len(permutations),
                }
            )
    panel_c = pd.DataFrame(concordance_rows)
    panel_c["q_bh_12_tests"] = _bh_adjust(panel_c["p_exact_two_sided"].to_numpy(dtype=float))

    panel_d = pd.DataFrame(
        [row for metric in METRIC_ORDER for row in _variance_partition(empirical, metric)]
    )

    simulation["relative_threshold_margin"] = _relative_margin(simulation["value"], simulation["threshold"])
    simulation["scenario_label"] = simulation["scenario"].map(SCENARIO_LABELS)
    simulation["metric_label"] = simulation["metric"].map(SIM_METRIC_LABELS)
    simulation["column_label"] = simulation["scenario_label"] + " | " + simulation["metric_label"]
    replicate_count = simulation.groupby(["method", "scenario", "metric"])["replicate"].nunique()
    if not replicate_count.eq(5).all():
        raise ValueError("Every simulation method-scenario-metric combination must contain five generated datasets.")
    panel_e = (
        simulation.groupby(
            [
                "method",
                "family",
                "scenario",
                "scenario_label",
                "metric",
                "metric_label",
                "column_label",
                "claim_axis",
                "threshold",
            ],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            value_sd_across_generated_datasets=("value", "std"),
            value_minimum=("value", "min"),
            value_maximum=("value", "max"),
            relative_threshold_margin=("relative_threshold_margin", "mean"),
            relative_margin_minimum=("relative_threshold_margin", "min"),
            relative_margin_maximum=("relative_threshold_margin", "max"),
            n_generated_datasets=("replicate", "nunique"),
            n_cells_per_generated_dataset=("n_cells", "first"),
        )
    )
    panel_e["value_definition"] = (
        "mean continuous known-truth metric across five independently generated datasets; "
        "no pass-fraction aggregation"
    )

    geometry = simulation.loc[
        simulation["metric"].isin(["truth_local_retention", "label_neighbor_recall"]),
        ["scenario", "replicate", "method", "family", "metric", "value", "threshold", "n_cells"],
    ].pivot(index=["scenario", "replicate", "method", "family", "n_cells"], columns="metric", values=["value", "threshold"])
    geometry.columns = [f"{outer}_{inner}" for outer, inner in geometry.columns]
    panel_f = geometry.reset_index()
    panel_f["local_relative_margin"] = (
        panel_f["value_truth_local_retention"] - panel_f["threshold_truth_local_retention"]
    ) / panel_f["threshold_truth_local_retention"]
    panel_f["label_relative_margin"] = (
        panel_f["value_label_neighbor_recall"] - panel_f["threshold_label_neighbor_recall"]
    ) / panel_f["threshold_label_neighbor_recall"]
    descriptive_rho = float(
        spearmanr(panel_f["value_truth_local_retention"], panel_f["value_label_neighbor_recall"]).statistic
    )
    panel_f["descriptive_spearman_rho_all_rows"] = descriptive_rho
    panel_f["dependence_note"] = "method-scenario-generated-dataset rows share methods and scenarios; no point-level P value is reported"

    sim_matrix = simulation.pivot(index="method", columns=["scenario", "replicate", "metric"], values="value")
    ordered_sim_columns = [
        (scenario, replicate, metric)
        for scenario in SCENARIO_ORDER
        for replicate in sorted(simulation["replicate"].unique())
        for metric in SIM_METRIC_ORDER
        if (scenario, replicate, metric) in sim_matrix.columns
    ]
    sim_matrix = sim_matrix.reindex(index=METHOD_ORDER, columns=pd.MultiIndex.from_tuples(ordered_sim_columns))
    if sim_matrix.isna().any().any():
        raise ValueError("Known-truth simulation profile matrix is incomplete.")
    signature_matrix = _objective_signature_matrix()
    objective_dist = squareform(pdist(signature_matrix.to_numpy(dtype=bool), metric="jaccard"))
    empirical_dist = _standardised_profile_distances(empirical_matrix)
    simulation_dist = _standardised_profile_distances(sim_matrix)
    triangle = np.triu_indices(len(METHOD_ORDER), 1)
    pair_rows = []
    for i, j in zip(*triangle):
        method_a = METHOD_ORDER[i]
        method_b = METHOD_ORDER[j]
        pair_rows.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "method_pair": f"{method_a} / {method_b}",
                "objective_signature_jaccard_distance": objective_dist[i, j],
                "empirical_profile_distance": empirical_dist[i, j],
                "simulation_profile_distance": simulation_dist[i, j],
                "n_objective_signature_features": signature_matrix.shape[1],
                "n_empirical_profile_variables": empirical_matrix.shape[1],
                "n_simulation_profile_variables": sim_matrix.shape[1],
            }
        )
    panel_g = pd.DataFrame(pair_rows)

    comparisons = [
        ("objective_vs_empirical", "Specification", "Empirical", objective_dist, empirical_dist, True),
        ("objective_vs_simulation", "Specification", "Known truth", objective_dist, simulation_dist, True),
        ("empirical_vs_simulation", "Empirical", "Known truth", empirical_dist, simulation_dist, False),
    ]
    mantel_rows = []
    for comparison, space_a, space_b, distance_a, distance_b, primary in comparisons:
        rho, p_exact = _exact_mantel(distance_a, distance_b, permutations)
        mantel_rows.append(
            {
                "comparison": comparison,
                "space_a": space_a,
                "space_b": space_b,
                "mantel_spearman_rho": rho,
                "p_exact_two_sided": p_exact,
                "primary_signature_profile_test": primary,
                "n_methods": len(METHOD_ORDER),
                "n_unique_method_pairs": len(panel_g),
                "n_label_permutations": len(permutations),
                "objective_distance_definition": f"Jaccard distance across {signature_matrix.shape[1]} binary features encoding the observation model, latent parameterisation and active explicit objective terms; navigation group excluded",
                "empirical_distance_definition": "Euclidean distance across 12 context-by-diagnostic variables after column-wise z standardisation, divided by sqrt(12)",
                "simulation_distance_definition": f"Euclidean distance across {sim_matrix.shape[1]} scenario-by-generated-dataset-by-metric variables after column-wise z standardisation, divided by sqrt({sim_matrix.shape[1]})",
            }
        )
    mantel_summary = pd.DataFrame(mantel_rows)
    mantel_summary["q_bh_two_signature_profile_tests"] = np.nan
    primary_mask = mantel_summary["primary_signature_profile_test"]
    mantel_summary.loc[primary_mask, "q_bh_two_signature_profile_tests"] = _bh_adjust(
        mantel_summary.loc[primary_mask, "p_exact_two_sided"].to_numpy(dtype=float)
    )

    signature_long = (
        signature_matrix.rename_axis("method")
        .reset_index()
        .melt(id_vars="method", var_name="encoded_feature", value_name="present")
    )
    signature_long[["feature_group", "feature"]] = signature_long["encoded_feature"].str.split(
        ":", n=1, expand=True
    )
    signature_long = signature_long[
        ["method", "feature_group", "feature", "present", "encoded_feature"]
    ]

    paths = {
        "a": _write_csv(panel_a, "Fig2a_empirical_threshold_margins.csv"),
        "b_scores": _write_csv(panel_b_scores, "Fig2b_empirical_profile_pca_scores.csv"),
        "b_loadings": _write_csv(panel_b_loadings, "Fig2b_empirical_profile_pca_loadings.csv"),
        "c": _write_csv(panel_c, "Fig2c_context_rank_concordance.csv"),
        "d": _write_csv(panel_d, "Fig2d_empirical_variance_partition.csv"),
        "e_signatures": _write_csv(signature_long, "Fig2e_objective_signature_matrix.csv"),
        "e": _write_csv(panel_g, "Fig2e_profile_distance_concordance.csv"),
        "e_summary": _write_csv(mantel_summary, "Fig2e_mantel_summary.csv"),
        "f": _write_csv(panel_e, "Fig2f_simulation_boundary_margins.csv"),
        "g": _write_csv(panel_f, "Fig2g_geometry_identity_decoupling.csv"),
    }
    manifest_rows = []
    for panel, path in paths.items():
        manifest_rows.append(
            {
                "panel_or_component": panel,
                "source_data_file": str(path.relative_to(ROOT)).replace("\\", "/"),
                "sha256": _hash(path),
                "plotting_code": "figures/plot_figure_2.py",
            }
        )
    paths["manifest"] = _write_csv(pd.DataFrame(manifest_rows), "Fig2_source_manifest.csv")
    paths["claim_matrix"] = _write_claim_matrix()
    return paths


def _colour_method_ticks(ax: plt.Axes) -> None:
    for tick, method in zip(ax.get_yticklabels(), METHOD_ORDER):
        tick.set_color(style.FAMILY_COLORS[config.METHOD_FAMILY[method]])
        tick.set_fontweight("bold")
    for boundary in [1.5, 3.5, 5.5]:
        ax.axhline(boundary, color="white", lw=1.1, zorder=3)


def _draw_margin_heatmap(
    fig: plt.Figure,
    ax: plt.Axes,
    matrix: pd.DataFrame,
    *,
    vlim: float,
    colorbar_label: str,
) -> mpl.image.AxesImage:
    clipped = np.clip(matrix.to_numpy(dtype=float), -vlim, vlim)
    im = ax.imshow(
        clipped,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-vlim, vcenter=0, vmax=vlim),
    )
    colorbar = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.012)
    colorbar.ax.set_ylabel(colorbar_label, rotation=270, labelpad=9, fontsize=5.4)
    colorbar.ax.tick_params(labelsize=5)
    colorbar.ax.axhline(0, color="#222222", lw=0.5)
    return im


def build_figure(paths: dict[str, Path]) -> None:
    _apply_style()
    empirical = pd.read_csv(paths["a"])
    pca_scores = pd.read_csv(paths["b_scores"])
    pca_loadings = pd.read_csv(paths["b_loadings"])
    concordance = pd.read_csv(paths["c"])
    variance = pd.read_csv(paths["d"])
    distance = pd.read_csv(paths["e"])
    mantel = pd.read_csv(paths["e_summary"])
    simulation = pd.read_csv(paths["f"])
    decoupling = pd.read_csv(paths["g"])

    fig = plt.figure(figsize=(7.45, 10.25))
    grid = fig.add_gridspec(
        5,
        10,
        height_ratios=[1.28, 1.06, 0.94, 1.50, 1.16],
        width_ratios=[1] * 10,
        hspace=0.88,
        wspace=0.78,
    )

    # a | Continuous empirical preservation fingerprint.
    ax_a = fig.add_subplot(grid[0, :])
    empirical["column_key"] = empirical["dataset_id"] + "__" + empirical["metric"]
    empirical_columns = [f"{dataset}__{metric}" for dataset in DATASET_ORDER for metric in METRIC_ORDER]
    empirical_matrix = empirical.pivot(index="method", columns="column_key", values="relative_threshold_margin").reindex(
        index=METHOD_ORDER, columns=empirical_columns
    )
    _draw_margin_heatmap(fig, ax_a, empirical_matrix, vlim=1.0, colorbar_label="relative margin")
    ax_a.set_yticks(np.arange(len(METHOD_ORDER)))
    ax_a.set_yticklabels(METHOD_ORDER)
    ax_a.set_xticks(np.arange(len(empirical_columns)))
    ax_a.set_xticklabels([METRIC_LABELS[metric] for _dataset in DATASET_ORDER for metric in METRIC_ORDER])
    _colour_method_ticks(ax_a)
    for boundary in [3.5, 7.5]:
        ax_a.axvline(boundary, color="white", lw=1.5)
    for dataset_index, dataset in enumerate(DATASET_ORDER):
        ax_a.text(
            dataset_index * 4 + 1.5,
            -0.68,
            DATASET_LABELS[dataset],
            ha="center",
            va="bottom",
            fontsize=6.1,
            fontweight="bold",
            color="#333333",
            clip_on=False,
        )
    ax_a.set_title("Empirical preservation fingerprints", loc="left", pad=23)
    _panel_label(ax_a, "a", x=-0.055, y=1.30)

    # b | Multivariate empirical profile separation.
    ax_b = fig.add_subplot(grid[1, 0:6])
    method_label_positions = {
        "PCA": (-2.82, 2.55),
        "GLM-PCA": (-2.45, 1.05),
        "scScope": (-2.55, -3.30),
        "SAUCIE": (-1.58, 0.92),
        "PHATE": (-0.58, -0.72),
        "UMAP": (2.58, -1.55),
        "PaCMAP": (2.62, 1.32),
        "t-SNE": (3.96, 0.58),
    }
    for row in pca_scores.itertuples(index=False):
        color = style.FAMILY_COLORS[row.family]
        ax_b.scatter(row.PC1, row.PC2, s=42, color=color, edgecolor="white", linewidth=0.45, zorder=3)
        label_x, label_y = method_label_positions[row.method]
        ax_b.annotate(
            row.method,
            (row.PC1, row.PC2),
            xytext=(label_x, label_y),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=5.1,
            color="#222222",
            arrowprops=dict(arrowstyle="-", color="#A5A5A5", lw=0.35, shrinkA=1.5, shrinkB=3.5),
        )
    loading_wide = pca_loadings.pivot_table(index="metric", columns="component", values="loading", aggfunc="mean").reindex(METRIC_ORDER)
    arrow_scale = max(np.ptp(pca_scores["PC1"]), np.ptp(pca_scores["PC2"])) * 0.62
    loading_colors = {
        "local_retention": "#0072B2",
        "trustworthiness": "#009E73",
        "global_rank_corr": "#D55E00",
        "label_neighbor_recall": "#CC79A7",
    }
    for metric, row in loading_wide.iterrows():
        x = float(row["PC1"] * arrow_scale)
        y = float(row["PC2"] * arrow_scale)
        ax_b.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", lw=0.8, color=loading_colors[metric]),
        )
    loading_handles = [
        Line2D([0], [0], color=loading_colors[metric], lw=1.0, label=METRIC_LABELS[metric])
        for metric in METRIC_ORDER
    ]
    ax_b.legend(
        handles=loading_handles,
        title="mean loading vector",
        loc="upper right",
        ncol=2,
        frameon=False,
        fontsize=4.4,
        title_fontsize=4.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.35,
    )
    pc1_var = float(pca_scores["PC1_variance_fraction"].iloc[0]) * 100
    pc2_var = float(pca_scores["PC2_variance_fraction"].iloc[0]) * 100
    ax_b.axhline(0, color="#D5D5D5", lw=0.55, zorder=0)
    ax_b.axvline(0, color="#D5D5D5", lw=0.55, zorder=0)
    ax_b.set_xlim(-3.55, 4.75)
    ax_b.set_ylim(-4.15, 3.55)
    ax_b.set_xlabel(f"PC1 ({pc1_var:.1f}% variance)")
    ax_b.set_ylabel(f"PC2 ({pc2_var:.1f}% variance)")
    ax_b.set_title("Empirical profile ordination", loc="left", pad=8)
    _panel_label(ax_b, "b")

    # c | Metric-specific context transfer.
    ax_c = fig.add_subplot(grid[1, 6:10])
    context_pairs = [
        "PBMC3k vs Paul15",
        "PBMC3k vs Heart atlas",
        "Paul15 vs Heart atlas",
    ]
    concordance_matrix = concordance.pivot(index="metric", columns="context_pair", values="spearman_rho").reindex(
        index=METRIC_ORDER, columns=context_pairs
    )
    concordance_q = concordance.pivot(index="metric", columns="context_pair", values="q_bh_12_tests").reindex(
        index=METRIC_ORDER, columns=context_pairs
    )
    im_c = ax_c.imshow(
        concordance_matrix.to_numpy(dtype=float),
        aspect="auto",
        cmap="PuOr_r",
        norm=mpl.colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1),
    )
    for i in range(concordance_matrix.shape[0]):
        for j in range(concordance_matrix.shape[1]):
            rho = float(concordance_matrix.iloc[i, j])
            q = float(concordance_q.iloc[i, j])
            marker = "*" if q < 0.05 else ""
            ax_c.text(j, i, f"{rho:.2f}{marker}", ha="center", va="center", fontsize=5.2, color="white" if abs(rho) > 0.62 else "#222222")
    ax_c.set_xticks(np.arange(len(context_pairs)))
    ax_c.set_xticklabels(["PBMC-Paul", "PBMC-Heart", "Paul-Heart"])
    ax_c.set_yticks(np.arange(len(METRIC_ORDER)))
    ax_c.set_yticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    colorbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.035, pad=0.025)
    colorbar_c.ax.set_ylabel("Spearman rho", rotation=270, labelpad=8, fontsize=5.3)
    colorbar_c.ax.tick_params(labelsize=5)
    ax_c.text(1.0, -0.28, "* BH-adjusted q < 0.05", transform=ax_c.transAxes, ha="right", va="top", fontsize=4.8, color="#555555")
    ax_c.set_title("Cross-context rank concordance", loc="left")
    _panel_label(ax_c, "c")

    # d | Descriptive variance decomposition.
    ax_d = fig.add_subplot(grid[2, 0:4])
    variance_matrix = variance.pivot(index="metric", columns="source", values="fraction_total_variation").reindex(
        index=METRIC_ORDER, columns=["method", "biological_context", "residual"]
    )
    variance_display = variance_matrix.T
    im_d = ax_d.imshow(variance_display.to_numpy(dtype=float), aspect="auto", cmap="Blues", vmin=0, vmax=1)
    for i in range(variance_display.shape[0]):
        for j in range(variance_display.shape[1]):
            value = float(variance_display.iloc[i, j])
            ax_d.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=5.1,
                color="white" if value >= 0.52 else "#222222",
            )
    ax_d.set_xticks(np.arange(len(METRIC_ORDER)))
    ax_d.set_xticklabels([METRIC_LABELS[m] for m in METRIC_ORDER])
    ax_d.set_yticks(np.arange(3))
    ax_d.set_yticklabels(["Method", "Context", "Interaction / residual"])
    colorbar_d = fig.colorbar(im_d, ax=ax_d, fraction=0.035, pad=0.025)
    colorbar_d.ax.set_ylabel("variance fraction", rotation=270, labelpad=7, fontsize=5.3)
    colorbar_d.ax.tick_params(labelsize=5)
    ax_d.set_title("Method and context contributions", loc="left")
    _panel_label(ax_d, "d", x=-0.10, y=1.12)

    # e | Pair-level distance concordance. The 28 pairwise observations are
    # displayed directly; Mantel inference still uses complete distance
    # matrices and never treats these points as independent replicates.
    ax_g = fig.add_subplot(grid[2, 5:10])
    ax_g.set_axis_off()
    ax_g.set_title("Distance concordance across method pairs", loc="left", pad=6)
    _panel_label(ax_g, "e", x=-0.07, y=1.12)

    comparison_specs = [
        (
            "objective_vs_empirical",
            "objective_signature_jaccard_distance",
            "empirical_profile_distance",
            "Specification vs empirical",
            "Specification distance",
            "Empirical distance",
            "#3572A5",
        ),
        (
            "objective_vs_simulation",
            "objective_signature_jaccard_distance",
            "simulation_profile_distance",
            "Specification vs known truth",
            "Specification distance",
            "Known-truth distance",
            "#C56A2D",
        ),
        (
            "empirical_vs_simulation",
            "empirical_profile_distance",
            "simulation_profile_distance",
            "Empirical vs known truth",
            "Empirical distance",
            "Known-truth distance",
            "#7A5195",
        ),
    ]
    inset_positions = [
        [0.00, 0.05, 0.285, 0.78],
        [0.355, 0.05, 0.285, 0.78],
        [0.710, 0.05, 0.285, 0.78],
    ]
    mantel_by_comparison = mantel.set_index("comparison")
    for spec, position in zip(comparison_specs, inset_positions):
        comparison, x_column, y_column, title, x_label, y_label, color = spec
        inset = ax_g.inset_axes(position)
        inset.scatter(
            distance[x_column],
            distance[y_column],
            s=14,
            color=color,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
        )
        summary = mantel_by_comparison.loc[comparison]
        rho = float(summary["mantel_spearman_rho"])
        if bool(summary["primary_signature_profile_test"]):
            probability = f"q = {float(summary['q_bh_two_signature_profile_tests']):.3f}"
        else:
            probability = f"P = {float(summary['p_exact_two_sided']):.3f}"
        inset.text(
            0.04,
            0.96,
            f"rho = {rho:.2f}\n{probability}",
            transform=inset.transAxes,
            ha="left",
            va="top",
            fontsize=4.45,
            color="#222222",
        )
        inset.set_title(title, loc="left", fontsize=5.25, pad=3)
        inset.set_xlabel(x_label, fontsize=4.65, labelpad=2)
        inset.set_ylabel(y_label, fontsize=4.65, labelpad=2)
        inset.tick_params(axis="both", labelsize=4.25, length=2.0, pad=1.5)
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)

    # f | Continuous known-truth fingerprints.
    ax_e = fig.add_subplot(grid[3, :])
    simulation["column_key"] = simulation["scenario"] + "__" + simulation["metric"]
    sim_columns = [
        f"{scenario}__{metric}"
        for scenario in SCENARIO_ORDER
        for metric in SIM_METRIC_ORDER
        if ((simulation["scenario"] == scenario) & (simulation["metric"] == metric)).any()
    ]
    sim_matrix = simulation.pivot(index="method", columns="column_key", values="relative_threshold_margin").reindex(
        index=METHOD_ORDER, columns=sim_columns
    )
    _draw_margin_heatmap(fig, ax_e, sim_matrix, vlim=1.0, colorbar_label="relative margin")
    ax_e.set_yticks(np.arange(len(METHOD_ORDER)))
    ax_e.set_yticklabels(METHOD_ORDER)
    ax_e.set_xticks(np.arange(len(sim_columns)))
    ax_e.set_xticklabels(
        [SIM_METRIC_LABELS[column.split("__", 1)[1]] for column in sim_columns],
        rotation=90,
        ha="center",
        va="top",
    )
    _colour_method_ticks(ax_e)
    cursor = 0
    for scenario in SCENARIO_ORDER:
        scenario_columns = [column for column in sim_columns if column.startswith(f"{scenario}__")]
        if not scenario_columns:
            continue
        center = cursor + (len(scenario_columns) - 1) / 2
        ax_e.text(center, -0.68, SCENARIO_LABELS[scenario], ha="center", va="bottom", fontsize=5.8, fontweight="bold", clip_on=False)
        cursor += len(scenario_columns)
        if cursor < len(sim_columns):
            ax_e.axvline(cursor - 0.5, color="white", lw=1.4)
    ax_e.set_title("Known-truth preservation fingerprints", loc="left", pad=23)
    _panel_label(ax_e, "f", x=-0.055, y=1.30)

    # g | Geometry-label dissociation under known truth.
    ax_f = fig.add_subplot(grid[4, :])
    for scenario in SCENARIO_ORDER:
        sub = decoupling.loc[decoupling["scenario"] == scenario]
        for row in sub.itertuples(index=False):
            ax_f.scatter(
                row.local_relative_margin,
                row.label_relative_margin,
                s=15,
                marker=SCENARIO_MARKERS[scenario],
                color=style.FAMILY_COLORS[row.family],
                edgecolor="white",
                linewidth=0.35,
                alpha=0.38,
            )
    ax_f.axvline(0, color="#555555", lw=0.7, ls="--")
    ax_f.axhline(0, color="#555555", lw=0.7, ls="--")
    ax_f.text(
        0.02,
        0.08,
        f"descriptive rho = {float(decoupling['descriptive_spearman_rho_all_rows'].iloc[0]):.2f}; n = {len(decoupling)} method-scenario-generated-dataset runs",
        transform=ax_f.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.2,
    )
    scenario_handles = [
        Line2D([0], [0], marker=SCENARIO_MARKERS[scenario], color="none", markerfacecolor="#777777", markeredgecolor="white", markersize=4.5, label=SCENARIO_LABELS[scenario])
        for scenario in SCENARIO_ORDER
    ]
    ax_f.legend(handles=scenario_handles, loc="lower right", bbox_to_anchor=(1.0, 1.01), ncol=6, frameon=False, fontsize=4.8, handletextpad=0.3, columnspacing=0.8)
    ax_f.set_xlabel("relative margin, truth local retention")
    ax_f.set_ylabel("relative margin, same-label neighbour fraction")
    ax_f.set_xlim(-0.83, 0.04)
    ax_f.set_ylim(-0.04, 0.88)
    ax_f.set_title("Geometry-identity decoupling under known truth", loc="left")
    _panel_label(ax_f, "g", x=-0.075, y=1.19)

    family_handles = [
        Patch(facecolor=style.FAMILY_COLORS[family], edgecolor="none", label=config.FAMILY_LABELS[family])
        for family in config.FAMILY_ORDER
    ]
    fig.legend(
        handles=family_handles,
        title="catalogue navigation group",
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=4,
        frameon=False,
        fontsize=5.2,
        title_fontsize=5.4,
    )
    fig.subplots_adjust(left=0.09, right=0.965, top=0.915, bottom=0.055)
    _save_figure(fig)


def write_run_metadata(paths: dict[str, Path]) -> Path:
    metadata = {
        "figure": "Figure_2",
        "main_claim": "The 17-feature pipeline record is associated with known-truth, but not empirical, profile distance under the primary coding.",
        "seed": config.SEED,
        "methods": METHOD_ORDER,
        "datasets": DATASET_ORDER,
        "simulation_scenarios": SCENARIO_ORDER,
        "source_manifest": str(paths["manifest"].relative_to(ROOT)).replace("\\", "/"),
        "claim_to_evidence_matrix": str(paths["claim_matrix"].relative_to(ROOT)).replace("\\", "/"),
        "exact_permutations": 40320,
        "objective_signature_features": int(pd.read_csv(paths["e_signatures"])["encoded_feature"].nunique()),
        "objective_distance": "Jaccard distance over binary observation-model, latent-parameterisation and active-objective features; navigation-group labels excluded",
        "missing_value_policy": "No values are imputed; generation fails if a required empirical or simulation profile is incomplete.",
    }
    path = METADATA_DIR / "Fig2_run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def main() -> None:
    SOURCE_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    paths = build_source_tables()
    build_figure(paths)
    metadata = write_run_metadata(paths)
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.pdf'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.svg'}")
    print(f"Wrote {OUTPUT_DIR / 'Figure_2.png'}")
    print(f"Wrote {paths['manifest']}")
    print(f"Wrote {metadata}")


if __name__ == "__main__":
    main()
