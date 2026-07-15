"""Build the sensitivity synthesis figure from auditable source tables."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figures.figure_style import (  # noqa: E402
    BORDER,
    DATASET_COLORS,
    DATASET_LABELS,
    FAIL_COLOR,
    FAMILY_COLORS,
    GRID,
    INK,
    PASS_COLOR,
    TEXT_MUTED,
    apply_final_style,
    clean_axis,
    panel_label,
)


SOURCE = ROOT / "data" / "source_data" / "generated"
FIGURE_DIR = ROOT / "outputs" / "main_figures"
META_DIR = ROOT / "metadata"

FAMILY_ORDER = ["factor", "deep", "graph", "relational"]
FAMILY_LABELS = {
    "factor": "Factor/count",
    "deep": "Deep latent",
    "graph": "Graph/diffusion",
    "relational": "Relational",
}
PROFILE_COLORS = {"empirical": "#3572A5", "simulation": "#C56A2D"}
ROOT_LABELS = {
    "stored_8Mk_root": "Stored 8Mk",
    "7MEP_centroid": "7MEP",
    "9GMP_centroid": "9GMP",
    "1Ery_centroid": "1Ery",
}
LINEAGE_LABELS = {
    "erythroid": "Erythroid",
    "megakaryocytic": "Megakaryocytic",
    "myeloid": "Myeloid",
    "basophil": "Basophil",
}
AXIS_LABELS = {
    "output_dimension": "Output\ndimension",
    "upstream_pca": "Upstream\nPCA",
    "expression_perturbation": "Expression\nperturbation",
}
AXIS_COLORS = {
    "output_dimension": "#3572A5",
    "upstream_pca": "#7A5195",
    "expression_perturbation": "#C56A2D",
}


def _write_claim_matrix() -> Path:
    rows = [
        ("a", "How do navigation-group support fractions change under joint boundary shifts?", "deterministic sensitivity", "SuppS7_boundary_sensitivity_family_summary.csv", "gate recomputation across five joint shifts", "pass fraction", "22 method-context-gate decisions per group and shift", "operational boundaries are not universal"),
        ("b", "How many baseline decisions change under each joint boundary shift?", "deterministic sensitivity", "SuppS7_boundary_sensitivity_changes.csv", "comparison with shift-zero decisions", "gained and lost decisions", "88 matched gate decisions per shift", "decisions share methods and contexts"),
        ("c", "Does the local-retention failure count depend on reference dimension and k?", "calibration", "SuppS10_boundary_count_sensitivity.csv", "20 reference-by-k definitions", "count below 0.30", "24 method-context analyses per definition", "fixed 1,000-cell calibration subset"),
        ("d", "Are local-retention method ranks stable across PCA reference definitions?", "calibration", "SuppS10_reference_rank_stability.csv", "pairwise Spearman correlations", "rank correlation", "8 methods per dataset and reference pair", "few methods limit correlation precision"),
        ("e", "Does the Paul15 pooled continuum decision depend on DPT root definition?", "trajectory sensitivity", "SuppS11_root_sensitive_continuum_metrics.csv", "four biologically named roots", "methods meeting each boundary", "8 methods per root and metric", "root definitions are not independent replicates"),
        ("f", "Is pseudotime-distance preservation consistent across Paul15 lineages?", "trajectory sensitivity", "SuppS11_lineage_restricted_metrics.csv", "lineage-restricted sampled-pair analysis", "Spearman correlation", "8 methods per lineage", "lineages differ in cell count and topology"),
        ("g", "Does specification-to-profile concordance depend on feature encoding and distance?", "coding sensitivity", "SuppS9_signature_definition_sensitivity.csv", "exact Mantel tests over alternative encodings", "Mantel Spearman rho", "8 methods and 28 method pairs", "many pair distances are tied"),
        ("h", "Is specification concordance driven by one method?", "influence analysis", "SuppS9_leave_one_method_out.csv", "leave-one-method-out exact Mantel tests", "Mantel Spearman rho", "7 methods and 21 pairs per omission", "omissions are overlapping analyses"),
        ("i", "Do random seeds affect local and global layout stability similarly?", "computational repeat", "SuppS8_multiseed_method_summary.csv", "five seeds in three contexts", "worst-context kNN overlap and distance-rank concordance", "30 seed pairs per method across contexts", "seeds are not biological replicates"),
        ("j", "Which analytical axes convert supported baselines into unsupported conditions?", "matched robustness transition", "SuppS12_axis_loss.csv", "matched baseline-to-condition comparison", "pass-to-fail fraction", "56-96 baseline-supported comparisons per axis", "conditions are computational, not biological replicates"),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "panel",
            "scientific_question",
            "evidence_level",
            "source_data",
            "analysis",
            "statistic",
            "n_definition",
            "reviewer_risk",
        ],
    )
    path = META_DIR / "claim_to_evidence_matrix_Fig7.tsv"
    frame.to_csv(path, sep="\t", index=False)
    return path


def _load() -> dict[str, pd.DataFrame]:
    names = {
        "boundary_family": "SuppS7_boundary_sensitivity_family_summary.csv",
        "boundary_changes": "SuppS7_boundary_sensitivity_changes.csv",
        "local_counts": "SuppS10_boundary_count_sensitivity.csv",
        "rank_stability": "SuppS10_reference_rank_stability.csv",
        "root_metrics": "SuppS11_root_sensitive_continuum_metrics.csv",
        "lineage_metrics": "SuppS11_lineage_restricted_metrics.csv",
        "signature": "SuppS9_signature_definition_sensitivity.csv",
        "leave_one_out": "SuppS9_leave_one_method_out.csv",
        "seed_summary": "SuppS8_multiseed_method_summary.csv",
        "robustness_loss": "SuppS12_axis_loss.csv",
    }
    return {key: pd.read_csv(SOURCE / filename) for key, filename in names.items()}


def _short_signature(value: str) -> str:
    mapping = {
        "full_pipeline_jaccard": "Full pipeline (J)",
        "full_pipeline_hamming": "Full pipeline (H)",
        "objective_terms_only": "Objective terms",
        "observation_model_only": "Observation model",
        "latent_parameterisation_only": "Latent model",
        "observation_plus_parameterisation": "Observation + latent",
        "without_not_explicit": "Explicit terms only",
        "legacy_scscope_stage_omitted": "scScope stage omitted",
        "execution_input_baseline": "Execution inputs",
        "navigation_group_baseline": "Navigation groups",
    }
    return mapping.get(value, value.replace("_", " "))


def plot(data: dict[str, pd.DataFrame]) -> list[Path]:
    apply_final_style()
    fig = plt.figure(figsize=(7.2, 7.0), constrained_layout=False)
    gs = fig.add_gridspec(3, 1, left=0.075, right=0.985, top=0.975, bottom=0.095, hspace=0.62)
    top = gs[0].subgridspec(1, 3, wspace=0.50)
    middle = gs[1].subgridspec(1, 4, width_ratios=[1.0, 1.0, 1.22, 1.22], wspace=0.78)
    bottom = gs[2].subgridspec(1, 3, wspace=0.50)
    axes = [
        fig.add_subplot(top[0, 0]),
        fig.add_subplot(top[0, 1]),
        fig.add_subplot(top[0, 2]),
        fig.add_subplot(middle[0, 0]),
        fig.add_subplot(middle[0, 1]),
        fig.add_subplot(middle[0, 2]),
        fig.add_subplot(middle[0, 3]),
        fig.add_subplot(bottom[0, 0]),
        fig.add_subplot(bottom[0, 1]),
        fig.add_subplot(bottom[0, 2]),
    ]
    axa, axb, axc, axd, axe, axf, axg, axh, axi, axj = axes

    boundary = data["boundary_family"]
    for family in FAMILY_ORDER:
        sub = boundary[boundary["family"].eq(family)].sort_values("boundary_shift")
        axa.plot(sub["boundary_shift"], sub["pass_fraction"], marker="o", markersize=3.5, linewidth=1.1, color=FAMILY_COLORS[family], label=FAMILY_LABELS[family])
    axa.axvline(0, color=BORDER, linestyle="--", linewidth=0.8)
    axa.set_xlabel("Joint boundary shift")
    axa.set_ylabel("Gate pass fraction")
    axa.set_ylim(0.35, 0.75)
    axa.set_title("Boundary-dependent support", loc="left")
    axa.legend(frameon=False, fontsize=5.2, ncol=2, loc="upper center", bbox_to_anchor=(0.50, -0.28))
    clean_axis(axa)

    changes = data["boundary_changes"]
    shifts = sorted(changes["boundary_shift"].unique())
    gained = [int(changes.loc[changes["boundary_shift"].eq(shift) & changes["decision_change"].eq("gained_support"), "decision_count"].sum()) for shift in shifts]
    lost = [int(changes.loc[changes["boundary_shift"].eq(shift) & changes["decision_change"].eq("lost_support"), "decision_count"].sum()) for shift in shifts]
    axb.bar(np.arange(len(shifts)), gained, color=PASS_COLOR, width=0.62, label="Gained")
    axb.bar(np.arange(len(shifts)), -np.asarray(lost), color=FAIL_COLOR, width=0.62, label="Lost")
    axb.axhline(0, color=BORDER, linewidth=0.8)
    axb.set_xticks(np.arange(len(shifts)), [f"{value:+.3f}" if value else "0" for value in shifts], rotation=30, ha="right")
    axb.set_xlabel("Joint boundary shift")
    axb.set_ylabel("Changed decisions")
    axb.set_title("Decision changes", loc="left")
    axb.legend(frameon=False, fontsize=5.5, loc="upper right")
    clean_axis(axb)

    local = data["local_counts"].copy()
    pca_order = sorted(local["reference_pca_dimensions"].unique())
    k_order = sorted(local["k"].unique())
    matrix = local.pivot(index="reference_pca_dimensions", columns="k", values="below_boundary").reindex(index=pca_order, columns=k_order)
    im = axc.imshow(matrix.to_numpy(), cmap="YlOrRd", vmin=0, vmax=24, aspect="auto")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix.iloc[row, col])
            axc.text(col, row, f"{value}/24", ha="center", va="center", fontsize=5.4, color="white" if value >= 18 else INK)
    axc.set_xticks(range(len(k_order)), [str(value) for value in k_order])
    axc.set_yticks(range(len(pca_order)), [str(value) for value in pca_order])
    axc.set_xlabel("Neighbourhood size, k")
    axc.set_ylabel("PCA reference dimensions")
    axc.set_title("Local-retention calibration", loc="left")
    cbar = fig.colorbar(im, ax=axc, fraction=0.047, pad=0.03)
    cbar.set_label("Below 0.30", fontsize=5.8)
    cbar.ax.tick_params(labelsize=5.2, length=2)

    ranks = data["rank_stability"]
    datasets = list(DATASET_LABELS)
    rng = np.random.default_rng(20260714)
    for x, dataset in enumerate(datasets):
        values = ranks.loc[ranks["dataset_id"].eq(dataset), "method_rank_spearman_rho"].to_numpy()
        jitter = rng.uniform(-0.14, 0.14, len(values))
        axd.scatter(x + jitter, values, s=12, alpha=0.65, color=DATASET_COLORS[dataset], edgecolor="white", linewidth=0.35)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axd.vlines(x, q1, q3, color=INK, linewidth=2.0)
        axd.scatter(x, med, s=26, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=4)
    axd.axhline(0, color=BORDER, linestyle="--", linewidth=0.7)
    axd.set_xticks(range(len(datasets)), [DATASET_LABELS[value] for value in datasets], rotation=30, ha="right")
    axd.set_ylabel("Method-rank Spearman rho")
    axd.set_title("Reference-sensitive ranks", loc="left")
    clean_axis(axd)

    root = data["root_metrics"]
    root_order = list(ROOT_LABELS)
    metric_specs = [
        ("pseudotime_distance_correlation", "Pooled distance", "#3572A5"),
        ("local_pseudotime_retention", "Local order", "#D38A35"),
    ]
    width = 0.34
    for offset, (metric, label, color) in zip([-width / 2, width / 2], metric_specs):
        counts = [int(root.loc[root["root_definition"].eq(value) & root["metric"].eq(metric), "meets_boundary"].sum()) for value in root_order]
        bars = axe.bar(np.arange(len(root_order)) + offset, counts, width=width, color=color, label=label)
        for bar, count in zip(bars, counts):
            axe.text(bar.get_x() + bar.get_width() / 2, count + 0.18, str(count), ha="center", va="bottom", fontsize=5.2)
    axe.set_xticks(range(len(root_order)), [ROOT_LABELS[value] for value in root_order], rotation=30, ha="right")
    axe.set_ylabel("Methods meeting boundary")
    axe.set_ylim(0, 9.4)
    axe.set_title("DPT-root sensitivity", loc="left")
    axe.legend(frameon=False, fontsize=5.0, loc="upper center", bbox_to_anchor=(0.50, -0.29), ncol=2)
    clean_axis(axe)

    lineage = data["lineage_metrics"]
    lineage_order = list(LINEAGE_LABELS)
    for x, lineage_name in enumerate(lineage_order):
        values = lineage.loc[lineage["lineage"].eq(lineage_name), "within_lineage_pseudotime_distance_correlation"].to_numpy()
        jitter = rng.uniform(-0.16, 0.16, len(values))
        axf.scatter(x + jitter, values, s=12, color="#4C78A8", alpha=0.7, edgecolor="white", linewidth=0.35)
        q1, med, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        axf.vlines(x, q1, q3, color=INK, linewidth=2.0)
        axf.scatter(x, med, s=24, facecolor="white", edgecolor=INK, linewidth=0.8, zorder=4)
    axf.axhline(0.45, color=BORDER, linestyle="--", linewidth=0.8)
    axf.set_xticks(range(len(lineage_order)), [LINEAGE_LABELS[value] for value in lineage_order], rotation=30, ha="right")
    axf.set_ylabel("Within-lineage rank rho")
    axf.set_ylim(0.25, 1.0)
    axf.set_title("Lineage preservation", loc="left")
    clean_axis(axf)

    signature = data["signature"].copy()
    signature_order = list(dict.fromkeys(signature["signature_definition"]))
    for profile, offset in [("empirical", -0.10), ("simulation", 0.10)]:
        sub = signature[signature["profile"].eq(profile)].set_index("signature_definition").reindex(signature_order)
        y = np.arange(len(signature_order)) + offset
        axg.scatter(sub["mantel_spearman_rho"], y, s=22, color=PROFILE_COLORS[profile], edgecolor="white", linewidth=0.5, label=profile.capitalize())
    axg.axvline(0, color=BORDER, linestyle="--", linewidth=0.7)
    axg.set_yticks(range(len(signature_order)), [_short_signature(value) for value in signature_order], fontsize=4.4)
    axg.tick_params(axis="y", pad=1)
    axg.set_xlabel("Exact Mantel rho")
    axg.set_title("Specification coding", loc="left")
    clean_axis(axg)

    loo = data["leave_one_out"]
    methods = list(dict.fromkeys(loo["omitted_method"]))
    for profile, offset in [("empirical", -0.10), ("simulation", 0.10)]:
        sub = loo[loo["profile"].eq(profile)].set_index("omitted_method").reindex(methods)
        y = np.arange(len(methods)) + offset
        axh.scatter(sub["mantel_spearman_rho"], y, s=24, color=PROFILE_COLORS[profile], edgecolor="white", linewidth=0.5, label=profile.capitalize())
    axh.axvline(0, color=BORDER, linestyle="--", linewidth=0.7)
    axh.set_yticks(range(len(methods)), methods)
    axh.set_xlabel("Leave-one-method-out Mantel rho")
    axh.set_title("Single-method influence", loc="left")
    clean_axis(axh)

    seeds = data["seed_summary"]
    offsets = {
        "GLM-PCA": (4, 4), "scScope": (4, -8), "SAUCIE": (4, 4),
        "UMAP": (4, -8), "PHATE": (4, 4), "PaCMAP": (4, 4),
    }
    for row in seeds.itertuples(index=False):
        color = FAMILY_COLORS[row.family]
        display_x = row.worst_context_knn_overlap
        if row.method == "PCA":
            display_x -= 0.012
        elif row.method == "t-SNE":
            display_x += 0.012
        axi.scatter(display_x, row.worst_context_distance_rank, s=34, color=color, edgecolor="white", linewidth=0.6)
        if row.method not in {"PCA", "t-SNE"}:
            dx, dy = offsets.get(row.method, (4, 4))
            axi.annotate(row.method, (display_x, row.worst_context_distance_rank), xytext=(dx, dy), textcoords="offset points", fontsize=5.0, color=INK)
    axi.annotate("PCA, t-SNE", (1.0, 1.0), xytext=(-4, -11), textcoords="offset points", ha="right", fontsize=5.0, color=INK)
    axi.set_xlabel("Worst-context kNN overlap")
    axi.set_ylabel("Worst-context distance-rank rho")
    axi.set_xlim(-0.03, 1.08)
    axi.set_ylim(0.70, 1.035)
    axi.set_title("Five-seed stability", loc="left")
    clean_axis(axi)

    loss = data["robustness_loss"].set_index("analysis_axis_resolved").reindex(list(AXIS_LABELS))
    bars = axj.bar(range(len(loss)), loss["support_loss_fraction"], color=[AXIS_COLORS[value] for value in loss.index], width=0.65)
    for bar, row in zip(bars, loss.itertuples()):
        axj.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.014, f"{row.n_pass_to_fail}/{row.n_baseline_supported}", ha="center", va="bottom", fontsize=5.6)
    axj.set_xticks(range(len(loss)), [AXIS_LABELS[value] for value in loss.index])
    axj.set_ylabel("Pass-to-fail fraction")
    axj.set_ylim(0, 0.24)
    axj.set_title("Perturbation-induced support loss", loc="left")
    clean_axis(axj)

    for letter, ax in zip("abcdefghij", axes):
        panel_label(ax, letter, x=-0.16, y=1.10)

    fig.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PROFILE_COLORS["empirical"], markeredgecolor="white", markersize=5, label="Empirical profile"),
            Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=PROFILE_COLORS["simulation"], markeredgecolor="white", markersize=5, label="Simulation profile"),
        ],
        frameon=False,
        fontsize=5.7,
        ncol=2,
        loc="lower center",
        bbox_to_anchor=(0.50, 0.006),
    )

    outputs = []
    for stem in ("Figure_7",):
        for ext in ("pdf", "svg", "png", "jpg"):
            path = FIGURE_DIR / f"{stem}.{ext}"
            kwargs: dict[str, object] = {"bbox_inches": "tight", "facecolor": "white"}
            if ext in {"png", "jpg"}:
                kwargs["dpi"] = 600
            if ext == "jpg":
                kwargs["pil_kwargs"] = {"quality": 95, "subsampling": 0}
            fig.savefig(path, **kwargs)
            outputs.append(path)
    plt.close(fig)
    return outputs


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    claim_matrix = _write_claim_matrix()
    data = _load()
    outputs = plot(data)
    metadata = {
        "figure": "Fig. 7",
        "core_conclusion": "Sensitivity analyses delimit which embedding interpretations are stable and which depend on diagnostic definition or analytical choice.",
        "archetype": "asymmetric quantitative grid",
        "panels": list("abcdefghij"),
        "claim_to_evidence_matrix": claim_matrix.name,
        "outputs": [path.name for path in outputs],
    }
    metadata_path = META_DIR / "Fig7_run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for path in [*outputs, claim_matrix, metadata_path]:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
