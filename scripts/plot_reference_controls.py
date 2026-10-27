"""Plot completed reference controls and a data-supplied conceptual map.

The empirical panels read public CSV results. No raw-data analysis, model fitting
or private manuscript/audit artifact is required. Use --check-only for read-only
schema/provenance checks, or --self-test for small semantic fixtures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from importlib.metadata import version
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import numpy as np
import pandas as pd

METHODS = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP", "scVI"]
ROOTS = ["7MEP_centroid", "9GMP_centroid", "1Ery_centroid", "stored_8Mk_root"]
ROOT_LABELS = ["7MEP", "9GMP", "1Ery", "8Mk"]
LINEAGES = ["erythroid", "megakaryocytic", "basophil", "myeloid"]
LINEAGE_LABELS = ["Ery", "MK", "Baso", "Myeloid"]
COUNTS = {}
SEEDS = []
SOURCE = OUT = None
DATASET_N = None
BLUE, ORANGE, GREEN = "#0072B2", "#D55E00", "#009E73"
INK, GREY = "#20242A", "#667078"
PLOT_ROWS = []
CHECKS = {}
READ_PATHS = {}
matplotlib = plt = DIVERGING = Line2D = FancyArrowPatch = Rectangle = Text = apply_style = None
REQUIRED_CSV = [
    "reference_root_concordance.csv", "crossed_root_agreement.csv",
    "fixed_side_contrasts.csv", "matched_root_seed_summary.csv",
    "marker_consistency.csv", "legacy_geometry_continuum.csv",
]


def configure_inputs(args):
    global SOURCE, OUT
    from analysis import paths
    candidates = [
        paths.RESULTS_DIR / "reference_attribution",
        paths.PUBLIC_SOURCE_DATA_DIR / "reference_attribution",
        paths.PUBLIC_SOURCE_DATA_DIR / "reference_controls",
        paths.PUBLIC_SOURCE_DATA_DIR / "generated/reference_attribution",
        paths.PUBLIC_SOURCE_DATA_DIR,
    ]
    SOURCE = args.source_dir
    if SOURCE is None:
        SOURCE = next((p for p in candidates if (p / "crossed_root_agreement.csv").is_file()), candidates[1])
    SOURCE = SOURCE.expanduser().resolve()
    OUT = (args.output_dir or CODE_ROOT / "outputs/reference_controls").expanduser().resolve()
    if SOURCE == OUT or SOURCE.is_relative_to(OUT):
        raise ValueError("Output must not contain or equal the source directory")
    concept_candidates = [
        SOURCE / "Supplementary_Fig_S20_conceptual_source.json",
        paths.PUBLIC_SOURCE_DATA_DIR / "Supplementary_Fig_S20_conceptual_source.json",
        paths.PUBLIC_SOURCE_DATA_DIR / "reference_attribution/Supplementary_Fig_S20_conceptual_source.json",
        paths.PUBLIC_SOURCE_DATA_DIR / "generated/Supplementary_Fig_S20_conceptual_source.json",
    ]
    concept = args.conceptual_source or next((p for p in concept_candidates if p.is_file()), concept_candidates[0])
    concept = concept.expanduser().resolve()
    if concept == (OUT / "Supplementary_Fig_S20_conceptual_source.json").resolve():
        raise ValueError("Conceptual source and exported conceptual record must be separate files")
    return concept


def load_plot_runtime():
    global matplotlib, plt, DIVERGING, Line2D, FancyArrowPatch, Rectangle, Text, apply_style
    os.environ["MPLCONFIGDIR"] = str(OUT / "Supplementary_Fig_S19_matplotlib_cache")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle
    from matplotlib.text import Text
    from figures.style import apply_style
    DIVERGING = LinearSegmentedColormap.from_list("reference_rho", [BLUE, "#FFFFFF", ORANGE])


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def assert_close(a, b, label: str, atol: float = 1e-10) -> None:
    if not np.allclose(a, b, atol=atol, rtol=0, equal_nan=False):
        raise AssertionError(label)




def validate_conceptual_source(value):
    ids = ["claim", "reference", "candidate", "endpoint", "stress", "interpret"]
    branches = ["reference_alternative", "endpoint_alternative", "stress_alternative"]
    if not value.get("conceptual") or value.get("empirical") is not False or value.get("data_source_class") != "D0":
        raise ValueError("S20 source must explicitly declare conceptual D0, not empirical data")
    nodes = value.get("nodes", [])
    mapping = {node.get("id"): node for node in nodes}
    if len(mapping) != len(nodes) or set(mapping) != set(ids + branches):
        raise ValueError("S20 needs six reasoning nodes and three alternative/stop nodes")
    if any(not isinstance(node.get(key), str) or not node[key].strip()
           for node in nodes for key in ["id", "title", "body"]):
        raise ValueError("Each conceptual node needs nonempty id, title and body")
    edges = value.get("edges", [])
    expected = set(zip(ids[:-1], ids[1:])) | {("reference", branches[0]), ("endpoint", branches[1]), ("stress", branches[2])}
    if len(edges) != len(expected) or {(edge.get("from"), edge.get("to")) for edge in edges} != expected:
        raise ValueError("Conceptual edges must match the declared decision path and alternatives")
    if any(not isinstance(edge.get("condition"), str) or not edge["condition"].strip() for edge in edges):
        raise ValueError("Each conceptual connection needs a condition")
    return mapping


def read_sources():
    global COUNTS, SEEDS, DATASET_N
    frames = {}
    for name in REQUIRED_CSV:
        path = SOURCE / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing public reference-control source: {path}; supply --source-dir")
        frames[name] = pd.read_csv(path)
        READ_PATHS[name] = path
    crossed = frames["crossed_root_agreement.csv"]
    keys = ["method", "seed", "expression_root", "representation_root", "scope", "mask"]
    if crossed.duplicated(keys).any() or set(crossed.method) != set(METHODS):
        raise ValueError("Crossed source has duplicate keys or a different method panel")
    if set(crossed.expression_root) != set(ROOTS) or set(crossed.representation_root) != set(ROOTS):
        raise ValueError("Both sides must contain all four root definitions")
    SEEDS = sorted(int(v) for v in crossed.seed.unique())
    if 0 not in SEEDS:
        raise ValueError("The fixed-side and legacy figure examples require seed zero")
    if len(crossed) != len(METHODS) * len(SEEDS) * 4 * 4 * 5 * 3:
        raise ValueError("Incomplete crossed roots, scopes, masks or seeds")
    common = crossed[crossed["mask"] == "global_common"]
    for scope in ["pooled", *LINEAGES]:
        values = common.loc[common.scope == scope, "n_cells"].unique()
        if len(values) != 1 or not int(values[0]) > 0:
            raise ValueError("A common-mask population must be fixed and nonempty within each scope")
        COUNTS[scope] = int(values[0])
    DATASET_N = int(crossed.loc[crossed.scope == "pooled", "n_scope"].max())
    hashes = {}
    optional = ["analysis_design.json", "summary.json", "validation.json", "common_cell_membership.csv.gz"]
    for name in optional:
        path = SOURCE / name
        if path.is_file():
            READ_PATHS[name] = path
    summary = {}
    if "summary.json" in READ_PATHS:
        summary = json.loads(READ_PATHS["summary.json"].read_text(encoding="utf-8"))
        if summary["status"] != "completed" or summary["n_global_common"] != COUNTS["pooled"] or summary["seeds"] != SEEDS:
            raise ValueError("Summary metadata disagree with CSV populations or seeds")
        if summary["global_common_lineage_n"] != COUNTS:
            raise ValueError("Summary lineage counts differ from the CSV")
    if "validation.json" in READ_PATHS:
        validation = json.loads(READ_PATHS["validation.json"].read_text(encoding="utf-8"))
        for key in ["all_primary_cell_counts_fixed_by_scope", "all_disconnected_values_missing", "all_inputs_hash_unchanged"]:
            if not validation.get(key):
                raise ValueError(f"Source validation failed: {key}")
    if "common_cell_membership.csv.gz" in READ_PATHS:
        membership = pd.read_csv(READ_PATHS["common_cell_membership.csv.gz"])
        frames["common_cell_membership.csv.gz"] = membership
        if membership.duplicated(["cell_id", "seed"]).any() or membership.cell_id.nunique() != DATASET_N:
            raise ValueError("Cell membership IDs are not unique/aligned by seed")
        for seed in SEEDS:
            subset = membership[membership.seed == seed]
            if int(subset.global_all_roots_all_methods_all_seeds_finite.sum()) != COUNTS["pooled"]:
                raise ValueError("Common cell membership differs from the plotted population")
    for name, path in READ_PATHS.items():
        hashes[name] = digest(path)
    manifest = SOURCE / "output_manifest.json"
    verified = []
    if manifest.is_file():
        entries = {row["path"]: row["sha256"] for row in json.loads(manifest.read_text(encoding="utf-8"))}
        for name, observed in hashes.items():
            if name in entries:
                if entries[name] != observed:
                    raise ValueError(f"Source manifest hash mismatch: {name}")
                verified.append(name)
        READ_PATHS["output_manifest.json"] = manifest
        hashes["output_manifest.json"] = digest(manifest)
    CHECKS.update(global_common_n=COUNTS["pooled"], excluded_n=DATASET_N - COUNTS["pooled"],
                  lineage_n=dict(COUNTS), source_validation_supplied="validation.json" in READ_PATHS,
                  membership_ids_supplied="common_cell_membership.csv.gz" in READ_PATHS,
                  unique_coordinate_hashes=summary.get("unique_coordinate_hashes_by_method", "not supplied"),
                  manifest_verified_files=verified, no_p_values=True, no_cutoff_on_crossed_dpt=True)
    check_record_semantics(frames)
    return frames, hashes

def check_record_semantics(frames):
    crossed = frames["crossed_root_agreement.csv"]
    common = crossed[crossed["mask"] == "global_common"]
    reference = frames["reference_root_concordance.csv"]
    reference = reference[reference["mask"] == "global_common"]
    marker = frames["marker_consistency.csv"]
    marker = marker[(marker.method == "expression_PCA50") & (marker["mask"] == "global_common")]
    for scope, count in COUNTS.items():
        rows = reference[reference.scope == scope]
        if len(rows) != 16 or not (rows.n_cells == count).all():
            raise ValueError("Reference and crossed figures must use the same common population")
        if scope != "pooled":
            rows = marker[marker.lineage == scope]
            if len(rows) != 4 or not (rows.n_cells == count).all():
                raise ValueError("Marker source does not match the common lineage population")
    same = common[common.expression_root == common.representation_root]
    keys = ["method", "scope", "expression_root"]
    computed = same.groupby(keys).agg(
        n_seeds=("seed", "nunique"), n_cells=("n_cells", "min"),
        signed_rho_median=("signed_spearman", "median"), signed_rho_min=("signed_spearman", "min"),
        signed_rho_max=("signed_spearman", "max"), distance_rho_median=("absolute_distance_spearman", "median"),
        distance_rho_min=("absolute_distance_spearman", "min"), distance_rho_max=("absolute_distance_spearman", "max"))
    stored = frames["matched_root_seed_summary.csv"]
    if stored.duplicated(keys).any():
        raise ValueError("Duplicate matched-root summary keys")
    stored = stored.set_index(keys)
    if set(computed.index) != set(stored.index):
        raise ValueError("Matched-root summary covers different methods, scopes or roots")
    assert_close(computed.to_numpy(), stored.loc[computed.index, computed.columns].to_numpy(),
                 "Seed-summary data disagree with the actual crossed comparisons")
    example = common[(common.method == "scVI") & (common.seed == 0) & (common.scope == "pooled")]
    contrasts = frames["fixed_side_contrasts.csv"]
    for side in ["expression", "representation"]:
        other = "representation" if side == "expression" else "expression"
        values = example[example[f"{other}_root"] == ROOTS[0]].set_index(f"{side}_root")
        for root in ROOTS[1:]:
            row = contrasts[(contrasts.method == "scVI") & (contrasts.seed == 0)
                            & (contrasts.scope == "pooled") & (contrasts.metric == "signed_spearman")
                            & (contrasts.varied_side == side) & (contrasts.fixed_root == ROOTS[0])
                            & (contrasts.root_a == ROOTS[0]) & (contrasts.root_b == root)]
            if len(row) != 1:
                raise ValueError("Fixed-side source must identify one actual matching comparison")
            assert_close([row.iloc[0].value_a, row.iloc[0].value_b],
                         [values.loc[ROOTS[0], "signed_spearman"], values.loc[root, "signed_spearman"]],
                         "Fixed-side source disagrees with crossed-DPT values")
    CHECKS.update(population_semantics_checked=True, seed_summaries_reconciled=True,
                  fixed_side_example_reconciled=True)


def record(panel: str, frame: pd.DataFrame, filename: str, kind: str = "source") -> None:
    copy = frame.copy()
    copy.insert(0, "panel", panel)
    copy.insert(1, "record_type", kind)
    copy.insert(2, "source_file", filename)
    PLOT_ROWS.append(copy)


def configure_style() -> None:
    apply_style()
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial"], "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "legend.fontsize": 8, "legend.frameon": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.edgecolor": GREY, "axes.linewidth": 0.6, "axes.unicode_minus": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "figure.dpi": 150,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    })


def heading(ax, letter: str, text: str) -> None:
    ax.set_title(text, loc="left", pad=11, fontsize=9, fontweight="semibold")
    ax.text(-0.25, 1.15, letter, transform=ax.transAxes, fontsize=10,
            fontweight="bold", ha="left", va="center", clip_on=False)


def horizontal_axis(ax, limits, label, *, zero=True) -> None:
    ax.set_xlim(*limits)
    ax.set_xlabel(label)
    if zero:
        ax.axvline(0, lw=0.65, color="#A4A9AD", zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def heatmap(fig, ax, values, xlabels, ylabels, *, signed: bool, label: str) -> None:
    vmin, vmax = (-1, 1) if signed else (0, 1)
    cmap = DIVERGING if signed else matplotlib.colormaps["cividis"]
    values = np.asarray(values, dtype=float)
    assert np.isfinite(values).all()
    image = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", interpolation="nearest")
    ax.set_xticks(np.arange(len(xlabels)), xlabels)
    ax.set_yticks(np.arange(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for (row, col), value in np.ndenumerate(values):
        rgba = cmap((value - vmin) / (vmax - vmin))
        luminance = sum(w * c for w, c in zip([0.2126, 0.7152, 0.0722], rgba[:3]))
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8,
                color="white" if luminance < 0.46 else INK)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.047, pad=0.04, ticks=[vmin, (vmin + vmax) / 2, vmax])
    colorbar.ax.set_ylabel(label, fontsize=8, labelpad=4)
    colorbar.ax.tick_params(labelsize=8, length=2)
    colorbar.outline.set_linewidth(0.5)


def interval_rows(ax, values: pd.DataFrame, metric_specs, ylabels) -> pd.DataFrame:
    summary_rows = []
    for y, method in enumerate(METHODS):
        rows = values[values.method == method].sort_values("seed")
        assert rows.seed.tolist() == SEEDS
        for metric, color, marker, offset in metric_specs:
            numbers = rows[metric].to_numpy(dtype=float)
            low, med, high = float(np.min(numbers)), float(np.median(numbers)), float(np.max(numbers))
            ax.plot([low, high], [y + offset, y + offset], lw=1.05, color=color)
            ax.scatter([med], [y + offset], s=17, marker=marker, facecolor="white",
                       edgecolor=color, linewidth=1, zorder=3)
            summary_rows.append({"method": method, "metric": metric, "minimum": low,
                                 "median": med, "maximum": high, "n_indexed_seeds": len(SEEDS)})
    ax.set_yticks(range(len(METHODS)), ylabels)
    ax.set_ylim(len(METHODS) - 0.5, -0.5)
    return pd.DataFrame(summary_rows)


def make_s19(frames: dict[str, pd.DataFrame]):
    fig, axes = plt.subplots(4, 2, figsize=(7, 11.1))
    fig.subplots_adjust(left=0.13, right=0.89, bottom=0.08, top=0.95, hspace=1.02, wspace=0.80)
    reference = frames["reference_root_concordance.csv"]
    primary = reference[reference["mask"] == "global_common"]
    crossed = frames["crossed_root_agreement.csv"]
    common = crossed[crossed["mask"] == "global_common"]
    matched = common[(common.expression_root == ROOTS[0]) & (common.representation_root == ROOTS[0])]

    ax = axes[0, 0]
    a = primary[primary.scope == "pooled"]
    assert len(a) == 16 and (a.n_cells == COUNTS["pooled"]).all()
    grid = a.pivot(index="expression_root_a", columns="expression_root_b", values="signed_spearman").loc[ROOTS, ROOTS]
    heatmap(fig, ax, grid, ROOT_LABELS, ROOT_LABELS, signed=True, label="Signed rho")
    ax.set_xlabel("Expression root B")
    ax.set_ylabel("Expression root A")
    heading(ax, "a", "Reference root changes")
    record("a", a, "reference_root_concordance.csv")

    ax = axes[0, 1]
    scopes = ["pooled", *LINEAGES]
    b = primary[(primary.expression_root_a == ROOTS[0]) & (primary.expression_root_b == ROOTS[2])].set_index("scope").loc[scopes].reset_index()
    assert b.n_cells.tolist() == [COUNTS[s] for s in scopes]
    for y, row in b.iterrows():
        ax.scatter(row.signed_spearman, y - 0.12, s=23, color=BLUE, marker="o")
        ax.scatter(row.absolute_distance_spearman, y + 0.12, s=23, color=ORANGE, marker="s")
    ax.set_yticks(range(5), ["Pooled", *LINEAGE_LABELS])
    ax.set_ylim(4.6, -0.6)
    horizontal_axis(ax, (-1.1, 1.1), "Spearman rho")
    ax.set_xticks([-1, 0, 1])
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", label="Order"),
               Line2D([], [], color=ORANGE, marker="s", ls="", label="Distance")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.35), ncol=2,
              handletextpad=0.35, columnspacing=0.7, borderaxespad=0)
    residuals = b.set_index("scope").rank_mae_best_orientation
    ax.text(0, -0.63, f"Rank MAE: pooled {residuals['pooled']:.3f}; Ery {residuals['erythroid']:.3f}",
            transform=ax.transAxes, fontsize=8, va="top")
    heading(ax, "b", "Reversal is lineage specific")
    record("b", b, "reference_root_concordance.csv")

    ax = axes[1, 0]
    marker = frames["marker_consistency.csv"]
    c = marker[(marker.method == "expression_PCA50") & (marker.seed == -1)
               & (marker["mask"] == "global_common") & marker.root_definition.isin([ROOTS[0], ROOTS[2]])]
    assert len(c) == 8
    for y, lineage in enumerate(LINEAGES):
        row = c[c.lineage == lineage].set_index("root_definition")
        x0, x1 = row.loc[ROOTS[0], "lineage_program_spearman"], row.loc[ROOTS[2], "lineage_program_spearman"]
        ax.plot([x0, x1], [y, y], color="#B6BBC0", lw=0.8)
        ax.scatter(x0, y - 0.055, color=BLUE, s=22, marker="o", zorder=3)
        ax.scatter(x1, y + 0.055, color=ORANGE, s=22, marker="s", zorder=3)
    ax.set_yticks(range(4), LINEAGE_LABELS)
    ax.set_ylim(3.5, -0.5)
    horizontal_axis(ax, (-1.05, 1.05), "Marker-pseudotime rho")
    ax.set_xticks([-1, 0, 1])
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", label="7MEP"),
               Line2D([], [], color=ORANGE, marker="s", ls="", label="1Ery")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.04, -0.43), ncol=2,
              handletextpad=0.35, columnspacing=0.8, borderaxespad=0)
    heading(ax, "c", "Marker anchors differ")
    record("c", c, "marker_consistency.csv")

    ax = axes[1, 1]
    legacy = frames["legacy_geometry_continuum.csv"]
    d = legacy[(legacy.seed == 0) & (legacy.scope == "pooled")]
    assert len(d) == len(METHODS) * len(ROOTS)
    grid = d.pivot(index="method", columns="expression_root", values="pseudotime_distance_correlation").loc[METHODS, ROOTS]
    heatmap(fig, ax, grid, ROOT_LABELS, METHODS, signed=bool(np.min(grid.to_numpy()) < 0), label="Distance rho")
    ax.set_xlabel("Expression root")
    heading(ax, "d", "Legacy distance diagnostic")
    record("d", d.drop(columns=[x for x in d.columns if "cutoff" in x]), "legacy_geometry_continuum.csv")

    ax = axes[2, 0]
    example = common[(common.method == "scVI") & (common.seed == 0) & (common.scope == "pooled")]
    contrasts = frames["fixed_side_contrasts.csv"]
    example_records = []
    for side, color, symbol, offset in [("expression", BLUE, "o", -0.11), ("representation", ORANGE, "s", 0.11)]:
        other = "representation" if side == "expression" else "expression"
        rows = example[example[f"{other}_root"] == ROOTS[0]].set_index(f"{side}_root").loc[ROOTS].reset_index()
        assert len(rows) == 4 and (rows.n_cells == COUNTS["pooled"]).all()
        for y, row in rows.iterrows():
            ax.scatter(row.signed_spearman, y + offset, color=color, marker=symbol, s=23)
            if y:
                audit = contrasts[(contrasts.method == "scVI") & (contrasts.seed == 0)
                                  & (contrasts.scope == "pooled") & (contrasts.metric == "signed_spearman")
                                  & (contrasts.varied_side == side) & (contrasts.fixed_root == ROOTS[0])
                                  & (contrasts.root_a == ROOTS[0]) & (contrasts.root_b == ROOTS[y])]
                assert len(audit) == 1
                assert_close(row.signed_spearman, audit.iloc[0].value_b, "Fixed-side example")
        rows["varied_side"] = side
        example_records.append(rows)
    e = pd.concat(example_records, ignore_index=True)
    ax.set_yticks(range(4), ROOT_LABELS)
    ax.set_ylim(3.5, -0.5)
    horizontal_axis(ax, (-1.05, 1.05), "Signed DPT order rho")
    ax.set_xticks([-1, 0, 1])
    handles = [Line2D([], [], color=BLUE, marker="o", ls="", label="Expression varied"),
               Line2D([], [], color=ORANGE, marker="s", ls="", label="Representation varied")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(-0.05, -0.44),
              handletextpad=0.4, borderaxespad=0, labelspacing=0.45)
    heading(ax, "e", "scVI seed 0, one root varied")
    record("e", e, "crossed_root_agreement.csv")
    CHECKS["fixed_side_example_scvi_seed0"] = {side: group.signed_spearman.tolist() for side, group in e.groupby("varied_side")}

    ax = axes[2, 1]
    f = matched[matched.scope == "pooled"]
    assert len(f) == len(METHODS) * len(SEEDS)
    summaries = interval_rows(ax, f, [("signed_spearman", BLUE, "o", -0.14),
                                     ("absolute_distance_spearman", ORANGE, "s", 0.14)], METHODS)
    horizontal_axis(ax, (min(-0.2, float(f[["signed_spearman", "absolute_distance_spearman"]].min().min()) - 0.03), 1.03), "DPT agreement (rho)")
    ax.set_xticks([0, 0.5, 1])
    heading(ax, "f", "Matched-root agreement")
    ax.legend(handles=[Line2D([], [], color=BLUE, marker="o", markerfacecolor="white", label="Order"),
                       Line2D([], [], color=ORANGE, marker="s", markerfacecolor="white", label="Distance")],
              loc="upper left", bbox_to_anchor=(-0.05, -0.44), ncol=2,
              handletextpad=0.35, columnspacing=0.7, borderaxespad=0)
    known = frames["matched_root_seed_summary.csv"]
    known = known[(known.scope == "pooled") & (known.expression_root == ROOTS[0])].set_index("method")
    for metric, prefix in [("signed_spearman", "signed_rho"), ("absolute_distance_spearman", "distance_rho")]:
        part = summaries[summaries.metric == metric].set_index("method").loc[METHODS]
        for generated, stored in [("minimum", "min"), ("median", "median"), ("maximum", "max")]:
            assert_close(part[generated], known.loc[METHODS, f"{prefix}_{stored}"], "Seed-summary reconciliation")
    record("f", f, "crossed_root_agreement.csv")
    record("f", summaries, "crossed_root_agreement.csv", "median_range")

    ax = axes[3, 0]
    g = matched[matched.scope.isin(LINEAGES)]
    assert len(g) == len(METHODS) * len(SEEDS) * len(LINEAGES)
    medians = g.groupby(["method", "scope"], as_index=False).signed_spearman.median()
    grid = medians.pivot(index="method", columns="scope", values="signed_spearman").loc[METHODS, LINEAGES]
    heatmap(fig, ax, grid, LINEAGE_LABELS, METHODS, signed=True, label="Signed rho")
    ax.set_xlabel("Lineage (7MEP roots)")
    heading(ax, "g", "Within-lineage order")
    record("g", g, "crossed_root_agreement.csv")
    record("g", medians, "crossed_root_agreement.csv", "median")

    ax = axes[3, 1]
    all_matched = crossed[(crossed.scope == "pooled") & (crossed.expression_root == ROOTS[0])
                          & (crossed.representation_root == ROOTS[0])]
    h = all_matched[all_matched["mask"] == "global_common"][["method", "seed", "signed_spearman", "n_cells"]].merge(
        all_matched[all_matched["mask"] == "pair_finite"][["method", "seed", "signed_spearman", "n_cells"]],
        on=["method", "seed"], suffixes=("_common", "_pair_finite"), validate="one_to_one")
    h["delta_signed_rho"] = h.signed_spearman_pair_finite - h.signed_spearman_common
    assert len(h) == len(METHODS) * len(SEEDS) and (h.n_cells_common == COUNTS["pooled"]).all()
    maximum = float(h.delta_signed_rho.abs().max())
    if not np.isfinite(h.delta_signed_rho).all():
        raise ValueError("Undefined mask differences cannot be silently plotted as zero")
    summaries = interval_rows(ax, h, [("delta_signed_rho", BLUE, "o", 0)], METHODS)
    limit = 0.09 if maximum <= 0.09 else maximum * 1.15
    horizontal_axis(ax, (-limit, limit), "Delta signed rho\nPair-finite minus common")
    ax.set_xticks([-0.08, 0, 0.08] if limit == 0.09 else [-limit, 0, limit])
    heading(ax, "h", "Finite-mask sensitivity")
    record("h", h, "crossed_root_agreement.csv", "paired_mask_difference")
    record("h", summaries, "crossed_root_agreement.csv", "median_range")
    CHECKS["maximum_absolute_mask_delta"] = maximum
    CHECKS["pair_finite_n_range"] = [int(h.n_cells_pair_finite.min()), int(h.n_cells_pair_finite.max())]
    CHECKS["common_reference_7MEP_vs_1Ery"] = b[["scope", "signed_spearman", "absolute_distance_spearman",
                                               "rank_mae_best_orientation", "n_cells", "n_pairs"]].to_dict("records")
    return fig


def make_s20(conceptual):
    """Render the supplied conceptual wording without replacing node titles."""
    mapping = validate_conceptual_source(conceptual)
    fig, ax = plt.subplots(figsize=(7, 9.3))
    fig.subplots_adjust(left=0.035, right=0.98, bottom=0.055, top=0.91)
    fig.text(0.5, 0.966, "Conceptual model; not a quantitative measurement.", ha="center", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    order = ["claim", "reference", "candidate", "endpoint", "stress", "interpret"]
    centers = [0.92, 0.76, 0.60, 0.44, 0.28, 0.12]
    colors = [BLUE, GREEN, BLUE, ORANGE, GREEN, INK]
    x, width, height = 0.025, 0.635, 0.112
    for i, (node_id, y, color) in enumerate(zip(order, centers, colors)):
        node = mapping[node_id]
        ax.add_patch(Rectangle((x, y - height / 2), width, height, facecolor="white", edgecolor="#A2ABB1", lw=0.7))
        ax.add_patch(Rectangle((x, y - height / 2), 0.008, height, facecolor=color, edgecolor="none"))
        ax.text(x + 0.023, y + 0.024, f"{i + 1}  {node['title']}", fontsize=9, fontweight="semibold", va="center")
        ax.text(x + 0.023, y - 0.018, node["body"], fontsize=8, va="center", linespacing=1.4)
        if i:
            ax.add_patch(FancyArrowPatch((x + width / 2, centers[i - 1] - height / 2 - 0.004),
                                        (x + width / 2, y + height / 2 + 0.004),
                                        arrowstyle="-|>", mutation_scale=10, color=GREY, lw=0.9))
    for index in [1, 3, 4]:
        node_id = order[index]
        branch_id = node_id + "_alternative"
        node = mapping[branch_id]
        edge = next(e for e in conceptual["edges"] if e["from"] == node_id and e["to"] == branch_id)
        y, bx, bw, bh = centers[index], 0.765, 0.23, 0.125
        ax.add_patch(Rectangle((bx, y - bh / 2), bw, bh, facecolor="#FAFBFC", edgecolor="#9EA7AE", lw=0.7))
        ax.text(bx + bw / 2, y + 0.035, node["title"], ha="center", va="center", fontsize=8, fontweight="semibold")
        ax.text(bx + bw / 2, y - 0.017, node["body"], ha="center", va="center", fontsize=8, linespacing=1.3)
        ax.add_patch(FancyArrowPatch((x + width + 0.006, y), (bx - 0.006, y),
                                    arrowstyle="-|>", mutation_scale=9, color=GREY, lw=0.8))
        ax.text((x + width + bx) / 2, y + 0.015, edge["condition"], ha="center", va="bottom", fontsize=8)
    fig.text(0.5, 0.027, "No fitted result, method ranking or universal reliability threshold is shown.", ha="center", fontsize=8)
    return fig

def text_audit(fig) -> dict[str, object]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    width, height = fig.canvas.get_width_height()
    boxes, small, outside, near_edge = [], [], [], []
    clearances = []
    seen = set()
    for item in fig.findobj(Text):
        if id(item) in seen or not item.get_visible() or not item.get_text().strip():
            continue
        seen.add(id(item))
        text = item.get_text()
        if item.get_fontsize() < 8:
            small.append(text)
        box = item.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
            outside.append(text)
        clearance = min(box.x0, box.y0, width - box.x1, height - box.y1) * 72 / fig.dpi
        clearances.append(float(clearance))
        if clearance < 6:
            near_edge.append(text)
        boxes.append((text, [float(v) for v in box.extents]))
    if small or outside or near_edge:
        raise AssertionError(f"Unreadable or unsafe-margin text: small={small}; outside={outside}; near_edge={near_edge}")
    overlaps = []
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            dx, dy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
            if dx > 2 and dy > 2:
                overlaps.append([ta, tb, round(dx, 1), round(dy, 1)])
    return {"minimum_font_pt": 8, "visible_text_count": len(boxes), "outside_page": outside,
            "required_canvas_clearance_pt": 6, "minimum_canvas_clearance_pt": min(clearances),
            "text_inside_safe_margin": True,
            "text_overlap_candidates": overlaps, "width_inches": float(fig.get_figwidth()),
            "height_inches": float(fig.get_figheight())}


def export_figure(fig, prefix: str) -> dict[str, object]:
    layout = text_audit(fig)
    if layout["text_overlap_candidates"]:
        print(prefix, "text overlaps:", layout["text_overlap_candidates"])
        raise AssertionError("Correct text overlaps before final export")
    outputs = {}
    for ext in ["png", "pdf", "svg"]:
        path = OUT / f"{prefix}.{ext}"
        kwargs = {"facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 300
        fig.savefig(path, **kwargs)
        outputs[path.name] = {"sha256": digest(path), "bytes": path.stat().st_size}
    plt.close(fig)
    return {"layout": layout, "outputs": outputs}


def export_common_values(frames: dict[str, pd.DataFrame]) -> Path:
    reference = frames["reference_root_concordance.csv"]
    reference = reference[reference["mask"] == "global_common"]
    columns = ["n_cells", "n_pairs", "signed_spearman", "absolute_distance_spearman",
               "rank_mae_direct", "rank_mae_reversed", "rank_mae_best_orientation"]
    pooled_rows = []
    for i, first in enumerate(ROOTS):
        for second in ROOTS[i + 1:]:
            row = reference[(reference.scope == "pooled") & (reference.expression_root_a == first)
                            & (reference.expression_root_b == second)].iloc[0]
            pooled_rows.append([first, second, *[row[c].item() if hasattr(row[c], "item") else row[c] for c in columns]])
    lineage_rows = []
    for scope in LINEAGES:
        row = reference[(reference.scope == scope) & (reference.expression_root_a == ROOTS[0])
                        & (reference.expression_root_b == ROOTS[2])].iloc[0]
        lineage_rows.append([scope, *[row[c].item() if hasattr(row[c], "item") else row[c] for c in columns]])
    markers = frames["marker_consistency.csv"]
    markers = markers[(markers.method == "expression_PCA50") & (markers["mask"] == "global_common")]
    marker_rows = []
    for scope in LINEAGES:
        values = markers[markers.lineage == scope].set_index("root_definition")
        marker_rows.append([scope, COUNTS[scope], float(values.loc[ROOTS[0], "lineage_program_spearman"]),
                            float(values.loc[ROOTS[2], "lineage_program_spearman"])])
    value = {"population": "global_common", "n_cells": COUNTS["pooled"], "dataset_n_cells": DATASET_N,
             "source": "reference_root_concordance.csv", "source_sha256": digest(SOURCE / "reference_root_concordance.csv"),
             "precision": "Values retained at the completed source CSV precision; no manual rounding.",
             "pooled_root_pair_columns": ["root_a", "root_b", *columns], "pooled_root_pairs": pooled_rows,
             "lineage_roots": [ROOTS[0], ROOTS[2]], "lineage_columns": ["lineage", *columns],
             "lineage_7MEP_vs_1Ery": lineage_rows,
             "marker_columns": ["lineage", "n_cells", "rho_7MEP", "rho_1Ery"], "marker_consistency": marker_rows,
             "interpretation": "Pooled root change is not a pure reversal; erythroid order is strongly but not exactly reversed. Masks overlap.",
             "do_not_mix": "Reference-only and common-cell populations can differ. Match the reported values to the plotted mask."}
    path = OUT / "Supplementary_Fig_S19_global_common_values.json"
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return path


def self_test():
    order = ["claim", "reference", "candidate", "endpoint", "stress", "interpret"]
    alternatives = ["reference_alternative", "endpoint_alternative", "stress_alternative"]
    nodes = [{"id": key, "title": "Conceptual test node", "body": "Schema fixture only"} for key in order + alternatives]
    task_titles = {"candidate": "Choose an analytical representation",
                   "endpoint": "Check the task outcome", "stress": "Check stability"}
    for node in nodes:
        if node["id"] in task_titles:
            node["title"] = task_titles[node["id"]]
    edges = [{"from": a, "to": b, "condition": "declared"} for a, b in zip(order[:-1], order[1:])]
    edges += [{"from": a, "to": a + "_alternative", "condition": "uncertain"} for a in ["reference", "endpoint", "stress"]]
    value = {"conceptual": True, "empirical": False, "data_source_class": "D0", "nodes": nodes, "edges": edges}
    mapping = validate_conceptual_source(value)
    assert len(mapping) == 9
    assert all(mapping[key]["title"] == title for key, title in task_titles.items())
    for invalid in [{**value, "empirical": True}, {**value, "edges": edges[:-1]},
                    {**value, "nodes": nodes + [nodes[0]]}]:
        try:
            validate_conceptual_source(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Conceptual provenance/schema mismatch accepted")
    samples = np.asarray([0.3, 0.1, 0.5, 0.4, 0.2])
    assert_close([samples.min(), np.median(samples), samples.max()], [0.1, 0.3, 0.5], "Computational range fixture")
    print(json.dumps({"status": "passed", "test": "conceptual provenance and range semantics",
                      "model_or_graph_fits": 0, "plots_rendered": 0, "files_written": 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="completed reference CSV directory; public data/results auto-discovery by default")
    parser.add_argument("--conceptual-source", type=Path, help="data-supplied S20 conceptual node/edge JSON")
    parser.add_argument("--output-dir", type=Path, help="default: outputs/reference_controls")
    parser.add_argument("--check-only", action="store_true", help="validate completed CSVs and conceptual source without rendering or writes")
    parser.add_argument("--self-test", action="store_true", help="small semantic fixtures only; no reads, fitting or writes")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    conceptual_path = configure_inputs(args)
    frames, input_hashes = read_sources()
    if not conceptual_path.is_file():
        raise FileNotFoundError(f"Supply public conceptual source JSON with --conceptual-source: {conceptual_path}")
    conceptual = json.loads(conceptual_path.read_text(encoding="utf-8"))
    validate_conceptual_source(conceptual)
    READ_PATHS["conceptual-source/" + conceptual_path.name] = conceptual_path
    input_hashes["conceptual-source/" + conceptual_path.name] = digest(conceptual_path)
    if args.check_only:
        print(json.dumps({"status": "passed", "global_common_n": COUNTS["pooled"],
                          "lineage_n": COUNTS, "seeds": SEEDS, "source_hashes": input_hashes,
                          "conceptual_nodes": len(conceptual["nodes"]), "plots_rendered": 0,
                          "files_written": 0}, indent=2))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    load_plot_runtime()
    configure_style()
    s19 = export_figure(make_s19(frames), "Supplementary_Fig_S19_reference_attribution")
    source_data = pd.concat(PLOT_ROWS, ignore_index=True, sort=False)
    source_path = OUT / "Supplementary_Fig_S19_source_data.csv"
    source_data.to_csv(source_path, index=False, float_format="%.12g")
    common_values_path = export_common_values(frames)
    s20 = export_figure(make_s20(conceptual), "Supplementary_Fig_S20_decision_map")
    public_concept = {key: conceptual[key] for key in ["data_source_class", "conceptual", "empirical", "nodes", "edges"]}
    (OUT / "Supplementary_Fig_S20_conceptual_source.json").write_text(json.dumps(public_concept, indent=2), encoding="utf-8")
    for name, expected in input_hashes.items():
        if digest(READ_PATHS[name]) != expected:
            raise ValueError(f"Source input changed while plotting: {name}")
    audit = {"script": str(Path(__file__).relative_to(CODE_ROOT)), "script_sha256": digest(Path(__file__)),
             "python": sys.version, "platform": platform.platform(),
             "versions": {name: version(name) for name in ["numpy", "pandas", "matplotlib"]},
             "input_hashes_at_start": input_hashes, "inputs_unchanged": True, "checks": CHECKS,
             "source_data_sha256": digest(source_path), "source_rows": len(source_data),
             "global_common_values_sha256": digest(common_values_path),
             "rows_by_panel_and_type": source_data.groupby(["panel", "record_type"]).size().rename("rows").reset_index().to_dict("records"),
             "S19": s19, "S20": s20,
             "evidence_counts": {"empirical_derived_sensitivity": 8, "conceptual_non_countable": 1,
                                 "simulation": 0, "independent_biological_validation": 0}}
    (OUT / "Supplementary_Fig_S19_plot_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"status": "completed", "n_common": COUNTS["pooled"], "S19_panels": 8,
                      "S20": "data-supplied conceptual map; not empirical", "source_rows": len(source_data),
                      "maximum_mask_delta": CHECKS["maximum_absolute_mask_delta"],
                      "outputs": [*s19["outputs"], *s20["outputs"]]}, indent=2))


if __name__ == "__main__":
    main()
