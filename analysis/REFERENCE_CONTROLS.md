# Portable Reference Controls

These controls evaluate root/reference dependence in saved Paul15 representations.
They do not train embedding models, establish biological truth, or estimate branch
topology. The numerical definitions of the crossed comparisons are unchanged.
Historical-array and published-table comparisons are optional audits, never inputs
to the newly calculated pseudotimes or a prerequisite for their calculation.

Run commands from the public code repository root, using the scientific environment
described in [README.md](README.md). Neither script needs a private checkout,
manuscript, correspondence, or review packet.

## Inputs and Paths

The analysis uses `analysis.paths`. CLI overrides are applied before importing the
trajectory module, so its marker-expression and processed-object paths follow the
same configuration.

| Setting | CLI override | Environment/default |
| --- | --- | --- |
| Analysis input root | `--analysis-data` | `SCDR_ANALYSIS_DATA`, otherwise `analysis/data` |
| scVI array directory | `--scvi-embeddings` | `SCDR_SCVI_EMBEDDINGS`, otherwise the `SCVI_EMBEDDINGS_DIR` in `analysis.paths` |
| Control output directory | `--output-dir` | `SCDR_ANALYSIS_RESULTS/reference_attribution`, otherwise `analysis/results/reference_attribution` |

The analysis input root must contain:

```text
analysis_objects/paul15_proc.h5ad
analysis_objects/paul15_counts_hvg.h5ad
expression_objects/paul15_expression.h5ad
anchor_embeddings/paul15_<method-stem>_seed<seed>.npy
```

Anchor stems are `pca`, `glmpca`, `scscope`, `saucie`, `umap`, `phate`, `tsne` and
`pacmap`. The separate scVI directory must contain
`paul15_scvi_dim2_seed<seed>.npy` and its `.json` metadata sidecar. The sidecar's
`input_sha256` must match the supplied count-file bytes. Its `latent_sha256` uses
the public scVI shape/dtype/content digest, not a `.npy` file digest or the graph
cache's different array digest. Processed and count objects must have exactly the
same ordered, unique cell IDs. All representations must follow that processed
object's row order and contain finite two-dimensional coordinates for its 2,730
cells. Arbitrary replacement datasets are not supported by this case-study module.

Use either the documented public release inputs or freshly fitted inputs from the
public workflow. See the archived and full-refit routes in [README.md](README.md).
For example, after obtaining release inputs with `python -m analysis.download_inputs`:

```bash
python -B -m analysis.scripts.reference_attribution_controls \
  --analysis-data analysis/data \
  --scvi-embeddings analysis/data/reference_results/embeddings \
  --output-dir analysis/results/reference_attribution
```

For a fresh workflow run, point the same CLI at that run's data and fitted scVI
directories rather than copying old fitted arrays into it:

```bash
python -B -m analysis.scripts.reference_attribution_controls \
  --analysis-data analysis/runs/public_refit/data \
  --scvi-embeddings analysis/runs/public_refit/results/embeddings \
  --output-dir analysis/runs/public_refit/results/reference_attribution
```

These commands are reproducibility instructions, not assertions that a new fit or
the complete controls were executed during a portability check. Five indexed
computational seeds, 0-4, are the default. `--seeds` permits sorted distinct subsets
including zero, but a smaller subset changes the global common mask and is not the
full five-seed comparison. No equality to an archived score or finite-cell count is
required of a fresh fit.

## Optional Audits

- If present, `analysis_object_manifest.csv` is recorded as input provenance.
- `data/source_data/fig3_paul15_embedding_coordinates.csv` is compared at seed zero
  when available. Supply another file with `--coordinate-table`. The audit checks
  cell-ID sets and matches coordinate rows by ID; it does not reorder the supplied
  arrays. Missing optional tables are not claimed as verified. A fresh stochastic
  fit can legitimately differ from published coordinates.
- A published S11 table at
  `data/source_data/generated/SuppS11_root_sensitive_continuum_metrics.csv` is
  audited when available. Override it with `--legacy-reproduction-table`. Expected
  columns are `method`, `root_definition`, `metric` and `value`. Missing historical
  tables produce a skipped status, zero checked rows and a null maximum difference,
  not a successful reproduction claim.
- `--historical-assignments` explicitly opts into auditing a previous single-root
  per-cell CSV. It must contain `method`, `cell_id`, `root_definition`,
  `method_dpt_pseudotime` and `reference_dpt_pseudotime`, with the expected seed-zero
  7MEP entries. It is never required or used as the source of current DPT values.
- `--strict-audits` requests equality checks for supplied historical tables and
  coordinate audits. This is useful for a matched archived-input run, not a
  prerequisite for analysis of fresh fitted coordinates. Without it, differences
  are recorded rather than silently equated with failed main calculations.

Core inputs such as scVI hash sidecars remain mandatory. Optional audits do not
relax those checks. Anchor-array shape/finiteness and a manifest alone cannot prove
row identity; absent matching coordinate evidence, the input row-order contract is
reported explicitly rather than described as independent verification.

## Preserved Calculation Definitions

- Nine methods, four roots, and the original overlapping lineage masks match the
  public `compute_trajectory` definitions. Runtime checks reject definition drift.
- Root cell IDs are selected once from the expression object: PCA20 centroid cells
  for 7MEP, 9GMP and 1Ery, plus the stored 8Mk root. They are identical on both sides.
- The expression reference uses all 50 saved PCA columns. Each fixed graph uses
  Euclidean UMAP-neighbour construction with 15 neighbours, graph seed zero, 15
  diffusion-map components with seed zero, ten DPT components and no branch calls.
  One graph/eigensystem is cached and rerooted; rerooting does not refit an embedding.
- Disconnected and other nonfinite pseudotimes remain missing. The primary mask
  intersects all roots, methods and requested seeds. Secondary masks are the
  within-seed intersection and the specific expression/representation finite pair.
- Pooled pairs use the original 5,000 nonself draws with seed 20260714. Lineage
  draws use `min(3000, 5*n_lineage)` and seed `20260714 + len(lineage)`. Draws occur
  before finite filtering and retain duplicate pairs.
- Signed Spearman and Kendall tau-b retain orientation. Absolute Spearman removes
  sign only. Absolute-distance Spearman compares `abs(t_i-t_j)` between DPT vectors.
  Direct/reversed rank MAE uses average-tie ranks scaled to `[0,1]`. A negative order
  coefficient alone does not identify a pure reversal. Constant correlations remain
  undefined, not zero.
- Crossed root comparisons, fixed-side deltas, lineage intersections and seed
  summaries preserve the original calculations. Marker scores use the existing
  public mean-log1p expression programmes, not fitted-map labels.
- The legacy geometry diagnostic is different: expression absolute-pseudotime
  differences versus fixed Euclidean embedding distances, plus local pseudotime
  retention. Its secondary 0.45/0.50 screening rules are not applied to crossed DPT.

Cells, pair draws, roots, seed labels and seed pairs are not independent biological
replicates. Median/minimum/maximum summaries describe computational variation, not
confidence intervals. Markers and alternative roots are internal consistency
evidence, not independent lineage tracing or causal embedding-failure validation.

## Outputs and Provenance

All analysis outputs, caches, status and manifests go under the selected output
directory. Inputs are read-only and are rehashed after calculation. Input records
use semantic `analysis-data/`, `scvi-embeddings/`, `code/` or `optional-audit/` labels
instead of embedding a checkout-specific absolute path. File bytes, graph array
content, and scVI shape/dtype/content hashes retain their distinct meanings.

The principal records are `reference_root_concordance.csv`,
`crossed_root_agreement.csv`, `representation_root_concordance.csv`,
`fixed_side_contrasts.csv`, `matched_root_seed_summary.csv`, `marker_consistency.csv`,
`legacy_geometry_continuum.csv`, `coverage.csv` and ID-level per-cell/mask/pair files.
`summary.json`, `validation.json`, `provenance.json` and `input_manifest.csv` record
the actual run and skipped/performed audits. No published sample size or score is
inserted as a computed result.

## Figure Source and Plotting

`scripts/plot_reference_controls.py` reads completed result tables. It does not
import the fitting runtime or read raw expression. With no `--source-dir`, it
searches the configured analysis results, then public source-data locations for a
directory containing `crossed_root_agreement.csv`. Supported public locations are
`data/source_data/reference_attribution`, `data/source_data/reference_controls`,
`data/source_data/generated/reference_attribution` and `data/source_data` itself.

Required CSVs in that directory are:

```text
reference_root_concordance.csv
crossed_root_agreement.csv
fixed_side_contrasts.csv
matched_root_seed_summary.csv
marker_consistency.csv
legacy_geometry_continuum.csv
```

Summary, validation, common-cell membership and analysis-design metadata are used
when supplied, but a private audit packet is not required. A supplied output
manifest is checked for the input files it covers. The plot audit distinguishes
CSV-structure checks from supplied ID/validation evidence; missing evidence is not
described as independently checked. Counts, seed numbers and display ranges are
read from the results, not fixed to the published 2,670-cell population.

S20 also requires the public data file
`Supplementary_Fig_S20_conceptual_source.json`. It can be colocated with those CSVs,
placed in the public source-data root/generated directory, or supplied with
`--conceptual-source`. The script reads its actual titles, text, conditions and
connections; it does not reconstruct the diagram from a manuscript. The JSON must
declare `conceptual: true`, `empirical: false`, `data_source_class: "D0"`, and
contain the six nodes `claim`, `reference`, `candidate`, `endpoint`, `stress`,
`interpret`, plus three `<reference|endpoint|stress>_alternative` nodes. Each node
has `id`, `title` and `body`. Eight edges with `from`, `to` and `condition` describe
the five sequential connections and the three alternatives. Conceptual status is
not an empirical measurement, method ranking or universally validated criterion.

```bash
python -B scripts/plot_reference_controls.py \
  --source-dir analysis/results/reference_attribution \
  --conceptual-source data/source_data/Supplementary_Fig_S20_conceptual_source.json \
  --output-dir outputs/reference_controls
```

This generates S19's eight empirical-derived sensitivity panels and S20's
non-empirical map in PNG300/PDF/SVG, plus panel source records, exact common-mask
values and a plot audit. Neither figure has an overall title in the artwork; S19
retains a-h headings and S20 retains its conceptual subtitle. Fonts are Arial at
least 8 pt. Undefined correlations trigger a clear failure rather than fabricated
zero values. Axes expand if newly fitted results exceed the archived ranges.
Output defaults to `outputs/reference_controls`, not the supplied publication
figure directory. Original source tables and published figures are not overwritten.

## Lightweight Tests

These are bounded semantic tests, not full analysis/refitting demonstrations:

```bash
python -B -m analysis.scripts.reference_attribution_controls --self-test
python -B scripts/plot_reference_controls.py --self-test
python -B scripts/plot_reference_controls.py \
  --source-dir analysis/results/reference_attribution \
  --conceptual-source data/source_data/Supplementary_Fig_S20_conceptual_source.json \
  --check-only
```

The analysis fixtures check exact reversal/distance invariance, missing-cell
exclusion, constant-vector missingness and semantic scVI provenance rejection for
different input hashes, digest conventions or row order. The plotting fixtures
check conceptual/non-empirical declarations, topology, duplicate nodes and range
definitions. Both self-tests fit zero models/graphs, render zero figures, and write
zero files. Check-only reads completed source tables and conceptual data, verifies
their structure/populations, supplied hashes, every matched-root seed summary and
the actual fixed-side example, and writes/renders nothing.
Numerical functions, finite-mask expressions and crossed/marker/legacy loops were
also compared structurally with the source implementation during this port.
Full analysis and figure regeneration are intentionally not part of these tests.
