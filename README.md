# Mathematical catalogue and diagnostic stress tests for single-cell dimensionality reduction

This repository contains the figure and analysis-reproduction materials for the manuscript *Mathematical characterization and empirical evaluation of single-cell transcriptomic dimensionality-reduction methods*.

## Contents

- `figures/`: plotting scripts and shared publication style.
- `data/source_data/`: supplied figure-level tables, with separate public control-source locations for S18-S20 described below.
- `scripts/`: portable command-line plotters for the additional control figures; no fitting is performed by these plotters.
- `metadata/`: mathematical specification records, public dataset sources and panel-level evidence mapping.
- `outputs/main_figures/`: final Figures 1-7 in publication formats.
- `outputs/supplementary_figures/`: S1-S21 outputs. S18-S21 have complete scalar sources and standalone plotters.
- `outputs/tables/`: Table 1 source and the Supplementary Tables S1-S21 workbook.
- `analysis/`: executable preprocessing, representation, downstream-analysis and verification workflow.

The lightweight checkout contains the established plotting inputs. Larger checksum-locked processed objects and fitted arrays for archived analysis are distributed through the [`analysis-reproducibility-v1.0.1` GitHub Release](https://github.com/wangsaidi/single-cell-dr-mechanism-catalogue/releases/tag/analysis-reproducibility-v1.0.1) and installed by `python -m analysis.download_inputs`. Do not assume that this historical release also contains the new S18-S20 control-source bundle. Public-source refitting instead starts from the access routes in `metadata/public_data_sources.csv`.

Supplementary Table S21 checks the Figure 4 association pattern using saved optimised scScope and reconstruction-only SAUCIE coordinates. Its complete [rescoring instructions](analysis/SAVED_REPRESENTATION_SENSITIVITY.md), measured records and 18 replacement coordinate arrays are included in this checkout. The other seven seed-zero representations remain fixed. This is a conditional implementation sensitivity, not another independent benchmark or a replacement of the historical main figures.

## Choose An Execution Route

Run commands from the repository directory containing `make_figures.py` and `analysis/` with the appropriate Python environment activated.

| Route | Consumed inputs | What is recomputed |
| --- | --- | --- |
| Figure-only | Supplied publication tables and, optionally, completed control-source files | Figure panels and plotting summaries; no model fitting or count simulation |
| Archived analysis | Exact checksum-verified processed objects and fitted representations | Diagnostics and downstream metrics, compared with read-only archived references; not a fresh fit |
| Public-source full refit | Public expression objects, including their documented supplied labels/subsets | Preprocessing, model fitting, six-scenario simulation generation, robustness, metrics and calibration |

The reduced `smoke` profile exercises all stage types with smaller budgets. A completed smoke run checks executable coverage only: it does not establish convergence, publication-scale agreement, a fresh environment build, or biological validity. Availability of an executable profile is not evidence that a full refit has completed.

A [verified functional-execution receipt](metadata/public_source_functional_validation.json) records successful coverage of all 21 required stages across an initial SAUCIE import-path failure and a forced resume, using reduced budgets and the complete original analysis cell populations. It retains the honest partial/resume flags, installed versions and source/evidence hashes, without private paths or smoke scores. This validates execution coverage only, not publication-scale refitting, convergence, numerical equivalence or a fresh environment build.

`FILE_MANIFEST.csv` records byte counts and SHA-256 hashes of the files as stored by GitHub, with LF line endings for Git-normalised text files. Windows checkouts may use CRLF line endings; compare the canonical Git contents when verifying text-file hashes. Binary figure and workbook hashes are unaffected.

## Reproducing the figures

Create a Python environment, install the tested dependencies and run the wrapper from the repository root.

Using Conda:

```bash
conda env create -f environment.yml
conda activate single-cell-dr-figures
python make_figures.py
```

Using a virtual environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python make_figures.py
```

Figure 1 is supplied as final conceptual artwork. With no options, the wrapper retains the established Figures 2-7 and S1-S17 module sequence; no S18-S20 assets are needed. Generated panel tables go to `data/source_data/generated/`. Figure 4 reports downstream outcomes for the eight anchor implementations plus scVI; the previous threshold-gate Figure 4 is retained as S14. S15 reports random-correspondence calibration, S16 reports scVI geometry/reference sensitivity/stability, and S17 reports the separate nine-method empirical specification sensitivity. The supplied workbook and combined supplementary PDF are not rebuilt or extended by the wrapper.

### Optional S18-S20 Figures

These additions use completed control results, not training commands. S18 reports deep-model training/configuration/projection controls; S19 reports crossed-root reference attribution; S20 is a data-supplied **conceptual decision map, not an empirical measurement**.

The public source layout is:

```text
data/source_data/
  deep_controls/
    matched_endpoint_scores.csv
    matched_population_scores.csv
    checkpoint_representation_stability.csv
    ... per-fit metadata, histories and other control provenance
  reference_attribution/
    reference_root_concordance.csv
    crossed_root_agreement.csv
    fixed_side_contrasts.csv
    matched_root_seed_summary.csv
    marker_consistency.csv
    legacy_geometry_continuum.csv
    Supplementary_Fig_S20_conceptual_source.json
    ... design, validation and provenance records
```

These locations contain completed control measurements, not smoke scores or partial fits. S18 includes per-fit provenance and histories. Explicit S18-S20 selections check required inputs before launching any figure. The wrapper never downloads, trains or invents missing control evidence.

Validate the supplied scalar plotting inputs without plotting or writing outputs:

```bash
python make_figures.py --revision-supplements-only --check-only
```

Render only the additions, or append them to the established figure sequence:

```bash
python make_figures.py --revision-supplements-only
python make_figures.py --include-revision-supplements
```

Both routes call `scripts.plot_deep_controls` for S18 and `scripts.plot_reference_controls` for S19/S20 in separate Python processes. New figures and their plot audits go to `outputs/supplementary_figures/`; neither the existing combined PDF nor the workbook is modified. Source schemas and experiment completeness are checked by the plotters, not inferred from folder existence. All subprocess failures propagate to the wrapper.

The larger saved arrays and pickle-free control inputs are distributed in the
[`analysis-workflow-v1.1.0` release](https://github.com/wangsaidi/single-cell-dr-mechanism-catalogue/releases/tag/analysis-workflow-v1.1.0).
Install the older processed-object bundle and the separate new control bundle with:

```bash
python -m analysis.download_inputs
python -m analysis.download_control_inputs
```

The new downloader verifies its registry-pinned ZIP hash and internal checksums
before extracting into a fresh directory. It never overwrites an existing input
directory. Original training input hashes are retained separately from the
Unicode-ID portable serializations; numeric X members are byte-identical.
See `analysis/DEEP_CONTROLS.md` for fitting and scoring and
`analysis/REFERENCE_CONTROLS.md` for crossed-root analysis. Saved-input metric
recomputation is distinct from fitting new models.

Use a separately exported public bundle or an isolated output location with:

```bash
python make_figures.py --revision-supplements-only --revision-source-root path/to/source_data --revision-output-dir path/to/new_figures
python make_figures.py --help
```

`--revision-source-root` contains both named source subdirectories. These overrides affect only S18-S20, not the established figure inputs/outputs. The source and output directories must not overlap.

The S17 script uses the original three coding fields, retaining scVI-specific features in a 20-feature union without changing the original eight-anchor Jaccard submatrix. It separates the original eight-method result, uniformly rescored eight-method control and nine-method extension. Exact tests permute whole method labels, not the dependent pair dots. The pooled and three context tests form a four-test BH family; six alternate encodings form a separate exploratory family. Method and endpoint omissions are descriptive. The primary eight-method Mantel tests and six-scenario simulation outputs remain unchanged. Run S17 alone with `python -m figures.plot_supplementary_figure_s17`; its results, complete control/context permutation vectors and design records are included under `data/source_data/generated/s17_*` and `metadata/s17_*`.

See `TUTORIAL.md` for standalone figure commands, output checks and the boundary between figure reproduction and raw-data reanalysis.

The wrapper first refreshes the heart donor-summary input from the canonical 15-non-self-neighbour diagnostic table. This keeps the continuous donor margins in Figure 5 and Supplementary Figure S5 consistent with the diagnostic definitions; it does not refit any model. Annotation colours are shared between Figure 3 and Supplementary Figure S3. Paul15's coarse display groups are used only for colouring, while diagnostic calculations use the original 19 annotations.

Expression perturbations in Figure 6c-d,j and Supplementary Figure S6f-g use PBMC3k only, four methods and two computational repeats. Dropout independently masks non-zero counts. The noise implementation draws one Gaussian gene-wise log-count shift vector per repeat and shares it across all cells; it is not independent cell-gene noise. Its scale is the indicated multiplier of each gene's baseline log-count standard deviation.

Figure 6f and Supplementary Figure S6h-i include one auxiliary generated count dataset with 1,600 cells and 600 genes at seed zero. Its generator is `analysis/scripts/generate_robustness_embeddings.py`, and its scores and design are supplied as `fig6_known_truth_simulation.csv` and `fig6_known_truth_simulation_design.json`. It is distinct from the independently replicated six-scenario suite. The four non-latent minima pool 28 conditions for PCA, 37 each for UMAP, PHATE and PaCMAP, and 10 for t-SNE; latent-distance correlation has one condition per method. `fig6_worst_case_condition_manifest.json` identifies the four input tables and per-component counts. Unequal pools describe the worst evaluated condition, not an equal-coverage method ranking.

Run the focused figure-source and exact-test regressions with `python -m unittest discover -s tests -v`. The lightweight orchestration/cache/entry-point tests can be run without datasets, fitting or plotting:

```bash
python -B -m unittest discover -s tests -p 'test_refit*.py' -v
```

## Reproducing the analysis

Use the modern analysis environment for archived analysis or public-source refitting. The environment creation commands below are setup instructions, not a claim that rebuilding them has been executed on every platform:

```bash
conda env create -f analysis/environment/analysis-environment.yml
conda activate single-cell-dr-analysis
python -m pip check
```

For the exact-input archived route, install the archive and use a new output run directory:

```bash
python -m analysis.download_inputs
python -B -m analysis.workflow.run_analysis --profile archived --archive-data analysis/data --archive-results analysis/results --run-dir analysis/runs/archived
```

Inputs remain read-only. The archived route retains a checksum-verified supplied eight-anchor scalar geometry baseline; it does not regenerate every historical scalar input from public counts. See [analysis/README.md](analysis/README.md) for that boundary and result-comparison tolerances. `refit-scvi` is an explicitly mixed route that fits scVI while consuming other archived inputs.

For a clean public-source refit, first follow the separate legacy environment and source setup in [analysis/README.md](analysis/README.md): Python 3.7/TensorFlow 1.15 for scScope/SAUCIE, `SCDR_LEGACY_PYTHON` selecting its executable, and a clean SAUCIE checkout at `5ab7976fc8d19a3823b6005f51736dc70f2017fb`. The modern environment and legacy environment are not interchangeable. Preflight checks the actual legacy imports and selected source. No analysis-input archive is needed for this route:

```bash
python -B -m analysis.workflow.run_analysis --profile full-refit --run-dir analysis/runs/public_refit --force
```

To check all stage types with smaller computational budgets instead:

```bash
python -B -m analysis.workflow.run_analysis --profile smoke --run-dir analysis/runs/smoke --force
```

Choose a new empty directory for each run. The workflow records consumed inputs, source/environment versions, stage start/finish/failure logs and current-run validation. `--force` reaches the inner latent/projection and simulation/robustness caches. Fresh-refit validation is separate from old archived fitted-array hashes. Public-source means public expression/count objects plus documented supplied selections and annotations, not FASTQ reprocessing or independently inferred biological truth. Smoke results must not replace publication inputs.

The complete budgets, exact commands, resume rules and expected outputs are in [analysis/README.md](analysis/README.md). Separate deep-configuration and reference-attribution controls supply S18/S19; they are not automatically run by `make_figures.py` or implied complete by a primary-workflow smoke run.

### Scheduled optimisation controls and execution evidence

Supplementary Figure S21 reports 36 fixed-budget scScope/SAUCIE fits, 90 matched
representation endpoints and 45 read-only saved-state gradient checks. The
complete measured scalar inputs are in
[`data/source_data/optimisation_controls`](data/source_data/optimisation_controls).
To reproduce this figure without model fitting:

```bash
python -B -m analysis.optimization_controls.plot_controls --controls-dir data/source_data/optimisation_controls --output-dir outputs/scheduled_controls
```

Fitting, saved-state auditing, scoring, schedules and separate environments are
documented in [analysis/OPTIMISATION_CONTROLS.md](analysis/OPTIMISATION_CONTROLS.md).
The standalone command is separate from `make_figures.py`. The supplied
[`Supplementary_Figures.pdf`](outputs/supplementary_figures/Supplementary_Figures.pdf)
contains S1-S21, and the [results workbook](outputs/tables/Supplementary_Tables.xlsx)
contains Tables S1-S20.

The [publication-budget execution receipt](metadata/public_source_publication_budget_validation.json)
records successful completion of all 24 public-source workflow stage types and
the current-run validator in the recorded installed environments. It is not a
fresh environment build. The [endpoint comparison](metadata/public_refit_endpoint_summary.csv)
quantifies differences from archived outputs, including rare-population changes;
the original figure values have not been silently replaced. The
[specification comparison](metadata/public_refit_specification_comparison.csv)
reports the actual whole-method permutation analyses on freshly fitted outputs.

The historical scScope wrapper did not fully seed Python sampling and the new
TensorFlow graph created inside scScope. Exact historical replay therefore uses
the supplied fitted arrays. The public wrapper now seeds those sources, with
an actual separate-process repeat check documented in
[`metadata/scscope_seed_repeat_check.json`](metadata/scscope_seed_repeat_check.json).
The completed publication-budget execution used the preceding wrapper; the
targeted seed correction check is not a complete rerun of that execution. The
scheduled optimisation controls already seed Python, NumPy and graph-local
TensorFlow explicitly. Loss stability and seed repeatability do not establish
a global optimum or biological reliability.

## Scope

The specification records describe terms traceable to published mathematical formulations. They are multi-component records rather than mutually exclusive method classes or performance rankings. Figures 2, 3 and 5-7 reproduce the focused eight-anchor analyses. Figure 4 adds scVI as a ninth implementation for direct clustering, population-recovery, trajectory and donor-information outcomes. Supplementary Figs. S15-S17 additionally include scVI in matched empirical permutation calibration, geometry and a separate empirical specification sensitivity. scVI does not enter the original eight-anchor primary specification test or controlled-simulation comparisons. No panel estimates performance for catalogue methods that were not executed.

## License

The analysis and figure-generation code is released under the MIT License. The underlying public biological datasets remain subject to the terms of their source repositories.
