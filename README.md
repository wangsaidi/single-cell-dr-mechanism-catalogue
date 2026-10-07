# Mathematical catalogue and diagnostic stress tests for single-cell dimensionality reduction

This repository contains the figure and analysis-reproduction materials for the manuscript *Mathematical characterization and empirical evaluation of single-cell transcriptomic dimensionality-reduction methods*.

## Contents

- `figures/`: plotting scripts and shared publication style.
- `data/source_data/`: figure-level source-data tables used by the scripts.
- `metadata/`: mathematical specification records, public dataset sources and panel-level evidence mapping.
- `outputs/main_figures/`: final Figures 1-7 in publication formats.
- `outputs/supplementary_figures/`: final Supplementary Figs. S1-S16 as individual files and a combined 17-page PDF.
- `outputs/tables/`: Table 1 source and the submission-ready Supplementary Tables S1-S16 workbook.
- `analysis/`: executable preprocessing, representation, downstream-analysis and verification workflow.

The lightweight repository checkout contains the plotting inputs. The larger checksum-locked analysis objects and evaluated representation arrays are distributed through the [`analysis-reproducibility-v1.0.1` GitHub Release](https://github.com/wangsaidi/single-cell-dr-mechanism-catalogue/releases/tag/analysis-reproducibility-v1.0.1) and are installed by `python -m analysis.download_inputs`. Raw-data refitting can begin from the public access routes listed in `metadata/public_data_sources.csv`.

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

Figure 1 is supplied as final conceptual artwork. The wrapper rebuilds Figures 2-7 and Supplementary Figs. S1-S16 from the included source-data tables. Generated panel tables are written to `data/source_data/generated/`. Figure 4 reports direct downstream outcomes for the eight anchor implementations plus scVI. The previous threshold-gate Figure 4 is retained as Supplementary Fig. S14. Supplementary Fig. S15 separates random correspondence from the original operational cutoffs; Supplementary Fig. S16 reports scVI geometry, reference sensitivity and computational stability. The curated Supplementary Tables S1-S16 workbook is supplied as a publication-facing output and is not rebuilt by the figure wrapper.

See `TUTORIAL.md` for standalone figure commands, output checks and the boundary between figure reproduction and raw-data reanalysis.

The wrapper first refreshes the heart donor-summary input from the canonical 15-non-self-neighbour diagnostic table. This keeps the continuous donor margins in Figure 5 and Supplementary Figure S5 consistent with the diagnostic definitions; it does not refit any model. Annotation colours are shared between Figure 3 and Supplementary Figure S3. Paul15's coarse display groups are used only for colouring, while diagnostic calculations use the original 19 annotations.

Expression perturbations in Figure 6c-d,j and Supplementary Figure S6f-g use PBMC3k only, four methods and two computational repeats. Dropout independently masks non-zero counts. The noise implementation draws one Gaussian gene-wise log-count shift vector per repeat and shares it across all cells; it is not independent cell-gene noise. Its scale is the indicated multiplier of each gene's baseline log-count standard deviation.

Figure 6f and Supplementary Figure S6h-i include one auxiliary generated count dataset with 1,600 cells and 600 genes at seed zero. Its generator is `analysis/scripts/generate_robustness_embeddings.py`, and its scores and design are supplied as `fig6_known_truth_simulation.csv` and `fig6_known_truth_simulation_design.json`. It is distinct from the independently replicated six-scenario suite. The four non-latent minima pool 28 conditions for PCA, 37 each for UMAP, PHATE and PaCMAP, and 10 for t-SNE; latent-distance correlation has one condition per method. `fig6_worst_case_condition_manifest.json` identifies the four input tables and per-component counts. Unequal pools describe the worst evaluated condition, not an equal-coverage method ranking.

Run the focused figure-source regressions with `python -m unittest discover -s tests -p 'test_panel_sources.py'`.

## Reproducing the analysis

The reviewer-facing archived-input profile recomputes all diagnostics and downstream results from the exact processed objects and evaluated representations used in the manuscript, then checks the generated tables against read-only references.

```bash
conda env create -f analysis/environment/analysis-environment.yml
conda activate single-cell-dr-analysis
python -m analysis.download_inputs
python -m analysis.workflow.run_analysis --profile archived
```

Optional `refit-scvi` and `full-refit` profiles are documented in `analysis/README.md`. The full profile includes public-data acquisition and all model-fitting code; scScope and SAUCIE use the separately recorded Python 3.7 environment.

## Scope

The specification records describe terms traceable to published mathematical formulations. They are multi-component records rather than mutually exclusive method classes or performance rankings. Figures 2, 3 and 5-7 reproduce the focused eight-anchor analyses. Figure 4 adds scVI as a ninth implementation for direct clustering, population-recovery, trajectory and donor-information outcomes. Supplementary Figs. S15 and S16 additionally include scVI in matched empirical permutation calibration and geometric analysis, but not in the eight-anchor specification or controlled-simulation comparisons. No panel estimates performance for catalogue methods that were not executed.

## License

The analysis and figure-generation code is released under the MIT License. The underlying public biological datasets remain subject to the terms of their source repositories.
