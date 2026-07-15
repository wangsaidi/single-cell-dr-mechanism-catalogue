# Figure reproduction tutorial

## Reproduction scope

This release rebuilds Figures 2-7 and Supplementary Figs. S1-S13 from the included figure-level source tables. Figure 1 is conceptual artwork and is supplied as a final file. The release does not claim to reconstruct every processed analysis object from raw sequencing files.

Public dataset identifiers, retrieval routes and preprocessing decisions are listed in `metadata/public_data_sources.csv`, `metadata/empirical_dataset_registry.csv` and `metadata/preprocessing_and_analysis_inputs.csv`. Evaluated software settings and robustness-axis coverage are listed in `metadata/evaluated_method_settings.csv` and `metadata/method_axis_coverage.csv`.

## Create the tested environment

```bash
conda env create -f environment.yml
conda activate single-cell-dr-figures
```

The equivalent pip workflow is documented in `README.md`.

## Rebuild every scripted figure

```bash
python make_figures.py
```

The main figures are written to `outputs/main_figures/`, supplementary figures to `outputs/supplementary_figures/`, and generated panel tables to `data/source_data/generated/`.

## Rebuild one figure group

```bash
python -m figures.plot_figure_2
python -m figures.plot_figures_3_4
python -m figures.plot_figures_5_6
python -m figures.plot_figure_7
python -m figures.plot_supplementary_figure_s1
python -m figures.plot_supplementary_figure_s2
python -m figures.plot_supplementary_figures_s3_s6
python -m figures.plot_supplementary_figures_s7_s8
python -m figures.plot_supplementary_figure_s9
python -m figures.plot_supplementary_figures_s10_s11
python -m figures.plot_supplementary_figure_s12
python -m figures.plot_supplementary_figure_s13
```

## Verify files

`FILE_MANIFEST.csv` records the path, byte count and SHA-256 digest of every release file at packaging time. Exact PDF or raster hashes can differ after regeneration because graphics backends may embed metadata, so scientific verification should compare the generated source tables, panel values and visible figure content.

## Interpretation boundary

The 26-method catalogue records published mathematical specifications. Figure 2 uses a 17-feature binary record of each evaluated pipeline, so optional objective terms that were inactive in a reported execution are not attributed to its observed behaviour. The robustness analyses are explicit method subsets; omitted method-axis combinations are not interpreted as passes or failures.
