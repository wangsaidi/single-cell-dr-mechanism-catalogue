# Figure reproduction tutorial

## Reproduction scope

The lightweight workflow rebuilds Figures 2-7 and Supplementary Figs. S1-S14 from the included figure-level source tables. Figure 1 is conceptual artwork and is supplied as a final file. The separate workflow in `analysis/` recomputes the empirical diagnostics and downstream results from checksum-locked processed objects and evaluated representations. Its optional full-refit profile reacquires the public datasets and reruns model fitting.

Public dataset identifiers, retrieval routes and preprocessing decisions are listed in `metadata/public_data_sources.csv`, `metadata/empirical_dataset_registry.csv` and `metadata/preprocessing_and_analysis_inputs.csv`. Evaluated software settings and robustness-axis coverage are listed in `metadata/evaluated_method_settings.csv` and `metadata/method_axis_coverage.csv`.

For the analysis workflow, follow `analysis/README.md`. The exact-input profile is the recommended verification route because it isolates the reported downstream calculations from platform-sensitive refitting of legacy neural implementations.

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
python -m figures.plot_figure_4_downstream
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

`figures.plot_figures_3_4` rebuilds Figure 3 and the original threshold-gate composite. `figures.plot_figure_4_downstream` then retains that composite as Supplementary Fig. S14 and writes the revised downstream-outcome Figure 4. The top-level wrapper runs the modules in this order.

## Verify files

`FILE_MANIFEST.csv` records the path, byte count and SHA-256 digest of every release file at packaging time. Exact PDF or raster hashes can differ after regeneration because graphics backends may embed metadata, so scientific verification should compare the generated source tables, panel values and visible figure content.

## Interpretation boundary

The 26-method catalogue records published mathematical specifications. Figure 2 uses a 17-feature binary record of each of the eight anchor pipelines, so optional objective terms that were inactive in a reported execution are not attributed to its observed behaviour. Figure 4 adds scVI only for the expanded downstream comparisons. The robustness analyses are explicit method subsets; omitted method-axis combinations are not interpreted as passes or failures.
