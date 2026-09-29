# Mathematical catalogue and diagnostic stress tests for single-cell dimensionality reduction

This repository release contains the figure-reproduction materials for the manuscript *Mathematical characterization and empirical evaluation of single-cell transcriptomic dimensionality-reduction methods*.

## Contents

- `figures/`: plotting scripts and shared publication style.
- `data/source_data/`: figure-level source-data tables used by the scripts.
- `metadata/`: mathematical specification records, public dataset sources and panel-level evidence mapping.
- `outputs/main_figures/`: final Figures 1-7 in publication formats.
- `outputs/supplementary_figures/`: final Supplementary Figs. S1-S14 as individual files and a combined 15-page PDF.
- `outputs/tables/`: Table 1 source and the submission-ready Supplementary Tables S1-S14 workbook.

The release contains plotting inputs rather than raw or processed AnnData objects. Raw-data reanalysis should begin from the public dataset access routes listed in `metadata/public_data_sources.csv`.

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

Figure 1 is supplied as final conceptual artwork. The wrapper rebuilds Figures 2-7 and Supplementary Figs. S1-S14 from the included source-data tables. Generated panel tables are written to `data/source_data/generated/`. Revised Figure 4 reports direct downstream outcomes for the eight anchor implementations plus scVI. The previous threshold-gate Figure 4 is retained unchanged as Supplementary Fig. S14. The curated Supplementary Tables S1-S14 workbook is supplied as a publication-facing output and is not rebuilt by the figure wrapper.

See `TUTORIAL.md` for standalone figure commands, output checks and the boundary between figure reproduction and raw-data reanalysis.

## Scope

The specification records describe terms traceable to published mathematical formulations. They are multi-component records rather than mutually exclusive method classes or performance rankings. Figures 2, 3 and 5-7 reproduce the focused eight-anchor analyses. Revised Figure 4 adds scVI as a ninth implementation for direct clustering, population-recovery, trajectory and donor-information outcomes. Neither panel estimates performance for catalogue methods that were not executed.

## License

The figure-generation code is released under the MIT License. The underlying public biological datasets remain subject to the terms of their source repositories.
