# Mathematical catalogue and diagnostic stress tests for single-cell dimensionality reduction

This repository release contains the figure-reproduction materials for the manuscript *A 26-method mathematical catalogue for evidence-aware interpretation of single-cell embeddings*.

## Contents

- `figures/`: plotting scripts and shared publication style.
- `data/source_data/`: figure-level source-data tables used by the scripts.
- `metadata/`: mathematical specification records, public dataset sources and panel-level evidence mapping.
- `outputs/main_figures/`: final Figures 1-7 in publication formats.
- `outputs/supplementary_figures/`: final Supplementary Figs. S1-S13 as individual files and a combined 14-page PDF.
- `outputs/tables/`: Table 1 source and the combined Supplementary Tables workbook.

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

Figure 1 is supplied as final conceptual artwork. The wrapper rebuilds Figures 2-7 and Supplementary Figs. S1-S13 from the included source-data tables. Generated panel tables are written to `data/source_data/generated/`.

See `TUTORIAL.md` for standalone figure commands, output checks and the boundary between figure reproduction and raw-data reanalysis.

## Scope

The specification records describe terms traceable to published mathematical formulations. They are multi-component records rather than mutually exclusive method classes or performance rankings. The empirical and simulation files reproduce the focused eight-method analyses and do not estimate performance for the remaining catalogue methods.

## License

The figure-generation code is released under the MIT License. The underlying public biological datasets remain subject to the terms of their source repositories.
