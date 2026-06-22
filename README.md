# Figure and source-data release

This repository contains the figure-reproduction materials for the manuscript:

**A mechanistic catalogue of 26 methods and an evidence framework for interpreting single-cell dimensionality reduction**

The package is limited to plotting code, final figures, supplementary figures, supplementary tables, figure-level source data and publication-facing metadata. Manuscript drafts, cover letters, response documents, local cache files and submission working files are not included.

## Contents

```text
figures/                         Plotting scripts and shared plotting style
data/source_data/                Figure-level source-data tables used by the scripts
data/metadata/                   Metadata used by the supplementary-table builder
metadata/                        Public dataset and claim-to-evidence metadata
outputs/main_figures/            Final main figures
outputs/supplementary_figures/   Final supplementary figures
outputs/tables/                  Table 1 source table and supplementary-table workbook
outputs/source_data/             Public copy of the figure-level source-data tables
outputs/legends/                 Supplementary figure legends
make_figures.py                  Wrapper that rebuilds the adopted figure set
requirements_minimal.txt         Minimal Python package list for plotting
```

## Reproducing the figures

Create a Python environment and install the plotting dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_minimal.txt
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements_minimal.txt
```

Then rebuild the figure set.

```bash
python make_figures.py
```

The wrapper rebuilds Figure 2 to Figure 7, Supplementary Fig. S2 to S6 and the supplementary-table workbook from the included source-data tables. Figure 1 is supplied as final conceptual artwork at `outputs/main_figures/Figure_1.jpg`. Supplementary Fig. S1 is supplied as a final static dataset-context figure in `outputs/supplementary_figures`.

## Figure provenance

| Output | Primary script | Main source-data location |
| --- | --- | --- |
| Figure 1 | Final artwork only | `outputs/main_figures/Figure_1.jpg` |
| Figure 2 | `figures/plot_figure_2.py` | `data/source_data/` |
| Figures 3 and 4 | `figures/plot_figures_3_4.py` | `data/source_data/` |
| Figures 5 to 7 | `figures/plot_figures_5_7.py` | `data/source_data/` |
| Supplementary Fig. S1 | Final static figure | `outputs/supplementary_figures/` |
| Supplementary Fig. S2 | `figures/plot_supplementary_figure_s2.py` | `data/source_data/` |
| Supplementary Figs. S3 to S6 | `figures/plot_supplementary_figures_s3_s6.py` | `data/source_data/` |
| Supplementary Tables | `figures/build_supplementary_tables.py` | `data/source_data/`, `metadata/` |

The file `metadata/main_figure_claim_to_evidence_matrix.tsv` records the panel-level claim-to-evidence mapping for the main result figures. The file `metadata/public_data_sources.csv` records public dataset sources and access routes.

## Data scope

This release uses figure-level source-data tables rather than full raw or processed AnnData objects. Public dataset sources are listed in `metadata/public_data_sources.csv`. The plotting release is intended to reproduce the submitted figures and tables from the included source-data tables; full raw-data reanalysis should start from the public datasets listed in the metadata.

The source-data tables have been cleaned for public release and do not contain local workstation paths, cache locations or private file hashes.
