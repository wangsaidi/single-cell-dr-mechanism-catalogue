# Analysis-level reproducibility workflow

This directory closes the path between the public biological inputs, the evaluated representations, the downstream statistics and the publication figures. It is separate from the lightweight figure-only workflow in the repository root.

## Exact reviewer-facing workflow

The default profile starts from the checksum-locked processed objects and evaluated representation arrays used in the manuscript. It recomputes diagnostics, Leiden clustering across the recorded resolution grid, population recovery, marker concordance, Paul15 trajectory outcomes, donor predictability, dimensionality sensitivity and all panel-level summaries.

```bash
conda env create -f analysis/environment/analysis-environment.yml
conda activate single-cell-dr-analysis
python -m analysis.download_inputs
python -m analysis.workflow.run_analysis --profile archived
python make_figures.py
```

`analysis/verify_reproducibility.py` checks every installed input against `analysis/ARCHIVE_SHA256SUMS.txt` and compares recomputed CSV tables with the read-only reference results. The plotting wrapper then rebuilds the final figures from the publication source tables.

## Matched null calibration and scVI geometry

The archived profile also runs `threshold_calibration` using the same numerical inputs, then `verify_calibration` compares all eight generated record tables with the publication references without overwriting them. It draws 1,999 whole-cell or reference-label permutations on a fixed label-stratified 1,000-cell subset of each dataset. Unrestricted and annotation-conditioned geometry, donor-conditioned annotation recovery, root-resolved pseudotime and cell-type-conditioned donor mixing are tested separately. The null tests address random correspondence, not universal biological fidelity. All original operational cutoffs remain unchanged. Their crossing decisions are reported alongside the null effects and separately BH-adjusted one-sided probabilities.

```bash
python -m analysis.scripts.calibrate_operational_thresholds
python -m analysis.verify_calibration
python -m figures.plot_supplementary_figures_s15_s16
```

Complete summaries and the sequential raw permutation arrays are included in `data/source_data/calibration/`. Seed repeats and cell pairs are computational comparisons, not biological replicates. Supplementary Table S15 contains the 351 calibrated diagnostic rows. Supplementary Table S16 contains 216 full-object geometric values, 150 reference-dimensionality values and 120 seed-pair values. scVI is evaluated without labels or donor covariates and is not presented as a batch-corrected model. The specification-distance and controlled-simulation panels retain their original eight-method scope.

## Optional refitting profiles

`--profile refit-scvi` refits all two- and ten-dimensional scVI models over five seeds before rerunning the downstream workflow. `--profile full-refit` also reacquires the three public datasets, reconstructs the analysis objects, fits the eight anchor implementations over five seeds, recomputes dimension and perturbation inputs, reruns the six controlled stress-test scenarios and refits scVI.

The scScope and SAUCIE fits require the separate recorded Python 3.7 environment. Create it from `analysis/environment/legacy-python37-environment.yml`, run `python -m analysis.setup_legacy_tools`, and set `SCDR_LEGACY_PYTHON` to that environment's Python executable. The SAUCIE source is fixed to commit `5ab7976fc8d19a3823b6005f51736dc70f2017fb`. The archived profile does not require this legacy environment because it uses the exact evaluated arrays.

## Scope and provenance

The archive is distributed through the [`analysis-reproducibility-v1.0.0` GitHub Release](https://github.com/wangsaidi/single-cell-dr-mechanism-catalogue/releases/tag/analysis-reproducibility-v1.0.0). It contains derived analysis objects and numerical method outputs, not newly collected participant data. Dataset provenance and access routes are listed in `metadata/public_data_sources.csv`. The empirical analysis covers the three datasets and evaluated implementations reported in the manuscript; it does not estimate performance for the other catalogue entries.
