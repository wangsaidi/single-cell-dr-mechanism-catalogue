# Saved representation sensitivity

This analysis checks the Figure 4 diagnostic-outcome associations after replacing only scScope and reconstruction-only SAUCIE with the saved scheduled outputs. It does not train dimensionality-reduction models or redraw the main figures. Clustering, marker measurements, DPT and exact association tests are recalculated with the original analysis functions.

## Inputs and execution

Use the modern analysis environment described in `analysis/README.md`. Install the archived processed-object and fitted-representation bundle once using:

```bash
python -m analysis.download_inputs
```

The small checked-in `data/saved_representation_sensitivity/scheduled_coordinates.zip` contains the 18 actual replacement coordinate arrays. Its hash and every member hash are recorded in `metadata/saved_representation_sensitivity_validation.json`. The three dataset prefixes are pb, pa and ht. Each has scScope t-SNE2 and reconstruction-only SAUCIE coordinates at three control seeds. These are the saved measured outputs, not regenerated or illustrative coordinates.

Extract the supplied archive into a new directory and rescore from the repository root:

```bash
python -m zipfile -e data/saved_representation_sensitivity/scheduled_coordinates.zip analysis/runs/saved_sensitivity_inputs
python -B -m analysis.scripts.evaluate_saved_representations --repo . --data analysis/data --controls analysis/runs/saved_sensitivity_inputs --output analysis/runs/saved_sensitivity
```

Choose new output and extraction directories rather than overwriting an earlier experiment. An existing separately extracted archived bundle may be passed through `--data`. The source control models can instead be regenerated using `analysis/OPTIMISATION_CONTROLS.md`; saved-array rescoring and fresh fitting are distinct routes.

## Statistical scope

The original panel uses nine archived seed-zero implementations. Each of three conditional panels replaces only the two deep representations, using scheduled control seed 0, 1 or 2, while the other seven remain fixed. The analysis scores 45 representations across three datasets. Paul15 uses 2,671 cells finite across every original and replacement output, identical to the original-only common population.

Each panel retains seven comparisons and the original two-sided exact Spearman test over all 362,880 whole-method-label permutations, including the identity. Benjamini-Hochberg adjustment is separate within each panel. The conditional panels share anchors and are not independent samples from an algorithm population. They do not isolate training duration from re-execution or implementation changes. Labels enter only post hoc mapping and scoring. No root, resolution or seed is selected by its outcome, and undefined order values remain missing.

`data/source_data/saved_representation_sensitivity/` supplies all 28 association tests, their 252 method-level input pairs, per-resolution and per-population outcomes, geometry, individual pseudotimes, comparison masks and input identities. Supplementary Table S21 contains the principal results. The historical primary figures are retained rather than silently overwritten.

All four associations that passed adjustment in the original panel retained that status in all three conditional substitutions; the remaining three did not pass. This conditional result does not imply invariant population F1 values, independent predictive validation or biological ground truth. The separate public-source rerun comparison in `metadata/public_refit_endpoint_comparison.csv` retains substantial rare-population changes, including the scScope megakaryocyte five-seed median changing from zero to approximately 0.632.
