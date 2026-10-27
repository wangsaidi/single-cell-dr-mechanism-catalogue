# Deep Configuration Controls

These standalone commands register and execute real deep-model controls, then
score saved representations on matched analysis objects. They do not change
the primary refit workflow or its historical fitting runner. The Python 3.7
child `analysis/round2_deep_controls.py` supplies the existing TensorFlow
instrumentation and numerical-sensitivity implementation unchanged.

Run the commands from the code repository directory containing `analysis/`.
Use the modern analysis environment for orchestration and scoring and the
separate legacy environment for model fitting. See `analysis/README.md` and
`analysis/environment/` for the pinned environments and source setup.

## Inputs And Portable Paths

The objects directory must contain both files for each of `pbmc3k`, `paul15`
and `heart_cell_atlas_subsampled`:

- `<dataset>_counts_hvg.h5ad`: nonnegative counts, original ordered cell and
  gene identities; labels are not used for fitting.
- `<dataset>_proc.h5ad`: the identical cell order, the `X_pca_ref` PCA50
  representation, and the recorded annotation field for scoring.

For the checksum-locked archived objects and evaluated two-dimensional arrays,
use the existing release downloader:

```bash
python -m analysis.download_inputs
```

It checks the versioned release archive checksum before extraction. The default
locations are `analysis/data/analysis_objects` and
`analysis/data/anchor_embeddings`. No count objects are downloaded by either
control script, and neither silently switches to another dataset or archive.

**Do not assume the release also contains scScope's native 100D predictions.**
The scorer requires the original, unprojected legacy cache for the native versus
projection comparison. Supply an archived legacy cache if available, or
regenerate the native outputs through the documented full-refit route:

```bash
python -B -m analysis.workflow.run_analysis --profile full-refit --run-dir analysis/runs/public_refit --force
```

That route writes count/processed objects in `data/analysis_objects`, native
scScope outputs in `data/legacy_method_cache`, and two-dimensional outputs in
`data/anchor_embeddings`, all under the selected run directory. Use that same
run's objects for control fitting and scoring. A new stochastic fit is not
recovery of the exact historical latent array or proof of numerical agreement.
The source route starts from public count objects and supplied annotations or
subsets, not FASTQ processing or independent biological annotation.

The nominal scScope layer width is 50 with recurrence depth two. The official
prediction function concatenates the recurrent states, yielding **100 columns**.
Do not invert a two-dimensional map, zero-pad it, substitute expression PCA,
or use a failed model's replacement coordinates as the missing native output.
`--without-originals` explicitly requests only completed new controls and does
not reproduce the original-native comparison.

| Setting | CLI | Environment/default |
| --- | --- | --- |
| Count/processed objects | `--objects-dir` (both commands) | `SCDR_CONTROL_OBJECTS_DIR`, otherwise `analysis.paths.ANALYSIS_OBJECTS_DIR` |
| Fit output | launcher `--output-dir` | `SCDR_DEEP_CONTROLS_DIR`, otherwise `analysis.paths.RESULTS_DIR/deep_configuration_controls` |
| Saved control fits | scorer `--controls-dir` | same control directory default |
| Original native arrays | scorer `--original-latent-dir` | `SCDR_ORIGINAL_LATENT_DIR`, otherwise `analysis.paths.LEGACY_CACHE_DIR` |
| Original 2D arrays | scorer `--original-embedding-dir` | `SCDR_ORIGINAL_EMBEDDING_DIR`, otherwise `analysis.paths.ANCHOR_EMBEDDINGS_DIR` |
| Score output | scorer `--output-dir` | a fresh `scores` directory inside the control directory |
| Legacy executable | launcher `--legacy-python` | `SCDR_LEGACY_PYTHON` |
| SAUCIE checkout | launcher `--saucie-source` | `SCDR_SAUCIE_SOURCE`, otherwise `analysis.legacy_runtime.legacy_source()` |

`analysis.paths` also respects the existing `SCDR_ANALYSIS_DATA` and
`SCDR_ANALYSIS_RESULTS` roots. Explicit command-line directories take precedence.
No repository-external project layout or machine-specific interpreter path is
required. Run records contain the actual resolved paths as provenance.

The default original filenames follow the full-refit route:

```text
scscope_seed{seed}_analysis_{dataset}_recommended_v1.npy
{dataset}_{method}_seed{seed}.npy
```

The second pattern uses lowercase `scscope` or `saucie`. For a differently named
archived cache, pass `--original-latent-pattern` and/or
`--original-embedding-pattern`. Patterns accept `dataset`, `method`, and `seed`
and must produce filenames only. This changes file lookup, not mathematical
definitions. Original scScope comparisons use seeds 0-4; original SAUCIE
reconstruction comparisons use seeds 0-2. Missing required arrays fail clearly;
they are not silently skipped.

## Register And Fit

Configure the legacy interpreter, for example in PowerShell:

```powershell
$env:SCDR_LEGACY_PYTHON = (conda run -n single-cell-dr-legacy-deep python -c "import sys; print(sys.executable)" | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
```

Or in a Bash-compatible shell:

```bash
export SCDR_LEGACY_PYTHON="$(conda run -n single-cell-dr-legacy-deep python -c 'import sys; print(sys.executable)')"
```

Prepare the clean fixed SAUCIE source using the existing setup command:

```bash
python -m analysis.setup_legacy_tools
```

The runtime helper verifies the fixed commit, clean source and actual import,
and uses the same legacy DLL/search paths for preflight and fitting. A directory,
missing executable, incompatible Python/NumPy/TensorFlow/scscope environment or
invalid source checkout stops execution. Preflight records belong to the new
control invocation, not the primary analysis log directory.

Register inputs and the full plan without fitting:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --objects-dir analysis/data/analysis_objects --output-dir analysis/results/deep_configuration_controls --prepare-only
```

The default plan has **36 fits**, with seeds 0, 1 and 2 in each of three datasets:

| Method/configuration | Fits | Interpretation |
| --- | ---: | --- |
| scScope `recurrent_latent` | 9 | Original recurrent model, exported checkpoints and 100D prediction |
| SAUCIE `reconstruction` | 9 | Original reconstruction architecture, penalties inactive |
| SAUCIE `task_architecture_reconstruction` | 9 | Task architecture with penalty losses inactive |
| SAUCIE `task_enabled` | 9 | Original task architecture and active penalties, including known numerical failure conditions |

scScope inputs use per-cell normalization to the median positive library size
without a log transform. SAUCIE uses log1p counts. The launcher retains the
parent control definitions: scScope batch size 64, width 50, recurrence two,
checks every 25 epochs, maximum 600 epochs; SAUCIE batch size 256, a 2D
bottleneck, maximum 10,000 steps, and a fixed diagnostic subset. The task
configuration uses `lambda_c=.1`, `lambda_d=.2`, `lambda_b=0`; enabling the task
branch changes architecture too, hence the architecture-matched control.
Stopping requires three consecutive relative diagnostic-loss changes below
.001 after at least 300 scScope epochs or 3,000 SAUCIE updates. Completion at a
budget limit is not proof that this rule or biological convergence was achieved.

Before any fit, `analysis_manifest.json` fixes all 36 jobs, transformed-input
hashes, count-object hashes, source hashes, settings and stopping definitions.
Run the registered jobs with:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --objects-dir analysis/data/analysis_objects --output-dir analysis/results/deep_configuration_controls --resume --workers 2
```

Each invocation records its selected jobs, explicit filters, preflight and
success/failure records under `executions/`. `--pilot` selects the four PBMC3k
seed-zero jobs; it does not change the registered full plan. Configuration and
job exclusions are explicit and recorded. Repeat `--seed` to select registered
seeds, for example `--seed 0 --seed 1`; omitting it retains seeds 0, 1 and 2.
Filtering does not create duplicate fits when a seed is repeated. The default does **not** omit the
unmodified task-enabled jobs because they are known to encounter nonfinite
values. Failed jobs retain their diagnostics and cause a nonzero exit; no score
or successful model is substituted.

Use a fresh output root, or `--resume` only with unchanged registered inputs and
source. Completed matching jobs may be reused. Existing failed or partial job
directories are never cleared, relabelled as complete or restarted in place.
Choose a fresh `--attempt` label for another actual execution. Repeating a
registered attempt requires `--resume` and identical filters. Attempt names use
letters, digits and underscores; `original` is reserved for scorer filtering.

## Declared Longer Budget Attempt

The `longer_budget` extension is a separately declared six-fit experiment:
scScope `recurrent_latent` and SAUCIE `reconstruction`, all three datasets,
computational seed 0, with caps of **2,000 epochs** and **20,000 steps**,
respectively. It leaves the original 36-job registration and all original
outputs intact. These are fresh fits at the declared larger caps, not in-place
continuation of saved model weights or replacement of an earlier outcome.

The declared rationale is that the initial training budgets did not meet the
strict loss-stability criterion and residual last-checkpoint neighbourhood
drift remained (neighbour overlap reported at approximately .91). This is a
loss/checkpoint-based budget amendment, not selection by annotation F1,
preferred downstream conclusions or the appearance of a map. It is not
relabelled as part of the original plan. No new neighbourhood pass threshold
is introduced. The loss definition, model architecture, objective, checkpoint
cadence and three-consecutive-change stopping criterion remain unchanged;
only the declared maximum budgets change. A cap does not imply that every
fit reaches it or that convergence is achieved.

Register the additional jobs before fitting:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --objects-dir analysis/data/analysis_objects --output-dir analysis/results/deep_configuration_controls --configuration recurrent_latent --configuration reconstruction --seed 0 --max-epochs 2000 --max-steps 20000 --attempt longer_budget --prepare-only
```

Run that exact registration with the same filters and caps:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --objects-dir analysis/data/analysis_objects --output-dir analysis/results/deep_configuration_controls --configuration recurrent_latent --configuration reconstruction --seed 0 --max-epochs 2000 --max-steps 20000 --attempt longer_budget --resume --workers 2
```

Explicit `--max-epochs` or `--max-steps` requires a labelled `--attempt` and a
positive value. Omitted caps remain at the original 600 epochs and 10,000
steps. Both cap fields are recorded in each selected job; scScope consumes
`max_epochs` and SAUCIE consumes `max_steps`. Changed-cap jobs carry
`maximum_budget_extension:true` and have new attempt-suffixed output
directories. `attempt_longer_budget.json` registers them separately before
any child process starts. Existing attempt settings cannot be changed with
`--resume`, and failed/partial directories cannot be restarted in place.

A fresh labelled attempt may use an updated orchestration launcher while
retaining the original immutable manifest; each execution records the current
launcher and child hashes. The registered child, SAUCIE model and loader
hashes must still match. If those sources changed, use a new control output
root, preserving the earlier root and its records. Do not rewrite an earlier
registration to make source changes appear identical.

Only completed metadata produces score rows. The scorer keeps
`reconstruction_extended`, `retrained_latent_extended` and
`retrained_PCA2_extended` separate from their original-budget counterparts.
It does not pool the additional seed-zero fits as extra independent seeds.
Checkpoint-stability rows include `budget_extension=true` for these scScope
fits and `false` for original-budget fits; the numerical stability calculations
are unchanged. A budget extension combined explicitly with numerical distance
stabilization would carry both suffixes, in the order `_stabilized_extended`.

The extension has been declared and launched in the ongoing analysis, but its
completion, strict-criterion status, checkpoint stability and downstream
effects remain **unverified pending the actual run records**. These public
commands provide the execution route, not a claim that the extension succeeded
or that adequate convergence has been demonstrated.

## Separate Numerical Implementation Sensitivity

The following is an **additional nine-job control**, not replacement of the nine
unmodified task-enabled jobs:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --objects-dir analysis/data/analysis_objects --output-dir analysis/results/deep_configuration_controls --configuration task_enabled --attempt nonnegative_distance --stabilize-distances --workers 2
```

The explicit flag sets `saucie_distance_stabilization=true` in a separately
registered attempt. It enables the existing child's finite-check and
nonnegative projection of the original float32 squared distances before the RBF
and square-root operations; the original square-root offset remains. It does
not add denominator epsilons, change the optimizer, suppress NaNs/infinities or
silently switch the training objective. Nonfinite values remain fatal.

This is a declared numerical implementation sensitivity, not evidence that the
unmodified objective succeeded. The scorer labels it `task_enabled_stabilized`.
The flag requires both a labelled attempt and explicit SAUCIE-only selection.
Source and raw failure records from the original jobs remain separate and
unchanged. No PCA fallback or automatic stabilized retry is performed.

## Score Saved Outputs

Supply the matching original native cache and 2D arrays, if available:

```bash
python -B -m analysis.scripts.score_deep_configuration_controls --objects-dir analysis/data/analysis_objects --controls-dir analysis/results/deep_configuration_controls --original-latent-dir analysis/data/legacy_method_cache --original-embedding-dir analysis/data/anchor_embeddings --output-dir analysis/results/deep_control_scores
```

For objects obtained from a full refit, use that run's `data/analysis_objects`,
`data/legacy_method_cache` and `data/anchor_embeddings` in both commands instead.
The scorer checks count-file hashes, transformed control-input hashes and cell
order against fitting metadata. Original saved arrays have no intrinsic cell
IDs: their supplied order must correspond to the checked analysis objects.
The scorer validates dimensions and finiteness, not an invented original
array-to-cell identity mapping.

If no original native cache is available, an explicit control-only score run is:

```bash
python -B -m analysis.scripts.score_deep_configuration_controls --objects-dir analysis/data/analysis_objects --controls-dir analysis/results/deep_configuration_controls --without-originals --output-dir analysis/results/deep_control_scores_new_only
```

Only successful child metadata supplies new endpoint candidates. Registered
failed or unfinished jobs remain listed as unscored in the endpoint design.
Missing arrays for a purportedly complete fit are errors. Multiple successful
attempts for the same configuration and seed are not pooled as extra replicates;
select one with scorer `--attempt NAME`. Repeat the option to select distinct
configurations, and use `--attempt original` for unlabelled base jobs. Budget
extensions are distinct configurations, so base and `longer_budget` outputs
can coexist without mixing their seed summaries. To score just the completed
extension controls, use a new score directory:

```bash
python -B -m analysis.scripts.score_deep_configuration_controls --objects-dir analysis/data/analysis_objects --controls-dir analysis/results/deep_configuration_controls --without-originals --attempt longer_budget --output-dir analysis/results/deep_control_scores_longer_budget
```

An unstarted, failed or incomplete extension contributes no invented endpoint
or stability row. Its registered job remains listed as not completed or failed.

The endpoint and population CSV schemas and numerical definitions match the
source scorer for identical inputs and software:

- Full-cell PCA50 reference and 15 nearest nonself neighbours, with the same
  local overlap and same-label-neighbour fraction.
- Trustworthiness on the same sorted random subset of at most 1,000 cells.
  The random generator seed is 20261027; subsequent 6,000 pair draws exclude
  self pairs and retain the first 5,000, preserving the source draw order.
- Spearman distance-rank correlation on those pairs.
- Scanpy 15-neighbour Euclidean graph, graph seed 0, undirected igraph Leiden
  until partition stability, and resolutions .2, .4, .6, .8, 1, 1.2, 1.5 and 2.
- Majority annotation assignment after clustering, zero-division F1 equal to
  zero, and trapezoidal ARI/macro-F1/per-label F1 integrals divided by 1.8.
- Original scScope native, supplied t-SNE2 and an explicit PCA2 control from
  the identical native array; newly fitted native and PCA2 representations at
  epoch 300 and the final checkpoint. SAUCIE uses step 300 and final outputs.
  PCA2 uses `PCA(n_components=2, random_state=0)` and float32 output. The scorer
  does not manufacture a missing t-SNE output or refit a deep model.

All derived projections go into the fresh score directory, not original or
training folders. The scorer refuses a nonempty score directory; rerun into a
new one when fits or selected attempts change. `--limit` limits only endpoint
candidates and is recorded as such, not as full-panel completion.

Outputs are `matched_endpoint_scores.csv`, `matched_population_scores.csv`,
`endpoint_seed_summary.csv`, `checkpoint_representation_stability.csv`,
`endpoint_design.json` and a final `scoring_complete.json`. Interrupted scores
lacking the final marker are incomplete and are not resumed by appending old
rows.

Checkpoint stability retains the latest source definition: for each completed
scScope fit, select the last saved epoch **strictly before** `final_epoch`,
compare its 100D coordinates with `latent_final.npy`, and calculate 15-neighbour
overlap and pairwise distance-rank agreement using seed 20261027. With all three
datasets and three computational seeds complete, this yields **nine checkpoint
stability rows** for the original budget, not one pilot row. If all three
extended scScope seed-zero fits also complete and are selected, there are
three additional rows, distinguished by `budget_extension=true`, not three
new biological replicates. This calculation covers all selected
completed scScope fits even if endpoint `--limit` is set. Missing preceding
checkpoints are errors, not comparisons of the final array with itself.

No cell, shared pair, seed, checkpoint or generated score is a biological
replicate. Seed summaries are observed median/minimum/maximum, not confidence
intervals. These scripts score geometry and annotation recovery only; they do
not add trajectory/donor-classification endpoints, independent biological
validation, a universal ranking or a completed primary-refit claim.

## Lightweight Checks

CLI help does not import TensorFlow or start a model:

```bash
python -B -m analysis.scripts.run_deep_configuration_controls --help
python -B -m analysis.scripts.score_deep_configuration_controls --help
```

Check syntax without writing bytecode:

```bash
python -B -c "import ast; from pathlib import Path; files=[Path('analysis/scripts/run_deep_configuration_controls.py'), Path('analysis/scripts/score_deep_configuration_controls.py')]; [ast.parse(p.read_text(encoding='utf-8'), filename=str(p)) for p in files]; print('Syntax OK')"
```

Syntax and plan checks do not establish that model runs complete, that either
environment can be rebuilt, or that numerical results agree after refitting.
