# Analysis Reproducibility

Three scientific routes answer different questions. Figure regeneration uses supplied result
tables. Archived analysis uses verified processed objects and fitted arrays.
Only public-source full refitting reruns preprocessing, model fitting, simulation
generation and downstream metrics. Do not describe either of the first two routes
as a fresh raw-source refit.

## Working Directory And Environments

Run all commands from the repository directory containing `analysis/` and
`make_figures.py`. First create the software environments:

```bash
conda env create -f analysis/environment/analysis-environment.yml
conda env create -f analysis/environment/legacy-python37-environment.yml
conda run -n single-cell-dr-analysis python -m pip check
conda run -n single-cell-dr-legacy-deep python -m pip check
```

Activate the modern environment so that `python` selects its interpreter:

```bash
conda activate single-cell-dr-analysis
```

Set `SCDR_LEGACY_PYTHON` to the absolute Python executable in the separate legacy
environment. For example, resolve it from the named Conda environment in
PowerShell:

```powershell
$env:SCDR_LEGACY_PYTHON = (conda run -n single-cell-dr-legacy-deep python -c "import sys; print(sys.executable)" | Where-Object { $_.Trim() } | Select-Object -Last 1).Trim()
```

Or in a Bash-compatible shell:

```bash
export SCDR_LEGACY_PYTHON="$(conda run -n single-cell-dr-legacy-deep python -c 'import sys; print(sys.executable)')"
```

An equivalent existing environment may be selected instead. Archived analysis
and figure-only reproduction do not need the legacy environment.

The modern environment targets Python 3.14, Scanpy 1.12.1, scvi-tools 1.4.3,
NumPy 2.4.4 and the other exact pins in `environment/requirements-analysis.txt`.
The analysis requirements include GLM-PCA 0.1.0. `torch==2.12.0` also admits a
CPU local build such as `2.12.0+cpu`; it does not require CUDA.
For a pip-based CPU installation, install PyTorch from its CPU index first:

```bash
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r analysis/environment/requirements-analysis.txt
python -m pip check
```

The legacy environment is separate: Python 3.7.16, TensorFlow 1.15.0,
NumPy 1.18.5, scscope 0.1.5, h5py 2.10.0 and scikit-learn 0.23.2.
Its YAML, historical package list and platform-specific Conda lock are in
`environment/`. Old Python/TensorFlow packages may require platform-specific
solver or wheel work. Keep the resolved installation manifests with each run.
The launcher prepends the legacy environment root, `Library/bin`,
`Library/mingw-w64/bin`, `DLLs` and `Scripts` to PATH for both validation and fits.
An unset path, directory such as `Path()`, non-executable file, incompatible
Python or failed TensorFlow/scscope import produces a clear preflight error.

## Fixed Legacy Source

SAUCIE must be a clean checkout at
`5ab7976fc8d19a3823b6005f51736dc70f2017fb`. A short source path is useful on
Windows; it can be outside the run directory:

```powershell
$env:SCDR_SAUCIE_SOURCE = Join-Path $HOME '.cache/single-cell-dr/SAUCIE'
```

The equivalent Bash setting is:

```bash
export SCDR_SAUCIE_SOURCE="$HOME/.cache/single-cell-dr/SAUCIE"
```

Then retrieve the fixed source revision:

```bash
python -m analysis.setup_legacy_tools
```

Setup uses per-command `git -c core.longpaths=true` for clone, fetch and checkout,
and refuses to overwrite a dirty checkout. If a prior long-path clone is
incomplete, preserve it and choose another empty source path. Do not disable TLS
or delete unrelated files. An existing valid checkout can be fetched with the
explicit transport settings when required by the local Git installation:

```powershell
git -c core.longpaths=true -c http.sslBackend=openssl -c http.version=HTTP/1.1 -C $env:SCDR_SAUCIE_SOURCE fetch origin
```

The workflow verifies the commit, clean source state and actual `SAUCIE` import
before fitting. The checkout directory is the package itself; `PYTHONPATH`
therefore points to its parent directory, while `--saucie-parent` remains the
checkout path passed to the standalone runner. The import-only preflight uses
the legacy interpreter, DLL paths and script-directory context used by fitting,
checks the resolved package file and required interfaces, and does not construct
or train a model. Do not substitute an unversioned source download or a different
commit.

## Clean Public-Source Refit

Do not download the analysis-input archive for this route. Choose a new empty
run directory, not the archived data/results directory:

```bash
python -B -m analysis.workflow.run_analysis --profile full-refit --run-dir analysis/runs/public_refit --force
```

Omitting `--run-dir` creates a new timestamped `analysis/runs/<profile>_<UTC>`
directory.

This runs the installed-metadata check and legacy preflight first, acquires
public expression objects, reconstructs processed/count-HVG and all-gene marker
objects, fits the eight anchors for seeds 0-4, generates six simulation scenarios
with five independently generated datasets each, generates robustness inputs,
fits direct 2D/10D scVI for seeds 0-4, calculates metrics and calibration, exports
six Figure 4 panel tables inside the run, and validates completeness.
Simulations precede robustness synthesis so their outputs are available to it.

"Public-source" is not FASTQ-to-count reprocessing or de novo annotation:
PBMC3k uses public raw counts plus Scanpy's published processed cell/gene
selection and Louvain labels; Paul15 uses the public Scanpy count/annotation
object; the heart dataset is the public scvi-tools atlas subset with supplied
cell types and donors, followed by the documented deterministic 8,000-cell
selection. Source accessions and limitations remain in
`../metadata/public_data_sources.csv`. The published labels/subsets are explicit
upstream dependencies, not independently inferred ground truth.

The workflow fixes `SCDR_ANALYSIS_DATA`, `SCDR_ANALYSIS_RESULTS`,
`SCDR_ANALYSIS_LOGS` and the scVI input directory to the chosen run's isolated
paths. Inherited cache overrides cannot silently select old fits. Scanpy's
download directory is inside the run's raw cache. Every fitting stage forces
outer embedding, legacy latent and scScope t-SNE caches; simulations and both
cached robustness axes also receive `--force`. Dropout/noise and the auxiliary
simulation already recompute without a cache. The legacy fitting configurations
are specified below. Supplied publication tables and archived fits are not a full-refit
fallback. Partial fit failures are retained and stop the workflow.

Historical implementation details must remain explicit:

- scScope: nominal layer width 50, recurrence depth 2 and batch size 64. Official
  `predict` concatenates recurrent representations, giving 100 columns in the
  existing evaluated latents, not a 50-column exported matrix. Each projection
  records the actual latent shape/hash before t-SNE to 2D.
- SAUCIE: reconstruction-only configuration, 300 optimization steps. The runner
  does **not** forward `--batch-size 64` to `train`; effective official default
  batch size is 256. Enabling `lambda_c` also changes activation architecture,
  so that comparison is not a pure loss-only ablation. This workflow does not
  establish convergence from a fixed training budget. Training-duration and
  architecture/penalty control experiments are separate analyses.

A nonempty run is rejected unless `--resume` is explicit. To rerun stages after
an interruption, use the same profile/archive locations and recorded run:

```bash
python -B -m analysis.workflow.run_analysis --profile full-refit --run-dir analysis/runs/public_refit --resume --from-step prepare_objects
```

Selected fitting stages still force their caches on resume. Do not resume an
old dataset into a different experiment. `--from-step`/`--through-step` selections
are recorded as partial, not certified as a complete clean execution. Review all
execution records and numerical agreement before reporting a completed refit.

## Reduced All-Stage Smoke Test

After source setup, run a fresh execution check with every stage
type, all eight anchors, all six simulated scenarios and scVI retained:

```bash
python -B -m analysis.workflow.run_analysis --profile smoke --run-dir analysis/runs/smoke --force
```

This profile still acquires/preprocesses the original three public objects and
retains their original analysis cell populations. It reduces computational
budgets, not biological scope: seeds 0-1, one generated replicate per scenario,
scScope 3 epochs, SAUCIE 10 steps, scVI 2 epochs without early stopping, GLM-PCA
3 iterations, UMAP 20 epochs, 2D/10D robustness outputs, one upstream PCA setting
(20 PCs), one perturbation repeat, two Leiden resolutions (0.4/1.0), one
neighbourhood sensitivity (k=5), 20 donor bootstrap draws, and calibration with
19 permutations on up to 200 cells and 500 sampled pairs. Other algorithm
defaults remain unchanged. Acquisition and some fitting stages can still require
substantial time. Two seeds retain
the seed-stability code path. Exact whole-method association tests remain exact.

Smoke outputs and audits are marked `smoke_not_publication`.
`publication_scale_full_refit_completed` remains false even if every smoke
stage completes. The original `full-refit` and archived budgets are unchanged,
and they explicitly reset an inherited smoke setting to publication scale.
Do not use smoke scores, p-values or bootstrap intervals in the manuscript;
these budgets do not establish convergence or scientific robustness. The smoke
profile checks executable coverage, not publication-scale numerical agreement.
Retain its success/failure logs and investigate failed stages; passing a smoke
test does not establish convergence or validate the scientific results.

## Archived Analysis

Install the checksum-locked bundle into the repository once, or select an
already extracted bundle's `analysis/data` and `analysis/results` locations:

```bash
python -m analysis.download_inputs
python -B -m analysis.workflow.run_analysis --profile archived --archive-data analysis/data --archive-results analysis/results --run-dir analysis/runs/archived
```

No legacy environment is needed. Inputs are read-only; checksums are checked
before analysis and again with result comparisons. scVI audits normalize epoch
counts in the audit view only. Recomputed results, calibration and panel exports
go into the run, never into publication tables or supplied fit metadata.
The eight-anchor scalar geometry baseline is a verified supplied table; scVI
geometry, trajectory/neighbourhood diagnostics and downstream metrics are
recomputed. Thus even this exact-input route is not a claim that all scalar
inputs were regenerated from public counts.

`ARCHIVE_SHA256SUMS.txt` retains the original binary/object input hashes.
`ARCHIVED_TABLE_SHA256SUMS.txt` additionally fixes the consumed publication
baseline and eight calibration references, canonicalizing only CRLF to LF for
these Git-tracked CSVs. Reference result comparisons use their existing numeric
tolerances, not byte-identical CSV output. Missing references fail verification.

The optional `--profile refit-scvi --run-dir analysis/runs/scvi_only`
checks archived data/tables, refits only scVI and runs the same metrics. It is an
explicitly mixed-input route, not a clean full refit. Fresh refits never run old
fitted-array hash comparisons or exact archived-result/calibration comparisons.

## Figure-Only Route

In a figure environment built from the code-root `environment.yml` or
`requirements.txt`:

```bash
python make_figures.py
```

This uses supplied publication tables and writes the figure wrapper's normal
outputs. It does not fit representations, generate new count simulations or
establish analysis-level reproducibility. The analysis workflow does not invoke it or overwrite its inputs.
Figure 1 is supplied conceptual artwork. See the code-root README/TUTORIAL for
the exact plotting scope.

## Outputs And Evidence

Every analysis run has `run.json` with selected stages, execution status,
isolated paths, initially-empty status and file hashes. `logs/stages.jsonl`
records stage start/finish/failure, UTC times, commands, duration and exit codes;
`logs/<stage>.log` retains stdout/stderr. `runtime_environment.json` records
installed versions and analysis-source/configuration hashes. Legacy preflight
records source revision, interpreter/import checks and actual batch conventions.

Full runs write `data/raw_cache`, `analysis_objects`, `expression_objects`,
`anchor_embeddings`, `legacy_method_cache`, `simulation_cache` and
`robustness_embeddings`; scVI arrays/history/metadata go in `results/embeddings`.
Per-seed anchor tables live in `results/source_data/*_seed{seed}.csv`.
Metric directories include diagnostics,
dimension sensitivity, clustering, marker concordance, trajectory, donor
predictability, panel summaries, calibration and `figure4_downstream`.

`logs/<stage>.input_reads.json` classifies observed Python file reads as public
source cache (including processed annotations), processed expression, fitted or
generated input, or table/configuration, with hashes. Prepared-input and final
output inventories cover HDF5 files too. The Python audit hook is **not** an OS
sandbox: native HDF5 accesses and legacy child reads are not exhaustively
observed. Legacy transformed-input metadata and output/projection hashes supply
additional provenance. Isolated paths, initially-empty runs and reviewed source
records are required alongside these logs; an empty read log is not proof of
no external input consumption.

`logs/refit_validation.json` checks current-run audits/coverage, not old binary
hashes, numerical equivalence, biological validity or training convergence.
Archive checks and refit validation must remain separate. Five computational
seeds are not biological replicates; stochastic agreement requires separately
declared numerical comparison criteria and actual completed refits.

## Small Regression Tests

```bash
python -B -m unittest discover -s tests -p 'test_refit*.py' -v
```

These standard-library tests use scientific stand-ins for cache/fit calls and
small real child CLIs for stage logging. They do not download data or train a
model. A metadata-only live check is safe to run separately:

```bash
python -B -m analysis.workflow.run_analysis --profile full-refit --run-dir analysis/runs/metadata_only --through-step record_environment
```

Do not infer a completed fresh-environment or full public-source experiment from
these tests. Network access, SAUCIE source setup, old dependency availability,
runtime/memory, current public-object versions, numerical agreement and the
training-control evidence require separate verification.
