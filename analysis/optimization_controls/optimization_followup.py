"""Run the declared scheduled-learning-rate sensitivity in isolated outputs.

This portable launcher is installed as analysis/optimization_controls/optimization_followup.py.
The model runner and numerical schedules match the executed controls. No result
is selected by labels, endpoint performance or a favourable early stopping time.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import subprocess


DATASETS = ('pbmc3k', 'paul15', 'heart_cell_atlas_subsampled')
CONFIGS = (('scScope', 'recurrent_latent', 'sc'), ('SAUCIE', 'reconstruction', 'rec'),
    ('SAUCIE', 'task_architecture_reconstruction', 'arc'), ('SAUCIE', 'task_enabled', 'task'))


def learning_rate(method, position):
    if position < 1:
        raise ValueError('Positions are one-based training epochs or updates.')
    if method == 'scScope':
        return 1e-4 if position <= 1000 else 1e-5 if position <= 1750 else 1e-6
    if method == 'SAUCIE':
        return 1e-3 if position <= 10000 else 1e-4 if position <= 20000 else 1e-5
    raise ValueError('Unknown method.')


def strict_stable(values):
    if len(values) < 4 or not all(math.isfinite(x) for x in values[-4:]):
        return False
    return all(abs(b-a) / max(abs(a), 1e-12) < .001 for a, b in zip(values[-4:-1], values[-3:]))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def design(input_dir, output_dir, saucie_source, threads=4):
    prior = Path(__file__).resolve().parents[1] / 'round2_deep_controls.py'
    jobs = []
    for dataset, short in zip(DATASETS, ('pb', 'pa', 'ht')):
        for seed in (0, 1, 2):
            for method, config, code in CONFIGS:
                source = input_dir / ('input_%s_%s.npz' % (dataset, method))
                if not source.is_file():
                    raise FileNotFoundError(str(source))
                jobs.append({'dataset': dataset, 'method': method, 'configuration': config,
                    'seed': seed, 'input': str(source),
                    'out_dir': str(output_dir / ('%s_%s_%d' % (short, code, seed))),
                    'saucie_source': str(saucie_source), 'prior_runner': str(prior),
                    'max_epochs': 2500, 'max_steps': 30000, 'threads': threads,
                    'saucie_distance_stabilization': config == 'task_enabled',
                    'fixed_budget_no_early_stop': True,
                    'schedule': '1000/1750 epochs:1e-4/1e-5/1e-6' if method == 'scScope'
                        else '10000/20000 updates:1e-3/1e-4/1e-5',
                    'seeds_are_computational': True, 'input_sha256': sha(source)})
    return jobs


def legacy_env(python, seed, threads):
    env = os.environ.copy()
    env['PATH'] = os.pathsep.join(str(python.parent / p) for p in
        ('', 'Library/bin', 'Library/mingw-w64/bin', 'DLLs', 'Scripts')) + os.pathsep + env.get('PATH', '')
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
        KMP_AFFINITY='disabled', TF_CPP_MIN_LOG_LEVEL='2', PYTHONHASHSEED=str(seed),
        PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='')
    return env


def run_job(job, python, runner):
    out = Path(job['out_dir'])
    out.mkdir(exist_ok=True)
    meta = out / 'run_metadata.json'
    if meta.exists():
        value = json.loads(meta.read_text())
        if value.get('status') != 'complete' or value.get('job') != job:
            raise ValueError('Existing incompatible outcome: ' + str(out))
        return {'job': out.name, 'status': 'previously_complete'}
    if (out / 'training.log').exists():
        raise ValueError('Retain and inspect the incomplete attempt before using a new output directory.')
    (out / 'job.json').write_text(json.dumps(job, indent=2))
    with (out / 'training.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([str(python), '-B', str(runner), str(out / 'job.json')],
            env=legacy_env(python, job['seed'], job['threads']), stdout=log, stderr=subprocess.STDOUT)
    return {'job': out.name, 'status': 'complete' if result.returncode == 0 else 'failed',
        'returncode': result.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--saucie-source', type=Path, required=True)
    parser.add_argument('--legacy-python', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error('Worker and thread counts must be positive.')
    inputs, output, source, python = [p.resolve() for p in
        (args.input_dir, args.output_dir, args.saucie_source, args.legacy_python)]
    for protected in (inputs, source, Path(__file__).resolve().parent):
        if output == protected or protected in output.parents or output in protected.parents:
            raise ValueError('Outputs must not overlap input or source directories.')
    runner = Path(__file__).with_name('optimization_followup_legacy.py')
    jobs = design(inputs, output, source, args.threads)
    registration = {'jobs': jobs, 'runner_sha256': sha(runner), 'launcher_sha256': sha(__file__),
        'registered_before_new_fitting': True, 'fixed_budgets': True,
        'scope': 'Explicit optimisation sensitivity, not replacement of the primary fitted outputs.'}
    manifest = output / 'followup_registration.json'
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != registration:
            raise ValueError('Resume requires identical registered source, inputs and jobs.')
    else:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Choose a new empty output directory.')
        output.mkdir(parents=True, exist_ok=True)
        manifest.write_text(json.dumps(registration, indent=2))
    if args.prepare_only:
        print('Registered all 36 fixed-budget jobs. No model was fitted.')
        return
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(run_job, j, python, runner) for j in jobs]):
            row = future.result()
            records.append(row)
            (output / 'execution.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(row), flush=True)
    if any(r['status'] == 'failed' for r in records):
        raise SystemExit('A failed fit is preserved, not replaced by a scored fallback.')


if __name__ == '__main__':
    main()
