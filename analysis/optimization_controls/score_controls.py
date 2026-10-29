"""Score all scheduled controls with the published matched-endpoint definitions."""
from pathlib import Path
import argparse
import hashlib
import json
import os

import anndata as ad
import numpy as np
import pandas as pd

from . import scoring_core as core
from .optimization_followup import DATASETS, CONFIGS, learning_rate, strict_stable, sha
from analysis.scripts import score_deep_configuration_controls as prior


def read_fits(output):
    registry = json.loads((output / 'followup_registration.json').read_text())
    source = Path(__file__).parent
    if registry['launcher_sha256'] != sha(source / 'optimization_followup.py') or \
            registry['runner_sha256'] != sha(source / 'optimization_followup_legacy.py'):
        raise ValueError('Registered portable fitting source changed.')
    expected = {(d, c, s) for d in DATASETS for _, c, _ in CONFIGS for s in (0, 1, 2)}
    observed = {(j['dataset'], j['configuration'], j['seed']) for j in registry['jobs']}
    if len(registry['jobs']) != 36 or observed != expected:
        raise ValueError('The complete declared grid has changed.')
    fits = []
    for job in registry['jobs']:
        folder = Path(job['out_dir'])
        meta = json.loads((folder / 'run_metadata.json').read_text())
        if meta['status'] != 'complete' or meta['job'] != job or sha(job['input']) != job['input_sha256'] or \
                meta['runner_sha256'] != registry['runner_sha256']:
            raise ValueError('A registered model or input is incomplete or mismatched.')
        h = pd.read_csv(folder / 'training_history.csv')
        position, end, interval, metric = ('epoch', 2500, 25, 'last_recurrence_relative_reconstruction') \
            if job['method'] == 'scScope' else ('step', 30000, 100, 'total_diagnostic_loss')
        if h[position].tolist() != list(range(interval, end+1, interval)):
            raise ValueError('A fixed fitting budget was not completed.')
        if not np.array_equal(h.learning_rate_used, [learning_rate(job['method'], int(n)) for n in h[position]]):
            raise ValueError('Recorded learning rates differ from the fixed schedule.')
        if strict_stable(h[metric].tolist()) != bool(meta['strict_original_monitor_met_at_final']):
            raise ValueError('Strict stability outcome does not reproduce from the raw history.')
        fits.append((job, meta, h))
    return fits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-dir', type=Path, required=True)
    parser.add_argument('--analysis-objects', type=Path, required=True)
    args = parser.parse_args()
    out, objects = args.controls_dir.resolve(), args.analysis_objects.resolve()
    fits = read_fits(out)
    gradients = pd.read_csv(out / 'full_gradient_diagnostics.csv')
    audit = json.loads((out / 'full_gradient_audit_summary.json').read_text())
    if not audit['all_36_complete'] or len(gradients) != 45:
        raise ValueError('Run all read-only saved-state gradient audits before final scoring.')
    if not np.allclose(gradients.objective_recomputed, gradients.objective_recorded, rtol=1e-5, atol=1e-8):
        raise ValueError('Saved-state monitored objectives do not match training records.')
    core.OUT = out
    core.prior.OBJECTS = objects
    for dataset in DATASETS:
        proc = ad.read_h5ad(objects / (dataset + '_proc.h5ad'))
        for job, _, _ in [v for v in fits if v[0]['dataset'] == dataset]:
            with np.load(job['input'], allow_pickle=False) as value:
                if not np.array_equal(value['cell_ids'], proc.obs_names.to_numpy().astype(str)):
                    raise ValueError('Training and scoring cell order differ.')
    optimisation = core.scalar_records(fits)
    endpoints = core.endpoints(fits)
    stability = core.checkpoint_stability(fits)
    if len(optimisation) != 36 or len(endpoints) != 90:
        raise ValueError('Expected all 36 fits and 90 matched representation endpoints.')
    summary = {'complete_fits': 36, 'expected_fits': 36, 'missing_fits': [],
        'all_fits_completed': True, 'endpoint_rows': 90, 'expected_endpoint_rows': 90,
        'all_endpoints_scored': True,
        'strict_original_monitor_met': sum(r['strict_original_monitor_met'] for r in optimisation),
        'strict_full_monitor_met': sum(r['strict_full_monitor_met'] for r in optimisation),
        'stability_comparisons': len(stability),
        'scope': 'Computational optimisation sensitivity, not biological replication or global optimality.',
        'source_sha256': {p.name: sha(p) for p in out.glob('*.csv')}}
    (out / 'verified_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
