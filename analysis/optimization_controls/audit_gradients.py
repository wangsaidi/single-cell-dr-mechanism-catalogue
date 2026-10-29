"""Audit actual saved states without fitting or reconstructing missing checkpoints."""
from pathlib import Path
import argparse
import csv
import json
import subprocess

from .optimization_followup import legacy_env, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controls-dir', type=Path, required=True)
    parser.add_argument('--legacy-python', type=Path, required=True)
    args = parser.parse_args()
    out, python = args.controls_dir.resolve(), args.legacy_python.resolve()
    registry = json.loads((out / 'followup_registration.json').read_text())
    runner = Path(__file__).with_name('optimization_gradient_audit_legacy.py')
    rows = []
    for job in registry['jobs']:
        folder = Path(job['out_dir'])
        if not (folder / 'run_metadata.json').exists():
            raise ValueError('Finish all registered fitting jobs before this audit.')
        target = folder / 'full_gradient_diagnostics_v2.csv'
        if not target.exists():
            with (folder / 'gradient_audit.log').open('a', encoding='utf-8') as log:
                subprocess.run([str(python), '-B', str(runner), str(folder / 'job.json')],
                    check=True, env=legacy_env(python, job['seed'], 1), stdout=log, stderr=subprocess.STDOUT)
        receipt = json.loads((folder / 'full_gradient_audit_v2.json').read_text())
        if receipt['auditor_sha256'] != sha(runner) or not receipt['no_training_ops_executed'] or \
                not receipt['checkpoint_bytes_unchanged']:
            raise ValueError('A saved-state audit is stale or incomplete.')
        if any(sha(name) != digest for name, digest in receipt['checkpoint_inputs_sha256'].items()):
            raise ValueError('A saved checkpoint changed after its read-only audit.')
        with target.open(newline='', encoding='utf-8') as stream:
            rows.extend(csv.DictReader(stream))
    if len(rows) != 45:
        raise ValueError('Expected 18 scScope and 27 SAUCIE actual state evaluations.')
    with (out / 'full_gradient_diagnostics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / 'full_gradient_audit_summary.json').write_text(json.dumps({
        'completed_fits': 36, 'missing_fits': [], 'all_36_complete': True,
        'scope': 'Read-only saved-state monitor and gradient checks, not a convergence threshold.'}, indent=2))
    print('Verified 45 saved states from all 36 fits without training updates.')


if __name__ == '__main__':
    main()
