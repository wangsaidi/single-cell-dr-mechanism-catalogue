"""Python 3.7-compatible, instrumented original-model configuration controls."""
from __future__ import print_function
import csv
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ['KMP_AFFINITY'] = 'disabled'

import numpy as np
import tensorflow as tf


def save_rows(path, rows):
    if not rows:
        return
    with open(str(path), 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hash(module):
    return hashlib.sha256(Path(inspect.getsourcefile(module)).read_bytes()).hexdigest()


def numeric_summary(values):
    values = np.asarray(values)
    finite = np.isfinite(values)
    return {'shape': list(values.shape), 'dtype': str(values.dtype),
            'n_nan': int(np.isnan(values).sum()),
            'n_posinf': int(np.isposinf(values).sum()),
            'n_neginf': int(np.isneginf(values).sum()),
            'n_zero': int((values == 0).sum()),
            'finite_min': float(values[finite].min()) if finite.any() else None,
            'finite_max': float(values[finite].max()) if finite.any() else None}


def saucie_distance_control_class(base_class, enabled=False):
    if type(enabled) is not bool:
        raise ValueError('saucie_distance_stabilization must be a JSON boolean')
    if not enabled:
        return base_class

    class NonnegativeSquaredDistanceSAUCIE(base_class):
        def _pairwise_dists(self, x1, x2):
            distances = super(NonnegativeSquaredDistanceSAUCIE, self)._pairwise_dists(x1, x2)
            # Opt-in numerical sensitivity: project finite squared distances only.
            distances = tf.check_numerics(distances, 'Non-finite raw SAUCIE squared distances',
                                         name='round2_finite_squared_distances')
            return tf.maximum(distances, tf.zeros_like(distances),
                              name='round2_nonnegative_squared_distances')

    return NonnegativeSquaredDistanceSAUCIE


def saucie_probe_tensors(graph):
    # Fetch existing forward ops, including their unmodified division operands.
    probes = {name: graph.get_tensor_by_name(name) for name in
              ('layer_c:0', 'embeddings:0', 'output:0')}
    for op in graph.get_operations():
        if op.name.startswith(('clustering/', 'intracluster_distances/')) and op.type in ('Div', 'RealDiv', 'Sqrt', 'Log', 'Exp', 'CheckNumerics', 'Maximum'):
            for tensor in list(op.inputs) + list(op.outputs):
                probes[tensor.name] = tensor
    return probes


def append_row(path, row):
    exists = path.exists()
    with open(str(path), 'a', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def configure_session():
    original = tf.Session
    def session(*args, **kwargs):
        config = kwargs.get('config') or tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.inter_op_parallelism_threads = 1
        kwargs['config'] = config
        return original(*args, **kwargs)
    tf.Session = session


def scscope_control(X, job, out):
    import scscope
    source = inspect.getsource(scscope.train)
    namespace = dict(scscope.__dict__)
    rows = []
    started = time.time()
    def checkpoint(epoch, error, sess, codes, holder, batches, data, batch_ids):
        arrays = sess.run(codes, feed_dict={holder: data, batches: batch_ids})
        latent = np.concatenate(arrays, axis=1).astype(np.float32)
        np.save(str(out / ('latent_epoch%d.npy' % epoch)), latent)
        delta = abs(error - rows[-1]['reconstruction_error']) / max(abs(rows[-1]['reconstruction_error']), 1e-12) if rows else None
        rows.append({'epoch': int(epoch), 'reconstruction_error': float(error),
                     'relative_checkpoint_change': delta, 'latent_dimension': int(latent.shape[1]),
                     'elapsed_seconds': time.time() - started})
        save_rows(out / 'training_history.csv', rows)
        print('CHECKPOINT', job['dataset'], job['seed'], epoch, error, flush=True)
    needle = "with tf.Graph().as_default(), tf.device('/cpu:0'):"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n        tf.set_random_seed(_control_seed)', 1)
    needle = "print('Current reconstruction error is: '+str(recon_error))"
    assert source.count(needle) == 1
    source = source.replace(needle, needle + '\n                _checkpoint(step, recon_error, sess, train_latent_code, all_input, exp_batch_idx_all, train_data_set, exp_batch_idx_input)', 1)
    needle = 'if (abs(reconstruction_error[-1] - reconstruction_error[-2])/reconstruction_error[-2] < 1e-3):'
    assert source.count(needle) == 1
    source = source.replace(needle, 'if step >= 300 and len(reconstruction_error) >= 4 and all(abs(reconstruction_error[-i] - reconstruction_error[-i-1]) / max(abs(reconstruction_error[-i-1]), 1e-12) < 1e-3 for i in (1,2,3)):', 1)
    namespace.update({'_control_seed': job['seed'], '_checkpoint': checkpoint})
    exec(compile(source, '<instrumented original scScope.train>', 'exec'), namespace)
    model = namespace['train'](X, latent_code_dim=50, use_mask=True, batch_size=64,
                               max_epoch=job['max_epochs'], epoch_per_check=25, T=2,
                               encoder_layers=[], decoder_layers=[], num_gpus=1)
    latent, reconstruction, _ = scscope.predict(X, model)
    np.save(str(out / 'latent_final.npy'), np.asarray(latent, dtype=np.float32))
    model['latent_code_session'].close()
    stable = len(rows) >= 4 and all(r['relative_checkpoint_change'] is not None and r['relative_checkpoint_change'] < .001 for r in rows[-3:])
    return {'source_sha256': source_hash(scscope.train), 'checkpoints': len(rows),
            'final_epoch': rows[-1]['epoch'], 'final_reconstruction_error': rows[-1]['reconstruction_error'],
            'checkpoint_stability_rule_met': stable, 'latent_dimension': int(latent.shape[1]),
            'instrumentation': 'original train function with in-graph seed, checkpoint export, and three-consecutive-change stopping rule after epoch300; architecture and training objective unchanged',
            'loss_definition': 'full-training-data masked relative reconstruction Frobenius norm',
            'batch_size': 64, 'maximum_epochs': job['max_epochs']}


def saucie_control(X, job, out):
    source_path = Path(job['saucie_source'])
    sys.path.insert(0, str(source_path.parent))
    import SAUCIE
    tf.reset_default_graph()
    tf.set_random_seed(job['seed'])
    class ArchitectureMatchedReconstruction(SAUCIE.SAUCIE):
        def _build_losses(self):
            old_c, old_d = self.lambda_c, self.lambda_d
            self.lambda_c, self.lambda_d = 0, 0
            super(ArchitectureMatchedReconstruction, self)._build_losses()
            self.lambda_c, self.lambda_d = old_c, old_d
    cls = ArchitectureMatchedReconstruction if job['configuration'] == 'task_architecture_reconstruction' else SAUCIE.SAUCIE
    distance_stabilization = job.get('saucie_distance_stabilization', False)
    cls = saucie_distance_control_class(cls, distance_stabilization)
    enabled_branch = job['configuration'] != 'reconstruction'
    h1 = min(512, max(16, X.shape[1] // 2))
    h2 = min(256, max(8, h1 // 2))
    h3 = min(128, max(4, h2 // 2))
    model = cls(X.shape[1], layers=[h1, h2, h3, 2], no_gpu=True,
                lambda_c=.1 if enabled_branch else 0,
                lambda_d=.2 if enabled_branch else 0, lambda_b=0)
    train = SAUCIE.Loader(X, shuffle=True)
    rng = np.random.RandomState(20261027)
    diagnostic = X[rng.permutation(len(X))[:min(1024, len(X))]]
    losses = tf.get_collection('losses')
    names = [t.name.split(':')[0].replace('/', '_') for t in losses]
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('x:0')
    y = graph.get_tensor_by_name('y:0')
    is_training = graph.get_tensor_by_name('is_training:0')
    training_flag_consumers = [op.name for op in is_training.consumers()]
    trace_steps = int(job.get('saucie_trace_steps', 100 if job['configuration'] == 'task_enabled' else 0))
    if trace_steps < 0:
        raise ValueError('saucie_trace_steps must be nonnegative')
    variables = tf.trainable_variables()
    gradients = []
    if trace_steps:
        gradients = [(variable.name, gradient) for variable, gradient in
                     zip(variables, tf.gradients(model.loss, variables)) if gradient is not None]
    probes = saucie_probe_tensors(graph)

    def numeric_failure(reason, batch, phase, batch_start, extra=None):
        feed = {x: batch, y: batch, is_training: phase == 'training_batch_pre_update'}
        tensors = dict(probes)
        tensors.update({variable.name: variable for variable in tf.global_variables()})
        tensors.update({'gradient/' + name: gradient for name, gradient in gradients})
        tensors.update({loss.name: loss for loss in losses})
        state = model.sess.run(tensors, feed_dict=feed)
        state['failing_batch'] = batch
        if extra:
            state.update(extra)
        keys = sorted(state)
        # Raw NaNs/infinities survive in NPZ; JSON counts are not replacements.
        with open(str(out / 'nonfinite_state.npz'), 'xb') as stream:
            np.savez_compressed(stream, **{'tensor_%04d' % i: state[key] for i, key in enumerate(keys)})
        divisions = []
        for op in graph.get_operations():
            if op.type in ('Div', 'RealDiv') and op.name.startswith(('clustering/', 'intracluster_distances/')):
                divisions.append({'operation': op.name, 'numerator': op.inputs[0].name,
                                  'denominator': op.inputs[1].name, 'output': op.outputs[0].name})
        mode_values = {str(flag): [str(value) for value in model.sess.run(losses,
                       feed_dict={x: batch, y: batch, is_training: flag})] for flag in (False, True)}
        record = {'status': 'failed_nonfinite', 'reason': reason, 'phase': phase,
                  'saucie_distance_stabilization': distance_stabilization,
                  'numerical_implementation': 'float32_nonnegative_squared_distance_sensitivity' if distance_stabilization else 'original',
                  'step': int(model.iteration), 'batch_start': int(batch_start),
                  'step_is_pre_update_attempt': phase == 'training_batch_pre_update',
                  'training_flag_consumers': training_flag_consumers, 'losses_by_training_flag': mode_values,
                  'division_operands': divisions,
                  'arrays': [{'key': 'tensor_%04d' % i, 'tensor': key, 'summary': numeric_summary(state[key])}
                             for i, key in enumerate(keys)]}
        with open(str(out / 'nonfinite_diagnostics.json'), 'x', encoding='utf-8') as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
        print('NONFINITE', json.dumps({'step': model.iteration, 'phase': phase, 'reason': reason}), flush=True)
        model.sess.close()
        raise RuntimeError('Non-finite SAUCIE %s at step %d (%s); raw state and diagnostic row preserved.' %
                           (reason, model.iteration, phase))

    def audit_batch(batch, phase, batch_start=-1, check_gradients=False):
        feed = {x: batch, y: batch, is_training: phase == 'training_batch_pre_update'}
        fetches = list(losses) + ([gradient for _, gradient in gradients] if check_gradients else [])
        fetched = model.sess.run(fetches, feed_dict=feed)
        values = np.asarray(fetched[:len(losses)], dtype=float)
        gradient_finite = all(np.isfinite(value).all() for value in fetched[len(losses):])
        row = {'step': int(model.iteration), 'phase': phase, 'batch_start': int(batch_start),
               'batch_n': len(batch), 'total_diagnostic_loss': float(values.sum()),
               'objective_finite': bool(np.isfinite(values).all()),
               'gradients_checked': bool(check_gradients), 'gradients_finite': bool(gradient_finite) if check_gradients else None}
        row.update({name: float(value) for name, value in zip(names, values)})
        append_row(out / 'diagnostic_batches.csv', row)
        if not np.isfinite(values).all() or not gradient_finite:
            numeric_failure('objective' if not np.isfinite(values).all() else 'gradient', batch, phase, batch_start)
        return values

    class ObservedLoader(object):
        def next_batch(self, batch_size=256):
            batch = train.next_batch(batch_size=batch_size)
            if model.iteration <= trace_steps:
                audit_batch(batch[0], 'training_batch_pre_update', check_gradients=True)
            return batch

    observed_train = ObservedLoader() if trace_steps else train
    for start in range(0, len(diagnostic), 256):
        audit_batch(diagnostic[start:start + 256], 'initial_diagnostic', start, check_gradients=bool(trace_steps))
    rows = []
    started = time.time()
    stable = False
    while model.iteration < job['max_steps']:
        model.train(observed_train, steps=min(100, job['max_steps'] - model.iteration), batch_size=256)
        total = np.zeros(len(losses), dtype=float)
        n = 0
        for start in range(0, len(diagnostic), 256):
            batch = diagnostic[start:start + 256]
            total += len(batch) * audit_batch(batch, 'checkpoint_diagnostic', start)
            n += len(batch)
        values = total / n
        score = float(values.sum())
        delta = abs(score - rows[-1]['total_diagnostic_loss']) / max(abs(rows[-1]['total_diagnostic_loss']), 1e-12) if rows else None
        row = {'step': int(model.iteration), 'total_diagnostic_loss': score,
               'relative_checkpoint_change': delta, 'effective_data_passes': model.iteration * 256. / len(X),
               'elapsed_seconds': time.time() - started}
        row.update({name: float(value) for name, value in zip(names, values)})
        rows.append(row)
        save_rows(out / 'training_history.csv', rows)
        if not np.isfinite(values).all():
            numeric_failure('aggregate objective', diagnostic[:256], 'checkpoint_aggregate', -1,
                            {'aggregate_objective_components': values})
        if model.iteration in (100, 300, 1000, 3000, 6000, 10000):
            coords = model.get_embedding(SAUCIE.Loader(X, shuffle=False))
            np.save(str(out / ('embedding_step%d.npy' % model.iteration)), np.asarray(coords, dtype=np.float32))
            if not np.isfinite(coords).all():
                numeric_failure('embedding', diagnostic[:256], 'checkpoint_embedding', -1,
                                {'embedding': np.asarray(coords)})
            print('CHECKPOINT', job['dataset'], job['configuration'], job['seed'], model.iteration, score, flush=True)
        stable = model.iteration >= 3000 and len(rows) >= 4 and all(r['relative_checkpoint_change'] is not None and r['relative_checkpoint_change'] < .001 for r in rows[-3:])
        if stable:
            break
    coords = model.get_embedding(SAUCIE.Loader(X, shuffle=False))
    np.save(str(out / 'embedding_final.npy'), np.asarray(coords, dtype=np.float32))
    if not np.isfinite(coords).all():
        numeric_failure('embedding', diagnostic[:256], 'final_embedding', -1, {'embedding': np.asarray(coords)})
    model.sess.close()
    return {'source_sha256': source_hash(SAUCIE.SAUCIE), 'checkpoints': len(rows),
            'final_step': rows[-1]['step'], 'final_diagnostic_loss': rows[-1]['total_diagnostic_loss'],
            'checkpoint_stability_rule_met': stable, 'latent_dimension': 2,
            'active_clustering_penalties': job['configuration'] == 'task_enabled',
            'task_architecture': enabled_branch, 'lambda_c': .1 if job['configuration'] == 'task_enabled' else 0,
            'lambda_d': .2 if job['configuration'] == 'task_enabled' else 0, 'lambda_b': 0,
            'batch_size': 256, 'maximum_steps': job['max_steps'],
            'loss_definition': 'raw objective components averaged over fixed diagnostic training subset; not independent validation',
            'diagnostic_subset_n': len(diagnostic), 'diagnostic_subset_seed': 20261027,
            'training_flag_consumers': training_flag_consumers, 'pre_update_trace_steps': trace_steps,
            'saucie_distance_stabilization': distance_stabilization,
            'numerical_implementation_sensitivity': distance_stabilization,
            'pairwise_distance_implementation': 'max(finite original float32 squared distance,0) before both RBF and sqrt; original sqrt +1e-3 retained' if distance_stabilization else 'original float32 squared-distance formula',
            'numerical_instrumentation': 'explicit opt-in squared-distance nonnegativity projection; no denominator epsilon or optimizer changes; all detected nonfinite values remain fatal' if distance_stabilization else 'original losses, gradients and division operands observed without epsilon/clipping or optimizer changes; failure is fatal and raw state is retained'}


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(job['out_dir'])
    out.mkdir(parents=True, exist_ok=True)
    for name in ('training_history.csv', 'diagnostic_batches.csv', 'run_metadata.json',
                 'nonfinite_state.npz', 'nonfinite_diagnostics.json', 'failure_metadata.json'):
        if (out / name).exists():
            raise FileExistsError('Preserve previous run artifacts; choose a fresh out_dir: %s' % out)
    np.random.seed(job['seed'])
    random.seed(job['seed'])
    tf.set_random_seed(job['seed'])
    configure_session()
    input_path = Path(job['input'])
    X = np.load(str(input_path))['X'].astype(np.float32)
    started = time.time()
    try:
        if not np.isfinite(X).all():
            raise ValueError('Non-finite model input; no replacement or model fit is allowed.')
        result = scscope_control(X, job, out) if job['method'] == 'scScope' else saucie_control(X, job, out)
    except Exception as exc:
        failure = {'status': 'failed', 'exception': repr(exc), 'job': job,
                   'input_file_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                   'input_shape': list(X.shape), 'elapsed_seconds': time.time() - started}
        with open(str(out / 'failure_metadata.json'), 'x', encoding='utf-8') as stream:
            json.dump(failure, stream, indent=2, allow_nan=False)
        raise
    result.update(job)
    result.update({'input_file_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                   'input_shape': list(X.shape), 'elapsed_seconds': time.time() - started,
                   'python': sys.version, 'numpy': np.__version__, 'tensorflow': tf.__version__,
                   'kmp_affinity': os.environ.get('KMP_AFFINITY'),
                   'status': 'complete', 'biological_replicates': False})
    (out / 'run_metadata.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
