"""Python 3.7 runner for explicit fixed-budget learning-rate sensitivities."""
from __future__ import print_function
import csv
import hashlib
import importlib.util
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

from optimization_followup import learning_rate, strict_stable


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows_write(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rng_state():
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': [state[0], state[1].tolist(),
            int(state[2]), int(state[3]), float(state[4])]}


def finite(*arrays):
    if not all(np.isfinite(value).all() for value in arrays):
        raise FloatingPointError('Nonfinite objective, gradient or representation; no replacement allowed.')


def grad_norm(values):
    values = [value.values if hasattr(value, 'values') else value for value in values]
    finite(*values)
    return float(np.sqrt(sum(np.square(np.asarray(value, dtype=np.float64)).sum() for value in values)))


def session_threads(threads):
    original = tf.Session
    def session(*args, **kwargs):
        config = kwargs.get('config') or tf.ConfigProto()
        config.intra_op_parallelism_threads = threads
        config.inter_op_parallelism_threads = 1
        kwargs['config'] = config
        return original(*args, **kwargs)
    tf.Session = session


def scscope_run(X, job, out):
    import scscope
    text = inspect.getsource(scscope.train)
    source_hash = sha(inspect.getsourcefile(scscope.train))
    namespace = dict(scscope.__dict__)
    histories = []
    started = time.time()
    fixed = np.random.RandomState(20261028).permutation(len(X))[:64]
    norm_input = float(np.linalg.norm(np.sign(X)*X))
    if norm_input == 0:
        raise ValueError('scScope input is entirely zero.')
    saved = set((300, 600, 1000, 1500, 1750, 2000, 2250, 2500))

    def checkpoint(epoch, original_error, sess, codes, outputs, holder, batches, data, ids,
                   gradients, training_holder, training_batches):
        values, recons = sess.run([codes, outputs], feed_dict={holder: data, batches: ids})
        latent = np.concatenate(values, axis=1).astype(np.float32)
        components = [float(np.linalg.norm(np.sign(data)*(r-data))/norm_input) for r in recons]
        finite(latent, np.asarray(components), np.asarray(original_error))
        if not np.isclose(components[-1], original_error, rtol=1e-6, atol=1e-7):
            raise ValueError('Last-recurrence monitor differs from the official callback value.')
        gradient_values = sess.run([pair[0] for pair in gradients], feed_dict={
            training_holder: data[fixed], training_batches: ids[fixed]})
        full = sum(components)
        delta = abs(full-histories[-1]['full_relative_reconstruction'])/max(
                    abs(histories[-1]['full_relative_reconstruction']), 1e-12) if histories else None
        row = {'epoch': int(epoch), 'last_recurrence_relative_reconstruction': float(original_error),
               'full_relative_reconstruction': full, 'recurrence1_relative_reconstruction': components[0],
               'recurrence2_relative_reconstruction': components[1], 'full_relative_checkpoint_change': delta,
               'gradient_norm_fixed64': grad_norm(gradient_values),
               'learning_rate_used': learning_rate('scScope', epoch),
               'elapsed_seconds': time.time()-started}
        histories.append(row)
        rows_write(out/'training_history.csv', histories)
        if epoch in saved or epoch >= 2000 and epoch % 100 == 0:
            np.save(str(out/('latent_epoch%d.npy'%epoch)), latent)
        if epoch in saved:
            states = out/('state%d'%epoch)
            states.mkdir(exist_ok=True)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            saver.save(sess, str(states/'model'), write_meta_graph=epoch==job['max_epochs'])
            (states/'rng.json').write_text(json.dumps(rng_state()), encoding='utf-8')
        print('CHECKPOINT', job['dataset'], job['seed'], epoch, full, flush=True)

    replacements = [
        ("with tf.Graph().as_default(), tf.device('/cpu:0'):",
         "with tf.Graph().as_default(), tf.device('/cpu:0'):\n        tf.set_random_seed(_seed)"),
        ('grads = average_gradients(tower_grads)',
         'grads = average_gradients(tower_grads)\n        followup_rate = tf.placeholder_with_default(tf.constant(learning_rate, dtype=tf.float32), shape=(), name="followup_rate")\n        opt = tf.train.AdamOptimizer(followup_rate, beta1=beta1)'),
        ("print('Current reconstruction error is: '+str(recon_error))",
         "print('Current reconstruction error is: '+str(recon_error))\n                _callback(step, recon_error, sess, train_latent_code, layer_output, all_input, exp_batch_idx_all, train_data_set, exp_batch_idx_input, grads, train_data, exp_batch_idx)"),
        ('if (abs(reconstruction_error[-1] - reconstruction_error[-2])/reconstruction_error[-2] < 1e-3):',
         'if False:'),
    ]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError('Pinned scScope source changed: ' + old)
        text = text.replace(old, new, 1)
    old = 'exp_batch_idx: cur_exp_batch_idx})'
    if text.count(old) != 2:
        raise ValueError('Unexpected scScope training-feed structure.')
    text = text.replace(old, 'exp_batch_idx: cur_exp_batch_idx, followup_rate: _rate(step)})')
    namespace.update(_seed=job['seed'], _callback=checkpoint,
                     _rate=lambda epoch: learning_rate('scScope', epoch))
    exec(compile(text, '<fixed-budget scScope learning-rate sensitivity>', 'exec'), namespace)
    model = namespace['train'](X, latent_code_dim=50, use_mask=True, batch_size=64,
        max_epoch=job['max_epochs'], epoch_per_check=25, T=2, encoder_layers=[],
        decoder_layers=[], num_gpus=1)
    latent, _, _ = scscope.predict(X, model)
    finite(latent)
    np.save(str(out/'latent_final.npy'), np.asarray(latent, dtype=np.float32))
    model['latent_code_session'].close()
    return {'source_sha256': source_hash, 'final_epoch': histories[-1]['epoch'],
        'strict_original_monitor_met_at_final': strict_stable([r['last_recurrence_relative_reconstruction'] for r in histories]),
        'strict_full_monitor_met_at_final': strict_stable([r['full_relative_reconstruction'] for r in histories]),
        'loss_definition': 'sum of both masked relative Frobenius reconstruction norms; last recurrence retained separately',
        'representation_dimension': 100, 'checkpoints': len(histories)}


def saucie_run(X, job, out):
    sys.path.insert(0, str(Path(job['saucie_source']).parent))
    import SAUCIE
    spec = importlib.util.spec_from_file_location('prior_control', job['prior_runner'])
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    tf.reset_default_graph()
    tf.set_random_seed(job['seed'])
    architecture = job['configuration'] != 'reconstruction'
    active = job['configuration'] == 'task_enabled'

    class ArchitectureMatched(SAUCIE.SAUCIE):
        def _build_losses(self):
            c, d = self.lambda_c, self.lambda_d
            self.lambda_c, self.lambda_d = 0, 0
            super(ArchitectureMatched, self)._build_losses()
            self.lambda_c, self.lambda_d = c, d

    base = ArchitectureMatched if job['configuration']=='task_architecture_reconstruction' else SAUCIE.SAUCIE
    base = previous.saucie_distance_control_class(base, job['saucie_distance_stabilization'])

    class Scheduled(base):
        def _build_optimization(self, norm_clip=5.):
            self.followup_rate = tf.placeholder_with_default(
                tf.constant(self.learning_rate, dtype=tf.float32), shape=(), name='followup_rate')
            opt = tf.train.AdamOptimizer(self.followup_rate)
            self.train_op = opt.minimize(self.loss, name='train_op')

    h1 = min(512, max(16, X.shape[1]//2))
    h2 = min(256, max(8, h1//2))
    h3 = min(128, max(4, h2//2))
    model = Scheduled(X.shape[1], layers=[h1,h2,h3,2], no_gpu=True,
        lambda_c=.1 if architecture else 0, lambda_d=.2 if architecture else 0, lambda_b=0)
    loader = SAUCIE.Loader(X, shuffle=True)
    idx = np.random.RandomState(20261027).permutation(len(X))[:min(1024,len(X))]
    np.save(str(out/'diagnostic_indices.npy'), idx)
    diagnostic = X[idx]
    graph = tf.get_default_graph()
    x,y,training = [graph.get_tensor_by_name(n) for n in ('x:0','y:0','is_training:0')]
    terms = tf.get_collection('losses')
    names = [t.name.split(':')[0].replace('/','_') for t in terms]
    gradients = [g for g in tf.gradients(model.loss, tf.trainable_variables()) if g is not None]
    histories = []
    started = time.time()
    saved = set((300,1000,3000,6000,10000,15000,20000,25000,30000))

    def diagnostic_row(step):
        total = np.zeros(len(terms), dtype=np.float64)
        for start in range(0,len(diagnostic),256):
            batch = diagnostic[start:start+256]
            values = model.sess.run(terms, feed_dict={x:batch,y:batch,training:False})
            finite(*values)
            total += len(batch)*np.asarray(values)
        total /= len(diagnostic)
        gradient_values = model.sess.run(gradients, feed_dict={x:diagnostic[:256],y:diagnostic[:256],training:False})
        score = float(total.sum())
        row = {'step': step, 'total_diagnostic_loss': score,
            'relative_checkpoint_change': abs(score-histories[-1]['total_diagnostic_loss'])/max(
                abs(histories[-1]['total_diagnostic_loss']),1e-12) if histories else None,
            'gradient_norm_fixed256':grad_norm(gradient_values),
            'learning_rate_used': learning_rate('SAUCIE', max(step,1)),
            'elapsed_seconds':time.time()-started}
        row.update({name:float(value) for name,value in zip(names,total)})
        return row

    try:
        diagnostic_row(0)
        for step in range(1,job['max_steps']+1):
            batch = loader.next_batch(batch_size=256)
            model.iteration += 1
            feed = {x:batch[0],y:batch[0],training:True,
                    graph.get_tensor_by_name('learning_rate_tensor:0'):model.learning_rate,
                    model.followup_rate:learning_rate('SAUCIE',step)}
            model.sess.run(model.train_op, feed_dict=feed)
            if step%100:
                continue
            row = diagnostic_row(step)
            histories.append(row)
            rows_write(out/'training_history.csv',histories)
            if step in saved or step>=25000 and step%1000==0:
                latent = model.get_embedding(SAUCIE.Loader(X,shuffle=False))
                finite(latent)
                np.save(str(out/('embedding_step%d.npy'%step)), np.asarray(latent,dtype=np.float32))
            if step in saved:
                states = out/('state%d'%step)
                states.mkdir(exist_ok=True)
                model.saver.save(model.sess,str(states/'model'),write_meta_graph=step==job['max_steps'])
                value = rng_state()
                value['loader'] = {'start':loader.start,'epoch':loader.epoch,'initial_order':loader.r}
                (states/'rng.json').write_text(json.dumps(value),encoding='utf-8')
                print('CHECKPOINT',job['dataset'],job['configuration'],job['seed'],step,row['total_diagnostic_loss'],flush=True)
        latent = model.get_embedding(SAUCIE.Loader(X,shuffle=False))
        finite(latent)
        np.save(str(out/'embedding_final.npy'),np.asarray(latent,dtype=np.float32))
    except Exception:
        state = model.sess.run({v.name:v for v in tf.global_variables()})
        np.savez_compressed(str(out/'failed_model_state.npz'),**state)
        raise
    finally:
        model.sess.close()
    return {'source_sha256':sha(inspect.getsourcefile(SAUCIE.SAUCIE)),
        'final_step':histories[-1]['step'],'checkpoints':len(histories),
        'strict_original_monitor_met_at_final':strict_stable([r['total_diagnostic_loss'] for r in histories]),
        'active_task_penalties':active,'task_architecture':architecture,
        'lambda_c':.1 if active else 0,'lambda_d':.2 if active else 0,'lambda_b':0,
        'representation_dimension':2,'diagnostic_subset_n':len(diagnostic),
        'numerical_sensitivity':job['saucie_distance_stabilization'],
        'loss_definition':'same objective components and fixed1024 diagnostic training groups as preceding controls'}


def main():
    job = json.loads(Path(sys.argv[1]).read_text())
    out = Path(job['out_dir'])
    if (out/'training_history.csv').exists() or (out/'run_metadata.json').exists():
        raise ValueError('Preserve prior artifacts; training output must be fresh.')
    if sha(job['input']) != job['input_sha256']:
        raise ValueError('Input checksum differs from registration.')
    np.random.seed(job['seed'])
    random.seed(job['seed'])
    tf.set_random_seed(job['seed'])
    session_threads(job['threads'])
    with np.load(job['input'],allow_pickle=False) as archive:
        X = np.asarray(archive['X'],dtype=np.float32)
        cell_ids = archive['cell_ids']
        gene_ids = archive['gene_ids']
    finite(X)
    if X.shape != (len(cell_ids),len(gene_ids)):
        raise ValueError('Ordered identifiers do not match the numeric input.')
    started=time.time()
    try:
        result = scscope_run(X,job,out) if job['method']=='scScope' else saucie_run(X,job,out)
    except Exception as exc:
        failure={'status':'failed','job':job,'exception':repr(exc),'elapsed_seconds':time.time()-started}
        (out/'failure_metadata.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
        raise
    result.update(status='complete',job=job,input_shape=list(X.shape),input_sha256=sha(job['input']),
        python=sys.version,numpy=np.__version__,tensorflow=tf.__version__,
        runner_sha256=sha(__file__),elapsed_seconds=time.time()-started,
        no_early_stopping=True,biological_replicates=False)
    (out/'run_metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
