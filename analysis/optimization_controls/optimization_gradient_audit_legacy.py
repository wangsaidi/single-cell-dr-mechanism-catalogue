"""Re-evaluate saved objectives/gradients without fitting or changing model states."""
from __future__ import print_function
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['KMP_AFFINITY']='disabled'
import numpy as np
import tensorflow as tf


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    job=json.loads(Path(sys.argv[1]).read_text())
    folder=Path(job['out_dir'])
    meta=json.loads((folder/'run_metadata.json').read_text())
    if meta['status']!='complete' or meta['job']!=job:
        raise ValueError('Only complete, registered fits may be audited.')
    target=folder/'full_gradient_diagnostics_v2.csv'
    if target.exists():
        raise ValueError('Do not overwrite a completed gradient audit.')
    with np.load(job['input'],allow_pickle=False) as value:
        X=np.asarray(value['X'],dtype=np.float32)
    final=2500 if job['method']=='scScope' else 30000
    baseline=600 if job['method']=='scScope' else 10000
    positions=(baseline,final) if (folder/('state%d'%baseline)/'model.index').exists() else (final,)
    if job['method']=='scScope' and len(positions)!=2:
        raise ValueError('An expected scScope baseline state is missing.')
    histories=list(csv.DictReader((folder/'training_history.csv').open()))
    template=folder/('state%d'%final)/'model.meta'
    inputs=[template]
    for position in positions:
        inputs.extend((folder/('state%d'%position)).glob('model.*'))
    hashes={str(p):sha(p) for p in inputs}
    graph=tf.Graph()
    with graph.as_default():
        saver=tf.train.import_meta_graph(str(template),clear_devices=True)
        variables=tf.trainable_variables()
        if job['method']=='scScope':
            import scscope
            variables=[v for v in variables if '/inference/' in v.name]
            holder=tf.placeholder(tf.float32,shape=[None,X.shape[1]],name='audit_X')
            ids=tf.placeholder(tf.float32,shape=[None,1],name='audit_batches')
            restored={v.op.name:v for v in tf.global_variables()}
            def restored_getter(getter,name,*args,**kwargs):
                if name not in restored:
                    raise ValueError('Audit requested a variable absent from the restored state: '+name)
                return restored[name]
            # Imported variables exist in collections, not the get_variable creation store.
            with tf.variable_scope(tf.get_variable_scope(),reuse=True,custom_getter=restored_getter):
                outputs,_,_=scscope.Inference(holder,50,2,[],[],ids,re_use=True)
            if {v.op.name for v in tf.global_variables()}!=set(restored):
                raise ValueError('Read-only auditing unexpectedly created a model variable.')
            norm=tf.norm(tf.sign(holder)*holder)
            objective=tf.add_n([tf.norm(tf.sign(holder)*(out-holder)) for out in outputs])/norm
            feeds=[({holder:X,ids:np.zeros((len(X),1),dtype=np.float32)},len(X))]
            recorded='full_relative_reconstruction'
        else:
            terms=tf.get_collection('losses')
            objective=tf.add_n(terms)
            x=graph.get_tensor_by_name('x:0')
            y=graph.get_tensor_by_name('y:0')
            train=graph.get_tensor_by_name('is_training:0')
            subset=X[np.random.RandomState(20261027).permutation(len(X))[:min(1024,len(X))]]
            feeds=[({x:subset[start:start+256],y:subset[start:start+256],train:False},len(subset[start:start+256]))
                   for start in range(0,len(subset),256)]
            recorded='total_diagnostic_loss'
        gradients=tf.gradients(objective,variables)
        pairs=[(v,g) for v,g in zip(variables,gradients) if g is not None]
        variables=[v for v,g in pairs]
        gradients=[g for v,g in pairs]
    config=tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1,
                         device_count={'GPU':0},allow_soft_placement=True)
    rows=[]
    with tf.Session(graph=graph,config=config) as session:
        for position in positions:
            saver.restore(session,str(folder/('state%d'%position)/'model'))
            total_n=sum(n for _,n in feeds)
            combined=None
            loss=0.
            for feed,n in feeds:
                values=session.run([objective]+gradients,feed_dict=feed)
                if not all(np.isfinite(v).all() for v in values):
                    raise FloatingPointError('Nonfinite saved-state objective or gradient.')
                weight=n/float(total_n)
                loss+=float(values[0])*weight
                weighted=[np.asarray(v,dtype=np.float64)*weight for v in values[1:]]
                combined=weighted if combined is None else [a+b for a,b in zip(combined,weighted)]
            norm=float(np.sqrt(sum(np.square(v).sum() for v in combined)))
            parameters=session.run(variables)
            parameter_norm=float(np.sqrt(sum(np.square(np.asarray(v,dtype=np.float64)).sum() for v in parameters)))
            historical=float(next(r for r in histories if int(r['epoch' if job['method']=='scScope' else 'step'])==position)[recorded])
            if not np.isclose(loss,historical,rtol=1e-5,atol=1e-8):
                raise ValueError('Restored objective differs from the recorded checkpoint: %r %r'%(loss,historical))
            rows.append({'dataset':job['dataset'],'method':job['method'],'configuration':job['configuration'],
                'seed':job['seed'],'position':position,'stage':'baseline' if position==baseline else 'scheduled_final',
                'objective_recomputed':loss,'objective_recorded':historical,
                'gradient_of_monitor_norm':norm,'parameter_norm':parameter_norm,
                'n_diagnostic_cells':total_n,'n_batches':len(feeds),'optimised_parameter_tensors':len(variables),
                'input_sha256':sha(job['input']),'checkpoint_index_sha256':sha(folder/('state%d'%position)/'model.index')})
    if any(sha(Path(name))!=digest for name,digest in hashes.items()):
        raise ValueError('A checkpoint changed during read-only auditing.')
    with target.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder/'full_gradient_audit_v2.json').write_text(json.dumps({
        'status':'verified','no_training_ops_executed':True,'checkpoint_bytes_unchanged':True,
        'auditor_sha256':sha(__file__),'checkpoint_inputs_sha256':hashes,
        'scope':'Gradient of full-data two-state normalised scScope loss or fixed-four-batch mean SAUCIE diagnostic loss; no universal stationarity threshold',
        'baseline_checkpoint_available':len(positions)==2,
        'gradient_norm_ratio':rows[1]['gradient_of_monitor_norm']/rows[0]['gradient_of_monitor_norm']
            if len(rows)==2 and rows[0]['gradient_of_monitor_norm']>0 else None},indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    main()
