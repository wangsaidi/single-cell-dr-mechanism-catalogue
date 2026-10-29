from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from analysis.scripts import score_deep_configuration_controls as prior
from .optimization_followup import learning_rate, strict_stable
OUT = None

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scalar_records(fits):
    result=[]
    prefix=[]
    histories=[]
    input_checks={}
    for job,meta,h in fits:
        keys={k:job[k] for k in ('dataset','method','configuration','seed')}
        if job['method']=='scScope':
            position='epoch'
            original_metric='last_recurrence_relative_reconstruction'
            full_metric='full_relative_reconstruction'
            baseline=600
            tail=h.loc[h.epoch>=2000]
            old_name='%s_scScope_recurrent_latent_seed%d'%(job['dataset'],job['seed'])
            old_metric='reconstruction_error'
        else:
            position='step'
            original_metric=full_metric='total_diagnostic_loss'
            baseline=10000
            tail=h.loc[h.step>=25000]
            old_name='%s_SAUCIE_%s_seed%d%s'%(job['dataset'],job['configuration'],job['seed'],
                '_nonnegative_distance' if job['configuration']=='task_enabled' else '')
            old_metric='total_diagnostic_loss'
        baseline_row=h.loc[h[position]==baseline].iloc[0]
        gradient=[c for c in h if c.startswith('gradient_norm_fixed')][0]
        tail_first=float(tail[full_metric].iloc[:5].mean())
        tail_last=float(tail[full_metric].iloc[-5:].mean())
        result.append({**keys,'completed_budget':int(h[position].iloc[-1]),
            'baseline_budget':baseline,'baseline_full_loss':float(baseline_row[full_metric]),
            'final_full_loss':float(h[full_metric].iloc[-1]),
            'full_loss_relative_change':float((h[full_metric].iloc[-1]-baseline_row[full_metric])/abs(baseline_row[full_metric])),
            'baseline_gradient_norm':float(baseline_row[gradient]),
            'final_gradient_norm':float(h[gradient].iloc[-1]),
            'gradient_norm_ratio':float(h[gradient].iloc[-1]/max(baseline_row[gradient],1e-12)),
            'tail_first5_loss_mean':tail_first,'tail_last5_loss_mean':tail_last,
            'tail_relative_drift':(tail_last-tail_first)/max(abs(tail_first),1e-12),
            'tail_relative_range':float((tail[full_metric].max()-tail[full_metric].min())/tail[full_metric].mean()),
            'strict_original_monitor_met':bool(meta['strict_original_monitor_met_at_final']),
            'strict_full_monitor_met':bool(meta.get('strict_full_monitor_met_at_final',meta['strict_original_monitor_met_at_final'])),
            'training_history_sha256':sha(Path(job['out_dir'])/'training_history.csv'),
            'seed_is_biological_replicate':False})
        for row in h.to_dict('records'):
            histories.append({**keys,**row})
    pd.DataFrame(result).to_csv(OUT/'optimisation_diagnostics.csv',index=False)
    pd.DataFrame(histories).to_csv(OUT/'complete_training_histories.csv',index=False)
    return result

def candidates(fits):
    for job,meta,h in fits:
        folder=Path(job['out_dir'])
        keys={k:job[k] for k in ('dataset','method','configuration','seed')}
        if job['method']=='scScope':
            stages=[('latent_epoch600.npy','baseline'),('latent_final.npy','scheduled_final')]
        else:
            stages=[('embedding_step10000.npy','baseline'),('embedding_final.npy','scheduled_final')]
        for name,stage in stages:
            yield {**keys,'stage':stage,'path':folder/name}
        if job['method']=='scScope':
            latent=np.load(folder/'latent_final.npy',allow_pickle=False)
            for method in ('PCA2','tSNE2'):
                target=folder/('final_'+method+'.npy')
                if not target.exists():
                    coords=PCA(n_components=2,random_state=0).fit_transform(latent) if method=='PCA2' else TSNE(
                        n_components=2,perplexity=30,learning_rate='auto',init='pca',
                        random_state=job['seed'],n_jobs=1).fit_transform(latent)
                    np.save(target,np.asarray(coords,dtype=np.float32))
                yield {**keys,'configuration':'recurrent_'+method,'stage':'scheduled_final','path':target}

def endpoints(fits):
    rows_path=OUT/'matched_endpoint_scores.csv'
    pop_path=OUT/'matched_population_scores.csv'
    rows=pd.read_csv(rows_path).to_dict('records') if rows_path.exists() else []
    populations=pd.read_csv(pop_path).to_dict('records') if pop_path.exists() else []
    key=lambda r:tuple(r[k] for k in ('dataset','configuration','seed','stage','array_sha256'))
    completed={key(r) for r in rows}
    items=list(candidates(fits))
    for dataset in sorted({r['dataset'] for r in items}):
        proc=ad.read_h5ad(prior.OBJECTS/(dataset+'_proc.h5ad'))
        field=proc.uns.get('analysis_label_field',{'pbmc3k':'louvain','paul15':'paul15_clusters',
              'heart_cell_atlas_subsampled':'cell_type'}[dataset])
        labels=proc.obs[field].astype(str).to_numpy()
        reference=prior.neighbors(np.asarray(proc.obsm['X_pca_ref'])[:,:50])
        for item in [r for r in items if r['dataset']==dataset]:
            if key({**item,'array_sha256':sha(item['path'])}) in completed:
                continue
            row,pop=prior.score_one(item,proc,reference,labels)
            rows.append(row)
            populations.extend(pop)
            pd.DataFrame(rows).to_csv(rows_path,index=False)
            pd.DataFrame(populations).to_csv(pop_path,index=False)
            print('SCORED',dataset,item['configuration'],item['seed'],item['stage'],flush=True)
    return rows

def checkpoint_stability(fits):
    rows=[]
    for job,meta,h in fits:
        folder=Path(job['out_dir'])
        if job['method']=='scScope':
            paths=sorted((p for p in folder.glob('latent_epoch*.npy') if int(p.stem.removeprefix('latent_epoch'))>=2000),
                         key=lambda p:int(p.stem.removeprefix('latent_epoch')))
        else:
            paths=sorted((p for p in folder.glob('embedding_step*.npy') if int(p.stem.removeprefix('embedding_step'))>=25000),
                         key=lambda p:int(p.stem.removeprefix('embedding_step')))
        for first,last in zip(paths[:-1],paths[1:]):
            A,B=np.load(first,allow_pickle=False),np.load(last,allow_pickle=False)
            na,nb=prior.neighbors(A),prior.neighbors(B)
            local=float(np.mean([len(set(a)&set(b))/15 for a,b in zip(na,nb)]))
            rng=np.random.default_rng(20261027)
            i=rng.integers(0,len(A),6000)
            j=rng.integers(0,len(A),6000)
            valid=i!=j
            i,j=i[valid][:5000],j[valid][:5000]
            rho=float(spearmanr(np.linalg.norm(A[i]-A[j],axis=1),np.linalg.norm(B[i]-B[j],axis=1)).statistic)
            rows.append({**{k:job[k] for k in ('dataset','method','configuration','seed')},
                'first_checkpoint':first.stem,'second_checkpoint':last.stem,
                'neighbour_overlap':local,'distance_rank_agreement':rho,
                'n_cells':len(A),'n_distance_pairs':len(i),'k':15,
                'first_array_sha256':sha(first),'second_array_sha256':sha(last)})
    pd.DataFrame(rows).to_csv(OUT/'tail_checkpoint_stability.csv',index=False)
    return rows
