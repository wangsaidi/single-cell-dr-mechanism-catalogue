"""Supplementary S21 from complete, measured optimisation controls only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import argparse
OUT = None
FOLLOWUP = None

DATASETS=('pbmc3k','paul15','heart_cell_atlas_subsampled')
COLORS=dict(zip(DATASETS,('#0072B2','#009E73','#8C56A2')))
LABELS=dict(zip(DATASETS,('PBMC3k','Paul15','Heart atlas')))
CONFIGS=('recurrent_latent','reconstruction','task_architecture_reconstruction','task_enabled')
SHORT=('scScope','SAU-R','SAU-A','SAU-T')
MARKERS=('o','s','D','^')


def main():
    global OUT, FOLLOWUP
    parser = argparse.ArgumentParser()
    parser.add_argument("--controls-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    OUT, FOLLOWUP = args.controls_dir.resolve(), args.output_dir.resolve()
    FOLLOWUP.mkdir(parents=True, exist_ok=True)
    summary=json.loads((OUT/'verified_summary.json').read_text())
    if not summary['all_fits_completed'] or not summary['all_endpoints_scored']:
        raise ValueError('Never draw incomplete optimisation evidence.')
    for name,digest in summary['source_sha256'].items():
        if hashlib.sha256((OUT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed since scoring: '+name)
    history=pd.read_csv(OUT/'complete_training_histories.csv')
    diagnostics=pd.read_csv(OUT/'optimisation_diagnostics.csv')
    scores=pd.read_csv(OUT/'matched_endpoint_scores.csv')
    stability=pd.read_csv(OUT/'tail_checkpoint_stability.csv')
    if len(scores)!=90 or len(diagnostics)!=36:
        raise ValueError('Expected all 36 fits and 90 matched endpoints.')
    plt.rcParams.update({'font.family':'Arial','font.size':8,'axes.titlesize':9,
        'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,
        'axes.spines.right':False,'axes.linewidth':.65,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(4,2,figsize=(7.1,10.7))
    fig.subplots_adjust(left=.13,right=.98,top=.965,bottom=.105,hspace=.70,wspace=.47)
    titles=('scScope recurrent loss','SAUCIE reconstruction','SAUCIE architecture control',
            'SAUCIE task sensitivity','Fixed-batch gradients','Extended optimisation',
            'Native-to-display recovery','Late-checkpoint geometry')
    for i,(ax,title) in enumerate(zip(axes.flat,titles)):
        ax.set_title(title,loc='left',fontweight='bold',pad=9)
        ax.text(-.22,1.075,chr(97+i),transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')
        ax.tick_params(length=3,width=.6)
    for ax,config in zip(axes.flat[:4],CONFIGS):
        group=history.loc[history.configuration==config]
        position='epoch' if config=='recurrent_latent' else 'step'
        metric='full_relative_reconstruction' if config=='recurrent_latent' else 'total_diagnostic_loss'
        for dataset in DATASETS:
            records=[]
            for seed in range(3):
                h=group.loc[(group.dataset==dataset)&(group.seed==seed)].sort_values(position)
                if h.empty:
                    raise ValueError('Missing history; no blank panel is permitted.')
                reference=float(h.loc[h[position]==300,metric].iloc[0])
                if reference<=0:
                    raise ValueError('Nonpositive normalization reference; do not invent a replacement.')
                records.append(h[metric].to_numpy()/reference)
            values=np.vstack(records)
            x=h[position].to_numpy()/1000
            ax.plot(x,values.mean(0),lw=1.1,color=COLORS[dataset])
            ax.fill_between(x,values.min(0),values.max(0),color=COLORS[dataset],alpha=.12,lw=0)
        boundaries=(1,1.75) if config=='recurrent_latent' else (10,20)
        for boundary in boundaries:
            ax.axvline(boundary,color='#999999',ls=':',lw=.7,zorder=0)
        ax.set_xlabel('Training epoch (thousands)' if config=='recurrent_latent' else 'Training updates (thousands)')
        ax.set_ylabel('Loss / value at 300 epochs' if config=='recurrent_latent' else 'Objective / value at 300 updates')
        ax.set_xlim(left=0)
    e,f,g,h=axes.flat[4:]
    for ci,config in enumerate(CONFIGS):
        for di,dataset in enumerate(DATASETS):
            part=diagnostics.loc[(diagnostics.configuration==config)&(diagnostics.dataset==dataset)].sort_values('seed')
            if len(part)!=3:
                raise ValueError('Missing fit-level gradient diagnostics.')
            x=ci+(di-1)*.20+np.array([-.045,0,.045])
            e.scatter(x,part.gradient_norm_ratio,color=COLORS[dataset],s=14,alpha=.85,linewidths=.3,edgecolors='white')
            context=scores.loc[(scores.configuration==config)&(scores.dataset==dataset)]
            baseline=context.loc[context.stage=='baseline'].set_index('seed')
            final=context.loc[context.stage=='scheduled_final'].set_index('seed')
            delta=final.resolution_integrated_macro_f1-baseline.resolution_integrated_macro_f1
            f.scatter(x,delta.loc[range(3)],color=COLORS[dataset],s=14,alpha=.85,linewidths=.3,edgecolors='white')
    for ax in (e,f):
        ax.set_xticks(range(4),SHORT,rotation=25,ha='right')
        ax.set_xlabel('Model configuration')
        ax.set_xlim(-.55,3.55)
    e.set_ylabel('Final / baseline gradient norm')
    e.axhline(1,color='#999999',ls=':',lw=.7,zorder=0)
    e.set_ylim(bottom=0)
    f.set_ylabel('Change in integrated macro-F1')
    f.axhline(0,color='#999999',ls=':',lw=.7,zorder=0)
    for dataset in DATASETS:
        native=scores.loc[(scores.dataset==dataset)&(scores.configuration=='recurrent_latent')&
                          (scores.stage=='scheduled_final')].set_index('seed')
        for config,marker in (('recurrent_PCA2','+'),('recurrent_tSNE2','x')):
            projected=scores.loc[(scores.dataset==dataset)&(scores.configuration==config)].set_index('seed')
            g.scatter(native.loc[range(3),'resolution_integrated_macro_f1'],
                projected.loc[range(3),'resolution_integrated_macro_f1'],
                color=COLORS[dataset],marker=marker,s=28,linewidths=1)
    g.plot([0,1],[0,1],color='#999999',ls=':',lw=.7,zorder=0)
    g.set(xlim=(0,1),ylim=(0,1),xlabel='Native 100D integrated macro-F1',ylabel='Projected 2D integrated macro-F1')
    g.legend(handles=[Line2D([],[],color='#444444',marker='+',ls='None',label='PCA2'),
                      Line2D([],[],color='#444444',marker='x',ls='None',label='t-SNE2')],
             loc='upper left',frameon=False,fontsize=7,ncol=2,handletextpad=.35,columnspacing=.9)
    for ci,config in enumerate(CONFIGS):
        for dataset in DATASETS:
            part=stability.loc[(stability.configuration==config)&(stability.dataset==dataset)]
            grouped=part.groupby('seed')[['neighbour_overlap','distance_rank_agreement']].min()
            if len(grouped)!=3:
                raise ValueError('Missing late-checkpoint geometry.')
            h.scatter((1-grouped.distance_rank_agreement)*1e5,(1-grouped.neighbour_overlap)*100,
                color=COLORS[dataset],marker=MARKERS[ci],s=20,alpha=.8,
                edgecolors='white',linewidths=.35)
    rank_change=(1-float(stability.distance_rank_agreement.min()))*1e5
    neighbour_change=(1-float(stability.neighbour_overlap.min()))*100
    if rank_change<=0 or neighbour_change<=0:
        raise ValueError('Inspect a constant checkpoint response before choosing its presentation.')
    h.set_xlabel('Maximum distance-rank loss\n(units of $10^{-5}$)')
    h.set_ylabel('Maximum neighbour change (%)')
    h.set_ylim(0,neighbour_change*1.10)
    h.set_xlim(0,rank_change*1.10)
    handles=[Line2D([],[],color=COLORS[d],marker='o',ls='None',label=LABELS[d],ms=4) for d in DATASETS]
    handles += [Line2D([],[],color='#555555',marker=m,ls='None',label=s,ms=4) for m,s in zip(MARKERS,SHORT)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.012),ncol=4,
               frameon=False,fontsize=7.5,handletextpad=.4,columnspacing=1.4)
    target=FOLLOWUP/'figures'
    target.mkdir(exist_ok=True)
    stem=target/'Supplementary_Fig_S21_Optimisation_Controls'
    for ext in ('pdf','svg','png'):
        fig.savefig(stem.with_suffix('.'+ext),dpi=450,facecolor='white')
    plt.close(fig)
    (target/'S21_source_audit.json').write_text(json.dumps({
        'source_summary_sha256':hashlib.sha256((OUT/'verified_summary.json').read_bytes()).hexdigest(),
        'fits':36,'measured_endpoints':90,'data_are_computational_not_biological_replicates':True,
        'all_data_from_completed_measured_controls':True},indent=2),encoding='utf-8')
    print('Supplementary S21 generated from all measured controls; main figures untouched.')


if __name__=='__main__':
    main()
