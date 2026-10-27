"""Draw only complete matched, empirical deep-workflow control results."""
from pathlib import Path
import json
import argparse
import re
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA = FIG = None
DATASETS = ['pbmc3k', 'paul15', 'heart_cell_atlas_subsampled']
COLOURS = dict(zip(DATASETS, ['#0072B2', '#009E73', '#8C56A2']))
LABELS = dict(zip(DATASETS, ['PBMC3k', 'Paul15', 'Heart atlas']))


def meta_records():
    return [json.loads(p.read_text()) | {'folder': str(p.parent)} for p in DATA.glob('*/run_metadata.json')]


def endpoint(ax, scores, metric, ylabel):
    order = ['R1_latent', 'R1_tSNE2', 'R1_PCA2']
    labels = ['Native 100D', 't-SNE 2D', 'PCA 2D']
    for di, dataset in enumerate(DATASETS):
        part = scores[(scores.dataset == dataset) & scores.configuration.isin(order)]
        assert len(part) == 15
        for seed, values in part.groupby('seed'):
            values = values.set_index('configuration').loc[order]
            x = np.arange(3) + (di - 1) * .14
            ax.plot(x, values[metric], color=COLOURS[dataset], lw=.55, alpha=.25)
        med = part.groupby('configuration')[metric].median().loc[order]
        low = part.groupby('configuration')[metric].min().loc[order]
        high = part.groupby('configuration')[metric].max().loc[order]
        x = np.arange(3) + (di - 1) * .14
        ax.errorbar(x, med, yerr=[med-low, high-med], fmt='o', ms=3.5,
                    lw=.8, capsize=2, color=COLOURS[dataset], label=LABELS[dataset])
    ax.set_xticks(range(3), labels)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Fixed scScope output or projection')


def main():
    records = meta_records()
    scores = pd.read_csv(DATA / 'matched_endpoint_scores.csv')
    stability = pd.read_csv(DATA / 'checkpoint_representation_stability.csv')
    expected = [('scScope', 'recurrent_latent'), ('SAUCIE', 'reconstruction'),
                ('SAUCIE', 'task_architecture_reconstruction'), ('SAUCIE', 'task_enabled')]
    for method, configuration in expected:
        fits = [r for r in records if r['method'] == method and r['configuration'] == configuration and not r.get('maximum_budget_extension')]
        assert len(fits) == 9 and {(r['dataset'], r['seed']) for r in fits} == {(d,s) for d in DATASETS for s in range(3)}, (method, configuration, len(fits))
        if configuration == 'task_enabled':
            assert all(r.get('saucie_distance_stabilization') for r in fits)
    assert len([r for r in records if r.get('maximum_budget_extension')]) == 6
    assert len(stability) == 12
    assert all(np.isfinite(scores[c]).all() for c in ['local_retention', 'global_rank_correlation', 'resolution_integrated_macro_f1'])
    assert len(scores) == 162, 'A complete matched score table is required.'
    assert len(records) == 42
    if CHECK_ONLY:
        print('Verified 42 fits, 162 endpoints and 12 checkpoint comparisons; no plots written.')
        return
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'Arial', 'font.size': 8, 'axes.titlesize': 9,
                         'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.linewidth': .65, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(4, 2, figsize=(7.1, 10.5))
    fig.subplots_adjust(left=.12, right=.98, top=.97, bottom=.09, hspace=.67, wspace=.40)
    titles = ['SAUCIE reconstruction', 'Task-penalty sensitivity', 'scScope reconstruction',
              'Training-length effect', 'Projection and local geometry', 'Projection and global geometry',
              'Projection and population recovery', 'Checkpoint structure stability']
    for i, (ax, title) in enumerate(zip(axes.flat, titles)):
        ax.set_title(title, loc='left', pad=9, fontweight='bold')
        ax.text(-.22, 1.08, chr(97+i), transform=ax.transAxes, fontweight='bold', fontsize=11, va='bottom')
        ax.tick_params(length=3, width=.6)
    a,b,c,d,e,f,g,h = axes.flat
    for r in records:
        if (r['method'],r['configuration']) not in [('SAUCIE','reconstruction'),('scScope','recurrent_latent')]:
            continue
        frame = pd.read_csv(Path(r['folder']) / 'training_history.csv')
        if r['method'] == 'SAUCIE':
            base = frame.loc[frame.step.eq(300), 'total_diagnostic_loss'].iloc[0]
            a.plot(frame.step, frame.total_diagnostic_loss / base, color=COLOURS[r['dataset']], alpha=.7, lw=.7,
                   ls='--' if r.get('maximum_budget_extension') else '-')
        else:
            c.plot(frame.epoch, frame.reconstruction_error, color=COLOURS[r['dataset']], alpha=.7, lw=.7,
                   ls='--' if r.get('maximum_budget_extension') else '-')
    a.axvline(300, color='.5', ls='--', lw=.7)
    a.axhline(1, color='.75', lw=.6)
    a.set(xlabel='Training updates', ylabel='Loss / loss at update 300', xscale='log')
    c.axvline(300, color='.5', ls='--', lw=.7)
    c.set(xlabel='Training epochs', ylabel='Relative reconstruction error')
    order = ['reconstruction', 'task_architecture_reconstruction', 'task_enabled_stabilized']
    final = scores[(scores.method == 'SAUCIE') & scores.stage.eq('final') & scores.configuration.isin(order)]
    assert len(final) == 27
    for di, dataset in enumerate(DATASETS):
        part = final[final.dataset.eq(dataset)]
        for seed, values in part.groupby('seed'):
            vals = values.set_index('configuration').loc[order]
            b.plot(np.arange(3)+(di-1)*.12, vals.resolution_integrated_macro_f1,
                   color=COLOURS[dataset], lw=.7, alpha=.5, marker='o', ms=2.4)
    b.set_xticks(range(3), ['Recon.', 'Task arch.', 'Task + numeric'])
    b.set(xlabel='SAUCIE configuration', ylabel='Resolution-integrated macro-F1', ylim=(0,1))
    for method, config, stage, marker in [('SAUCIE','reconstruction','300_steps','s'),
                                           ('scScope','retrained_latent','300_epochs','o')]:
        part = scores[scores.configuration.eq(config)]
        early = part[part.stage.eq(stage)].set_index(['dataset','seed'])
        late = part[part.stage.eq('final')].set_index(['dataset','seed'])
        assert len(early) == len(late) == 9
        for key in early.index:
            d.scatter(early.loc[key,'resolution_integrated_macro_f1'], late.loc[key,'resolution_integrated_macro_f1'],
                      color=COLOURS[key[0]], marker=marker, s=18, edgecolors='white', linewidths=.35)
    d.plot([0,1], [0,1], color='.7', lw=.7, ls='--', zorder=0)
    d.set(xlabel='Macro-F1 at original budget', ylabel='Macro-F1 at primary final checkpoint', xlim=(0,1), ylim=(0,1))
    d.legend(handles=[Line2D([],[],marker='o',ls='',color='.3',label='scScope native'),
                      Line2D([],[],marker='s',ls='',color='.3',label='SAUCIE recon.')],
             frameon=False, fontsize=6.8, loc='upper left', handletextpad=.3)
    endpoint(e, scores, 'local_retention', 'PCA50 neighbour overlap')
    endpoint(f, scores, 'global_rank_correlation', 'Distance-rank correlation')
    endpoint(g, scores, 'resolution_integrated_macro_f1', 'Resolution-integrated macro-F1')
    e.set_ylim(0,float(scores[scores.configuration.isin(['R1_latent','R1_tSNE2','R1_PCA2'])].local_retention.max())*1.18)
    f.set_ylim(min(-.02,float(scores.global_rank_correlation.min())-.02),1)
    g.set_ylim(0,1)
    for dataset in DATASETS:
        sub = stability[stability.dataset.eq(dataset)]
        for extended, values in sub.groupby('budget_extension'):
            h.scatter(values.pairwise_neighbor_overlap, values.pairwise_distance_rank_rho,
                      color=COLOURS[dataset], s=25 if extended else 17, marker='^' if extended else 'o',
                      edgecolors='white', linewidths=.4)
    h.set(xlabel='Checkpoint neighbour overlap', ylabel='Checkpoint distance-rank rho',
          xlim=(max(0,float(stability.pairwise_neighbor_overlap.min())-.025),1.005),
          ylim=(max(0,float(stability.pairwise_distance_rank_rho.min())-.0004),1.00005))
    from matplotlib.ticker import FormatStrFormatter
    h.yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    handles = [Line2D([],[],marker='o',color=COLOURS[d],ls='-',label=LABELS[d],markersize=4) for d in DATASETS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.55,.015),ncol=3,frameon=False,fontsize=8)
    stem = FIG / 'Supplementary_Fig_S18_deep_controls'
    for ext in ['pdf','svg','png']:
        fig.savefig(stem.with_suffix('.'+ext), dpi=350)
    plt.close(fig)
    stable_count = sum(bool(r['checkpoint_stability_rule_met']) for r in records)
    primary_met = sum(bool(r['checkpoint_stability_rule_met']) for r in records if not r.get('maximum_budget_extension'))
    extended_met = stable_count - primary_met
    legend = ('Supplementary Fig. S18. Training, projection and active configuration alter deep-workflow interpretation. '
        'All panels use PBMC3k, Paul15 and heart analysis objects containing 2,638, 2,730 and 8,000 cells. '
        'Blue, green and purple identify these datasets. a, Genuine reconstruction-only SAUCIE losses divided by their update-300 value; '
        'each line is a fit. Solid curves use the primary control caps; dashed curves are separately labelled longer-budget fits. The vertical line marks 300 updates. b, Matched final population recovery for the default reconstruction branch, '
        'task-architecture reconstruction control and task-penalty numerical sensitivity. Lines connect the same dataset and seed. '
        'Enabling task penalties also changes activations; the middle control separates that architectural change. '
        'The task sensitivity explicitly enforces nonnegative finite squared distances before the original kernel and square root; '
        'unmodified task code produced nonfinite loss and is not silently replaced. c, Original scScope masked relative reconstruction errors; '
        'the vertical reference marks 300 epochs, and dashed curves denote longer-budget fits. d, Resolution-integrated macro-F1 at the original versus primary final budget for the same new fits; '
        'circles denote native scScope and squares reconstruction-only SAUCIE. The diagonal marks equality. '
        'e-g, Matched projection controls for saved 100-dimensional scScope predictions and their two-dimensional t-SNE and PCA projections, '
        'assessing PCA50 neighbour overlap, global distance-rank correlation and annotated-population macro-F1 separately. '
        'Thin lines connect individual seeds; points show medians and whiskers the full observed range. '
        'h, Neighbour overlap and distance-rank concordance between the last preterminal 25-epoch checkpoint and final native scScope output. Circles use the primary cap and triangles the longer budget. '
        'Primary controls use three computational seeds per dataset; e-g use five original seeds. The six longer-budget fits use seed zero only and are not additional independent seeds. '
        'Macro-F1 is the equal-label average of the harmonic mean of precision and recall after unsupervised Leiden clustering and post hoc majority mapping, '
        'integrated over the fixed resolution grid. Neighbour size is 15 and global scores use 5,000 fixed nonself pairs. '
        'Trustworthiness and additional population measurements are recorded in Supplementary Table S18. '
        'Primary caps are 10,000 SAUCIE updates and 600 scScope epochs; the loss-driven extension caps are 20,000 updates and 2,000 epochs. Early stopping requires three successive relative loss changes below 0.001 after the minimum budget. '
        f'The strict loss rule was met by {primary_met} of 36 primary fits and {extended_met} of six extensions. Unmet rules remain unmet; these plots do not establish optimum training or biological reliability. '
        'Seed ranges are computational variation, not confidence intervals; no biological significance tests are applied.')
    (FIG / 'Supplementary_Fig_S18_legend.json').write_text(json.dumps({'S18': {'legend':legend, 'title':legend.split('. ')[0]+'. '+legend.split('. ')[1]+'.',
        'n_fits':42, 'strict_rule_met':stable_count, 'training_settings': {'saucie_cap':10000,'scscope_cap':600,'extended_saucie_cap':20000,'extended_scscope_cap':2000}}},indent=2))
    manifest = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/'matched_endpoint_scores.csv',
                DATA/'matched_population_scores.csv', DATA/'checkpoint_representation_stability.csv']}
    (FIG / 'Supplementary_Fig_S18_sources.json').write_text(json.dumps({'source_hashes':manifest,'panels':list('abcdefgh'),
        'original_task_failure_preserved':True,'unmodified_task_not_scored':True},indent=2))
    print('S18 generated only after 36 primary controls, six labelled extensions and all required endpoints were verified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot completed deep-workflow configuration controls; no model fitting.')
    parser.add_argument('--source-dir',type=Path,default=CODE_ROOT/'data/source_data/deep_controls')
    parser.add_argument('--output-dir',type=Path,default=CODE_ROOT/'outputs/supplementary_figures')
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    DATA, FIG = args.source_dir.resolve(), args.output_dir.resolve()
    CHECK_ONLY = args.check_only
    if DATA == FIG or DATA.is_relative_to(FIG) or FIG.is_relative_to(DATA):
        parser.error('Source and output directories must not overlap.')
    main()
