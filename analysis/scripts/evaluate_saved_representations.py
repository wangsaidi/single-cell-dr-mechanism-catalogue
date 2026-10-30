"""Score saved representations with the published downstream routines; no model fitting."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import itertools
import json
import os
import sys
from datetime import datetime, timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--controls', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'verified_summary.json').exists():
        saved = json.loads((out / 'verified_summary.json').read_text())
        for name, digest in saved['result_sha256'].items():
            if sha(out / name) != digest:
                raise ValueError('Completed rescoring output changed: ' + name)
        print('Completed saved-array sensitivity already verified; no rerun.', flush=True)
        return
    os.environ.update(SCDR_ANALYSIS_DATA=str(args.data.resolve()),
        SCDR_ANALYSIS_RESULTS=str(out / 'unused_analysis_results'),
        SCDR_ANALYSIS_LOGS=str(out / 'unused_analysis_logs'),
        NUMBA_CACHE_DIR=str(out / 'cache/numba'), MPLCONFIGDIR=str(out / 'cache/matplotlib'),
        OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', NUMBA_NUM_THREADS='2')
    os.environ.pop('SCDR_EXECUTION_SCALE', None)
    sys.path.insert(0, str(args.repo.resolve()))
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scanpy as sc
    from scipy.stats import spearmanr
    from analysis.scripts import compute_clustering as cluster
    from analysis.scripts import compute_diagnostics as geometry
    from analysis.scripts import compute_marker_concordance as marker
    from analysis.scripts import compute_trajectory as trajectory
    from analysis.scripts import summarise_results as summary
    methods = cluster.METHODS
    resolutions = [.2, .4, .6, .8, 1., 1.2, 1.5, 2.]
    plan = {'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'methods': methods, 'control_seeds': [0, 1, 2], 'other_seven_seed': 0,
        'replaced_methods': ['scScope', 'SAUCIE'], 'datasets': cluster.DATASETS,
        'scScope': 'saved t-SNE2 of scheduled epoch-2500 native output',
        'SAUCIE': 'saved reconstruction-only step-30000 two-dimensional output',
        'model_fitting': False, 'graph_k': 15, 'graph_seed': 0, 'resolutions': resolutions,
        'exact_test': 'two-sided Spearman whole-method-label permutations, identity included',
        'multiplicity': 'seven original comparisons per conditional panel, BH within panel',
        'trajectory': '7MEP root, common finite cells across original and replacement outputs; original-only population separately retained',
        'scope': 'Post hoc configuration and re-execution sensitivity; three replacements share seven fixed anchors, not independent biological or algorithm samples.'}
    registration = out / 'scoring_plan.json'
    if not registration.exists():
        registration.write_text(json.dumps(plan, indent=2))
    def embedding(dataset, method, seed=None):
        if seed is None:
            if method == 'scVI':
                return args.data / 'reference_results/embeddings' / (dataset + '_scvi_dim2_seed0.npy')
            return args.data / 'anchor_embeddings' / (dataset + '_' + cluster.stem(method) + '_seed0.npy')
        short = {'pbmc3k': 'pb', 'paul15': 'pa', 'heart_cell_atlas_subsampled': 'ht'}[dataset]
        if method == 'scScope':
            return args.controls / ('%s_sc_%d' % (short, seed)) / 'final_tSNE2.npy'
        if method == 'SAUCIE':
            return args.controls / ('%s_rec_%d' % (short, seed)) / 'embedding_final.npy'
        raise ValueError('Only declared deep configurations are replaced.')
    source_rows, geometric_rows, cluster_rows, population_rows, marker_rows = [], [], [], [], []
    coords_by_dataset, proc_by_dataset, scored = {}, {}, {}
    for dataset in cluster.DATASETS:
        proc_path = args.data / 'analysis_objects' / (dataset + '_proc.h5ad')
        proc = ad.read_h5ad(proc_path)
        field = {'pbmc3k': 'louvain', 'paul15': 'paul15_clusters', 'heart_cell_atlas_subsampled': 'cell_type'}[dataset]
        labels = proc.obs[field].astype(str).to_numpy()
        ids = proc.obs_names.astype(str).to_numpy()
        high = np.asarray(proc.obsm['X_pca_ref'][:, :50], dtype=np.float32)
        reference_knn = geometry.knn_idx(high)
        proc_by_dataset[dataset] = proc
        source_rows.append({'role': 'analysis_object', 'dataset_id': dataset, 'file': proc_path.name,
            'sha256': sha(proc_path), 'n_cells': len(ids),
            'ordered_cell_ids_sha256': hashlib.sha256('\n'.join(ids).encode()).hexdigest()})
        if dataset != 'paul15':
            expression, marker_sets, expected = marker.load_expression(dataset, ids.tolist())
            marker_scores, _ = marker.module_scores(expression, marker_sets)
        candidates = [(m, None) for m in methods] + [(m, s) for s in range(3) for m in ('scScope', 'SAUCIE')]
        arrays = {}
        for method, seed in candidates:
            name = method + ('__original' if seed is None else '__scheduled%d' % seed)
            path = embedding(dataset, method, seed)
            coords = np.load(path, allow_pickle=False).astype(np.float32)
            if coords.shape != (len(ids), 2) or not np.isfinite(coords).all():
                raise ValueError('Nonfinite or misaligned representation: ' + name)
            arrays[name] = coords
            source_rows.append({'role': 'representation', 'dataset_id': dataset, 'variant': name,
                'file': path.name, 'sha256': sha(path), 'n_cells': len(ids), 'n_dimensions': 2})
            work_knn = geometry.knn_idx(coords)
            row = {'dataset_id': dataset, 'method': method, 'variant': name,
                'control_seed': seed if seed is not None else -1, 'n_cells': len(ids),
                'local_retention': float(np.mean([len(set(a) & set(b)) / 15 for a, b in zip(reference_knn, work_knn)])),
                'label_neighbor_recall': geometry.label_neighbour_recall(coords, labels),
                'global_rank_corr': geometry.global_rank(high, coords, 0)}
            geometric_rows.append(row)
            scored[(dataset, name)] = dict(row)
            if dataset != 'paul15':
                work = cluster.build_graph(coords, 15)
                metrics, pops, concordances = [], [], []
                for resolution in resolutions:
                    sc.tl.leiden(work, resolution=resolution, key_added='partition', random_state=0,
                        flavor='igraph', directed=False, n_iterations=-1)
                    groups = work.obs['partition'].astype(str).to_numpy()
                    glob, per_label = cluster.evaluate_partition(dataset, method,
                        seed if seed is not None else 0, 15, resolution, groups, labels)
                    glob.update(variant=name, control_seed=seed if seed is not None else -1)
                    cluster_rows.append(glob)
                    metrics.append(glob)
                    for pop in per_label:
                        pop.update(variant=name, control_seed=seed if seed is not None else -1)
                        population_rows.append(pop)
                    frame = pd.DataFrame({'leiden_cluster': groups, 'reference_label': labels})
                    valid_weight = 0
                    for group in frame.leiden_cluster.unique():
                        inside = groups == group
                        majority = frame.loc[inside, 'reference_label'].value_counts().idxmax()
                        effects = {program: marker.standardised_mean_difference(
                            marker_scores.loc[inside, program].to_numpy(dtype=float),
                            marker_scores.loc[~inside, program].to_numpy(dtype=float)) for program in marker_sets}
                        ordered = sorted(effects, key=lambda p: (-np.nan_to_num(effects[p], nan=-np.inf), p))
                        match = ordered[0] in expected.get(str(majority), [])
                        valid_weight += int(inside.sum()) * int(match)
                    concordance = float(valid_weight / len(ids))
                    concordances.append({'resolution': resolution, 'value': concordance})
                    marker_rows.append({'dataset_id': dataset, 'method': method, 'variant': name,
                        'resolution': resolution, 'cluster_weighted_marker_concordance': concordance})
                values = pd.DataFrame(metrics)
                scored[(dataset, name)].update(
                    ari_resolution_auc=cluster.resolution_auc(values, 'adjusted_rand_index'),
                    macro_f1_resolution_auc=cluster.resolution_auc(values, 'macro_f1_after_cluster_majority_mapping'),
                    cluster_weighted_marker_concordance_resolution_auc=summary.normalized_auc(
                        pd.DataFrame(concordances), 'resolution', 'value'))
            print('SCORED', dataset, name, flush=True)
        coords_by_dataset[dataset] = arrays
    # Hold the cell mask fixed when representations are substituted.
    proc = proc_by_dataset['paul15']
    root = trajectory.root_definitions(proc)[trajectory.PRIMARY_ROOT]
    reference = trajectory.reference_dpt(proc, root)
    pseudotimes = {}
    for name, coords in coords_by_dataset['paul15'].items():
        values, _ = trajectory.run_dpt(coords, root)
        pseudotimes[name] = values
        scored[('paul15', name)]['pseudotime_rank_corr'] = geometry.one_dimensional_rank(reference, coords, 0)
    original_times = {name: v for name, v in pseudotimes.items() if name.endswith('__original')}
    original_mask = trajectory.common_finite_mask(reference, original_times, 'original panel')
    shared_mask = trajectory.common_finite_mask(reference, pseudotimes, 'original and replacement panels')
    trajectory_rows = []
    for name, values in pseudotimes.items():
        method = name.split('__')[0]
        rho = trajectory.safe_spearman(reference[shared_mask], values[shared_mask])
        scored[('paul15', name)]['dpt_spearman_seed0_common'] = rho
        trajectory_rows.append({'method': method, 'variant': name,
            'dpt_spearman_joint_common': rho, 'n_joint_common': int(shared_mask.sum()),
            'n_finite': int(np.isfinite(values).sum()), 'n_total': len(values),
            'dpt_spearman_original_common': trajectory.safe_spearman(reference[original_mask], values[original_mask]),
            'n_original_common': int(original_mask.sum())})
    ids = proc.obs_names.astype(str).to_numpy()
    pd.DataFrame({'cell_id': ids, 'original_common': original_mask, 'joint_common': shared_mask}).to_csv(
        out / 'trajectory_comparison_population.csv', index=False)
    pd.DataFrame({'cell_id': ids, 'reference_pseudotime': reference, **pseudotimes}).to_csv(
        out / 'trajectory_cell_pseudotimes.csv', index=False)
    definitions = [('local_retention_vs_ari', 'local_retention', 'ari_resolution_auc', d)
        for d in ('pbmc3k', 'heart_cell_atlas_subsampled')]
    definitions += [('label_neighbour_vs_macro_f1', 'label_neighbor_recall', 'macro_f1_resolution_auc', d)
        for d in ('pbmc3k', 'heart_cell_atlas_subsampled')]
    definitions += [('marker_concordance_vs_macro_f1', 'cluster_weighted_marker_concordance_resolution_auc', 'macro_f1_resolution_auc', d)
        for d in ('pbmc3k', 'heart_cell_atlas_subsampled')]
    definitions += [('trajectory_geometry_vs_dpt_order', 'pseudotime_rank_corr', 'dpt_spearman_seed0_common', 'paul15')]
    association_rows, scatter_rows, panel_rows = [], [], []
    for panel in ('original', 'scheduled0', 'scheduled1', 'scheduled2'):
        estimates = []
        for comparison, x, y, dataset in definitions:
            rows = []
            for method in methods:
                variant = method + '__' + (panel if panel != 'original' and method in ('scScope', 'SAUCIE') else 'original')
                value = scored[(dataset, variant)]
                rows.append(value)
                scatter_rows.append({'panel': panel, 'dataset_id': dataset, 'method': method,
                    'variant': variant, 'comparison': comparison, 'diagnostic': x, 'outcome': y,
                    'diagnostic_value': value[x], 'outcome_value': value[y]})
            data = pd.DataFrame(rows)
            summary.validate_association_panel(data, [x, y])
            rho, p, permutations = summary.exact_spearman_permutation(data[x].to_numpy(), data[y].to_numpy())
            estimates.append({'panel': panel, 'dataset_id': dataset, 'comparison': comparison,
                'spearman_rho': rho, 'exact_two_sided_p': p, 'n_methods': 9,
                'n_permutations': permutations,
                'n_trajectory_joint_common': int(shared_mask.sum()) if dataset == 'paul15' else None})
        qs = summary.benjamini_hochberg(np.asarray([r['exact_two_sided_p'] for r in estimates]))
        for row, q in zip(estimates, qs):
            row.update(q_bh_seven=float(q), q_below_0_05=bool(q < .05))
        association_rows.extend(estimates)
        print('TESTED', panel, flush=True)
    frames = {
        'input_inventory.csv': pd.DataFrame(source_rows),
        'geometry_metrics.csv': pd.DataFrame(geometric_rows),
        'clustering_by_resolution.csv': pd.DataFrame(cluster_rows),
        'population_by_resolution.csv': pd.DataFrame(population_rows),
        'marker_by_resolution.csv': pd.DataFrame(marker_rows),
        'trajectory_summary.csv': pd.DataFrame(trajectory_rows),
        'representation_endpoints.csv': pd.DataFrame(scored.values()),
        'association_results.csv': pd.DataFrame(association_rows),
        'association_inputs.csv': pd.DataFrame(scatter_rows)}
    pop = frames['population_by_resolution.csv']
    population_summary = []
    for keys, frame in pop.groupby(['dataset_id', 'method', 'variant', 'label', 'n_cells_label'], sort=False):
        row = dict(zip(['dataset_id', 'method', 'variant', 'label', 'n_cells_label'], keys))
        row.update({metric + '_resolution_auc': summary.normalized_auc(frame, 'resolution', metric)
            for metric in ('precision', 'recall', 'f1')})
        population_summary.append(row)
    frames['population_summary.csv'] = pd.DataFrame(population_summary)
    for name, frame in frames.items():
        frame.to_csv(out / name, index=False)
    files = list(frames) + ['trajectory_comparison_population.csv', 'trajectory_cell_pseudotimes.csv']
    receipt = {'completed': True, 'model_fitting_performed': False, 'new_main_figures': 0,
        'number_of_representations_scored': len(geometric_rows),
        'conditional_panels': 4, 'association_tests': len(association_rows),
        'original_trajectory_common_cells': int(original_mask.sum()),
        'joint_trajectory_common_cells': int(shared_mask.sum()),
        'n_cells': {d: int(p.n_obs) for d, p in proc_by_dataset.items()},
        'scoring_scope': plan['scope'],
        'modules': {m.__name__: sha(m.__file__) for m in (cluster, geometry, marker, trajectory, summary)},
        'software': {p: importlib.metadata.version(p) for p in ('numpy', 'pandas', 'scipy', 'scanpy', 'anndata', 'igraph', 'scikit-learn')},
        'result_sha256': {name: sha(out / name) for name in files}}
    (out / 'verified_summary.json').write_text(json.dumps(receipt, indent=2, allow_nan=False))
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
