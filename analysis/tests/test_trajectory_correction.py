"""Regression tests for finite-cell trajectory comparisons and seed-zero tests."""

import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.scripts import compute_trajectory as trajectory
from analysis.scripts import summarise_results as summaries


class TrajectoryCorrectionTests(unittest.TestCase):
    def test_nonfinite_values_remain_missing_without_replacing_valid_endpoints(self):
        original = np.array([0.0, 0.3, 1.0, np.inf, -np.inf, np.nan])
        result = trajectory.preserve_missing_pseudotime(original)
        np.testing.assert_array_equal(result[:3], original[:3])
        self.assertTrue(np.isnan(result[3:]).all())
        self.assertTrue(np.isinf(original[3]))

    def test_actual_disconnected_graph_has_no_endpoint_fill_or_order(self):
        rng = np.random.default_rng(42)
        coords = np.vstack([rng.normal(0, .01, (24, 2)), rng.normal(100, .01, (24, 2))])
        values, order, metadata = trajectory.run_dpt_roots(coords, {"root": 0})["root"]
        self.assertTrue(np.isfinite(values[:24]).all())
        self.assertTrue(np.isnan(values[24:]).all())
        self.assertTrue(np.isnan(order[24:]).all())
        self.assertEqual(metadata["n_connected_components"], 2)
        self.assertEqual(int(metadata["disconnected_mask"].sum()), 24)

    def test_common_intersection_is_identical_for_methods_with_different_missingness(self):
        reference = np.arange(6, dtype=float)
        first = np.array([0., 1., np.nan, 3., 4., 5.])
        second = np.array([0., 1., 2., 3., np.nan, 5.])
        common = trajectory.common_finite_mask(reference, {"a": first, "b": second}, "test")
        np.testing.assert_array_equal(common, [True, True, False, True, False, True])
        fields = trajectory.coverage_fields(first, common, {
            "disconnected_mask": np.isnan(first), "n_connected_components": 2,
        })
        self.assertEqual((fields["n_total"], fields["n_finite"], fields["n_common"]), (6, 5, 4))
        self.assertEqual(fields["n_disconnected"], 1)
        self.assertAlmostEqual(trajectory.safe_spearman(reference[common], first[common]), 1.)

    def test_zero_common_intersection_errors_instead_of_dropping_a_method(self):
        with self.assertRaisesRegex(ValueError, "n_common=0"):
            trajectory.common_finite_mask(np.arange(4.), {
                "a": np.array([0., 1., np.nan, np.nan]),
                "b": np.array([np.nan, np.nan, 2., 3.]),
            }, "disjoint")

    def test_nonfinite_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Non-finite reference"):
            trajectory.require_finite_reference(np.array([0., np.inf]), "root")

    def test_marker_pair_mask_excludes_infinite_and_missing_values(self):
        first = np.array([0., 1., np.nan, np.inf, 4.])
        second = np.array([4., 3., 2., 1., 0.])
        np.testing.assert_array_equal(trajectory.finite_pair_mask(first, second), [True, True, False, False, True])
        self.assertAlmostEqual(trajectory.safe_spearman(first, second), -1.)

    def test_duplicate_seed_zero_inputs_are_rejected(self):
        frame = pd.DataFrame({"method": ["PCA", "PCA"], "seed": [0, 0]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            summaries.matched_seed_zero(frame, "seed", ["method"])

    def test_all_seven_associations_use_only_seed_zero_including_marker_and_trajectory(self):
        methods = trajectory.METHODS
        geometry, marker, cluster, time_geometry, time_outcomes = [], [], [], [], []
        for seed in [0, 1]:
            for i, method in enumerate(methods):
                value = i + 1 if seed == 0 else 100 - i
                for dataset in ["pbmc3k", "heart_cell_atlas_subsampled"]:
                    for metric in ["local_retention", "label_neighbor_recall"]:
                        geometry.append(dict(dataset_id=dataset, method=method, metric=metric,
                                             output_dimension=2, seed=seed, value=value))
                    cluster.append(dict(dataset_id=dataset, method=method, embedding_seed=seed,
                                        ari_resolution_auc=value / 10, macro_f1_resolution_auc=value / 20))
                    marker.append(dict(dataset_id=dataset, method=method, embedding_seed=seed,
                                       cluster_weighted_marker_concordance_resolution_auc=value / 30))
                time_geometry.append(dict(method=method, seed=seed, metric="pseudotime_rank_corr", value=value / 40))
                time_outcomes.append(dict(method=method, embedding_seed=seed, n_common=40,
                                          n_comparison_methods=9, reference_pseudotime_spearman_common=value / 50))
        inputs = {
            "geometry_metrics_all_methods.csv": pd.DataFrame(geometry),
            "trajectory_geometry_metrics_all_methods.csv": pd.DataFrame(time_geometry),
            "trajectory_outcomes.csv": pd.DataFrame(time_outcomes),
        }
        written = {}

        def capture(frame, path, **kwargs):
            written[Path(path).name] = frame.copy()

        with patch.object(summaries.pd, "read_csv", side_effect=lambda p: inputs[Path(p).name].copy()), \
             patch.object(pd.DataFrame, "to_csv", capture), \
             patch.object(summaries, "exact_spearman_permutation", return_value=(.5, .1, 362880)):
            summaries.diagnostic_outcome_links(pd.DataFrame(cluster), pd.DataFrame(marker), pd.DataFrame())
        links = written["diagnostic_outcome_associations.csv"]
        scatter = written["diagnostic_outcome_scatter_data.csv"]
        self.assertEqual(len(links), 7)
        self.assertEqual(len(scatter), 63)
        self.assertTrue(links[["embedding_seed", "diagnostic_seed", "outcome_seed"]].eq(0).all().all())
        np.testing.assert_allclose(links.q_bh_7_planned_tests, .1)
        value_scales = {
            "local_retention_vs_ari": (1, 10),
            "label_neighbour_vs_macro_f1": (1, 20),
            "marker_concordance_vs_macro_f1": (30, 20),
            "trajectory_geometry_vs_dpt_order": (40, 50),
        }
        for row in scatter.itertuples(index=False):
            diagnostic_scale, outcome_scale = value_scales[row.comparison]
            seed_zero_value = methods.index(row.method) + 1
            self.assertAlmostEqual(row.diagnostic_value, seed_zero_value / diagnostic_scale)
            self.assertAlmostEqual(row.outcome_value, seed_zero_value / outcome_scale)
        paul = scatter[scatter.dataset_id.eq("paul15")].set_index("method")
        for i, method in enumerate(methods):
            self.assertAlmostEqual(paul.loc[method, "diagnostic_value"], (i + 1) / 40)
            self.assertAlmostEqual(paul.loc[method, "outcome_value"], (i + 1) / 50)
        self.assertTrue(paul.outcome.eq("dpt_spearman_seed0_common").all())


if __name__ == "__main__":
    unittest.main()
