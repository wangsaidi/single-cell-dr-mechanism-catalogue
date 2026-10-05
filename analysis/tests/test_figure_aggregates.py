"""Regression checks for fitted-subset figure sources and aggregate provenance."""

from __future__ import annotations

import os
from pathlib import Path
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from figures.plot_supplementary_figure_s1 import ROOT, SOURCE_DIR, S1_DATASETS, derive_s1_sources
from figures.plot_supplementary_figures_s3_s6 import (
    FAILURE_SPECS, REV_SOURCE_DIR, derive_failure_summaries,
    failure_summary, validate_failure_summary,
)
from figures.plot_supplementary_figures_s15_s16 import METRICS, SOURCE, geometry_xlim


class FittedSubsetSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        default = ROOT / "analysis/data/analysis_objects"
        smoke = ROOT.parent / "public_release_smoke_20260930/analysis/data/analysis_objects"
        cls.objects = Path(os.environ.get("SCDR_S1_ANALYSIS_OBJECTS", smoke if smoke.exists() else default))
        cls.tables, cls.provenance = derive_s1_sources(cls.objects)

    def test_all_ten_persisted_tables_match_exact_objects(self):
        self.assertEqual(len(self.tables), 10)
        for name, expected in self.tables.items():
            with self.subTest(name=name):
                actual = pd.read_csv(SOURCE_DIR / name)
                pd.testing.assert_frame_equal(actual, expected.replace("", np.nan),
                    check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-14)

    def test_subset_sizes_and_aligned_fitted_counts(self):
        summary = self.tables["supp_s1_dataset_summary.csv"].set_index("dataset_id")
        for dataset, (cells, features, _, _) in S1_DATASETS.items():
            self.assertEqual((summary.loc[dataset, "n_obs"], summary.loc[dataset, "n_vars"]), (cells, features))
            self.assertTrue(self.provenance[dataset]["counts_layer_equal"])
            self.assertTrue(self.provenance[dataset]["cell_feature_order_equal"])

    def test_composition_denominators_are_fitted_cells(self):
        composition = self.tables["supp_s1_dataset_composition.csv"]
        for (dataset, role), frame in composition.groupby(["dataset_id", "field_role"]):
            self.assertEqual(frame.n_cells.sum(), S1_DATASETS[dataset][0], (dataset, role))
            self.assertAlmostEqual(frame.fraction.sum(), 1)
        heart = composition[composition.dataset_id.eq("heart_cell_atlas_subsampled")]
        self.assertEqual(len(heart[heart.field_role.eq("batch_or_donor")]), 14)
        rare = self.tables["supp_s1_rare_label_burden.csv"].set_index("dataset_id")
        self.assertEqual(rare.loc["heart_cell_atlas_subsampled", "n_rare_labels"], 3)
        self.assertAlmostEqual(rare.loc["heart_cell_atlas_subsampled", "rare_cell_fraction"], 315 / 8000)

    def test_count_depth_and_sparsity_use_identical_selected_features(self):
        complexity = self.tables["supp_s1_cell_complexity_metrics.csv"].set_index("dataset_id")
        expression = self.tables["supp_s1_expression_sparsity_metrics.csv"].set_index("dataset_id")
        depth = self.tables["supp_s1_total_count_depth.csv"].set_index("dataset_id")
        for dataset, (_, features, _, _) in S1_DATASETS.items():
            row = complexity.loc[dataset]
            self.assertAlmostEqual(row.zero_fraction, 1 - row.mean_detected_genes_per_cell / features)
            self.assertEqual(row.mean_total_counts_per_cell, expression.loc[dataset, "mean_total_counts_per_cell"])
            self.assertEqual(row.mean_total_counts_per_cell, depth.loc[dataset, "mean_total_counts_per_cell"])
            self.assertIn("not full-library totals", depth.loc[dataset, "source_definition"])

    def test_unreached_variance_thresholds_are_not_fabricated_as_50_pcs(self):
        pca = self.tables["supp_s1_pca_variance_structure.csv"]
        self.assertTrue(pca.n_saved_pcs.eq(50).all())
        self.assertTrue(pca.saved_pcs_cumulative_variance.lt(.5).all())
        self.assertTrue(pca[["n_pcs_for_50pct", "n_pcs_for_80pct"]].isna().all().all())
        self.assertTrue(pca.source_definition.str.contains("not reached").all())

    def test_pc20_separability_is_defined_and_finite(self):
        frame = self.tables["supp_s1_pc_label_separability.csv"]
        self.assertTrue(np.isfinite(frame[["pc_label_neighbor_recall", "pc20_label_silhouette"]]).all().all())
        self.assertTrue(frame.source_definition.str.contains("first 20 saved").all())
        self.assertTrue(frame.source_definition.str.contains("full-cell silhouette").all())


class RobustnessAggregateTests(unittest.TestCase):
    def test_canonical_flags_persisted_summaries_and_main_fig6_agree(self):
        tables = derive_failure_summaries()
        for name, summary in tables.items():
            source, groups, main_source, dimension = FAILURE_SPECS[name]
            rows = pd.read_csv(SOURCE_DIR / source)
            validate_failure_summary(pd.read_csv(REV_SOURCE_DIR / main_source), summary, groups)
            expected_n = 20 if "dataset" in name else 9 if "upstream" in name else 15
            self.assertTrue(summary.n_rows.eq(expected_n).all())
            self.assertEqual(int(np.round((summary.failure_fraction * summary.n_rows).sum())),
                int(rows.support.eq("below_threshold").sum()))
            self.assertEqual(len(rows), 144 if "upstream" in name else 240)

    def test_stale_aggregate_rejected(self):
        frame = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
        expected = failure_summary(frame, ["method", "metric"], "output_dimension")
        stale = expected.copy()
        stale.loc[0, "failure_fraction"] += .01
        with self.assertRaisesRegex(ValueError, "Stale"):
            validate_failure_summary(stale, expected, ["method", "metric"])

    def test_invalid_support_flag_and_boundary_disagreement_rejected(self):
        frame = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
        for flag in ["unknown", "below_threshold" if frame.loc[0, "support"] == "pass" else "pass"]:
            invalid = frame.copy()
            invalid.loc[0, "support"] = flag
            with self.assertRaises(ValueError):
                failure_summary(invalid, ["method", "metric"], "output_dimension")

    def test_duplicate_or_nonfinite_experiment_rejected(self):
        frame = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
        with self.assertRaisesRegex(ValueError, "duplicated"):
            failure_summary(pd.concat([frame, frame.iloc[:1]]), ["method", "metric"], "output_dimension")
        frame.loc[0, "value"] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            failure_summary(frame, ["method", "metric"], "output_dimension")

    def test_equality_to_boundary_is_pass(self):
        frame = pd.read_csv(SOURCE_DIR / "fig6_output_dimension_response.csv")
        frame.loc[0, "value"] = frame.loc[0, "threshold"]
        frame.loc[0, "support"] = "pass"
        failure_summary(frame, ["method", "metric"], "output_dimension")


class FigureTraceabilityTests(unittest.TestCase):
    def test_current_fig4_and_retired_s14_have_separate_responsibilities(self):
        matrix = pd.read_csv(ROOT / "metadata/main_figure_claim_to_evidence_matrix.csv")
        self.assertFalse(matrix.duplicated(["figure", "panel"]).any())
        current = matrix[matrix.figure.eq("Fig4")].set_index("panel")
        self.assertEqual(set(current.index), set("abcdef"))
        self.assertEqual(set(matrix[matrix.figure.eq("SuppS14")].panel), set("abcdefghij"))
        for panel, row in current.iterrows():
            self.assertIn("scVI", row["reviewer-risk control"])
            self.assertIn("9", row["n definition"])
            self.assertTrue((SOURCE_DIR / f"figure4_downstream/Figure4_panel_{panel}.csv").exists())
        self.assertIn("within each embedding seed", current.loc["d", "metric or statistical definition"])
        self.assertIn("without imputation", current.loc["d", "missing and exclusion rule"])
        self.assertIn("seven", current.loc["f", "metric or statistical definition"])

    def test_s15_axis_contains_full_null_intervals_and_observations(self):
        frame = pd.read_csv(SOURCE / "calibration_geometry.csv")
        for metric in METRICS:
            lower, upper = geometry_xlim(frame, metric)
            values = frame[frame.metric.eq(metric)][["null_q025", "null_q975", "null_mean", "observed"]]
            self.assertLess(lower, values.min().min())
            self.assertGreater(upper, values.max().max())
        self.assertLess(geometry_xlim(frame, METRICS[2])[0], -.0540061129138)

    def test_s15_bounds_follow_more_extreme_nulls_and_reject_nonfinite(self):
        frame = pd.read_csv(SOURCE / "calibration_geometry.csv")
        index = frame.index[frame.metric.eq(METRICS[2])][0]
        frame.loc[index, "null_q025"] = -.9
        self.assertLess(geometry_xlim(frame, METRICS[2])[0], -.9)
        frame.loc[index, "null_q025"] = np.nan
        with self.assertRaises(ValueError):
            geometry_xlim(frame, METRICS[2])


if __name__ == "__main__":
    unittest.main()
