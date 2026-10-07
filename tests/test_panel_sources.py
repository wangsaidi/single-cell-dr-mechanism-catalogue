"""Regression checks for donor definitions and minimum-condition provenance."""
from pathlib import Path
import json
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/source_data"


class PanelSourceTests(unittest.TestCase):
    def test_evidence_matrix_preserves_original_schema(self):
        matrix = pd.read_csv(ROOT / 'metadata/main_figure_claim_to_evidence_matrix.csv')
        self.assertEqual(len(matrix.columns), 11)
        self.assertIn('reviewer-risk control', matrix.columns)
        self.assertNotIn('reviewer risk control', matrix.columns)
        risk = matrix.loc[matrix.figure.eq('Fig6') & matrix.panel.eq('f'), 'reviewer-risk control']
        self.assertEqual(len(risk), 1)
        self.assertIn('Unequal evaluated condition pools', risk.iloc[0])

    def test_donor_table_uses_canonical_nonself_definitions(self):
        canonical = pd.read_csv(SOURCE / "fig4_heart_donor_gate_metrics.csv")
        canonical = canonical.pivot(index=["dataset_id", "method", "seed"], columns="metric", values="value")
        plotted = pd.read_csv(SOURCE / "fig5_heart_identity_donor_support.csv").set_index(["dataset_id", "method", "seed"])
        for column in ("cell_type_label_recall", "donor_entropy_norm"):
            pd.testing.assert_series_equal(plotted[column].sort_index(), canonical[column].sort_index(),
                                           check_names=False, rtol=1e-12, atol=1e-14)
        margins = pd.concat([(plotted.cell_type_label_recall - .55) / .55,
                             (plotted.donor_entropy_norm - .50) / .50], axis=1).min(axis=1)
        self.assertEqual(int((margins >= 0).sum()), 5)

    def test_minima_reproduce_from_documented_condition_pools(self):
        manifest = json.loads((SOURCE / "fig6_worst_case_condition_manifest.json").read_text())
        inputs = pd.concat([pd.read_csv(SOURCE / name) for name in manifest["inputs"]], ignore_index=True)
        summary = pd.read_csv(SOURCE / manifest["summary"])
        self.assertEqual(len(summary), 25)
        for row in summary.itertuples(index=False):
            values = inputs.loc[inputs.method.eq(row.method) & inputs.metric.eq(row.metric), "value"].dropna()
            self.assertEqual(len(values), row.n_conditions)
            self.assertAlmostEqual(float(values.min()), row.worst_value, places=12)
            self.assertEqual(manifest["methods"][row.method][row.metric], row.n_conditions)
            if row.metric == "latent_distance_corr":
                self.assertEqual(row.n_conditions, 1)
                self.assertEqual(row.threshold, .45)

    def test_auxiliary_simulation_is_separate_single_generated_dataset(self):
        design = json.loads((SOURCE / "fig6_known_truth_simulation_design.json").read_text())
        scores = pd.read_csv(SOURCE / "fig6_known_truth_simulation.csv")
        self.assertEqual(design["n_cells"], 1600)
        self.assertEqual(design["n_genes"], 600)
        self.assertEqual(sum(design["labels"].values()), 1600)
        self.assertEqual(sum(design["batches"].values()), 1600)
        self.assertEqual(set(scores.simulation_replicate), {0})
        self.assertEqual(len(scores), 25)


if __name__ == "__main__":
    unittest.main()
