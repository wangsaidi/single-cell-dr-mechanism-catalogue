"""Small synthetic validator fixtures; no scientific imports or private inputs."""

from contextlib import redirect_stdout
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from analysis.workflow import validate_refit as validator


class ValidationMetadataTests(unittest.TestCase):
    def metadata(self, root, smoke, errors):
        with patch.multiple(validator, LOGS_DIR=root, RESULTS_DIR=root / "results",
                            SOURCE_RESULTS_DIR=root / "source", SMOKE=smoke), \
                patch.dict(os.environ, {"SCDR_WORKFLOW_PROFILE": "full-refit"}), \
                patch.object(validator, "validate", return_value=errors) as validate, \
                redirect_stdout(io.StringIO()):
            if errors:
                with self.assertRaises(SystemExit) as raised:
                    validator.main()
                self.assertEqual(raised.exception.code, 1)
            else:
                validator.main()
        validate.assert_called_once_with(root, root / "results", root / "source", "full-refit")
        return json.loads((root / "refit_validation.json").read_text(encoding="utf-8"))

    def test_smoke_profile_is_not_the_full_refit_validation_alias(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.metadata(Path(temporary), True, [])
        self.assertEqual(record["profile"], "smoke")
        self.assertEqual(record["validation_mode"], "full-refit")
        self.assertEqual(record["execution_scale"], "smoke_not_publication")
        self.assertTrue(record["passed"])
        self.assertFalse(record["historical_binary_hash_comparison"])
        self.assertFalse(record["stochastic_agreement_or_convergence_established"])

    def test_full_refit_profile_and_validation_mode_remain_full_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.metadata(Path(temporary), False, [])
        self.assertEqual(record["profile"], "full-refit")
        self.assertEqual(record["validation_mode"], "full-refit")
        self.assertEqual(record["execution_scale"], "publication")
        self.assertFalse(record["stochastic_agreement_or_convergence_established"])

    def test_smoke_failure_is_written_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as temporary:
            record = self.metadata(Path(temporary), True, ["missing fit"])
        self.assertFalse(record["passed"])
        self.assertEqual(record["errors"], ["missing fit"])
        self.assertEqual(record["profile"], "smoke")

    def test_smoke_still_requires_complete_eight_method_six_scenario_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            logs, results, source = root / "logs", root / "results", root / "source"
            for directory in (logs, source, results / "calibration", results / "figure4_downstream"):
                directory.mkdir(parents=True)
            for name in ("input_audit_summary.json", "scvi_output_audit_summary.json", "revision_result_audit.json"):
                (logs / name).write_text(json.dumps({"audit_passed": True}), encoding="ascii")
            for seed in (0, 1):
                (logs / f"anchor_embedding_failures_seed{seed}.json").write_text("[]", encoding="ascii")
                (source / f"fig3_family_local_label_metrics_seed{seed}.csv").write_text("fixture", encoding="ascii")
            for name in ("fig6_output_dimension_response.csv", "fig6_upstream_pca_response.csv",
                         "fig6_dropout_noise_response.csv", "fig6_known_truth_simulation.csv"):
                (source / name).write_text("support\nfixture\n", encoding="ascii")
            (results / "calibration/metadata.json").write_text("{}", encoding="ascii")
            for index in range(6):
                (results / f"figure4_downstream/panel{index}.csv").write_text("fixture", encoding="ascii")
            methods = ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"]
            scenarios = ["linear_low_rank", "nonlinear_manifold", "branching_trajectory",
                         "dropout_stress", "batch_shift", "rare_population"]
            grid = [(method, scenario, 0) for method in methods for scenario in scenarios]
            table = source / "fig6_mechanism_simulation_suite.csv"

            def write_grid(rows):
                with table.open("w", newline="", encoding="ascii") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["method", "scenario", "replicate"])
                    writer.writerows(rows)

            with patch.multiple(validator, SMOKE=True, SEEDS=[0, 1]):
                write_grid(grid)
                self.assertEqual(validator.validate(logs, results, source, "full-refit"), [])
                write_grid(grid[:-1])
                self.assertTrue(any("coverage mismatch" in error for error in
                                    validator.validate(logs, results, source, "full-refit")))
                write_grid(grid)
                (logs / "anchor_embedding_failures_seed1.json").write_text('["failed"]', encoding="ascii")
                self.assertTrue(any("missing/failed anchor fits" in error for error in
                                    validator.validate(logs, results, source, "full-refit")))


if __name__ == "__main__":
    unittest.main()
