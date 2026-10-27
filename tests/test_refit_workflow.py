"""Standard-library orchestration tests; no dataset download or model training."""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from analysis import legacy_runtime
from analysis import paths
from analysis.workflow import run_analysis as workflow
from analysis.workflow.stage_runner import InputAudit
from analysis.workflow.validate_refit import validate


class RefitWorkflowTests(unittest.TestCase):
    def args(self, root, profile="full-refit"):
        return argparse.Namespace(run_dir=root / "run", archive_data=root / "archive-data",
                                  archive_results=root / "archive-results", profile=profile, resume=False)

    def test_refit_forces_all_fitting_stages_without_archived_verification(self):
        steps = dict(workflow.full_refit_steps())
        for name, arguments in steps.items():
            if name.startswith("anchor_embeddings") or name in {"known_truth_stress_tests", "robustness_embeddings", "fit_scvi"}:
                self.assertIn("--force", arguments, name)
        self.assertNotIn("verify_inputs", steps)
        self.assertNotIn("verify_results", steps)
        self.assertNotIn("verify_calibration", steps)
        self.assertIn("validate_refit", steps)
        self.assertLess(list(steps).index("known_truth_stress_tests"), list(steps).index("robustness_embeddings"))

    def test_smoke_reduces_budgets_but_retains_all_stage_types(self):
        full = dict(workflow.full_refit_steps())
        smoke = dict(workflow.full_refit_steps(smoke=True))
        self.assertEqual([name for name in smoke if name.startswith("anchor_embeddings")],
                         ["anchor_embeddings_seed_0", "anchor_embeddings_seed_1"])
        self.assertEqual({name for name in full if not name.startswith("anchor_embeddings")},
                         {name for name in smoke if not name.startswith("anchor_embeddings")})
        self.assertEqual(smoke["threshold_calibration"][-2:], ["--permutations", "19"])

    def test_mocked_completed_smoke_is_not_a_publication_scale_full_refit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "smoke"
            argv = ["run_analysis", "--profile", "smoke", "--run-dir", str(root)]
            with patch.object(sys, "argv", argv), patch.object(workflow, "execute_steps"):
                workflow.main()
            metadata = json.loads((root / "run.json").read_text())
            self.assertTrue(metadata["complete_workflow"])
            self.assertFalse(metadata["publication_scale_full_refit_completed"])
            self.assertEqual(metadata["execution_scale"], "smoke_not_publication")

    def test_archived_contains_no_fitting_stages(self):
        names = dict(workflow.ARCHIVED_STEPS)
        self.assertIn("verify_inputs", names)
        self.assertIn("verify_results", names)
        self.assertFalse(any("--force" in args for args in names.values()))
        self.assertFalse(any(name.startswith("anchor_embeddings") or name == "fit_scvi" for name in names))

    def test_isolated_paths_ignore_inherited_cache_overrides_and_reject_nonempty(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self.args(root)
            with patch.dict(os.environ, {"SCDR_ANALYSIS_DATA": "wrong-cache"}):
                env, run, metadata = workflow.prepare_run(args)
            self.assertEqual(Path(env["SCDR_ANALYSIS_DATA"]), run / "data")
            self.assertEqual(Path(env["SCDR_ANALYSIS_RESULTS"]), run / "results")
            self.assertTrue(metadata["initially_empty"])
            with self.assertRaisesRegex(ValueError, "not empty"):
                workflow.prepare_run(args)
            args.resume = True
            workflow.prepare_run(args)
            args.profile = "archived"
            with self.assertRaisesRegex(ValueError, "must match"):
                workflow.prepare_run(args)

    def test_overlap_with_archive_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary))
            args.run_dir = args.archive_data / "child"
            with self.assertRaisesRegex(ValueError, "overlap"):
                workflow.prepare_run(args)

    def test_archived_paths_leave_input_data_and_scvi_outside_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.args(Path(temporary), "archived")
            env, run, _ = workflow.prepare_run(args)
            self.assertEqual(Path(env["SCDR_ANALYSIS_DATA"]), args.archive_data)
            self.assertEqual(Path(env["SCDR_SCVI_EMBEDDINGS"]), args.archive_results / "embeddings")
            self.assertFalse((run / "data").exists())

    def test_mocked_stage_failure_is_logged_and_stops_remaining_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env, run, _ = workflow.prepare_run(self.args(root))
            with patch.object(workflow.subprocess, "run", side_effect=[subprocess.CompletedProcess([], 0),
                    subprocess.CalledProcessError(7, ["mock"]), None]) as runner:
                with self.assertRaises(subprocess.CalledProcessError):
                    workflow.execute_steps([("a", ["-m", "a"]), ("b", ["-m", "b"]), ("c", ["-m", "c"])], env, run)
            events = [json.loads(line) for line in (run / "logs/stages.jsonl").read_text().splitlines()]
            self.assertEqual([row["event"] for row in events], ["start", "finish", "start", "failure"])
            self.assertEqual(events[-1]["returncode"], 7)
            self.assertEqual(runner.call_count, 2)

    def test_lightweight_real_child_cli_logs_success_and_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            env, run, _ = workflow.prepare_run(self.args(Path(temporary)))
            workflow.execute_steps([("help_only", ["-m", "unittest", "--help"])], env, run)
            self.assertTrue((run / "logs/help_only.input_reads.json").is_file())
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.execute_steps([("expected_failure", ["-m", "analysis.workflow.validate_refit"])], env, run)
            self.assertTrue((run / "logs/expected_failure.input_reads.json").is_file())
            self.assertIn('"event": "failure"', (run / "logs/stages.jsonl").read_text())

    def test_cli_rejects_force_on_archived_and_invalid_step(self):
        for arguments in (["--force"], ["--through-step", "typo"]):
            result = subprocess.run([sys.executable, "-m", "analysis.workflow.run_analysis", "--run-dir", "unused", *arguments],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)

    def test_cli_partial_run_is_not_certified_as_full_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            argv = ["run_analysis", "--profile", "full-refit", "--run-dir", str(root), "--through-step", "record_environment"]
            with patch.object(sys, "argv", argv), patch.object(workflow, "execute_steps"):
                workflow.main()
            metadata = json.loads((root / "run.json").read_text())
            self.assertEqual(metadata["status"], "partial")
            self.assertFalse(metadata["complete_workflow"])

    def test_old_no_run_dir_cli_selects_new_isolated_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            argv = ["run_analysis", "--through-step", "record_environment"]
            with patch.object(sys, "argv", argv), patch.object(workflow, "ANALYSIS_ROOT", root), \
                    patch.object(workflow, "execute_steps"):
                workflow.main()
            directories = list((root / "runs").glob("archived_*"))
            self.assertEqual(len(directories), 1)
            self.assertTrue((directories[0] / "run.json").is_file())

    def test_input_audit_blocks_archived_writes_and_full_refit_table_reads(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            path = root / "source.csv"
            path.write_text("value\n1\n")
            archived = InputAudit("archived", root, root / "out", [root])
            with self.assertRaises(PermissionError):
                archived("open", (str(path), "w", os.O_WRONLY))
            archived("open", (str(path), "r", os.O_RDONLY))
            self.assertEqual(archived.reads[str(path)]["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            full = InputAudit("full-refit", root / "new", root / "out", [root])
            with self.assertRaises(PermissionError):
                full("open", (str(path), "r", os.O_RDONLY))

    def test_refit_validation_rejects_missing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertTrue(validate(root / "logs", root / "results", root / "source", "full-refit"))

    def test_relocated_archived_manifest_and_external_full_refit_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            with patch.object(paths, "DATA_DIR", root):
                self.assertEqual(paths.analysis_input_path("analysis/data/analysis_objects/ds.h5ad"), root / "analysis_objects/ds.h5ad")
                external = root / "fresh/ds.h5ad"
                self.assertEqual(paths.analysis_input_path(str(external)), external)
                self.assertEqual(paths.display_path(external), str(external))

    def test_analysis_python_sources_parse_without_importing_scientific_stack(self):
        for path in paths.ANALYSIS_ROOT.rglob("*.py"):
            if any(part in {"external", "runs"} for part in path.relative_to(paths.ANALYSIS_ROOT).parts):
                continue
            with self.subTest(path=path.name):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


class LegacyRuntimeTests(unittest.TestCase):
    def test_empty_and_directory_are_not_python_executables(self):
        for value in ("", str(Path.cwd())):
            with patch.dict(os.environ, {"SCDR_LEGACY_PYTHON": value}):
                with self.assertRaisesRegex(FileNotFoundError, "executable file"):
                    legacy_runtime.require_legacy_python()

    def test_incompatible_python_is_explained(self):
        with patch.dict(os.environ, {"SCDR_LEGACY_PYTHON": sys.executable}), patch.object(
                legacy_runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(1, [], stderr="Python 3.7 required")):
            with self.assertRaisesRegex(RuntimeError, "Python 3.7 required"):
                legacy_runtime.require_legacy_python()

    def test_dll_path_order_is_preserved(self):
        python = Path(sys.executable).resolve()
        env = legacy_runtime.legacy_environment(python, Path("SAUCIE"))
        prefixes = env["PATH"].split(os.pathsep)
        self.assertEqual(prefixes[:5], [str(python.parent / part) for part in
                         ("", "Library/bin", "Library/mingw-w64/bin", "DLLs", "Scripts")])

    def test_pythonpath_is_checkout_parent_not_package_folder(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {"PYTHONPATH": "unrelated-source"}):
            source = Path(temporary) / "external" / "SAUCIE"
            env = legacy_runtime.legacy_environment(Path(sys.executable), source)
            self.assertEqual(env["PYTHONPATH"], str(source.resolve().parent))
            self.assertNotEqual(env["PYTHONPATH"], str(source))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_import_probe_reproduces_old_failure_and_accepts_parent_search_path(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            source = root / "external" / "SAUCIE"
            source.mkdir(parents=True)
            (source / "__init__.py").write_text(
                "class SAUCIE:\n    def __init__(self):\n        raise AssertionError('must not fit')\n"
                "class Loader:\n    def __init__(self):\n        raise AssertionError('must not load training data')\n")
            (source / "tensorflow.py").write_text("__version__ = 'test-stand-in'\n")
            cwd = root / "unrelated_working_directory"
            cwd.mkdir()
            probe = paths.ANALYSIS_ROOT / "legacy_import_probe.py"
            command = [sys.executable, "-B", "-S", str(probe), "--saucie-parent", str(source)]
            env = legacy_runtime.legacy_environment(Path(sys.executable), source)
            broken = {**env, "PYTHONPATH": str(source)}
            result = subprocess.run(command, env=broken, cwd=cwd, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("No module named 'SAUCIE'", result.stderr)
            result = subprocess.run(command, env=env, cwd=cwd, text=True, capture_output=True, check=True)
            payload = json.loads(result.stdout)
            self.assertEqual(payload["package_file"], str((source / "__init__.py").resolve()))
            self.assertEqual(payload["script_directory"], str(paths.ANALYSIS_ROOT))
            self.assertEqual(payload["working_directory"], str(cwd))
            self.assertFalse(payload["model_instantiated"])
            self.assertFalse(payload["training_executed"])
            self.assertFalse((source / "__pycache__").exists())

    def test_import_probe_rejects_shadow_package_from_wrong_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            selected = root / "selected-checkout"
            selected.mkdir()
            (selected / "__init__.py").write_text("")
            (selected / "tensorflow.py").write_text("__version__ = 'test-stand-in'\n")
            shadow = root / "SAUCIE"
            shadow.mkdir()
            (shadow / "__init__.py").write_text("class SAUCIE: pass\nclass Loader: pass\n")
            result = subprocess.run([sys.executable, "-B", "-S", str(paths.ANALYSIS_ROOT / "legacy_import_probe.py"),
                                     "--saucie-parent", str(selected)],
                                    env=legacy_runtime.legacy_environment(Path(sys.executable), selected),
                                    cwd=root, text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("instead of", result.stderr)

    def test_import_preflight_failure_is_actionable(self):
        with patch.object(legacy_runtime.subprocess, "run", side_effect=subprocess.CalledProcessError(
                1, [], stderr="ModuleNotFoundError: No module named 'SAUCIE'")):
            with self.assertRaisesRegex(RuntimeError, "SAUCIE import preflight failed.*legacy fitting context"):
                legacy_runtime.probe_saucie_import(Path(sys.executable), Path("external/SAUCIE"))

    def test_preflight_uses_source_argument_and_fitting_environment(self):
        source = Path("external/SAUCIE").resolve()
        with patch.object(legacy_runtime.subprocess, "run", return_value=subprocess.CompletedProcess(
                [], 0, stdout='{"training_executed": false}')) as runner:
            self.assertFalse(legacy_runtime.probe_saucie_import(Path(sys.executable), source)["training_executed"])
        arguments = runner.call_args.args[0]
        self.assertEqual(arguments[1], str(paths.ANALYSIS_ROOT / "legacy_import_probe.py"))
        self.assertEqual(arguments[-2:], ["--saucie-parent", str(source)])
        self.assertEqual(runner.call_args.kwargs["env"]["PYTHONPATH"], str(source.parent))

    def test_main_preflight_requires_real_saucie_import_in_addition_to_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            imported = {"package_file": "expected/SAUCIE/__init__.py", "training_executed": False}
            with patch.object(paths, "LOGS_DIR", root), \
                    patch.object(legacy_runtime, "require_legacy_python", return_value=Path(sys.executable)), \
                    patch.object(legacy_runtime, "verify_legacy_source", return_value={"commit": legacy_runtime.SAUCIE_COMMIT}), \
                    patch.object(legacy_runtime, "probe_saucie_import", return_value=imported) as probe, \
                    patch("builtins.print"):
                legacy_runtime.main()
                probe.assert_called_once()
            payload = json.loads((root / "legacy_preflight.json").read_text())
            self.assertEqual(payload["SAUCIE_import"], imported)

    def test_failed_import_does_not_write_passing_preflight_record(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(paths, "LOGS_DIR", root), \
                    patch.object(legacy_runtime, "require_legacy_python", return_value=Path(sys.executable)), \
                    patch.object(legacy_runtime, "verify_legacy_source", return_value={"commit": legacy_runtime.SAUCIE_COMMIT}), \
                    patch.object(legacy_runtime, "probe_saucie_import", side_effect=RuntimeError("import failed")):
                with self.assertRaisesRegex(RuntimeError, "import failed"):
                    legacy_runtime.main()
            self.assertFalse((root / "legacy_preflight.json").exists())


if __name__ == "__main__":
    unittest.main()
