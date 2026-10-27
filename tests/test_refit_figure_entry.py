"""Standard-library plotting-entry tests; no figures, downloads or model fits."""

from contextlib import redirect_stderr
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import make_figures


class FigureEntryTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which("git"), "Git is needed to check ignore-rule semantics")
    def test_gitignore_excludes_run_artifacts_not_portable_sources_or_tests(self):
        gitignore = Path(make_figures.__file__).resolve().with_name(".gitignore")
        ignored = ["analysis/runs/full-refit/run.json", "analysis/runs/smoke/data/input.h5ad"]
        public = ["tests/test_refit_figure_entry.py", "analysis/README.md", "scripts/plot_deep_controls.py"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / ".gitignore").write_text(gitignore.read_text(encoding="utf-8"), encoding="utf-8")
            subprocess.run(["git", "init", "--quiet", "--template="], cwd=root,
                           check=True, capture_output=True, timeout=20)
            result = subprocess.run(
                ["git", "-c", "core.excludesFile=", "check-ignore", "--no-index", "--stdin", "-z"],
                input="\0".join([*ignored, *public]) + "\0", cwd=root,
                capture_output=True, text=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.rstrip("\0").split("\0"), ignored)

    def sources(self, root):
        source = root / "public source data"
        output = root / "figure outputs"
        for module, directory, filenames in make_figures.REVISION_PLOTS:
            script = root.joinpath(*module.split(".")).with_suffix(".py")
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("# CLI stand-in; never executed.\n", encoding="ascii")
            folder = source / directory
            folder.mkdir(parents=True, exist_ok=True)
            for filename in filenames:
                (folder / filename).write_text("fixture\n", encoding="ascii")
        return source, output

    def flags(self, source, output, mode="--revision-supplements-only"):
        return [mode, "--revision-source-root", str(source),
                "--revision-output-dir", str(output)]

    def test_default_keeps_established_module_sequence_without_source_discovery(self):
        with patch.object(make_figures, "revision_commands") as discover, \
                patch.object(make_figures.subprocess, "run") as run:
            make_figures.main([])
        discover.assert_not_called()
        modules = [call.args[0][2] for call in run.call_args_list]
        self.assertEqual(modules, ["analysis.scripts.refresh_biological_plot_inputs", *make_figures.MODULES])
        self.assertNotIn("scripts.plot_reference_controls", modules)
        self.assertTrue(all(call.kwargs == {"check": True, "cwd": make_figures.CODE_ROOT}
                            for call in run.call_args_list))

    def test_missing_sources_fail_before_any_subprocess_or_directory_creation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = root / "missing", root / "outputs"
            errors = io.StringIO()
            with patch.object(make_figures, "CODE_ROOT", root), \
                    patch.object(make_figures.subprocess, "run") as run, \
                    redirect_stderr(errors), self.assertRaises(SystemExit) as raised:
                make_figures.main(self.flags(source, output, "--include-revision-supplements"))
            self.assertEqual(raised.exception.code, 2)
            self.assertIn("Publish/export completed controls first", errors.getvalue())
            self.assertIn("plot_deep_controls.py", errors.getvalue())
            run.assert_not_called()
            self.assertFalse(source.exists())
            self.assertFalse(output.exists())

    def test_sources_alone_do_not_accept_missing_portable_script(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            (root / "scripts/plot_deep_controls.py").unlink()
            with patch.object(make_figures, "CODE_ROOT", root), self.assertRaisesRegex(ValueError, "plot_deep_controls.py"):
                make_figures.revision_commands(source, output)

    def test_empty_conceptual_source_is_not_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            (source / "reference_attribution/Supplementary_Fig_S20_conceptual_source.json").write_bytes(b"")
            with patch.object(make_figures, "CODE_ROOT", root), self.assertRaisesRegex(ValueError, "conceptual_source.json"):
                make_figures.revision_commands(source, output)

    def test_revision_only_passes_explicit_public_paths_and_never_refreshes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            with patch.object(make_figures, "CODE_ROOT", root), \
                    patch.object(make_figures.subprocess, "run") as run:
                make_figures.main(self.flags(source, output))
            self.assertEqual(run.call_count, 2)
            for call, (module, directory, _) in zip(run.call_args_list, make_figures.REVISION_PLOTS):
                self.assertEqual(call.args[0], [sys.executable, "-m", module,
                                 "--source-dir", str(source / directory), "--output-dir", str(output)])
                self.assertEqual(call.kwargs, {"check": True, "cwd": root})
            self.assertFalse(output.exists())

    def test_include_appends_revision_plots_after_established_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            with patch.object(make_figures, "CODE_ROOT", root), \
                    patch.object(make_figures.subprocess, "run") as run:
                make_figures.main(self.flags(source, output, "--include-revision-supplements"))
            modules = [call.args[0][2] for call in run.call_args_list]
            self.assertEqual(modules, ["analysis.scripts.refresh_biological_plot_inputs",
                             *make_figures.MODULES, *(r[0] for r in make_figures.REVISION_PLOTS)])

    def test_check_only_calls_plotters_read_only_and_skips_established_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            with patch.object(make_figures, "CODE_ROOT", root), \
                    patch.object(make_figures.subprocess, "run") as run:
                make_figures.main([*self.flags(source, output), "--check-only"])
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(c.args[0][-1] == "--check-only" for c in run.call_args_list))
            self.assertFalse(output.exists())

    def test_check_only_without_revision_selection_fails_before_subprocess(self):
        with patch.object(make_figures.subprocess, "run") as run, \
                redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            make_figures.main(["--check-only"])
        self.assertEqual(raised.exception.code, 2)
        run.assert_not_called()

    def test_source_output_overlap_is_rejected_both_ways(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, _ = self.sources(root)
            for output in [source, source / "plots", root]:
                with self.subTest(output=output), self.assertRaisesRegex(ValueError, "must not overlap"):
                    make_figures.revision_commands(source, output)

    def test_plot_failure_stops_later_plot_groups(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, output = self.sources(root)
            error = subprocess.CalledProcessError(1, "plotter")
            with patch.object(make_figures, "CODE_ROOT", root), \
                    patch.object(make_figures.subprocess, "run", side_effect=error) as run, \
                    self.assertRaises(subprocess.CalledProcessError):
                make_figures.main(self.flags(source, output))
            self.assertEqual(run.call_count, 1)

    def test_real_help_from_unrelated_directory_needs_no_scientific_imports(self):
        script = Path(make_figures.__file__).resolve()
        with tempfile.TemporaryDirectory() as temporary:
            before = set(Path(temporary).iterdir())
            result = subprocess.run([sys.executable, "-B", "-S", str(script), "--help"],
                                    cwd=temporary, capture_output=True, text=True, timeout=20)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("--revision-supplements-only", result.stdout)
            self.assertIn("--check-only", result.stdout)
            self.assertEqual(set(Path(temporary).iterdir()), before)


if __name__ == "__main__":
    unittest.main()
