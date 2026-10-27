"""Exercise real cache/CLI functions with standard-library scientific stand-ins."""

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch

from analysis.paths import ANALYSIS_ROOT


class Array:
    shape = (40, 2)

    def astype(self, *args, **kwargs):
        return self

    def copy(self):
        return self


@contextmanager
def scientific_standins(smoke=False):
    """Import actual modules but never import NumPy, Scanpy or TensorFlow."""
    names = ["numpy", "pandas", "scanpy", "scipy", "scipy.sparse", "scipy.stats",
             "sklearn", "sklearn.manifold", "analysis.config", "analysis.diagnostics", "analysis.execution"]
    stubs = {name: types.ModuleType(name) for name in names}
    config = stubs["analysis.config"]
    for key, value in {"N_NEIGHBORS": 15, "SEED": 0, "TSNE_PERPLEXITY": 30,
                       "ANCHOR_METHODS": ["PCA", "GLM-PCA", "scScope", "SAUCIE", "UMAP", "PHATE", "t-SNE", "PaCMAP"], "METHOD_FAMILY": {},
                       "SUPPORT_THRESHOLDS": {"local_retention": .3, "trustworthiness": .9,
                                              "global_rank_corr": .45, "label_recall": .55}}.items():
        setattr(config, key, value)
    stubs["analysis.execution"].SMOKE = smoke
    np = stubs["numpy"]
    np.float32 = "float32"
    np.nan = float("nan")
    np.asarray = MagicMock(side_effect=lambda value, **kwargs: value if isinstance(value, Array) else Array())
    np.load = MagicMock(return_value=Array())
    np.save = MagicMock()
    np.isfinite = MagicMock()
    pd = stubs["pandas"]
    pd.DataFrame = MagicMock()
    frame = pd.DataFrame.return_value
    frame.__getitem__.return_value.__eq__.return_value.any.return_value = False
    pd.Series = MagicMock()
    pd.Series.return_value.value_counts.return_value.to_dict.return_value = {}
    pd.Series.return_value.nunique.return_value = 2
    stubs["scipy"].sparse = stubs["scipy.sparse"]
    stubs["scipy.stats"].spearmanr = MagicMock()
    stubs["sklearn.manifold"].TSNE = MagicMock()
    import analysis
    # Also patch parent attributes: from analysis import config consults these.
    with patch.dict(sys.modules, stubs), patch.object(analysis, "config", config, create=True), \
            patch.object(analysis, "diagnostics", stubs["analysis.diagnostics"], create=True):
        modules = {}
        for name in ("generate_robustness_embeddings", "generate_anchor_embeddings", "generate_known_truth_stress_tests"):
            full_name = "analysis.scripts." + name
            spec = importlib.util.spec_from_file_location(full_name, ANALYSIS_ROOT / "scripts" / (name + ".py"))
            module = importlib.util.module_from_spec(spec)
            with patch.dict(sys.modules, {**modules, full_name: module}):
                spec.loader.exec_module(module)
            modules[full_name] = module
        yield modules, np, pd


class CacheTests(unittest.TestCase):
    def test_smoke_keeps_all_eight_methods_and_all_six_simulation_scenarios(self):
        with scientific_standins(smoke=True) as (modules, _, __):
            anchors = modules["analysis.scripts.generate_anchor_embeddings"]
            simulations = modules["analysis.scripts.generate_known_truth_stress_tests"]
            robustness = modules["analysis.scripts.generate_robustness_embeddings"]
            self.assertEqual(len(anchors.METHODS), 8)
            self.assertEqual(set(simulations.METHODS), set(anchors.METHODS))
            self.assertEqual(len(simulations.SCENARIOS), 6)
            self.assertEqual(simulations.SIM_REPLICATES, [0])
            self.assertEqual(robustness.OUTPUT_DIMS, [2, 10])
            self.assertEqual(robustness.UPSTREAM_PCS, [20])

    def test_anchor_force_reaches_deep_fit_and_nonforce_reuses_outer_cache(self):
        with scientific_standins() as (modules, _, __):
            module = modules["analysis.scripts.generate_anchor_embeddings"]
            proc = types.SimpleNamespace(n_obs=40)
            fresh, cached = Array(), Array()
            with patch.object(module, "_load_or_none", return_value=cached), \
                    patch.object(module, "_run_legacy", return_value=fresh) as legacy, \
                    patch.object(module, "_save", side_effect=lambda path, arr: arr):
                self.assertIs(module.compute_embedding("ds", "scScope", proc, None, "label", 0), cached)
                legacy.assert_not_called()
                self.assertIs(module.compute_embedding("ds", "scScope", proc, None, "label", 0, force=True), fresh)
                self.assertTrue(legacy.call_args.kwargs["force"])

    def test_empirical_force_bypasses_inner_latents_for_both_deep_methods(self):
        with scientific_standins() as (modules, _, __), tempfile.TemporaryDirectory() as temporary:
            module = modules["analysis.scripts.generate_anchor_embeddings"]
            root, fresh = Path(temporary), Array()
            for method in ("scScope", "SAUCIE"):
                with patch.object(module, "LEGACY_DIR", root), patch.object(module, "_load_or_none", return_value=fresh) as load, \
                        patch.object(module, "require_legacy_python", return_value=Path(sys.executable)), \
                        patch.object(module, "_prepare_legacy_input", return_value=root / "input.npz"), \
                        patch.object(module.subprocess, "run") as runner, \
                        patch.object(module, "_project_scscope_latent", return_value=fresh) as project:
                    module._run_legacy("ds", method, None, "label", 0, force=True)
                    self.assertEqual(runner.call_count, 1)
                    source = module.legacy_source()
                    self.assertEqual(runner.call_args.kwargs["env"]["PYTHONPATH"], str(source.parent))
                    command = runner.call_args.args[0]
                    self.assertEqual(command[command.index("--saucie-parent") + 1], str(source))
                    self.assertEqual(load.call_count, 1, "only reload the new runner output")
                    if method == "scScope":
                        self.assertTrue(project.call_args.kwargs["force"])

    def test_empirical_and_simulation_projection_force_bypass_cache(self):
        with scientific_standins() as (modules, np, _), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ("generate_anchor_embeddings", "generate_known_truth_stress_tests"):
                module = modules["analysis.scripts." + name]
                loader = "_load_or_none" if "anchor" in name else "_load_finite_cache"
                directory = "LEGACY_DIR" if "anchor" in name else "SIM_DIR"
                arguments = ("ds", Array(), 0) if "anchor" in name else (Array(), 0, "linear_low_rank", 0)
                with patch.object(module, directory, root), patch.object(module, loader, return_value=Array()) as load, \
                        patch.object(module, "_array_hash", return_value="test"), \
                        patch.object(module, "TSNE") as tsne:
                    tsne.return_value.fit_transform.return_value = Array()
                    module._project_scscope_latent(*arguments)
                    tsne.assert_not_called()
                    load.reset_mock()
                    module._project_scscope_latent(*arguments, force=True)
                    load.assert_not_called()
                    tsne.assert_called_once()

    def test_simulation_force_bypasses_deep_latents_and_propagates_to_projection(self):
        with scientific_standins() as (modules, _, __), tempfile.TemporaryDirectory() as temporary:
            module = modules["analysis.scripts.generate_known_truth_stress_tests"]
            for method in ("scScope", "SAUCIE"):
                with patch.object(module, "SIM_DIR", Path(temporary)), \
                        patch.object(module, "_load_finite_cache", return_value=Array()) as load, \
                        patch.object(module, "require_legacy_python", return_value=Path(sys.executable)), \
                        patch.object(module, "_prepare_legacy_input", return_value=Path(temporary) / "input.npz"), \
                        patch.object(module.subprocess, "run") as runner, \
                        patch.object(module, "_project_scscope_latent", return_value=Array()) as project:
                    module._run_legacy_method(Array(), Array(), method, 0, "linear_low_rank", 0, force=True)
                    runner.assert_called_once()
                    source = module.legacy_source()
                    self.assertEqual(runner.call_args.kwargs["env"]["PYTHONPATH"], str(source.parent))
                    command = runner.call_args.args[0]
                    self.assertEqual(command[command.index("--saucie-parent") + 1], str(source))
                    self.assertEqual(load.call_count, 1)
                    if method == "scScope":
                        self.assertTrue(project.call_args.kwargs["force"])

    def test_simulation_outer_cache_force_reaches_embed(self):
        with scientific_standins() as (modules, np, pd), tempfile.TemporaryDirectory() as temporary:
            module = modules["analysis.scripts.generate_known_truth_stress_tests"]
            simulation = {key: Array() for key in ("counts", "labels", "batch", "pseudotime")}
            simulation["counts"].shape = (40, 5)
            with patch.multiple(module, SOURCE_DIR=Path(temporary), SIM_DIR=Path(temporary),
                                SCENARIOS=["linear_low_rank"], SIM_REPLICATES=[0], METHODS=["scScope"]), \
                    patch.object(module, "_simulate_scenario", return_value=simulation), \
                    patch.object(module, "_preprocess_counts_to_pca", return_value=Array()), \
                    patch.object(module, "_metric_rows", return_value=[]), \
                    patch.object(module, "_load_finite_cache", return_value=Array()) as cache, \
                    patch.object(module, "_embed", return_value=Array()) as embed:
                module.run_suite()
                embed.assert_not_called()
                cache.reset_mock()
                module.run_suite(force=True)
                cache.assert_not_called()
                embed.assert_called_once()
                self.assertTrue(embed.call_args.kwargs["force"])

    def test_robustness_force_bypasses_both_cached_axes(self):
        with scientific_standins() as (modules, np, _), tempfile.TemporaryDirectory() as temporary:
            module = modules["analysis.scripts.generate_robustness_embeddings"]
            root = Path(temporary)
            for dataset in ("pbmc3k", "paul15", "heart_cell_atlas_subsampled"):
                (root / f"dim_{dataset}_PCA_2.npy").touch()
                (root / f"upstream_{dataset}_PCA_pca20.npy").touch()
            proc = MagicMock()
            proc.n_obs = 40
            with patch.multiple(module, SOURCE_DIR=root, ROBUST_DIR=root, ROBUST_METHODS=["PCA"],
                                UPSTREAM_METHODS=["PCA"], OUTPUT_DIMS=[2], UPSTREAM_PCS=[20]), \
                    patch.object(module, "_load_proc", return_value=(proc, {"label_field": "label"})), \
                    patch.object(module, "_compute_pca_space", return_value=Array()), \
                    patch.object(module, "_metric_bundle", return_value=[]), \
                    patch.object(module, "_embed_from_reference", return_value=Array()) as embed:
                module.write_output_dimension_response()
                module.write_upstream_pca_response()
                embed.assert_not_called()
                np.load.reset_mock()
                module.write_output_dimension_response(force=True)
                module.write_upstream_pca_response(force=True)
                np.load.assert_not_called()
                self.assertEqual(embed.call_count, 6)

    def test_archived_hash_verification_resolves_isolated_input_paths(self):
        with scientific_standins(), tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, embeddings, public = root / "data", root / "scvi", root / "public"
            for directory in (data, embeddings, public):
                directory.mkdir()
            paths = [data / "object.csv", embeddings / "fit.npy", public / "reference.csv"]
            for path in paths:
                path.write_bytes(b"original")
            digest = hashlib.sha256(b"original").hexdigest()
            (root / "ARCHIVE_SHA256SUMS.txt").write_text(f"{digest}  data/object.csv\n{digest}  results/embeddings/fit.npy\n")
            (root / "ARCHIVED_TABLE_SHA256SUMS.txt").write_text(f"{digest}  reference.csv\n")
            spec = importlib.util.spec_from_file_location("verify_mock", ANALYSIS_ROOT / "verify_reproducibility.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with patch.multiple(module, ANALYSIS_ROOT=root, DATA_DIR=data, SCVI_EMBEDDINGS_DIR=embeddings,
                                PUBLIC_SOURCE_DATA_DIR=public, REFERENCE_RESULTS_DIR=root / "missing"):
                self.assertEqual(module.verify_inputs(), [])
                paths[1].write_bytes(b"refitted")
                self.assertTrue(module.verify_inputs())
                self.assertEqual(module.verify_inputs(data_only=True), [])
                self.assertTrue(module.verify_results(), "missing references must not silently pass")
                paths[0].write_bytes(b"changed")
                self.assertTrue(module.verify_inputs(data_only=True))


if __name__ == "__main__":
    unittest.main()
