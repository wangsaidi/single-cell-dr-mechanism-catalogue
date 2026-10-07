"""Protect the exact method-level sensitivity kernel and matching decisions."""
import itertools
import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from figures.plot_supplementary_figure_s17 import (
    ROOT, METHODS, METHOD_ORDER, METADATA, encode_specifications,
    exact_mantel, profile_matrix, specification_distance,
)
from figures.plot_figure_2 import _objective_signature_matrix, _bh_adjust


class SpecificationExtensionTests(unittest.TestCase):
    def test_tied_exact_kernel_against_direct_relabeling(self):
        x = squareform(pdist(np.array([[0, 1], [0, 1], [1, 0], [1, 1]]), "jaccard"))
        y = squareform(pdist(np.array([[0, 2], [1, 1], [3, 1], [1, 2]])))
        upper = np.triu_indices(4, 1)
        direct = np.array([spearmanr(x[upper], y[np.ix_(p, p)][upper]).statistic
                           for p in itertools.permutations(range(4))])
        for batch in (1, 7, 100):
            result, null = exact_mantel(x, y, batch)
            np.testing.assert_allclose(null, direct, atol=1e-14)
            self.assertEqual(result["n_permutations"], 24)
            self.assertEqual(result["tail_count"], int((abs(direct) >= abs(direct[0])-1e-12).sum()))

    def test_original_submatrix_unchanged_with_scvi_features(self):
        matrix = encode_specifications(METADATA / "method_mathematical_specifications.csv", METHODS)
        self.assertEqual(matrix.shape, (9, 20))
        np.testing.assert_array_equal(specification_distance(matrix)[:8, :8],
                                      specification_distance(_objective_signature_matrix()))

    def test_complete_seed_zero_source(self):
        data = pd.read_csv(ROOT / "data/source_data/calibration/scvi_geometry_full.csv")
        data = data.loc[data.output_dimension.eq(2) & data.seed.eq(0)]
        self.assertEqual(profile_matrix(data, METHODS).shape, (9, 12))
        self.assertTrue(data.k.eq(15).all())
        self.assertTrue(data.reference_dimension.eq(50).all())
        self.assertTrue(data.sampled_pairs.eq(5000).all())
        with self.assertRaises(ValueError):
            profile_matrix(data.iloc[1:], METHODS)
        with self.assertRaises(ValueError):
            profile_matrix(pd.concat([data, data.iloc[:1]]), METHODS)

    def test_recovery_and_declared_multiplicity(self):
        source = ROOT / "data/source_data/generated"
        control = pd.read_csv(source / "s17_controls.csv")
        self.assertAlmostEqual(control.iloc[0].rho, .163781707312566, places=12)
        self.assertAlmostEqual(control.iloc[0].p_exact, .475496031746032, places=12)
        influence = pd.read_csv(source / "s17_method_influence.csv")
        recovery = influence.loc[influence.omitted_method.eq("scVI") & influence.scaling.eq("Rescaled eight-method subset")].iloc[0]
        self.assertAlmostEqual(recovery.rho, control.iloc[1].rho, places=13)
        for name, count in [("contexts", 4), ("encodings", 6)]:
            frame = pd.read_csv(source / f"s17_{name}.csv")
            self.assertEqual(len(frame), count)
            np.testing.assert_allclose(frame.q_bh, _bh_adjust(frame.p_exact.to_numpy()), atol=1e-14)


if __name__ == "__main__":
    unittest.main()
