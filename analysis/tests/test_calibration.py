"""Focused tests of permutation units and probability calculations."""
import unittest
import numpy as np
import pandas as pd
from analysis.scripts.calibrate_operational_thresholds import bh, groups_for, null_summary, permutation


class CalibrationTests(unittest.TestCase):
    def test_conditional_assignment_is_bijective_and_preserves_groups(self):
        labels = np.repeat(np.arange(4), 8)
        p = permutation(np.random.default_rng(9), len(labels), groups_for(labels))
        np.testing.assert_array_equal(np.sort(p), np.arange(len(labels)))
        np.testing.assert_array_equal(labels[p], labels)

    def test_graph_relabelling_preserves_degree_and_no_self_edges(self):
        n = 32
        graph = (np.arange(n)[:, None] + np.arange(1, 5)) % n
        p = permutation(np.random.default_rng(7), n)
        inverse = np.empty(n, int); inverse[p] = np.arange(n)
        result = inverse[graph[p]]
        self.assertEqual(result.shape, graph.shape)
        self.assertTrue(np.all(result != np.arange(n)[:, None]))
        self.assertTrue(all(len(np.unique(row)) == 4 for row in result))

    def test_plus_one_probability_has_correct_direction_and_never_zero(self):
        high = null_summary(2, [0, 1], .3, "example")
        low = null_summary(-1, [0, 1], .3, "example", tail="lower")
        self.assertAlmostEqual(high["p_one_sided"], 1 / 3)
        self.assertAlmostEqual(low["p_one_sided"], 1 / 3)

    def test_bh_adjustment_is_separate_by_null_family(self):
        frame = pd.DataFrame({"null_family": ["a", "a", "a", "b"], "p_one_sided": [.01, .04, .9, .01]})
        np.testing.assert_allclose(bh(frame).q_bh, [.03, .06, .9, .01])


if __name__ == "__main__":
    unittest.main()
