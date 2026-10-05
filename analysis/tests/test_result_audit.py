"""Completeness guards for the five-seed root-sensitivity design."""
from itertools import product
import unittest
import pandas as pd
from analysis.scripts.audit_results import METHODS, ROOTS, SEEDS, root_grid_errors


class ResultAuditTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame(list(product(METHODS, SEEDS, ROOTS)),
                                  columns=["method", "embedding_seed", "root_definition"])

    def test_complete_five_seed_grid_passes(self):
        self.assertEqual(len(self.frame), 180)
        self.assertEqual(root_grid_errors(self.frame), [])

    def test_old_single_seed_grid_fails(self):
        self.assertTrue(root_grid_errors(self.frame[self.frame.embedding_seed.eq(0)]))

    def test_duplicate_cannot_replace_missing_key(self):
        frame = self.frame.copy()
        frame.iloc[-1] = frame.iloc[0]
        errors = root_grid_errors(frame)
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("missing" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
