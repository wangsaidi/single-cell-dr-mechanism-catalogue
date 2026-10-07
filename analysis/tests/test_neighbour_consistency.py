"""Ensure Fig. 5j uses the declared k nearest non-self neighbours."""
import unittest
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from figures.plot_figures_5_6 import heart_marker_neighbour_results


class NeighbourDefinitionTests(unittest.TestCase):
    def test_closest_nonself_neighbour_is_retained(self):
        coords = pd.DataFrame({'method': ['PCA'] * 3, 'x': [0., 1., 10.],
                               'y': [0.] * 3, 'label': ['A', 'A', 'B']})
        markers = pd.DataFrame({'label': ['A', 'B'], 'z_score': [1., 2.], 'n_cells': [2, 1]})
        result = heart_marker_neighbour_results(coords, markers, k=1).set_index('label')
        self.assertEqual(result.loc['A', 'label_knn_recall'], 1.)
        self.assertEqual(result.loc['B', 'label_knn_recall'], 0.)

    def test_matches_bruteforce_k15_definition(self):
        rng = np.random.default_rng(274)
        xy = rng.normal(size=(80, 2))
        labels = np.array(['A', 'B', 'C', 'D'] * 20)
        distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
        np.fill_diagonal(distances, np.inf)
        nearest = distances.argsort(axis=1)[:, :15]
        fractions = (labels[nearest] == labels[:, None]).mean(axis=1)
        coords = pd.DataFrame({'method': 'PCA', 'x': xy[:, 0], 'y': xy[:, 1], 'label': labels})
        markers = pd.DataFrame({'label': ['A', 'B', 'C', 'D'], 'z_score': [1., 2., 3., 4.], 'n_cells': 20})
        actual = heart_marker_neighbour_results(coords, markers).set_index('label')
        for label in np.unique(labels):
            self.assertAlmostEqual(actual.loc[label, 'label_knn_recall'], fractions[labels == label].mean())
            self.assertEqual(actual.loc[label, 'n_cells_coordinate'], 20)

    def test_rejects_nonfinite_or_insufficient_coordinates(self):
        coords = pd.DataFrame({'method': ['PCA'] * 3, 'x': [0., 1., np.nan],
                               'y': [0.] * 3, 'label': ['A', 'A', 'B']})
        markers = pd.DataFrame({'label': ['A', 'B'], 'z_score': [1., 2.], 'n_cells': [2, 1]})
        with self.assertRaises(ValueError):
            heart_marker_neighbour_results(coords, markers, k=1)
        coords.loc[2, 'x'] = 2.
        with self.assertRaises(ValueError):
            heart_marker_neighbour_results(coords, markers, k=3)


if __name__ == '__main__':
    unittest.main()
