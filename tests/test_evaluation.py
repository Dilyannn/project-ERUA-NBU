import unittest

import numpy as np

from image_model.evaluation import mask_scores


class MaskScoresTest(unittest.TestCase):
    def test_shifted_boundary_matches_only_at_given_tolerance(self):
        reference = np.zeros((10, 10), dtype=bool)
        reference[2:5, 2:5] = True
        predicted = np.roll(reference, 1, axis=1)

        exact = mask_scores(predicted, reference, tolerance=0)
        self.assertAlmostEqual(exact["dice"], 2 / 3)
        self.assertEqual(exact["iou"], 0.5)
        self.assertEqual(exact["false_positive_pixels"], 3)
        self.assertEqual(exact["false_negative_pixels"], 3)
        self.assertLess(exact["boundary_f1"], 1)
        self.assertEqual(mask_scores(predicted, reference, tolerance=1)["boundary_f1"], 1.0)

    def test_empty_masks_and_dimension_mismatch(self):
        empty = np.zeros((5, 5), dtype=bool)
        self.assertEqual(mask_scores(empty, empty, tolerance=3)["boundary_f1"], 1.0)
        self.assertEqual(mask_scores(~empty, empty, tolerance=3)["boundary_f1"], 0.0)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            mask_scores(empty, empty[:2], tolerance=3)
