import unittest

import numpy as np

from image_model.evaluation import mask_scores, region_scores, validate_region


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

    def test_region_metrics_distinguish_missing_and_empty_labels(self):
        reference = np.array([[True, True], [False, False]])
        predicted = np.array([[True, False], [True, False]])
        skin = ~reference
        validate_region("peri_limbal_sclera", reference, reference)
        validate_region("skin", skin, reference)
        scores = region_scores(predicted, {
            "peri_limbal_sclera": reference,
            "pigmented_sclera": np.zeros_like(reference),
            "skin": skin,
            "eyelid": None,
        })
        self.assertEqual(scores["peri_limbal_sclera_recall"], 0.5)
        self.assertIsNone(scores["pigmented_sclera_recall"])
        self.assertEqual(scores["skin_false_positive_pixels"], 1)
        self.assertIsNone(scores["eyelid_false_positive_pixels"])
        with self.assertRaisesRegex(ValueError, "outside"):
            validate_region("caruncle", reference, reference)
