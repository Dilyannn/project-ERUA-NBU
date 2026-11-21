import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from image_model.transfuse import Baseline, Sample, clean_mask, load_images, read_manifest, write_outputs


class TransFuseGeometryTest(unittest.TestCase):
    def test_outputs_round_trip_to_exif_oriented_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            original = np.full((16, 12, 3), (31, 47, 59), dtype=np.uint8)
            source_path = root / "source.png"
            source_image = Image.fromarray(original)
            exif = source_image.getexif()
            exif[274] = 6
            source_image.save(source_path, exif=exif)
            source_bytes = source_path.read_bytes()
            manifest = root / "crops.json"
            manifest.write_text(json.dumps({
                "orientation": "exif-transposed",
                "samples": [{"id": "eye", "source": "source.png", "box": [3, 2, 11, 8]}],
            }), encoding="utf-8")

            sample, = read_manifest(manifest)
            source, crop, box = load_images(sample)
            self.assertEqual(source.size, (16, 12))
            self.assertEqual(crop.size, (8, 6))
            probability = np.full((384, 384), 0.05, dtype=np.float32)
            probability[64:256, 96:288] = 0.8
            output = root / "results"
            write_outputs(output, source, box, probability)

            expected_mask = np.zeros((12, 16), dtype=np.uint8)
            expected_mask[3:6, 5:9] = 255
            expected_probability = np.zeros((12, 16), dtype=np.float32)
            expected_probability[2:8, 3:11] = 0.05
            expected_probability[3:6, 5:9] = 0.8
            with Image.open(output / "raw-mask.png") as raw:
                np.testing.assert_array_equal(np.asarray(raw), expected_mask)
            with Image.open(output / "cleaned-mask.png") as cleaned:
                np.testing.assert_array_equal(np.asarray(cleaned), expected_mask)
            with Image.open(output / "overlay.png") as overlay:
                outside = expected_mask == 0
                np.testing.assert_array_equal(np.asarray(overlay)[outside], np.asarray(source)[outside])
                self.assertFalse(np.array_equal(np.asarray(overlay)[~outside], np.asarray(source)[~outside]))
            np.testing.assert_array_equal(np.load(output / "probability.npy"), expected_probability)
            self.assertEqual(source_path.read_bytes(), source_bytes)

    def test_rejects_inconsistent_crop_metadata(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            crop = root / "crop.png"
            Image.new("RGB", (20, 12)).save(source)
            Image.new("RGB", (10, 6)).save(crop)
            with self.assertRaisesRegex(ValueError, "crop dimensions"):
                load_images(Sample("eye", source, crop, (0, 0, 8, 6)))
            with self.assertRaisesRegex(ValueError, "outside"):
                load_images(Sample("eye", source, None, (0, 0, 20, 13)))

    def test_eye_opening_padding_clips_to_image_edges(self):
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.png"
            Image.new("RGB", (30, 22)).save(source_path)
            sample = Sample("eye", source_path, None, (1, 4, 29, 21), box_is_eye_opening=True)
            _, crop, box = load_images(sample)
            self.assertEqual(box, (0, 0, 30, 22))
            self.assertEqual(crop.size, (30, 22))
            inner = Sample("eye", source_path, None, (7, 6, 22, 16), box_is_eye_opening=True)
            _, crop, box = load_images(inner)
            self.assertEqual(box, (5, 2, 24, 20))
            self.assertEqual(crop.size, (19, 18))

    def test_cleanup_keeps_two_components_and_fills_only_small_holes(self):
        raw = np.zeros((64, 64), dtype=np.uint8)
        raw[5:30, 5:30] = 1
        raw[9, 9] = 0
        raw[16:21, 16:21] = 0
        raw[35:55, 35:55] = 1
        raw[59:62, 59:62] = 1
        cleaned = clean_mask(raw, Baseline(closing_kernel=(1, 1)))
        self.assertEqual(cleaned[9, 9], 1)
        self.assertEqual(cleaned[18, 18], 0)
        self.assertEqual(cleaned[40, 40], 1)
        self.assertEqual(cleaned[60, 60], 0)
