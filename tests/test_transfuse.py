import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from image_model.segmentation import (
    MaskSettings,
    Sample,
    clean_mask,
    load_images,
    read_manifest,
    run_segmentation,
    write_outputs,
)
from image_model.transfuse import SEMINAR_MASK_SETTINGS


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
            write_outputs(output, source, box, probability, SEMINAR_MASK_SETTINGS)

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
        cleaned = clean_mask(raw, MaskSettings(0.125, 2, 0.02, (1, 1)))
        self.assertEqual(cleaned[9, 9], 1)
        self.assertEqual(cleaned[18, 18], 0)
        self.assertEqual(cleaned[40, 40], 1)
        self.assertEqual(cleaned[60, 60], 0)

    def test_shared_runner_accepts_a_non_square_probability_map(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (12, 8)).save(root / "source.png")
            manifest = root / "crops.json"
            manifest.write_text(json.dumps({
                "orientation": "exif-transposed",
                "samples": [{"id": "eye", "source": "source.png", "box": [2, 1, 8, 5]}],
            }), encoding="utf-8")
            probability = np.array([[0, 0.75, 0], [0, 0.75, 0]], dtype=np.float32)
            prepared = []

            def predict(crop):
                self.assertEqual((crop.mode, crop.size), ("RGB", (6, 4)))
                return probability

            def prepare():
                prepared.append(True)
                return predict, {"model": "fixture"}

            settings = MaskSettings(0.5, 2, 0, (1, 1))
            output = root / "results"
            run_segmentation(manifest, output, prepare, settings)
            metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model"], "fixture")
            self.assertEqual(metadata["samples"], [{"id": "eye", "box": [2, 1, 8, 5]}])
            expected = np.zeros((8, 12), dtype=np.float32)
            expected[1:5, 4:6] = 0.75
            np.testing.assert_array_equal(np.load(output / "eye" / "probability.npy"), expected)
            with self.assertRaises(FileExistsError):
                run_segmentation(manifest, output, prepare, settings)
            self.assertEqual(prepared, [True])

    def test_shared_runner_rejects_invalid_predictions_without_publishing_partial_results(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (8, 8)).save(root / "source.png")
            manifest = root / "crops.json"
            manifest.write_text(json.dumps({
                "orientation": "exif-transposed",
                "samples": [
                    {"id": name, "source": "source.png", "box": [0, 0, 8, 8]}
                    for name in ("first", "second")
                ],
            }), encoding="utf-8")
            probabilities = iter([
                np.full((2, 3), 0.5, dtype=np.float32),
                np.full((2, 3), 1.1, dtype=np.float32),
            ])
            output = root / "results"
            with self.assertRaisesRegex(ValueError, "probability map"):
                run_segmentation(
                    manifest, output, lambda: (lambda crop: next(probabilities), {}),
                    MaskSettings(0.5, 2, 0, (1, 1)),
                )
            self.assertFalse(output.exists())
            self.assertEqual(set(root.iterdir()), {manifest, root / "source.png"})

    def test_publication_preserves_destination_created_during_prediction(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            Image.new("RGB", (8, 8)).save(source)
            manifest = root / "crops.json"
            manifest.write_text(json.dumps({
                "orientation": "exif-transposed",
                "samples": [{"id": "eye", "source": source.name, "box": [0, 0, 8, 8]}],
            }), encoding="utf-8")
            output = root / "results"

            def predict(crop):
                output.mkdir()
                return np.zeros((2, 2), dtype=np.float32)

            with self.assertRaises(FileExistsError):
                run_segmentation(manifest, output, lambda: (predict, {}), SEMINAR_MASK_SETTINGS)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(set(root.iterdir()), {source, manifest, output})
