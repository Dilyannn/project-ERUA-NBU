import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from PIL import Image

from image_model.datasets.manifest_export import export_manifests
from image_model.datasets.ubiris_importer import inspect_pairs
from image_model.evaluation import score_run
from image_model.segmentation import load_images, read_manifest, sha256


class ExportTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.base = Path(temp.name)
        self.root = self.base / "synthetic"
        for directory, mode in (("sum/image", "RGB"), ("sum/sclera_mask", "L")):
            path = self.root / directory
            path.mkdir(parents=True)
            Image.new(mode, (10, 8)).save(path / "eye.png")
        self.inventory = inspect_pairs(self.root, foreground=255, mask_frame="oriented-source")
        self.dest = self.base / "manifests"

    def test_cli_round_trip_and_source_hashes(self):
        pair, = self.inventory.samples
        crops, refs = export_manifests(self.inventory, ["eye"], {"eye": (2, 2, 8, 6)}, self.dest)
        sample, = read_manifest(crops)
        source, _, padded = load_images(sample)
        self.assertEqual((source.size, padded), ((10, 8), (1, 0, 9, 8)))
        run = self.base / "results"
        (run / "eye").mkdir(parents=True)
        (run / "run.json").write_text(json.dumps({"samples": [{"id": "eye"}]}))
        for variant in ("cleaned", "raw"):
            Image.fromarray(np.zeros((8, 10), dtype=np.uint8)).save(run / "eye" / f"{variant}-mask.png")
        self.assertEqual([r["dice"] for r in score_run(run, refs, 1)], [1, 1])
        self.assertEqual(sha256(self.root / pair.image), pair.image_sha256)
        self.assertEqual(sha256(self.root / pair.mask), pair.mask_sha256)

    def test_invalid_selection_and_boxes_publish_nothing(self):
        box = [0, 0, 1, 1]
        for ids, boxes in (([], {}), (["eye"], {}), (["eye"], {"other": box}),
                           (["eye", "eye"], {"eye": box}), (["other"], {"other": box}),
                           (["eye"], {"eye": [0, 0, 11, 2]}), (["eye"], {"eye": [False, 0, 2, 1]})):
            with self.subTest(ids=ids, boxes=boxes), self.assertRaises(ValueError):
                export_manifests(self.inventory, ids, boxes, self.dest)
            self.assertFalse(self.dest.exists())

    def test_stale_reference_and_destination(self):
        pair, = self.inventory.samples
        with self.assertRaises(ValueError):
            export_manifests(self.inventory, ["eye"], {"eye": [0, 0, 1, 1]}, self.root / "out")
        Image.new("L", (10, 8), 255).save(self.root / pair.mask)
        with self.assertRaisesRegex(ValueError, "changed"):
            export_manifests(self.inventory, ["eye"], {"eye": [0, 0, 1, 1]}, self.dest)

    def test_csv_write_failure_cleans_staging(self):
        from unittest.mock import patch

        before = set(self.base.iterdir())
        with patch("image_model.datasets.manifest_export.csv.writer", side_effect=OSError("disk error")):
            with self.assertRaisesRegex(OSError, "disk error"):
                export_manifests(self.inventory, ["eye"], {"eye": [0, 0, 10, 8]}, self.dest)
        self.assertEqual(set(self.base.iterdir()), before)

    def test_relative_destination_and_existing_output(self):
        from contextlib import chdir
        import csv

        with chdir(self.base):
            crops, refs = export_manifests(self.inventory, ["eye"], {"eye": [0, 0, 10, 8]}, Path("manifests"))
        row, = json.loads(crops.read_text())["samples"]
        self.assertEqual(set(row), {"id", "source", "eye_opening_box"})
        self.assertFalse(Path(row["source"]).is_absolute())
        with refs.open(newline="", encoding="utf-8") as stream:
            header, reference = csv.reader(stream)
        self.assertEqual(header, ["id", "mask"])
        self.assertEqual(reference, ["eye", "../synthetic/sum/sclera_mask/eye.png"])
        contents = crops.read_bytes(), refs.read_bytes()
        with self.assertRaises(ValueError):
            export_manifests(self.inventory, ["eye"], {"eye": [0, 0, 1, 1]}, self.dest)
        self.assertEqual((crops.read_bytes(), refs.read_bytes()), contents)

    def test_cli_invalid_id_and_changed_source(self):
        for directory in ("sum/image", "sum/sclera_mask"):
            (self.root / directory / "eye.png").rename(self.root / directory / "bad.id.png")
        inventory = inspect_pairs(self.root, foreground=255, mask_frame="oriented-source")
        with self.assertRaisesRegex(ValueError, "invalid id"):
            export_manifests(inventory, ["bad.id"], {"bad.id": [0, 0, 10, 8]}, self.dest)
        Image.new("RGB", (10, 8), "red").save(self.root / inventory.samples[0].image)
        with self.assertRaisesRegex(ValueError, "changed"):
            export_manifests(inventory, ["bad.id"], {"bad.id": [0, 0, 10, 8]}, self.dest)
        self.assertFalse(self.dest.exists())
