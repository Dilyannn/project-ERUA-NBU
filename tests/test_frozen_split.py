import copy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
from image_model.datasets.ubiris_importer import inspect_pairs
from image_model.datasets.split_audit import (
    ManifestError, analyze_original_split_dirs, audit_manifest, inventory_split, validate_frozen,
)


class FrozenTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for folder, mode in (("image", "RGB"), ("mask", "L")):
            directory = self.root / folder
            directory.mkdir()
            for n in range(3):
                Image.new(mode, (4, 2), n if mode == "RGB" else 255).save(directory / f"s{n}.png")
        self.inv = inspect_pairs(self.root, images="image", masks="mask", foreground=255,
                                 mask_frame="oriented-source")
        self.mapping = {p.id: p.id for p in self.inv.samples}
        self.m = inventory_split(self.inv, self.mapping, mapping_status="verified")

    def check(self, m):
        return validate_frozen(m, self.inv, self.mapping, mapping_status="verified")

    def test_exact_regeneration(self):
        self.assertTrue(self.check(self.m).passed)
        self.assertEqual(len({r["split"] for r in self.m["records"]}), 3)
        for action in ("drop", "duplicate", "unexpected", "move", "seed"):
            m = copy.deepcopy(self.m)
            rows = m["records"]
            if action == "drop":
                rows.pop()
            elif action == "duplicate":
                rows.append(rows[0])
            elif action == "unexpected":
                rows[0]["image_id"] = "extra"
            elif action == "move":
                rows[0]["split"] = rows[1]["split"]
            else:
                m["split_generation"]["seed"] += 1
            with self.subTest(action=action), self.assertRaises(ManifestError):
                self.check(m)

    def test_content_changes(self):
        for path in (self.inv.samples[0].image, self.inv.samples[0].mask):
            p = self.root / path
            original = p.read_bytes()
            p.write_bytes(original + b"change")
            with self.assertRaises(ManifestError):
                self.check(self.m)
            p.write_bytes(original)

    def test_mapping_requirements(self):
        with self.assertRaises(ManifestError):
            validate_frozen(self.m, self.inv, self.mapping, mapping_status="provisional")
        for mapping in ({}, dict.fromkeys(self.mapping, "only")):
            with self.assertRaises(ManifestError):
                inventory_split(self.inv, mapping, mapping_status="verified")

    def test_paths_and_split_roots(self):
        for path in (str(self.root / "image/s0.png"), "image/../image/s0.png"):
            m = copy.deepcopy(self.m)
            m["records"][0]["image_path"] = path
            self.assertFalse(audit_manifest(m, base_dir=self.root).passed)
        dirs = dict.fromkeys(("train", "validation", "test"), self.root)
        self.assertEqual(analyze_original_split_dirs(dirs).record_counts["train"], 3)
