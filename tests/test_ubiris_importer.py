import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from PIL import Image

from image_model.datasets.ubiris_importer import inspect_pairs


class ImportTest(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.images = self.root / "sum/image"
        self.masks = self.root / "sum/sclera_mask"
        self.images.mkdir(parents=True)
        self.masks.mkdir()

    def pair(self, name, label=255, orientation=1):
        exif = Image.Exif()
        exif[274] = orientation
        Image.new("RGB", (6, 4)).save(self.images / f"{name}.png", exif=exif)
        size = (4, 6) if orientation == 6 else (6, 4)
        Image.new("L", size, label).save(self.masks / f"{name}.png")

    def scan(self, foreground=255):
        return inspect_pairs(self.root, foreground=foreground, mask_frame="oriented-source")

    def test_duplicate_labels_and_order(self):
        self.pair("z")
        self.pair("A")
        Image.new("1", (6, 4), 1).save(self.masks / "A.png")
        result = self.scan()
        self.assertEqual(result, self.scan())
        self.assertEqual([p.id for p in result.samples], ["A", "z"])
        self.assertEqual(result.duplicate_groups, (("A", "z"),))
        self.assertEqual(result.samples[0].image_sha256, hashlib.sha256((self.images / "A.png").read_bytes()).hexdigest())
        self.pair("b", 0)
        result = self.scan()
        self.assertFalse(result.samples)
        self.assertEqual([e.reason for e in result.exclusions], ["conflicting duplicate annotations"] * 3)

    def test_missing_ambiguous_corrupt_unsupported(self):
        for name in ("orphan", "missing", "bad", "double", "unsupported"):
            self.pair(name)
        (self.images / "orphan.png").unlink()
        (self.masks / "missing.png").unlink()
        (self.images / "bad.png").write_bytes(b"bad")
        Image.new("RGB", (6, 4)).save(self.images / "DOUBLE.bmp")
        Image.new("RGB", (6, 4)).save(self.masks / "unsupported.png")
        reasons = {e.id.lower(): e.reason for e in self.scan().exclusions}
        self.assertEqual(reasons["orphan"], "missing image")
        self.assertEqual(reasons["missing"], "missing mask")
        self.assertIn("unreadable image", reasons["bad"])
        self.assertEqual(reasons["double"], "ambiguous stem")
        self.assertEqual(reasons["unsupported"], "unsupported mask mode")

    def test_orientation_dimensions_encoding(self):
        self.pair("eye", 1, 6)
        self.assertIn("encoding", self.scan().exclusions[0].reason)
        pair, = self.scan(1).samples
        self.assertEqual((pair.width, pair.height), (4, 6))
        Image.new("L", (6, 4)).save(self.masks / "eye.png")
        self.assertIn("dimensions", self.scan().exclusions[0].reason)
        self.pair("eye", 42)
        self.assertIn("encoding", self.scan().exclusions[0].reason)
        with self.assertRaises(ValueError):
            self.scan(42)

    def test_duplicate_audit_includes_unusable_masks(self):
        for name in ("a", "b", "c"):
            self.pair(name)
        (self.masks / "b.png").unlink()
        (self.masks / "c.png").write_bytes(b"corrupt")
        result = self.scan()
        self.assertEqual(result.duplicate_groups, (("a", "b", "c"),))
        self.assertEqual([p.id for p in result.samples], ["a"])
        self.assertEqual([e.id for e in result.exclusions], ["b", "c"])
        self.pair("d", 0)
        result = self.scan()
        self.assertEqual(result.duplicate_groups, (("a", "b", "c", "d"),))
        self.assertFalse(result.samples)
        self.assertEqual([e.id for e in result.exclusions], ["a", "b", "c", "d"])

    def test_mask_pixels_agree_with_evaluator(self):
        from image_model.evaluation import read_mask

        self.pair("eye")
        path = self.masks / "eye.png"
        for mode, foreground in (("1", 1), ("L", 1), ("L", 255)):
            with self.subTest(mode=mode, foreground=foreground):
                mask = Image.new(mode, (6, 4))
                mask.putpixel((3, 1), foreground)
                mask.save(path)
                before = path.read_bytes()
                pair, = self.scan(foreground).samples
                actual = read_mask(self.root / pair.mask)
                self.assertTrue(actual[1, 3])
                self.assertEqual(actual.sum(), 1)
                self.assertEqual(pair.mask_sha256, hashlib.sha256(before).hexdigest())
                self.assertEqual(path.read_bytes(), before)

    def test_mask_orientation_unsupported_format_and_input_paths(self):
        self.pair("eye")
        tag = Image.Exif()
        tag[274] = 2
        Image.new("L", (6, 4)).save(self.masks / "eye.png", exif=tag)
        self.assertIn("orientation", self.scan().exclusions[0].reason)
        Image.new("RGB", (6, 4)).save(self.images / "eye.png", format="GIF")
        self.assertIn("unsupported image format", self.scan().exclusions[0].reason)
        for directory in (str(self.images), "sum/../sum/image"):
            with self.assertRaises(ValueError):
                inspect_pairs(self.root, images=directory, foreground=1, mask_frame="oriented-source")
