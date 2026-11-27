from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from image_model.datasets.split_audit import (  # noqa: E402
    DEFAULT_SPLIT_RATIOS,
    SCHEMA_VERSION,
    SPLIT_ALGORITHM,
    ManifestError,
    analyze_original_split_dirs,
    audit_manifest,
    generate_splits,
)


class SplitAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_manifest(self, subjects: int = 8, images_per_subject: int = 2) -> dict:
        records = []
        for subject_number in range(subjects):
            subject_id = f"synthetic-subject-{subject_number:02d}"
            for image_number in range(images_per_subject):
                record_id = f"record-{subject_number:02d}-{image_number:02d}"
                image_id = f"image-{subject_number:02d}-{image_number:02d}"
                image_path = self.root / "images" / f"{image_id}.png"
                mask_path = self.root / "masks" / f"{image_id}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                image_path.write_bytes(f"image-bytes-{record_id}".encode())
                mask_path.write_bytes(f"mask-bytes-{record_id}".encode())
                records.append(
                    {
                        "record_id": record_id,
                        "image_id": image_id,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "subject_id": subject_id,
                    }
                )
        return {"schema_version": SCHEMA_VERSION, "records": records}

    def assign_splits(self, manifest: dict, split_by_record: dict[str, str] | None = None) -> dict:
        split_by_record = split_by_record or {}
        for row in manifest["records"]:
            row["split"] = split_by_record.get(row["record_id"], "train")
        manifest["split_generation"] = {
            "algorithm": SPLIT_ALGORITHM,
            "seed": 42,
            "ratios": dict(DEFAULT_SPLIT_RATIOS),
        }
        return manifest

    def test_subject_grouping_keeps_eyes_and_sessions_together(self) -> None:
        manifest = self.make_manifest(subjects=5, images_per_subject=4)
        for index, row in enumerate(manifest["records"]):
            row["eye"] = "left" if index % 2 == 0 else "right"
            row["session"] = "session-1" if index % 4 < 2 else "session-2"

        generated = generate_splits(manifest, seed=19)
        split_by_subject: dict[str, set[str]] = {}
        for row in generated["records"]:
            split_by_subject.setdefault(row["subject_id"], set()).add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in split_by_subject.values()))
        self.assertEqual(len(split_by_subject), 5)
        self.assertTrue(audit_manifest(generated).passed)

    def test_subject_leakage_fails_audit(self) -> None:
        manifest = self.make_manifest(subjects=1, images_per_subject=2)
        first_id, second_id = [row["record_id"] for row in manifest["records"]]
        self.assign_splits(manifest, {first_id: "train", second_id: "test"})
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertEqual(report.subject_overlap_count, 1)

    def test_image_identifier_leakage_fails_audit(self) -> None:
        manifest = self.make_manifest(subjects=2, images_per_subject=1)
        manifest["records"][1]["image_id"] = manifest["records"][0]["image_id"]
        first_id, second_id = [row["record_id"] for row in manifest["records"]]
        self.assign_splits(manifest, {first_id: "train", second_id: "test"})
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertEqual(report.image_overlap_count, 1)
        self.assertTrue(any("Duplicate image identifiers" in issue for issue in report.issues))

    def test_exact_duplicate_content_crossing_splits_fails(self) -> None:
        manifest = self.make_manifest(subjects=1, images_per_subject=2)
        duplicate_path = self.root / "images" / "same-bytes-different-name.png"
        duplicate_path.write_bytes(Path(manifest["records"][0]["image_path"]).read_bytes())
        manifest["records"][1]["image_path"] = str(duplicate_path)
        first_id, second_id = [row["record_id"] for row in manifest["records"]]
        self.assign_splits(manifest, {first_id: "train", second_id: "test"})
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertEqual(report.exact_duplicate_images, 1)
        self.assertEqual(report.exact_duplicate_leakage_groups, 1)

    def test_exact_duplicates_in_one_split_are_reported_without_leakage(self) -> None:
        manifest = self.make_manifest(subjects=1, images_per_subject=2)
        duplicate_path = self.root / "images" / "same-bytes-different-name.png"
        duplicate_path.write_bytes(Path(manifest["records"][0]["image_path"]).read_bytes())
        manifest["records"][1]["image_path"] = str(duplicate_path)
        report = audit_manifest(self.assign_splits(manifest))
        self.assertTrue(report.passed)
        self.assertEqual(report.exact_duplicate_images, 1)
        self.assertEqual(report.exact_duplicate_leakage_groups, 0)

    def test_exact_duplicate_with_conflicting_subjects_fails_audit_and_generation(self) -> None:
        manifest = self.make_manifest(subjects=2, images_per_subject=1)
        duplicate_path = self.root / "images" / "copy-with-different-name.png"
        duplicate_path.write_bytes(Path(manifest["records"][0]["image_path"]).read_bytes())
        manifest["records"][1]["image_path"] = str(duplicate_path)
        report = audit_manifest(self.assign_splits(manifest))
        self.assertFalse(report.passed)
        self.assertTrue(any("conflicting subject" in issue for issue in report.issues))
        with self.assertRaisesRegex(ManifestError, "conflicting subject"):
            generate_splits(manifest)

    def test_original_split_overlap_and_duplicate_basename_counts(self) -> None:
        train = self.root / "original" / "train"
        validation = self.root / "original" / "validation"
        test = self.root / "original" / "test"
        for directory in (train, validation, test):
            directory.mkdir(parents=True)
        (train / "shared.png").write_bytes(b"a")
        (train / "nested").mkdir()
        (train / "nested" / "shared.png").write_bytes(b"b")
        (validation / "val-only.png").write_bytes(b"c")
        (test / "shared.png").write_bytes(b"d")
        audit = analyze_original_split_dirs(
            {"train": train, "validation": validation, "test": test}
        )
        self.assertEqual(audit.record_counts, {"train": 2, "validation": 1, "test": 1})
        self.assertEqual(audit.unique_basename_counts["train"], 1)
        self.assertEqual(audit.duplicate_record_counts["train"], 1)
        self.assertEqual(audit.pairwise_basename_overlaps["train/test"], 1)
        self.assertEqual(audit.basenames_in_multiple_splits, 1)
        self.assertNotIn("shared.png", audit.render())

    def test_original_split_requires_all_three_split_directories(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly the train, validation, and test"):
            analyze_original_split_dirs({"train": self.root})

    def test_known_original_split_shape_and_eight_train_test_overlaps(self) -> None:
        directories = {
            split: self.root / "known-shape" / split
            for split in ("train", "validation", "test")
        }
        for directory in directories.values():
            directory.mkdir(parents=True)
        shared = [f"shared-{index:02d}.png" for index in range(8)]
        for name in shared:
            (directories["train"] / name).write_bytes(b"train")
            (directories["test"] / name).write_bytes(b"test")
        for index in range(272):
            (directories["train"] / f"train-only-{index:03d}.png").write_bytes(b"train")
            (directories["test"] / f"test-only-{index:03d}.png").write_bytes(b"test")
        for index in range(140):
            (directories["validation"] / f"validation-only-{index:03d}.png").write_bytes(b"val")

        audit = analyze_original_split_dirs(directories)
        self.assertEqual(audit.record_counts, {"train": 280, "validation": 140, "test": 280})
        self.assertEqual(audit.pairwise_basename_overlaps["train/test"], 8)
        self.assertEqual(audit.basenames_in_multiple_splits, 8)

    def test_same_seed_and_input_order_produce_same_assignments(self) -> None:
        manifest = self.make_manifest(subjects=12, images_per_subject=2)
        first = generate_splits(manifest, seed=123)
        reversed_manifest = copy.deepcopy(manifest)
        reversed_manifest["records"].reverse()
        second = generate_splits(reversed_manifest, seed=123)
        first_assignments = {row["record_id"]: row["split"] for row in first["records"]}
        second_assignments = {row["record_id"]: row["split"] for row in second["records"]}
        self.assertEqual(first_assignments, second_assignments)
        self.assertEqual(first["split_generation"], second["split_generation"])

    def test_custom_seed_and_ratios_are_recorded(self) -> None:
        manifest = self.make_manifest()
        generated = generate_splits(
            manifest,
            seed=7,
            ratios={"train": 0.6, "validation": 0.2, "test": 0.2},
        )
        report = audit_manifest(generated)
        self.assertTrue(report.passed)
        self.assertEqual(report.seed, 7)
        self.assertEqual(report.split_ratios, {"train": 0.6, "validation": 0.2, "test": 0.2})
        self.assertEqual(sum(report.images_by_split.values()), report.total_images)
        self.assertEqual(sum(report.subjects_by_split.values()), report.unique_subjects)

    def test_invalid_seed_values_are_rejected(self) -> None:
        for seed in (True, False, 3.5, "42", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_splits(self.make_manifest(), seed=seed)

    def test_invalid_ratio_values_are_rejected(self) -> None:
        invalid_ratios = (
            {"train": 0.7, "validation": 0.3},
            {"train": 0.7, "validation": 0.15, "test": 0.15, "other": 0.0},
            {"train": 0.0, "validation": 0.5, "test": 0.5},
            {"train": -0.1, "validation": 0.6, "test": 0.5},
            {"train": float("nan"), "validation": 0.2, "test": 0.8},
            {"train": float("inf"), "validation": 0.2, "test": 0.8},
            {"train": True, "validation": 0.2, "test": 0.8},
            {"train": "0.7", "validation": 0.15, "test": 0.15},
            {"train": 0.5, "validation": 0.2, "test": 0.2},
            {"train": 10**10000, "validation": 0.1, "test": 0.1},
        )
        for ratios in invalid_ratios:
            with self.subTest(ratios=ratios), self.assertRaises(ValueError):
                generate_splits(self.make_manifest(), ratios=ratios)

    def test_empty_and_missing_subject_identifiers_are_rejected(self) -> None:
        empty = self.make_manifest()
        empty["records"][0]["subject_id"] = ""
        self.assertFalse(audit_manifest(self.assign_splits(empty)).passed)
        with self.assertRaises(ManifestError):
            generate_splits(empty)

        missing = self.make_manifest()
        del missing["records"][0]["subject_id"]
        self.assertFalse(audit_manifest(self.assign_splits(missing)).passed)
        with self.assertRaises(ManifestError):
            generate_splits(missing)

    def test_duplicate_record_identifiers_are_rejected(self) -> None:
        manifest = self.assign_splits(self.make_manifest(subjects=2, images_per_subject=1))
        manifest["records"][1]["record_id"] = manifest["records"][0]["record_id"]
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertTrue(any("Duplicate record identifiers" in issue for issue in report.issues))

    def test_invalid_identifier_types_fail_without_crashing(self) -> None:
        invalid_values = (None, 17, [], {}, True)
        for field in ("image_id", "subject_id", "record_id"):
            for value in invalid_values:
                with self.subTest(field=field, value=value):
                    manifest = self.assign_splits(self.make_manifest())
                    manifest["records"][0][field] = value
                    report = audit_manifest(manifest)
                    self.assertFalse(report.passed)
                    self.assertTrue(report.issues)

    def test_malformed_record_and_unhashable_split_fail_without_crashing(self) -> None:
        malformed_record = self.assign_splits(self.make_manifest())
        malformed_record["records"][0] = None
        self.assertFalse(audit_manifest(malformed_record).passed)

        invalid_split = self.assign_splits(self.make_manifest())
        invalid_split["records"][0]["split"] = []
        self.assertFalse(audit_manifest(invalid_split).passed)

    def test_missing_manifest_field_is_rejected(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        del manifest["records"][0]["mask_path"]
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertTrue(any("required" in issue for issue in report.issues))

    def test_missing_local_file_is_rejected(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        manifest["records"][0]["mask_path"] = str(self.root / "missing-mask.png")
        self.assertFalse(audit_manifest(manifest).passed)
        with self.assertRaises(ManifestError):
            generate_splits(manifest)

    def test_invalid_local_path_is_reported_without_crashing(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        manifest["records"][0]["image_path"] = "bad\x00path.png"
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertTrue(any("path" in issue.lower() for issue in report.issues))

    def test_invalid_split_value_and_missing_assigned_split_are_rejected(self) -> None:
        invalid = self.assign_splits(self.make_manifest())
        invalid["records"][0]["split"] = "holdout"
        self.assertFalse(audit_manifest(invalid).passed)

        missing = self.assign_splits(self.make_manifest())
        del missing["records"][0]["split"]
        report = audit_manifest(missing)
        self.assertFalse(report.passed)
        self.assertTrue(any("missing an assigned split" in issue for issue in report.issues))

    def test_unsupported_schema_version_is_rejected(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        manifest["schema_version"] = "99.0"
        self.assertFalse(audit_manifest(manifest).passed)
        with self.assertRaises(ManifestError):
            generate_splits(manifest)

    def test_missing_split_generation_metadata_is_rejected(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        del manifest["split_generation"]
        report = audit_manifest(manifest)
        self.assertFalse(report.passed)
        self.assertIsNone(report.seed)
        self.assertIsNone(report.split_ratios)
        self.assertTrue(any("metadata is missing" in issue for issue in report.issues))

    def test_malformed_split_generation_metadata_is_rejected(self) -> None:
        valid = {"algorithm": SPLIT_ALGORITHM, "seed": 42, "ratios": dict(DEFAULT_SPLIT_RATIOS)}
        invalid_metadata = (
            None,
            [],
            {"seed": 42, "ratios": dict(DEFAULT_SPLIT_RATIOS)},
            {**valid, "algorithm": None},
            {**valid, "algorithm": "unknown-v1"},
            {"algorithm": SPLIT_ALGORITHM, "ratios": dict(DEFAULT_SPLIT_RATIOS)},
            {**valid, "seed": None},
            {**valid, "seed": True},
            {**valid, "seed": 1.5},
            {**valid, "seed": "42"},
            {"algorithm": SPLIT_ALGORITHM, "seed": 42},
            {**valid, "ratios": None},
            {**valid, "ratios": []},
            {**valid, "ratios": {"train": 0.7, "validation": 0.3}},
            {**valid, "ratios": {"train": 0.7, "validation": 0.15, "test": 0.15, "extra": 0.0}},
            {**valid, "ratios": {"train": 0.0, "validation": 0.5, "test": 0.5}},
            {**valid, "ratios": {"train": float("nan"), "validation": 0.2, "test": 0.8}},
            {**valid, "ratios": {"train": float("inf"), "validation": 0.2, "test": 0.8}},
            {**valid, "ratios": {"train": 0.5, "validation": 0.2, "test": 0.2}},
        )
        for metadata in invalid_metadata:
            with self.subTest(metadata=metadata):
                manifest = self.assign_splits(self.make_manifest())
                manifest["split_generation"] = metadata
                report = audit_manifest(manifest)
                self.assertFalse(report.passed)
                self.assertTrue(report.issues)

    def test_audit_report_does_not_include_subject_ids_or_paths(self) -> None:
        manifest = self.assign_splits(self.make_manifest())
        report = audit_manifest(manifest)
        output = report.render() + str(report.to_dict())
        self.assertNotIn("synthetic-subject", output)
        self.assertNotIn(str(self.root), output)
        self.assertNotIn("image-00-00", output)


if __name__ == "__main__":
    unittest.main()
