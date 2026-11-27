"""Subject-level split generation and leakage auditing for dataset manifests.

Subject identifiers must come from an explicit trusted mapping; they are never
derived from filenames or paths.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .ubiris_importer import Inventory


SCHEMA_VERSION = "1.0"
SPLITS = ("train", "validation", "test")
SPLIT_ALGORITHM = "subject-greedy-v2"
# Provisional target proportions; confirm these with the ERUA team before use.
DEFAULT_SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
REQUIRED_RECORD_FIELDS = (
    "record_id",
    "image_id",
    "image_path",
    "mask_path",
    "subject_id",
)
_IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class ManifestError(ValueError):
    """Raised when a manifest cannot safely be used to generate splits."""


@dataclass(frozen=True)
class OriginalSplitAudit:
    """Aggregate inventory of image files found in original split folders."""

    record_counts: dict[str, int]
    unique_basename_counts: dict[str, int]
    duplicate_record_counts: dict[str, int]
    pairwise_basename_overlaps: dict[str, int]
    basenames_in_multiple_splits: int

    def to_dict(self) -> dict[str, Any]:
        """Return aggregate counts only; never expose source basenames."""
        return asdict(self)

    def render(self) -> str:
        lines = ["Original split inventory"]
        for split in SPLITS:
            lines.append(
                f"  {split}: {self.record_counts[split]} records, "
                f"{self.unique_basename_counts[split]} unique basenames, "
                f"{self.duplicate_record_counts[split]} duplicate records"
            )
        lines.append(f"  basenames in multiple splits: {self.basenames_in_multiple_splits}")
        for pair, count in self.pairwise_basename_overlaps.items():
            lines.append(f"  {pair} overlap: {count}")
        return "\n".join(lines)


@dataclass(frozen=True)
class AuditReport:
    """Privacy-preserving aggregate results from a manifest audit."""

    schema_version: str | None
    total_records: int
    total_images: int
    unique_subjects: int
    images_by_split: dict[str, int]
    subjects_by_split: dict[str, int]
    exact_duplicate_images: int
    exact_duplicate_leakage_groups: int
    image_overlap_count: int
    subject_overlap_count: int
    passed: bool
    issues: tuple[str, ...]
    seed: int | None
    split_ratios: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        """Return a report containing aggregates only, never IDs or paths."""
        return asdict(self)

    def render(self) -> str:
        lines = [
            "Split leakage audit",
            f"  schema version: {self.schema_version or 'unavailable'}",
            f"  total records: {self.total_records}",
            f"  unique images: {self.total_images}",
            f"  unique subjects: {self.unique_subjects}",
        ]
        for split in SPLITS:
            lines.append(
                f"  {split}: {self.images_by_split[split]} images, "
                f"{self.subjects_by_split[split]} subjects"
            )
        lines.extend(
            [
                f"  exact duplicate images: {self.exact_duplicate_images}",
                f"  exact duplicate groups crossing splits: {self.exact_duplicate_leakage_groups}",
                f"  image overlaps: {self.image_overlap_count}",
                f"  subject overlaps: {self.subject_overlap_count}",
                f"  leakage-only result: {'PASS' if self.passed else 'FAIL'}",
                f"  seed: {self.seed if self.seed is not None else 'unavailable'}",
                f"  split ratios: {json.dumps(self.split_ratios, sort_keys=True) if self.split_ratios else 'unavailable'}",
            ]
        )
        if self.issues:
            lines.append("  issues:")
            lines.extend(f"    - {issue}" for issue in self.issues)
        return "\n".join(lines)


def analyze_original_split_dirs(
    split_directories: Mapping[str, str | Path],
) -> OriginalSplitAudit:
    """Count original split records, duplicate basenames, and pairwise overlaps.

    Only files with common image extensions are counted. Reports contain counts,
    not filenames, so archive identifiers are not disclosed.
    """
    if not isinstance(split_directories, Mapping) or set(split_directories) != set(SPLITS):
        raise ValueError("Provide exactly the train, validation, and test directories")

    names_by_split: dict[str, list[str]] = {}
    record_counts: dict[str, int] = {}
    unique_counts: dict[str, int] = {}
    duplicate_counts: dict[str, int] = {}
    for split in SPLITS:
        try:
            root = Path(split_directories[split])
            if not root.is_dir():
                raise ValueError
        except (OSError, TypeError, ValueError):
            raise ValueError(f"Original split directory is unavailable: {split}") from None
        if (root / "image").is_dir():
            root = root / "image"
        names = [
            path.stem.casefold()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        ]
        counts = Counter(names)
        names_by_split[split] = names
        record_counts[split] = len(names)
        unique_counts[split] = len(counts)
        duplicate_counts[split] = sum(count - 1 for count in counts.values())

    pair_overlaps: dict[str, int] = {}
    overlap_membership: Counter[str] = Counter()
    for names in names_by_split.values():
        overlap_membership.update(set(names))
    for index, left in enumerate(SPLITS):
        left_names = set(names_by_split[left])
        for right in SPLITS[index + 1 :]:
            pair_overlaps[f"{left}/{right}"] = len(left_names & set(names_by_split[right]))

    return OriginalSplitAudit(
        record_counts=record_counts,
        unique_basename_counts=unique_counts,
        duplicate_record_counts=duplicate_counts,
        pairwise_basename_overlaps=pair_overlaps,
        basenames_in_multiple_splits=sum(1 for count in overlap_membership.values() if count > 1),
    )


def generate_splits(
    manifest: Mapping[str, Any],
    *,
    seed: int = 42,
    ratios: Mapping[str, float] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Assign records by explicit subject ID and return a new manifest.

    Assignment is deterministic for a given manifest, seed, and ratios and is
    independent of input record order. The greedy balancing target is image
    count; subjects remain indivisible.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    normalized_ratios = _normalize_ratios(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    rows, issues, hashes = _inspect_manifest(manifest, base_dir=base_dir, require_split=False)
    if issues:
        raise ManifestError("Manifest cannot be split: " + "; ".join(issues))

    digest_subjects: dict[str, set[str]] = defaultdict(set)
    for row, digest in zip(rows, hashes):
        subject_id = _identifier(row.get("subject_id"))
        if digest and subject_id is not None:
            digest_subjects[digest].add(subject_id)
    if any(len(subjects) > 1 for subjects in digest_subjects.values()):
        raise ManifestError("Exact duplicate image content has conflicting subject identifiers")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject_id = _identifier(row.get("subject_id"))
        # _inspect_manifest has already rejected invalid subject identifiers.
        assert subject_id is not None
        groups[subject_id].append(dict(row))

    total_images = len(rows)
    targets = {split: total_images * normalized_ratios[split] for split in SPLITS}
    image_counts = {split: 0 for split in SPLITS}
    seed_text = str(seed)

    def tie_hash(value: str) -> str:
        return hashlib.sha256(f"{seed_text}\0{value}".encode("utf-8")).hexdigest()

    # Stable hashes use the seed only for tie-breaking; groups remain indivisible.
    subjects = sorted(groups, key=lambda subject: (-len(groups[subject]), tie_hash(subject), subject))
    split_priority = sorted(SPLITS, key=lambda split: (tie_hash(f"split:{split}"), split))
    split_rank = {split: rank for rank, split in enumerate(split_priority)}
    assignments: dict[str, str] = {}
    if len(subjects) < len(SPLITS):
        raise ManifestError("At least three independent subject groups are required")
    for index, subject in enumerate(subjects):
        size = len(groups[subject])
        empty = [name for name in SPLITS if image_counts[name] == 0]
        candidates = empty if len(subjects) - index == len(empty) else SPLITS
        split = min(
            candidates,
            key=lambda candidate: (
                sum(
                    ((image_counts[name] + (size if name == candidate else 0) - targets[name]) / total_images) ** 2
                    for name in SPLITS
                ),
                split_rank[candidate],
            ),
        )
        assignments[subject] = split
        image_counts[split] += size

    output_records = []
    for row in rows:
        subject_id = _identifier(row.get("subject_id"))
        assert subject_id is not None
        output_row = dict(row)
        output_row["split"] = assignments[subject_id]
        output_records.append(output_row)
    output_records.sort(key=lambda row: (row["record_id"], row["image_id"]))

    result = dict(manifest)
    result["schema_version"] = SCHEMA_VERSION
    result["records"] = output_records
    result["split_generation"] = {
        "algorithm": SPLIT_ALGORITHM,
        "seed": seed,
        "ratios": normalized_ratios,
    }
    return result


def audit_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path | None = None,
) -> AuditReport:
    """Audit manifest integrity, reproducibility metadata, and split leakage."""
    schema_version = manifest.get("schema_version") if isinstance(manifest, Mapping) else None
    rows, issues, hashes = _inspect_manifest(manifest, base_dir=base_dir, require_split=True)
    seed, ratios, metadata_issues = _validate_generation_metadata(manifest)
    issues.extend(metadata_issues)

    ids_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    subjects_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    image_splits: dict[str, set[str]] = defaultdict(set)
    subject_splits: dict[str, set[str]] = defaultdict(set)
    digest_splits: dict[str, set[str]] = defaultdict(set)
    digest_subjects: dict[str, set[str]] = defaultdict(set)
    image_subjects: dict[str, set[str]] = defaultdict(set)
    unique_subject_ids: set[str] = set()
    unique_image_ids: set[str] = set()

    for row, digest in zip(rows, hashes):
        image_id = _identifier(row.get("image_id"))
        subject_id = _identifier(row.get("subject_id"))
        split = row.get("split")
        if image_id is not None:
            unique_image_ids.add(image_id)
            if subject_id is not None:
                image_subjects[image_id].add(subject_id)
        if subject_id is not None:
            unique_subject_ids.add(subject_id)
        if isinstance(split, str) and split in SPLITS:
            if image_id is not None:
                ids_by_split[split].add(image_id)
                image_splits[image_id].add(split)
            if subject_id is not None:
                subjects_by_split[split].add(subject_id)
                subject_splits[subject_id].add(split)
            if digest is not None:
                digest_splits[digest].add(split)
                if subject_id is not None:
                    digest_subjects[digest].add(subject_id)

    image_overlaps = sum(1 for splits in image_splits.values() if len(splits) > 1)
    subject_overlaps = sum(1 for splits in subject_splits.values() if len(splits) > 1)
    duplicate_groups = Counter(digest for digest in hashes if digest is not None)
    exact_duplicates = sum(count - 1 for count in duplicate_groups.values() if count > 1)
    duplicate_leakage_groups = sum(
        1 for digest, count in duplicate_groups.items()
        if count > 1 and len(digest_splits.get(digest, set())) > 1
    )

    if image_overlaps:
        issues.append("Image identifiers overlap across splits")
    if subject_overlaps:
        issues.append("Subject identifiers overlap across splits")
    if duplicate_leakage_groups:
        issues.append("Exact duplicate image content overlaps across splits")
    if any(len(subjects) > 1 for subjects in digest_subjects.values()):
        issues.append("Exact duplicate image content has conflicting subject identifiers")
    if any(len(subjects) > 1 for subjects in image_subjects.values()):
        issues.append("An image identifier has conflicting subject identifiers")

    total_records = len(manifest.get("records", [])) if isinstance(manifest, Mapping) and isinstance(manifest.get("records"), list) else 0
    return AuditReport(
        schema_version=schema_version if isinstance(schema_version, str) else None,
        total_records=total_records,
        total_images=len(unique_image_ids),
        unique_subjects=len(unique_subject_ids),
        images_by_split={split: len(ids_by_split[split]) for split in SPLITS},
        subjects_by_split={split: len(subjects_by_split[split]) for split in SPLITS},
        exact_duplicate_images=exact_duplicates,
        exact_duplicate_leakage_groups=duplicate_leakage_groups,
        image_overlap_count=image_overlaps,
        subject_overlap_count=subject_overlaps,
        passed=not issues,
        issues=tuple(dict.fromkeys(issues)),
        seed=seed,
        split_ratios=ratios,
    )


def _validate_generation_metadata(
    manifest: Mapping[str, Any],
) -> tuple[int | None, dict[str, float] | None, list[str]]:
    if not isinstance(manifest, Mapping):
        return None, None, ["Split generation metadata is unavailable"]
    metadata = manifest.get("split_generation")
    if not isinstance(metadata, Mapping):
        return None, None, ["Split generation metadata is missing or is not an object"]

    issues: list[str] = []
    algorithm = metadata.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        issues.append("Split generation algorithm is missing or invalid")
    elif algorithm != SPLIT_ALGORITHM:
        issues.append("Split generation algorithm is unsupported")

    seed = metadata.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int):
        seed = None
        issues.append("Split generation seed is missing or invalid")

    try:
        ratios = _normalize_ratios(metadata.get("ratios"))
    except (TypeError, ValueError):
        ratios = None
        issues.append("Split generation ratios are missing or invalid")
    return seed, ratios, issues


def _normalize_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(ratios, Mapping):
        raise ValueError("Ratios must be an object")
    if set(ratios) != set(SPLITS):
        raise ValueError("Ratios must define exactly train, validation, and test")

    normalized: dict[str, float] = {}
    for split in SPLITS:
        value = ratios[split]
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError("Split ratios must be finite positive numbers")
        try:
            value = float(value)
        except (OverflowError, TypeError, ValueError):
            raise ValueError("Split ratios must be finite positive numbers") from None
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Split ratios must be finite positive numbers")
        normalized[split] = value
    if not math.isclose(sum(normalized.values()), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Split ratios must sum to 1")
    return normalized


def _identifier(value: Any) -> str | None:
    """Return only already-normalized, nonempty string identifiers."""
    if isinstance(value, str) and value and value == value.strip():
        return value
    return None


def _inspect_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path | None,
    require_split: bool,
) -> tuple[list[dict[str, Any]], list[str], list[str | None]]:
    issues: list[str] = []
    if not isinstance(manifest, Mapping):
        return [], ["Manifest must be an object"], []
    schema_version = manifest.get("schema_version")
    if not isinstance(schema_version, str) or schema_version != SCHEMA_VERSION:
        issues.append("Manifest schema version is missing or unsupported")
    records = manifest.get("records")
    if not isinstance(records, list):
        return [], issues + ["Manifest records must be a list"], []
    if not records:
        issues.append("Manifest records must not be empty")

    root: Path | None = None
    if base_dir is not None:
        try:
            root = Path(base_dir).resolve(strict=True)
        except (OSError, TypeError, ValueError):
            issues.append("Manifest base directory is invalid")

    rows: list[dict[str, Any]] = []
    hashes: list[str | None] = []
    seen_record_ids: set[str] = set()
    seen_image_ids: set[str] = set()
    duplicate_record_ids = False
    duplicate_image_ids = False
    for record in records:
        if not isinstance(record, Mapping):
            issues.append("Manifest contains a record that is not an object")
            continue
        row = dict(record)
        absent = [field for field in REQUIRED_RECORD_FIELDS if field not in row]
        if absent:
            issues.append("Manifest records are missing required fields")
        for field in ("record_id", "image_id", "subject_id"):
            if _identifier(row.get(field)) is None:
                label = "subject identifiers" if field == "subject_id" else f"{field} values"
                issues.append(f"Manifest contains empty or invalid {label}")

        record_id = _identifier(row.get("record_id"))
        if record_id is not None:
            if record_id in seen_record_ids:
                duplicate_record_ids = True
            seen_record_ids.add(record_id)
        image_id = _identifier(row.get("image_id"))
        if image_id is not None:
            if image_id in seen_image_ids:
                duplicate_image_ids = True
            seen_image_ids.add(image_id)

        resolved_paths: dict[str, Path] = {}
        for path_field in ("image_path", "mask_path"):
            raw_path = row.get(path_field)
            if not isinstance(raw_path, str) or not raw_path:
                issues.append("Manifest records are missing required local paths")
                continue
            try:
                if "\x00" in raw_path:
                    raise ValueError
                path = Path(raw_path)
                if root is not None:
                    if path.anchor or ".." in path.parts:
                        raise ValueError
                    path = (root / path).resolve()
                    if not path.is_relative_to(root):
                        raise ValueError
                if not path.is_file():
                    issues.append("Manifest references a missing image or mask file")
                else:
                    resolved_paths[path_field] = path
            except (OSError, TypeError, ValueError):
                issues.append("Manifest contains an invalid image or mask path")

        for field in ("split", "original_split"):
            if field not in row:
                if field == "split" and require_split:
                    issues.append("Manifest record is missing an assigned split")
                continue
            value = row.get(field)
            if value is None and field == "original_split":
                continue
            if not isinstance(value, str) or value not in SPLITS:
                if field == "split" or value is not None:
                    issues.append("Manifest contains an invalid split value")

        digest = None
        image_path = resolved_paths.get("image_path")
        if image_path is not None:
            try:
                digest = _sha256(image_path)
                if "image_sha256" in row and digest != row["image_sha256"]:
                    issues.append("Image hash differs from inventory")
                if "mask_sha256" in row and "mask_path" in resolved_paths:
                    if _sha256(resolved_paths["mask_path"]) != row["mask_sha256"]:
                        issues.append("Mask hash differs from inventory")
            except (OSError, ValueError):
                issues.append("Image file could not be hashed")
        rows.append(row)
        hashes.append(digest)

    if duplicate_record_ids:
        issues.append("Duplicate record identifiers are present")
    if duplicate_image_ids:
        issues.append("Duplicate image identifiers are present")
    return rows, list(dict.fromkeys(issues)), hashes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory_split(inventory: Inventory, mapping: Mapping[str, str], *,
                    mapping_status: str, seed: int = 42,
                    ratios: Mapping[str, float] | None = None) -> dict[str, Any]:
    """Build a reproducible split from a fresh inventory and an external mapping.

    Verified status is an explicit custodian assertion. Keep the evidence private.
    """
    if mapping_status not in {"verified", "provisional"}:
        raise ManifestError("Declare mapping status")
    if set(mapping) != {pair.id for pair in inventory.samples}:
        raise ManifestError("Mapping must cover exactly the eligible inventory")
    records = []
    for pair in inventory.samples:
        records.append(dict(record_id=pair.id, image_id=pair.id, subject_id=mapping[pair.id],
                            image_path=pair.image.as_posix(), mask_path=pair.mask.as_posix(),
                            image_sha256=pair.image_sha256, mask_sha256=pair.mask_sha256,
                            width=pair.width, height=pair.height))
    manifest = generate_splits({"schema_version": SCHEMA_VERSION, "records": records},
                               base_dir=inventory.root, seed=seed, ratios=ratios)
    manifest["mapping_status"] = mapping_status
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest["snapshot_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    return manifest


def validate_snapshot(manifest: Mapping[str, Any], inventory: Inventory, mapping: Mapping[str, str],
                    *, mapping_status: str, seed: int = 42,
                    ratios: Mapping[str, float] | None = None) -> AuditReport:
    """Validate against independent trusted inputs and configuration before freezing."""
    expected = inventory_split(inventory, mapping, mapping_status=mapping_status,
                               seed=seed, ratios=ratios)
    if json.dumps(expected, sort_keys=True) != json.dumps(manifest, sort_keys=True):
        raise ManifestError("Manifest differs from regenerated trusted inventory split")
    report = audit_manifest(expected, base_dir=inventory.root)
    if not report.passed:
        raise ManifestError("Snapshot failed file integrity or leakage checks")
    return report


def validate_frozen(manifest: Mapping[str, Any], inventory: Inventory, mapping: Mapping[str, str],
                    *, mapping_status: str, seed: int = 42,
                    ratios: Mapping[str, float] | None = None) -> AuditReport:
    """Require independently verified mapping before accepting a final partition."""
    if mapping_status != "verified":
        raise ManifestError("Freezing requires a verified external subject mapping")
    return validate_snapshot(manifest, inventory, mapping, mapping_status=mapping_status,
                             seed=seed, ratios=ratios)
