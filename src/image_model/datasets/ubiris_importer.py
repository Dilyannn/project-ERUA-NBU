"""Inspect a private, flat pair of source directories without modifying their files."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import struct
from typing import Literal

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Pair:
    id: str
    image: Path
    mask: Path
    image_sha256: str
    mask_sha256: str
    width: int
    height: int


@dataclass(frozen=True)
class Exclusion:
    id: str
    paths: tuple[Path, ...]
    reason: str


@dataclass(frozen=True)
class Inventory:
    root: Path
    samples: tuple[Pair, ...]
    exclusions: tuple[Exclusion, ...]
    duplicate_groups: tuple[tuple[str, ...], ...]


def _index(root: Path, directory: str) -> dict[str, list[Path]]:
    folder = (root / directory).resolve()
    if Path(directory).anchor or ".." in Path(directory).parts or not folder.is_relative_to(root):
        raise ValueError("Input directory must be inside the dataset root.")
    groups = defaultdict(list)
    for path in sorted(folder.iterdir(), key=lambda p: p.name):
        groups[path.stem.casefold()].append(path.relative_to(root))
    return groups


def _inspect(data: bytes, role: str, foreground: int) -> tuple[tuple[int, int], str]:
    with Image.open(BytesIO(data)) as probe:
        probe.verify()
    with Image.open(BytesIO(data)) as image:
        if image.format not in {"PNG", "JPEG", "BMP", "TIFF"} or getattr(image, "n_frames", 1) != 1:
            raise ValueError(f"unsupported {role} format or frame count")
        orientation = image.getexif().get(274, 1)
        if role == "image":
            if image.mode != "RGB" or orientation not in range(1, 9):
                raise ValueError("unsupported source mode or EXIF orientation")
            return ImageOps.exif_transpose(image).size, ""
        if orientation != 1:
            raise ValueError("mask requires orientation review")
        if image.mode not in {"1", "L"}:
            raise ValueError("unsupported mask mode")
        pixels = np.asarray(image)
        if image.mode == "L" and not np.isin(pixels, (0, foreground)).all():
            raise ValueError("mask values outside configured binary encoding")
        return image.size, hashlib.sha256((pixels != 0).tobytes()).hexdigest()


def _inspect_member(root: Path, path: Path, role: str, foreground: int) -> tuple[tuple[int, int], str, str]:
    absolute = root / path
    if not absolute.resolve().is_relative_to(root) or not absolute.is_file():
        raise ValueError(f"invalid {role} path")
    try:
        data = absolute.read_bytes()
        size, annotation = _inspect(data, role, foreground)
        return size, annotation, hashlib.sha256(data).hexdigest()
    except (OSError, EOFError, SyntaxError, struct.error, Image.DecompressionBombError) as error:
        raise ValueError(f"unreadable {role}: {type(error).__name__}") from error


def inspect_pairs(root: Path, *, foreground: Literal[1, 255],
                  mask_frame: Literal["oriented-source"], images: str = "sum/image",
                  masks: str = "sum/sclera_mask") -> Inventory:
    """Require caller-confirmed oriented masks and white/positive foreground.

    One-bit masks use their logical values. For L masks foreground is exact.
    Paths in records are root-relative. Only exact photo bytes define duplicates.
    Duplicate groups include unambiguous readable photos with excluded masks.
    Only comparable valid annotations can establish a label conflict.
    Both directories are flat. An unreadable directory fails the whole scan.
    """
    if type(foreground) is not int or foreground not in (1, 255) or mask_frame != "oriented-source":
        raise ValueError("Unsupported mask encoding or coordinate frame.")
    root = root.resolve(strict=True)
    if (root / images).resolve() == (root / masks).resolve():
        raise ValueError("Image and mask directories must be distinct.")
    photos, labels = _index(root, images), _index(root, masks)
    samples: list[Pair] = []
    exclusions = []
    by_hash = defaultdict(list)
    annotation = {}
    for key in sorted(photos.keys() | labels.keys()):
        sources, references = photos.get(key, []), labels.get(key, [])
        paths = tuple(sources + references)
        name = paths[0].stem
        try:
            if len(sources) > 1:
                raise ValueError("ambiguous stem")
            if not sources:
                raise ValueError("missing image")
            image = sources[0]
            source_size, _, image_hash = _inspect_member(root, image, "image", foreground)
            by_hash[image_hash].append(name)
            if len(references) > 1:
                raise ValueError("ambiguous stem")
            if not references:
                raise ValueError("missing mask")
            mask = references[0]
            if image.stem != mask.stem:
                raise ValueError("inconsistent stem spelling")
            mask_size, digest, mask_hash = _inspect_member(root, mask, "mask", foreground)
            if source_size != mask_size:
                raise ValueError("oriented source and mask dimensions differ")
            pair = Pair(name, image, mask, image_hash, mask_hash, *source_size)
        except (ValueError, OSError) as error:
            exclusions.append(Exclusion(name, paths, str(error)))
            continue
        samples.append(pair)
        annotation[name] = digest

    duplicates = tuple(sorted(tuple(sorted(ids)) for ids in by_hash.values() if len(ids) > 1))
    conflicts = set()
    for group in duplicates:
        known_labels = {annotation[name] for name in group if name in annotation}
        if len(known_labels) > 1:
            conflicts.update(group)
    for pair in samples:
        if pair.id in conflicts:
            exclusions.append(Exclusion(pair.id, (pair.image, pair.mask), "conflicting duplicate annotations"))
    return Inventory(root, tuple(sorted((p for p in samples if p.id not in conflicts), key=lambda p: p.id)),
                     tuple(sorted(exclusions, key=lambda e: e.id)), duplicates)
