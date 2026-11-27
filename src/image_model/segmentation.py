"""Shared crop handling, mask post-processing, and source-coordinate output."""

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageOps


ORIENTATION = "exif-transposed"
HORIZONTAL_EYE_PADDING = 0.10
VERTICAL_EYE_PADDING = 0.35


@dataclass(frozen=True)
class MaskSettings:
    threshold: float
    max_components: int
    max_hole_fraction: float
    closing_kernel: tuple[int, int]


@dataclass(frozen=True)
class Sample:
    name: str
    source: Path
    crop: Path | None
    box: tuple[int, int, int, int]
    box_is_eye_opening: bool = False


def read_manifest(path: Path) -> list[Sample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("orientation") != ORIENTATION:
        raise ValueError(f"Manifest orientation must be {ORIENTATION!r}.")
    rows = data.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must contain a nonempty samples list.")

    samples = []
    seen = set()
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError(f"Sample #{index} must be an object.")
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name):
            raise ValueError(f"Sample #{index} has an invalid id.")
        if name.casefold() in seen:
            raise ValueError(f"Duplicate sample id: {name!r}.")
        seen.add(name.casefold())
        opening = "eye_opening_box" in row
        if ("box" in row) == opening:
            raise ValueError(f"{name}: provide exactly one of box or eye_opening_box.")
        box_key = "eye_opening_box" if opening else "box"
        box = row[box_key]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) is not int for value in box)
        ):
            raise ValueError(f"{name}: {box_key} must be four integers [left, top, right, bottom].")
        source, crop = row.get("source"), row.get("crop")
        if not isinstance(source, str) or not source:
            raise ValueError(f"{name}: source must be an image path.")
        if "crop" in row and (not isinstance(crop, str) or not crop):
            raise ValueError(f"{name}: crop must be an image path when supplied.")
        if crop and opening:
            raise ValueError(f"{name}: a pre-made crop requires explicit box coordinates.")
        samples.append(Sample(name, path.parent / source, path.parent / crop if crop else None, tuple(box), opening))
    return samples


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_images(sample: Sample) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    source = open_rgb(sample.source)
    left, top, right, bottom = sample.box
    if not (0 <= left < right <= source.width and 0 <= top < bottom <= source.height):
        raise ValueError(f"{sample.name}: box {sample.box} lies outside the {source.size} source.")
    if sample.box_is_eye_opening:
        horizontal = math.ceil((right - left) * HORIZONTAL_EYE_PADDING)
        vertical = math.ceil((bottom - top) * VERTICAL_EYE_PADDING)
        left, right = max(0, left - horizontal), min(source.width, right + horizontal)
        top, bottom = max(0, top - vertical), min(source.height, bottom + vertical)
    box = (left, top, right, bottom)
    crop = open_rgb(sample.crop) if sample.crop else source.crop(box)
    if crop.size != (right - left, bottom - top):
        raise ValueError(f"{sample.name}: crop dimensions {crop.size} do not match the box {box}.")
    return source, crop, box


def clean_mask(raw: np.ndarray, settings: MaskSettings) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    largest = sorted(range(1, count), key=lambda label: (-stats[label, cv2.CC_STAT_AREA], label))
    result = np.zeros_like(raw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, settings.closing_kernel)
    for label in largest[:settings.max_components]:
        component = (labels == label).astype(np.uint8)
        n_background, background, hole_stats, _ = cv2.connectedComponentsWithStats(
            1 - component, connectivity=4,
        )
        border = set(np.concatenate((background[0], background[-1], background[:, 0], background[:, -1])))
        max_area = settings.max_hole_fraction * stats[label, cv2.CC_STAT_AREA]
        holes = np.zeros(n_background, dtype=bool)
        for region in range(1, n_background):
            if region not in border and hole_stats[region, cv2.CC_STAT_AREA] <= max_area:
                holes[region] = True
        component[holes[background]] = 1
        result |= cv2.morphologyEx(component, cv2.MORPH_CLOSE, kernel)
    return result


def project_to_source(values: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    width, height = size
    left, top, right, bottom = box
    canvas = np.zeros((height, width), dtype=values.dtype)
    canvas[top:bottom, left:right] = cv2.resize(
        values, (right - left, bottom - top), interpolation=cv2.INTER_NEAREST,
    )
    return canvas


def write_outputs(
    destination: Path,
    source: Image.Image,
    box: tuple[int, int, int, int],
    probability: np.ndarray,
    settings: MaskSettings,
) -> None:
    if (
        probability.ndim != 2
        or not probability.size
        or not np.issubdtype(probability.dtype, np.floating)
        or not np.isfinite(probability).all()
        or probability.min() < 0
        or probability.max() > 1
    ):
        raise ValueError("Expected a nonempty 2D float probability map in [0, 1].")
    raw = (probability >= settings.threshold).astype(np.uint8)
    cleaned = clean_mask(raw, settings)
    raw = project_to_source(raw, box, source.size)
    cleaned = project_to_source(cleaned, box, source.size)
    projected = project_to_source(probability.astype(np.float32), box, source.size)

    destination.mkdir()
    np.save(destination / "probability.npy", projected, allow_pickle=False)
    Image.fromarray(raw * 255).save(destination / "raw-mask.png")
    Image.fromarray(cleaned * 255).save(destination / "cleaned-mask.png")
    tint = Image.blend(source, Image.new("RGB", source.size, (0, 255, 0)), 0.35)
    Image.composite(tint, source, Image.fromarray(cleaned * 255)).save(destination / "overlay.png")


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


type Predictor = Callable[[Image.Image], np.ndarray]
type PreparePredictor = Callable[[], tuple[Predictor, dict[str, object]]]


def run_segmentation(
    manifest: Path,
    output: Path,
    prepare_predictor: PreparePredictor,
    settings: MaskSettings,
) -> None:
    """Run a predictor whose probability map covers the entire oriented RGB crop."""
    samples = read_manifest(manifest)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output path already exists: {output}")
    predict, model_metadata = prepare_predictor()
    recorded_samples: list[dict[str, object]] = []
    metadata = {
        **model_metadata,
        "manifest_sha256": sha256(manifest),
        "orientation": ORIENTATION,
        "eye_opening_padding": {"horizontal": HORIZONTAL_EYE_PADDING, "vertical": VERTICAL_EYE_PADDING},
        "samples": recorded_samples,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        destination = Path(scratch)
        for sample in samples:
            source, crop, box = load_images(sample)
            write_outputs(destination / sample.name, source, box, predict(crop), settings)
            recorded_samples.append({"id": sample.name, "box": box})
        (destination / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        if os.name == "nt":
            # Windows rename atomically fails if another writer already published.
            destination.rename(output)
            return
        output.mkdir()
        try:
            destination.replace(output)
        except OSError:
            output.rmdir()
            raise
