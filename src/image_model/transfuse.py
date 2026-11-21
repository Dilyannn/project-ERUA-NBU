"""Run the seminar's Sclera-TransFuse baseline on a local crop manifest."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class Baseline:
    input_size: int = 384
    threshold: float = 0.125
    max_components: int = 2
    max_hole_fraction: float = 0.02
    closing_kernel: tuple[int, int] = (25, 13)


SEMINAR_BASELINE = Baseline()
MODEL_MODULE = "Sclera_Transfuse_Seg"
ORIENTATION = "exif-transposed"
HORIZONTAL_EYE_PADDING = 0.10
VERTICAL_EYE_PADDING = 0.35
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


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


def clean_mask(raw: np.ndarray, baseline: Baseline) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    largest = sorted(range(1, count), key=lambda label: (-stats[label, cv2.CC_STAT_AREA], label))
    result = np.zeros_like(raw)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, baseline.closing_kernel)
    for label in largest[:baseline.max_components]:
        component = (labels == label).astype(np.uint8)
        n_background, background, hole_stats, _ = cv2.connectedComponentsWithStats(
            1 - component, connectivity=4,
        )
        border = set(np.concatenate((background[0], background[-1], background[:, 0], background[:, -1])))
        max_area = baseline.max_hole_fraction * stats[label, cv2.CC_STAT_AREA]
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


def write_outputs(destination: Path, source: Image.Image, box: tuple[int, int, int, int], probability: np.ndarray) -> None:
    baseline = SEMINAR_BASELINE
    if probability.shape != (baseline.input_size, baseline.input_size) or not np.isfinite(probability).all():
        raise ValueError(f"Expected a finite {baseline.input_size} × {baseline.input_size} probability map.")
    raw = (probability >= baseline.threshold).astype(np.uint8)
    cleaned = clean_mask(raw, baseline)
    raw = project_to_source(raw, box, source.size)
    cleaned = project_to_source(cleaned, box, source.size)
    projected = project_to_source(probability.astype(np.float32), box, source.size)

    destination.mkdir()
    np.save(destination / "probability.npy", projected, allow_pickle=False)
    Image.fromarray(raw * 255).save(destination / "raw-mask.png")
    Image.fromarray(cleaned * 255).save(destination / "cleaned-mask.png")
    tint = Image.blend(source, Image.new("RGB", source.size, (0, 255, 0)), 0.35)
    Image.composite(tint, source, Image.fromarray(cleaned * 255)).save(destination / "overlay.png")


def predict(model, crop: Image.Image, device) -> np.ndarray:
    import torch

    size = SEMINAR_BASELINE.input_size
    rgb = np.asarray(crop.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    normalized = np.ascontiguousarray(((rgb - MEAN) / STD).transpose(2, 0, 1))
    inputs = torch.from_numpy(normalized).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(inputs)
        if tuple(logits.shape) != (1, 1, size, size):
            raise ValueError(f"Unexpected TransFuse output shape: {tuple(logits.shape)}.")
        return logits.sigmoid()[0, 0].cpu().numpy()


def load_model(upstream: Path, checkpoint: Path, device):
    import torch

    sys.path.insert(0, str(upstream))
    try:
        model = importlib.import_module(MODEL_MODULE).scleratransfuse_seg(128, 1)
    finally:
        sys.path.pop(0)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(isinstance(value, torch.Tensor) for value in state.values())
    ):
        raise ValueError("Expected the authors' TransFuse tensor state dict.")
    if all(key.startswith("module.") for key in state):
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state, "module.")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def run(manifest: Path, output: Path, upstream: Path, checkpoint: Path, device_name: str) -> None:
    import torch

    samples = read_manifest(manifest)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output path already exists: {output}")
    sources = (upstream / f"{MODEL_MODULE}.py", upstream / "module.py")
    device = torch.device(device_name)
    recorded_samples = []
    metadata = {
        "baseline": asdict(SEMINAR_BASELINE),
        "checkpoint_sha256": sha256(checkpoint),
        "upstream_sha256": {source.name: sha256(source) for source in sources},
        "manifest_sha256": sha256(manifest),
        "device": str(device),
        "orientation": ORIENTATION,
        "eye_opening_padding": {"horizontal": HORIZONTAL_EYE_PADDING, "vertical": VERTICAL_EYE_PADDING},
        "samples": recorded_samples,
    }
    model = load_model(upstream.resolve(), checkpoint, device)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as scratch:
        destination = Path(scratch)
        for sample in samples:
            source, crop, box = load_images(sample)
            write_outputs(destination / sample.name, source, box, predict(model, crop, device))
            recorded_samples.append({"id": sample.name, "box": box})
        (destination / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        output.mkdir()
        try:
            destination.replace(output)
        except OSError:
            output.rmdir()
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True, help="Path to the external Sclera-TransFuse-Seg directory.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to transfuse2.pth.")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the generated results.")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda:0.")
    args = parser.parse_args()
    try:
        run(args.manifest, args.output, args.upstream, args.checkpoint, args.device)
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f"erua-transfuse: {error}\n")
