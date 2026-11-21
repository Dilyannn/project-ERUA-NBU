"""Score source-coordinate sclera masks against binary reference masks."""

import argparse
import csv
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.mode not in {"1", "L"}:
            raise ValueError(f"{path}: expected a monochrome binary mask.")
        pixels = np.asarray(image)
        if not np.isin(pixels, (0, 1, 255)).all():
            raise ValueError(f"{path}: expected a binary mask with values 0 and 1 or 255.")
        return pixels != 0


def contour(mask: np.ndarray) -> np.ndarray:
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return mask & (eroded == 0)


def matched_fraction(edge: np.ndarray, other: np.ndarray, tolerance: float) -> float:
    distance = cv2.distanceTransform((~other).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return float(np.mean(distance[edge] <= tolerance))


def mask_scores(predicted: np.ndarray, reference: np.ndarray, tolerance: float) -> dict[str, float | int]:
    if predicted.shape != reference.shape:
        raise ValueError(f"Mask dimensions differ: {predicted.shape} and {reference.shape}.")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Boundary tolerance must be finite and nonnegative.")

    true_positive = int(np.count_nonzero(predicted & reference))
    false_positive = int(np.count_nonzero(predicted & ~reference))
    false_negative = int(np.count_nonzero(~predicted & reference))
    union = true_positive + false_positive + false_negative
    if not union:
        boundary = 1.0
    elif not predicted.any() or not reference.any():
        boundary = 0.0
    else:
        predicted_edge, reference_edge = contour(predicted), contour(reference)
        precision = matched_fraction(predicted_edge, reference_edge, tolerance)
        recall = matched_fraction(reference_edge, predicted_edge, tolerance)
        boundary = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "dice": 2 * true_positive / (true_positive + union) if union else 1.0,
        "iou": true_positive / union if union else 1.0,
        "boundary_f1": boundary,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
    }


def score_run(results: Path, references: Path, tolerance: float) -> list[dict[str, str | float | int]]:
    with references.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != ["id", "mask"]:
            raise ValueError("Reference CSV must have id,mask columns.")
        expected = {}
        for row in reader:
            name, mask = row["id"], row["mask"]
            if not name or not mask or name in expected or None in row:
                raise ValueError("Each row must contain a unique sample id and a mask path.")
            expected[name] = references.parent / mask

    data = json.loads((results / "run.json").read_text(encoding="utf-8"))
    names = [sample["id"] for sample in data["samples"]]
    if not names or len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError("Result and reference sample IDs must match exactly.")

    rows = []
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"Invalid sample id: {name!r}.")
        reference = read_mask(expected[name])
        for variant in ("raw", "cleaned"):
            predicted = read_mask(results / name / f"{variant}-mask.png")
            rows.append({
                "id": name, "mask": variant, "tolerance_px": tolerance,
                **mask_scores(predicted, reference, tolerance),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True, help="TransFuse output directory.")
    parser.add_argument("--references", type=Path, required=True, help="CSV mapping sample IDs to binary masks.")
    parser.add_argument("--tolerance-px", type=float, default=3.0, help="Boundary distance in source-image pixels.")
    args = parser.parse_args()
    try:
        rows = score_run(args.results, args.references, args.tolerance_px)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"erua-score-masks: {error}\n")
    writer = csv.DictWriter(sys.stdout, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
