"""Export existing CLI manifests into a new private directory outside the dataset."""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
import csv
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from image_model.segmentation import ORIENTATION, load_images, read_manifest, sha256
from .ubiris_importer import Inventory


def export_manifests(inventory: Inventory, ids: Iterable[str], boxes: Mapping[str, Sequence[int]],
                     destination: Path) -> tuple[Path, Path]:
    """Write crops.json and references.csv; return their paths in that order.

    Supply one RGB-only eye-opening box per selected ID, in oriented source
    coordinates [left, top, right, bottom), before the baseline adds padding.
    The destination's parent must exist on the same drive as the source files.
    """
    destination = destination.resolve()
    if destination.is_relative_to(inventory.root) or destination.exists():
        raise ValueError("Choose a new output directory outside the dataset.")
    if isinstance(ids, str):
        raise ValueError("Pass a collection of sample IDs.")
    selected = sorted(ids)
    pairs = {pair.id: pair for pair in inventory.samples}
    if not selected or len(set(selected)) != len(selected) or set(selected) - pairs.keys():
        raise ValueError("Selection must contain unique eligible IDs.")
    if set(boxes) != set(selected):
        raise ValueError("Box IDs must match the selection exactly.")
    rows = []
    references = []
    for name in selected:
        pair = pairs[name]
        relative = []
        for path, digest in ((pair.image, pair.image_sha256), (pair.mask, pair.mask_sha256)):
            source = (inventory.root / path).resolve(strict=True)
            if not source.is_relative_to(inventory.root) or sha256(source) != digest:
                raise ValueError(f"{name}: file changed since inspection.")
            relative.append(Path(os.path.relpath(source, destination)).as_posix())
        rows.append({"id": name, "source": relative[0], "eye_opening_box": list(boxes[name])})
        references.append((name, relative[1]))

    # Publish the two validated files together.
    with TemporaryDirectory(dir=destination.parent) as temp:
        stage = Path(temp) / "export"
        stage.mkdir()
        crops = stage / "crops.json"
        crops.write_text(json.dumps({"orientation": ORIENTATION, "samples": rows}, indent=2), encoding="utf-8")
        for sample in read_manifest(crops):
            source_path = (destination / sample.source.relative_to(stage)).resolve()
            source, _, _ = load_images(replace(sample, source=source_path))
            pair = pairs[sample.name]
            if source.size != (pair.width, pair.height):
                raise ValueError(f"{sample.name}: inventory dimensions differ.")
        with (stage / "references.csv").open("w", encoding="utf-8", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(("id", "mask"))
            writer.writerows(references)
        stage.rename(destination)
    return destination / "crops.json", destination / "references.csv"
