"""Run the seminar's Sclera-TransFuse baseline on a local crop manifest."""

import argparse
from dataclasses import asdict
from functools import partial
import importlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from .segmentation import MaskSettings, Predictor, run_segmentation, sha256


INPUT_SIZE = 384
SEMINAR_MASK_SETTINGS = MaskSettings(
    threshold=0.125,
    max_components=2,
    max_hole_fraction=0.02,
    closing_kernel=(25, 13),
)
MODEL_MODULE = "Sclera_Transfuse_Seg"
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict(model, crop: Image.Image, device) -> np.ndarray:
    import torch

    size = INPUT_SIZE
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


def prepare_transfuse(
    upstream: Path, checkpoint: Path, device_name: str,
) -> tuple[Predictor, dict[str, object]]:
    import torch

    sources = (upstream / f"{MODEL_MODULE}.py", upstream / "module.py")
    device = torch.device(device_name)
    metadata = {
        "baseline": {"input_size": INPUT_SIZE, **asdict(SEMINAR_MASK_SETTINGS)},
        "checkpoint_sha256": sha256(checkpoint),
        "upstream_sha256": {source.name: sha256(source) for source in sources},
        "device": str(device),
    }
    model = load_model(upstream.resolve(), checkpoint, device)
    return partial(predict, model, device=device), metadata


def run(manifest: Path, output: Path, upstream: Path, checkpoint: Path, device_name: str) -> None:
    run_segmentation(
        manifest, output, partial(prepare_transfuse, upstream, checkpoint, device_name), SEMINAR_MASK_SETTINGS,
    )


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
