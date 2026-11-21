# Run the TransFuse baseline

ERUA owns the crop interface, mask cleanup, output format, and evaluation. The model architecture runs from a separate checkout of the authors' [Sclera-TransFuse](https://github.com/lhqqq/Sclera-TransFuse) repository. The checkout revision used for this integration is `6ae698bce7554cceeb6a0f7f8a806b28c9793e3f`. Use the model definition in `Sclera-TransFuse-Seg/Sclera_Transfuse_Seg.py` and download `transfuse2.pth` from the [authors' checkpoint folder](https://drive.google.com/drive/folders/104TAazlhPCHCWI1IcPqgUJKLjj1alGpr). Keep the checkout, checkpoints, and any participant files outside tracked source.

The authors' download used in the synthetic smoke test had SHA-256 `585f6e2a765cae1e6edd840bd5e10b569865ae92cab775443d178379141fc589`. Compare this with the seminar team's checkpoint hash before recording a seminar reproduction.

From the project root with your Python 3.12 environment active:

```sh
python -m pip install -e '.[transfuse]'
erua-transfuse \
  --upstream ../Sclera-TransFuse/Sclera-TransFuse-Seg \
  --checkpoint checkpoints/transfuse2.pth \
  --manifest data/crops.json \
  --output runs/transfuse-baseline
```

Inference defaults to CPU. Use `--device cuda:0` on a compatible PyTorch GPU environment.

## Crop manifest

```json
{
  "orientation": "exif-transposed",
  "samples": [
    {
      "id": "eye-001",
      "source": "images/eye-001.jpg",
      "box": [40, 32, 424, 288]
    }
  ]
}
```

Paths are relative to the manifest directory. Coordinates are `[left, top, right, bottom]` in the EXIF-oriented source, with right and bottom excluded. The command crops the source using the box. To use an existing crop, add a `"crop"` image path whose oriented dimensions match the box exactly. Use unique, non-identifying sample IDs.

To use a consistent eye crop, replace `box` with `eye_opening_box`. Mark a rectangle around the visible eye opening on the RGB photo using the inner and outer corners and the upper and lower lid edges. The command adds 10% of the opening width to each side and 35% of the opening height above and below, rounding up and clipping to the image. Define the rectangle before looking at segmentation masks or predictions. The command records the final box in `run.json`. Explicit `box` coordinates remain available for a fixed crop or full-image control.

## Results

The output path must be new. Each sample gets a float32 `probability.npy`, a `raw-mask.png`, a `cleaned-mask.png`, and a separate `overlay.png` at source resolution. Values outside the crop are zero in the probability and mask files. The overlay leaves pixels outside the cleaned mask unchanged, and the command preserves the source file.

`run.json` records the checkpoint, manifest, and upstream file hashes along with the device, crop coordinates, and baseline settings. The named baseline uses 384 × 384 bilinear RGB resizing with ImageNet normalization, sigmoid probabilities, threshold 0.125, up to two connected components, holes up to 2% of component area, and per-component 25 × 13 elliptical closing. Outputs map back via nearest-neighbor resizing.

Run the synthetic geometry and mask tests with `python -m unittest discover -s tests -v`.

## Real-image smoke test

The [UBIPr project page](https://iris.di.ubi.pt/) publishes an unmodified [example JPEG](https://iris.di.ubi.pt/ubipr_i1.jpg) under the dataset's [CC BY-NC-SA 4.0 terms](https://iris.di.ubi.pt/ubipr.html). With the authors' checkpoint, both the full image and the hand-selected box `[112, 130, 398, 265]` produced all four outputs at the original 535 × 369 resolution. The full-image mask included large areas of skin. The tighter crop put the cleaned mask over the eye on visual inspection. This single example checks inference and illustrates crop sensitivity. Use a labeled test panel for accuracy measurements.

The seminar slide images include rendered probability maps, masks, or overlays. Use them as references for the team's reported failure cases. Reproducing their results requires the original RGB inputs and crop settings.

## Score a run

Prepare a CSV with the header `id,mask` and one row per sample in the run. Each path is relative to the CSV and points to a source-coordinate binary reference mask with foreground value 1 or 255. The IDs must match the corresponding results exactly.

```sh
erua-score-masks \
  --results runs/transfuse-baseline \
  --references data/references.csv \
  --tolerance-px 3 > runs/transfuse-scores.csv
```

The command reports Dice, IoU, boundary F1, and false-positive and false-negative pixel counts for raw and cleaned masks. Boundary F1 matches contour pixels within a Euclidean distance of 3 source-image pixels in both directions. Both empty masks score 1, and an empty mask paired with a nonempty one scores 0.

## Crop pilot

The four single-eye JPEG examples on the [UBIPr project page](https://iris.di.ubi.pt/) were scored both full-frame and with the padded eye crop. The displayed reference JPEG labels use grayscale levels near 0, 85, 170, and 255. For this pilot, values 128 through 212 were assigned to the visually identified sclera class (170). Use the original lossless annotations for a formal evaluation.

The eye-opening boxes in source pixels, in the order of examples 1 through 4, were `[123,154,368,237]`, `[119,130,340,212]`, `[350,334,884,506]`, and `[315,462,941,662]`. Model, normalization, and post-processing were identical across runs.

| Input | Mask | Mean Dice | Mean IoU | Mean boundary F1 at 3 px |
| --- | --- | ---: | ---: | ---: |
| Full image | Raw | 0.383 | 0.255 | 0.101 |
| Full image | Cleaned | 0.407 | 0.283 | 0.116 |
| Padded eye crop | Raw | 0.612 | 0.447 | 0.205 |
| Padded eye crop | Cleaned | 0.604 | 0.439 | 0.197 |

Cropping improved mean overlap and reduced false-positive pixels across the four examples. On padded crops, cleanup recovered some missed pixels while adding false positives overall. For cleaned masks, boundary F1 fell on two of the four images after cropping, even though the mean improved. Keep both mask variants and use a subject-separated validation set to evaluate post-processing changes. These examples are exploratory and provide no estimate of performance on the seminar panel.
