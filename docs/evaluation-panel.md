# Evaluation panel and annotation protocol

Protocol version 1 covers anatomical sclera masks on RGB eye photographs. Track the seminar panel and the public UBIPr comparison as separate datasets.

## Prepare the seminar panel

1. Record authorization, dataset revision, image and annotation hashes, and a stable pseudonymous sample ID. Store images, masks, linking keys, and subject manifests in approved storage outside Git.
2. Group all images of a person together, including both eyes, repeated captures, and augmented copies. Assign groups to validation or test before comparing models. Preserve the split in the versioned manifest and record coverage by pigmentation, capture lighting, device, gaze, and occlusion where known. Record missing metadata as unknown.
3. Define the crop from the original EXIF-oriented RGB image using the visible eye opening. Save the box before opening reference masks or predictions. Process the source image with the documented crop padding and map predictions back to source resolution.
4. Annotate exposed sclera, including visibly dark scleral tissue. Exclude iris and cornea, eyelids, eyelashes, skin, caruncle, tear-lake tissue, and strong reflections. A second reviewer should check disagreements around the limbus and eyelids and record what was adjudicated.
5. Provide optional masks for true sclera near the limbus, visibly pigmented sclera, and separate skin, eyelid, and caruncle regions where the annotation supports them. Mark exposure and pigmentation by visual review of the source and do not use darkness alone as an anatomical label.

Use one row per photo in the private panel manifest. Record the sample ID, pseudonymous subject group, split, capture metadata, source and mask paths or hashes, crop coordinates, annotation revision, and review status. Keep the test split fixed. Use validation data for decisions about crop, threshold, or post-processing.

## Score the panel

Use [the baseline commands](transfuse-baseline.md) for inference and scoring. The reference CSV begins with `id,mask`. Append any of these optional columns with relative paths to binary masks:

| Column | Measurement |
| --- | --- |
| `peri_limbal_sclera` | Recall of the annotated sclera near the limbus |
| `pigmented_sclera` | Recall of the annotated pigmented sclera |
| `skin`, `eyelid`, `caruncle` | False-positive pixels per exclusion region |

Use source-coordinate masks with matching dimensions. The two sclera subsets must sit inside the sclera reference. The three exclusion regions must sit outside it. A blank cell signifies unavailable annotation. An empty region has zero reference pixels and undefined recall. Keep these distinctions when summarizing results and report the number of contributing images for each metric.

## Public UBIPr comparison

The [UBIPr single-eye segmented release](https://iris.di.ubi.pt/ubipr.html) is a public research substitute for running the evaluation workflow. It contains lossless reference masks and is distributed under CC BY-NC-SA 4.0. Cite Chandrashekhar Padole and Hugo Proença, “Periocular Recognition: Analysis of Performance Degradation Factors,” ICB 2012.

Version `ubipr-panel-v1` was sampled from complete image-mask pairs in the first 64 MiB of the release archive (prefix SHA-256 `8346f28cf559e9cb79131f950f4e8a380eebe92deec439a35625fe3bf788eb7a`). We took one pair per archive subject prefix. Groups and captures were ranked by SHA-256 of `erua-ubipr-panel-v1:<identifier>`, with the first eight groups assigned to validation and the next eight to test. One validation reference was excluded before prediction because its sclera label was disconnected from the eye. The final panel has seven validation and eight test images.

The public case uses class 170 for sclera and class 85 for iris. Its peri-limbal proxy is reference sclera within five source-image pixels of annotated iris. Available labels support that proxy and whole-sclera scores. Separate pigmented sclera, skin, eyelid, and caruncle error counts require reviewed annotations. The public test groups are held out from ERUA tuning. Overlap with upstream TransFuse training subjects is unverified.

### TransFuse results

The table reports means over the eight test images with the authors' checkpoint and the same thresholds and cleanup settings in each run. Boundary F1 uses a three-pixel tolerance at source resolution.

| Input | Mask | Dice | IoU | Boundary F1 | Peri-limbal recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Full image | Raw | 0.334 | 0.214 | 0.090 | 0.660 |
| Full image | Cleaned | 0.305 | 0.205 | 0.085 | 0.418 |
| Padded eye crop | Raw | 0.630 | 0.462 | 0.200 | 0.803 |
| Padded eye crop | Cleaned | 0.645 | 0.478 | 0.219 | 0.803 |

On cropped inputs, cleanup improved boundary F1 on seven test cases and reduced it on one. On full images, cleanup reduced mean peri-limbal recall. Use the padded crop for the public baseline and retain raw and cleaned masks for comparison on the seminar panel. The reviewed seminar panel is still needed to assess performance on dark sclera and leakage into specific tissue.
