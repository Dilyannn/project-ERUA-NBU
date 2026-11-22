# Sclera segmentation

Produce an original-coordinate mask of exposed sclera in ordinary RGB eye photos. Include true dark scleral tissue near the limbus. Exclude the iris and cornea, eyelids, eyelashes, skin, caruncle, tear-lake tissue, and strong reflections. Preserve the original RGB values for later pigmentation measurements.

## Reproduce the existing baseline

Use the released [Sclera-TransFuse](https://github.com/lhqqq/Sclera-TransFuse) `transfuse2.pth` checkpoint. Reproduce the seminar team's EXIF orientation and eye crop, 384 × 384 inference with ImageNet RGB normalization, and sigmoid probability map. Record the crop coordinates and orientation so outputs can be mapped back consistently.

Freeze the existing post-processing as the control. Threshold at 0.125, retain up to two principal scleral components, fill enclosed holes no larger than 2% of each component, and apply component-wise 25 × 13 elliptical closing. Map the binary mask back with nearest-neighbor interpolation. Save the probability map, raw and cleaned binary masks, and a separate overlay. Measure the effect of each post-processing step on the validation split before changing it.

## Add a model to the shared pipeline

`image_model.segmentation` handles the crop manifest, eye-opening padding, source-coordinate projection, and output files. A model prepares a predictor once per run. The predictor takes an oriented RGB crop and returns a 2D floating-point sclera probability map in [0, 1] covering the entire crop. Its output grid can differ from TransFuse's 384 × 384 grid. If model preprocessing pads or crops the input, remove that padding or restore the crop extent before returning the probabilities.

Keep checkpoint loading, preprocessing, and conversion to sclera probabilities inside each model's module. Use the same image-space outputs and reference masks for evaluation. Report raw masks for the model comparison and report model-specific cleanup separately, with settings chosen on the validation split.

## Compare segmentation models

| Candidate | Role |
| --- | --- |
| [Swin Tiny + UPerNet](https://huggingface.co/openmmlab/upernet-swin-tiny) | First new experiment. The hierarchical backbone feeds a multiscale decoder. Adapt the pretrained model with a sclera segmentation head and evaluate it against TransFuse on the same split. |
| [DINOv2 Small](https://huggingface.co/docs/transformers/model_doc/dinov2) | Use spatial patch features from `facebook/dinov2-small` and train a dense decoder with higher-resolution boundary refinement. Freeze the backbone initially and evaluate before fine-tuning. |
| [SAM2-UNet+FDA](https://lmi.fe.uni-lj.si/wp-content/uploads/2026/09/IJCB_SSBC_2026.pdf#page=5) | Competition reference. The adapted pipeline won SSBC 2026 Track 1 with 0.870 harmonic-mean F1 across three datasets. Verify reproducibility and access before planning it as a local comparator. |

[Hugging Face Transformers](https://huggingface.co/docs/transformers/index) provides model-loading and preprocessing APIs for DINOv2 and Swin. Keep resizing and normalization specific to each model and compare their masks at the same source resolution.

SSBC 2026 reported 0.813 harmonic-mean F1 for the Swin-Tiny-based VLS-SegNet and 0.781 for DINOv2Refine. Those entries used different decoders and adaptation methods, so the scores compare complete systems. Neither result measures performance of the proposed configurations on the seminar images.

## Evaluate on the seminar images

1. Establish a subject-separated validation and held-out test panel of clinician-reviewed masks spanning scleral pigmentation, lighting, camera, gaze, and occlusion where metadata is available. Record how strongly pigmented areas and boundary exclusions should be annotated.
2. Report Dice and IoU, boundary F1 at a fixed tolerance, recall on annotated dark and peri-limbal sclera, and false-positive inclusion of eyelid, skin, and caruncle. Report the sample counts for each subgroup and inspect failures with overlays. Use one decision policy across phenotypes.
3. For capture robustness, test lighting and white-balance variation and preserve the original photo alongside the mask. For later color measurements, standardize acquisition and validate color calibration separately from segmentation.

Use [SSBC 2026](https://sites.google.com/hyderabad.bits-pilani.ac.in/ssbc2026/home) as a published benchmark. The 2026 submission deadline has passed. Its two tracks covered foundation-model adaptation and label-efficient learning. SSBC scores quantify segmentation performance under that protocol. Clinical screening would require a separate dataset, clinical endpoints, and validation.
