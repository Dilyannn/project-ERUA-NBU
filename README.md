# project-ERUA-NBU

This project evaluates methods for segmenting exposed sclera in ordinary RGB eye photographs across a range of ocular pigmentation. It produces masks in the source image's coordinates and preserves the original RGB values for later analysis.

- `src/image_model/` holds the Python model and evaluation code as it is developed.
- `docs/model-selection.md` records the sclera segmentation baseline, model candidates, and evaluation plan.
- `.github/workflows/` holds pull request policy checks.

Open feature pull requests against `dev`. Promote changes to `main` through a pull request from this repository's `dev` branch.

Keep local datasets, downloaded weights, and experiment outputs out of Git. Record dataset and checkpoint versions alongside evaluation results.
