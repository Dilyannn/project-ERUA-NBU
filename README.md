# project-ERUA-NBU

This project evaluates methods for segmenting exposed sclera in ordinary RGB eye photographs across a range of ocular pigmentation. It produces masks in the source image's coordinates and preserves the original RGB values for later analysis.

- `src/image_model/` holds the Python model and evaluation code as it is developed.
- `docs/model-selection.md` records the sclera segmentation baseline, model candidates, and evaluation plan.
- `docs/evaluation-panel.md` defines the reference masks and subject-separated evaluation protocol.
- `.github/workflows/` holds pull request policy checks.

## Local setup

Use Python 3.12. From the repository root, create an isolated environment and install the local package in editable mode.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -c "import image_model"
```

For the TransFuse command, see [baseline setup and crop format](docs/transfuse-baseline.md).

## Contributing

Open feature pull requests against `dev`. Promote changes to `main` through a pull request from this repository's `dev` branch.

Keep local datasets, downloaded weights, and experiment outputs out of Git. Record dataset and checkpoint versions alongside evaluation results.
