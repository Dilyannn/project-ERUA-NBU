# Prepare and audit the UBIRIS.v2 sclera dataset

## Purpose and scope

[Issue #15](https://github.com/Dilyannn/project-ERUA-NBU/issues/15) uses an annotated UBIRIS.v2 subset to check sclera segmentation, crop handling, and post-processing. The subset supports whole-sclera overlap and boundary scoring. Pigmentation-specific recall and tissue-specific false positives require the reviewed region labels in the separate [evaluation panel](evaluation-panel.md).

This draft describes the workflow at integration commit `73b38b8de0e1f000624d31a1d952c772c7e762c5`. Use a checkout containing that implementation. Institutional authorization, archive authenticity, subject mapping, independent real-data verification, and checkpoint training independence are pending.

## UBIRIS.v2 dataset provenance

The University of Beira Interior (UBI) collected UBIRIS.v2 to study visible-light iris recognition under less constrained conditions. The original release contains 11,102 images from 261 volunteers and 522 irises. The two acquisition sessions captured volunteers in motion at distances of 4–8 m. The manually cropped images are 400 × 300 pixels in 24-bit sRGB TIFF. These figures describe the original release. [Official dataset page](https://iris.di.ubi.pt/ubiris2.html) · [Original publication](https://doi.org/10.1109/TPAMI.2009.66)

## Sclera-TransFuse subset and annotation provenance

The Sclera-TransFuse authors published a UBIRIS.v2 subset of 683 eye images with manually labeled sclera masks. Their 2024 paper and project repository are the public sources for this release. The exact archive revision, annotation quality-control procedure, and correspondence between the published release and the ERUA copy remain to be established. [Author project page](https://nlpr-sir.github.io/publication/wang-tbiom-2024-1/) · [Official implementation and release information](https://github.com/lhqqq/Sclera-TransFuse)

UBI's original paper describes NICE.I masks for visible, unobscured iris regions. The Federal University of Paraná also released a separate sclera annotation collection for 300 UBIRIS.v2 images and 1,000 MICHE-I images. These sources describe different annotation releases. [UBIRIS.v2 paper, §4.1](https://di.ubi.pt/~hugomcp/doc/UBIRISv2_IEEETPAMI.pdf) · [UFPR annotation release](https://web.inf.ufpr.br/vri/databases/sclera-segmentation-annotations/)

## Research-use terms and access restrictions

The UBI license provides for internal research at the institution represented by the principal investigator. The investigator submits the signed application to UBI and receives access instructions from UBI. The license prohibits redistribution, publication, or dissemination of the dataset, including parts of it. It requires citation and delivery of publicly released research reports to SOCIA Lab. The form lists a DOI for a different paper. Use the dataset citation confirmed by UBI's dataset page and IEEE. [UBI agreement](https://iris.di.ubi.pt/ubiris2AppForm.pdf) · [Access procedure and citation](https://iris.di.ubi.pt/ubiris2.html) · [IEEE record](https://ieeexplore.ieee.org/document/4815254)

Obtain written confirmation of NBU's authorization, the approved team members, and the terms for the specific subset and its masks before acquiring or opening it. Keep archives, images, masks, weights, participant identifiers, per-sample manifests, and generated outputs in approved private storage outside Git. The approved team storage location is pending. Share only authorized aggregate results.

## Dataset acquisition and SHA-256 verification procedure

After authorization, obtain the approved archive through the custodian. In a private record, note the source URL, release identifier, acquisition date, filename, and any publisher or custodian checksum. Hash the unopened archive before extraction and hash the exact checkpoint separately. Compare each local SHA-256 with its trusted expected value. Resolve any mismatch before use. If a trusted expected value is unavailable, retain the local SHA-256 for reproducibility and keep authenticity pending.

Activate the Python 3.12 environment described below before running the checksum command. Set these placeholders in the same shell used for the remaining commands. Use absolute paths. Create the approved private run directory before running the examples. Keep the run directory and dataset on the same drive for portable relative paths.

```sh
export ERUA_ARCHIVE='/path/to/approved/dataset-archive.zip'
export ERUA_DATASET='/path/to/private/extracted-dataset'
export ERUA_RUN='/path/to/private/existing-run-directory'
export ERUA_UPSTREAM='/path/to/Sclera-TransFuse/Sclera-TransFuse-Seg'
export ERUA_CHECKPOINT='/path/to/approved/transfuse2.pth'

python -c 'import hashlib, sys; from pathlib import Path
for name in sys.argv[1:]:
    with Path(name).open("rb") as stream:
        print(hashlib.file_digest(stream, "sha256").hexdigest(), name)
' "$ERUA_ARCHIVE" "$ERUA_CHECKPOINT"
```

| Artifact | Trusted expected SHA-256 | Measured SHA-256 |
| --- | --- | --- |
| Approved subset archive | `<archive-expected-sha256>` | `<archive-measured-sha256>` |
| Exact `transfuse2.pth` used for the run | `<checkpoint-expected-sha256>` | `<checkpoint-measured-sha256>` |

Extract the verified or custodian-approved copy into the private dataset directory, preserving the directory structure and original files and names. The importer computes image and mask SHA-256 values for the extracted pairs. Those hashes support later integrity checks on the individual files.

## Python environment and dependency setup

Use Python 3.12. From the integration checkout root on macOS or Linux, create and activate a virtual environment, or activate an existing Python 3.12 environment. On Windows PowerShell, the activation script is `.venv\Scripts\Activate.ps1`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python --version
python -m pip install -e '.[transfuse]'
python -m pip check
python -m unittest discover -s tests -v
```

The `transfuse` extra installs NumPy, Pillow, OpenCV, PyTorch, torchvision, timm, and einops. Record the resolved versions with `python -m pip list --format=freeze`. Install the upstream model code separately at revision `6ae698bce7554cceeb6a0f7f8a806b28c9793e3f`, as described in the [baseline setup](transfuse-baseline.md).

## Dataset importer usage

Import, splitting, and export use Python functions. Run the inventory, subject mapping, split generation, and export blocks in order in the same Python session, or save them together in a private script outside Git. Run the original-folder audit separately. The examples use the shell variables above. They assume the approved archive has the layout and encoding reported by the integration team. Verify those settings on the authorized release before relying on them.

```python
import json
import os
from pathlib import Path

from image_model.datasets.ubiris_importer import inspect_pairs

dataset = Path(os.environ["ERUA_DATASET"])
run = Path(os.environ["ERUA_RUN"])

inventory = inspect_pairs(
    dataset,
    images="sum/image",
    masks="sum/SegmentationClass",
    foreground=255,
    mask_frame="oriented-source",
)
print("Eligible pairs:", len(inventory.samples))
print("Exclusions:", len(inventory.exclusions))
print("Exact-image duplicate groups:", len(inventory.duplicate_groups))
```

The importer scans flat folders, matches file stems, validates RGB sources and binary references, and checks dimensions after applying the source EXIF orientation. Review the exclusion reasons, duplicate groups, and hashes in the private inventory. Exclusion details can include paths and names. The importer uses the declared foreground polarity and coordinate frame. Independently verify that white marks sclera and that the masks align with the oriented images.

**Team-reported.** The integration team used `sum/SegmentationClass` with foreground `255`. It reported mixed encodings in the importer's default `sum/sclera_mask` folder, 683 eligible pairs, zero exact-image duplicate groups, and one empty reference retained for expert review. Preserve the original masks and record the review decision before using the empty reference in an evaluation. [PR #26](https://github.com/Dilyannn/project-ERUA-NBU/pull/26)

## Subject mapping requirements

Prepare `subject-mapping.json` in the private run directory as a JSON object whose keys are eligible sample IDs from the inventory and whose string values are the original subject IDs confirmed by the custodian, or a custodian-confirmed subject crosswalk. The mapping must cover the eligible inventory exactly. Preserve IDs as strings, including leading zeros. Both eyes and all sessions of one person must share one subject ID. Treat eye codes and frame indices as provisional until the custodian confirms how they map to people, including the three reported unusual frames.

```python
mapping = json.loads((run / "subject-mapping.json").read_text(encoding="utf-8"))
mapping_status = "provisional"
```

Set `mapping_status = "verified"` only after receiving and recording independent confirmation. If no suitable mapping is available, stop before generating splits.

## Deterministic split generation and leakage auditing

The `subject-greedy-v2` algorithm assigns whole subject groups, aims for the requested proportions by image count, and uses the seed for deterministic tie-breaking. The example uses the provisional targets of 70/15/15 and seed 42. At least three subject groups are required. Confirm the final settings with the team.

```python
from image_model.datasets.split_audit import (
    audit_manifest,
    inventory_split,
    validate_frozen,
    validate_snapshot,
)

seed = 42
ratios = {"train": 0.70, "validation": 0.15, "test": 0.15}
settings = {"mapping_status": mapping_status, "seed": seed, "ratios": ratios}
snapshot = inventory_split(inventory, mapping, **settings)

audit = audit_manifest(snapshot, base_dir=inventory.root)
print(audit.render())
if not audit.passed or any(count == 0 for count in audit.images_by_split.values()):
    raise RuntimeError("Split audit failed or a split is empty.")

validate = validate_frozen if mapping_status == "verified" else validate_snapshot
validate(snapshot, inventory, mapping, **settings)

with (run / "split-manifest.json").open("x", encoding="utf-8") as stream:
    json.dump(snapshot, stream, indent=2, sort_keys=True)
    stream.write("\n")
```

Record the algorithm, seed, ratios, snapshot SHA-256, and actual image and subject counts per split. The snapshot binds assignments to the inventory, file hashes, and mapping status. To verify a stored snapshot, load its JSON and call the applicable validation function with a fresh inventory, the independent mapping, and the agreed settings. Reserve validation for tuning and test for the final comparison once splits are frozen.

### Audit the original release folders

Run this standalone example after access is authorized. Replace the three placeholder paths with the original split directories. Each key is required. If a split directory contains an `image` subdirectory, the function scans that subdirectory. If it does not, the function scans the supplied directory. Point directly to the image-only directory if the release uses a different layout.

```python
from pathlib import Path

from image_model.datasets.split_audit import analyze_original_split_dirs

original_dirs = {
    "train": Path("/path/to/approved/original-train-directory"),
    "validation": Path("/path/to/approved/original-validation-directory"),
    "test": Path("/path/to/approved/original-test-directory"),
}
original_audit = analyze_original_split_dirs(original_dirs)
print(original_audit.render())
```

The aggregate report counts image files, unique case-insensitive stems, and repeated stems within each split. It also reports pairwise stem overlaps and the number of stems present in more than one split. The new split audit separately checks subject IDs, image IDs, and exact image bytes. The team reported original folder counts of 280/140/280 and eight shared train/test basenames. Its provisional split had 478/103/102 images. Independent verification on the authorized release is pending. [Issue #15](https://github.com/Dilyannn/project-ERUA-NBU/issues/15) · [PR #26](https://github.com/Dilyannn/project-ERUA-NBU/pull/26)

## Crop and reference manifest export

Prepare `eye-opening-boxes.json` as a private JSON object mapping selected sample IDs to four integer coordinates `[left, top, right, bottom]`. Mark the visible eye opening on the EXIF-oriented RGB image before viewing the reference or prediction. Right and bottom are exclusive. The baseline applies padding automatically, so enter the unpadded box.

Prefer synthetic samples for development smoke tests and use validation samples for authorized real-data checks. Record the selection for repeatability. The example below selects from validation. Before opening the test split, fix the checkpoint, crop protocol, thresholds, post-processing, and evaluation rules. Reserve test samples for the final comparison. Test-split smoke-test outputs and scores must never guide model or checkpoint selection, tuning, or changes to the evaluation protocol.

```python
from image_model.datasets.manifest_export import export_manifests

boxes = json.loads((run / "eye-opening-boxes.json").read_text(encoding="utf-8"))
validation_ids = {
    row["image_id"] for row in snapshot["records"] if row["split"] == "validation"
}
if not boxes or not set(boxes) <= validation_ids:
    raise ValueError("Choose smoke-test samples from the validation split.")

crops, references = export_manifests(inventory, boxes.keys(), boxes, run / "manifests")
```

The destination must be new and outside the dataset root. Export rechecks source and mask hashes and writes `crops.json` and `references.csv` together. File paths are relative to the manifest directory. Keep the directory layout consistent when reusing the manifests. Use a separate private run directory to repeat the workflow without overwriting the saved snapshot or exports.

## TransFuse smoke test and mask evaluation

From the activated environment, run the baseline and score the exported references. The output directory must be new.

```sh
erua-transfuse \
  --manifest "$ERUA_RUN/manifests/crops.json" \
  --upstream "$ERUA_UPSTREAM" \
  --checkpoint "$ERUA_CHECKPOINT" \
  --output "$ERUA_RUN/predictions" \
  --device cpu

erua-score-masks \
  --results "$ERUA_RUN/predictions" \
  --references "$ERUA_RUN/manifests/references.csv" \
  --tolerance-px 3 > "$ERUA_RUN/scores.csv"
```

Check that `run.json` records the expected hashes, device, crop coordinates, and settings. Confirm that probability maps are finite and within [0, 1], both masks are binary and at the oriented source resolution, scoring covers exactly the selected samples, and original image and mask hashes are unchanged. The scorer reports Dice, IoU, boundary F1 at the chosen source-pixel tolerance, and false-positive and false-negative pixel counts for raw and cleaned masks. Two empty masks score 1 on the three overlap and boundary metrics. A single empty mask scores 0. Record the policy for an empty reference before reporting aggregates. [Baseline outputs and scoring](transfuse-baseline.md)

**Team-reported.** Three real samples completed CPU inference and scoring, with binary output masks at 400 × 300, finite probabilities in range, and unchanged hashes for all six inputs. This run has yet to be independently reproduced. [PR #26](https://github.com/Dilyannn/project-ERUA-NBU/pull/26)

## Training overlap and diagnostic evaluation limitations

The IJCB 2023 author poster lists UBIRIS.v2 for both training and testing in published Sclera-TransFuse experiments. The exact training, validation, model-selection, and testing inputs of the released `transfuse2.pth` are unresolved. The checkpoint filename and the upstream code revision do not establish its training history. Establish independence using provenance tied to the exact checkpoint and an authorized comparison by image and person. A subject-separated ERUA split only controls overlap among ERUA's own splits. [IJCB 2023 author poster](https://wylcasia.github.io/papers/IJCB2023-2-poster.pdf) · [Upstream implementation](https://github.com/lhqqq/Sclera-TransFuse)

Use the results as **diagnostic segmentation measurements** for the selected inputs and pipeline settings. Prior exposure to the same image, a transformed copy, or another frame of the same person can inflate apparent generalization. A three-image smoke test measures workflow operation. Generalization to unseen participants and clinical claims require separate evidence.

## Independent technical QA results

**Team-reported.** Bozhidar (`b0jkata14`) reported **54 of 54 tests passed on Windows and Linux** at integration commit `73b38b8`. [PR #26 verification comment](https://github.com/Dilyannn/project-ERUA-NBU/pull/26#discussion_r4724254833)

**Independently verified.** At the same integration commit (`73b38b8de0e1f000624d31a1d952c772c7e762c5`), `python -m unittest discover -s tests -v` ran **54 tests, all passed on macOS**, with zero failures, errors, or skips. The run took 1.010 seconds on Python 3.12.16 and macOS 27.0.1 (build `26A434`, arm64). The suite included checks that a destination created during export or segmentation is preserved. All inputs were synthetic.

| Dependency | Tested version |
| --- | --- |
| NumPy | 2.5.5 |
| Pillow | 13.0.0 |
| opencv-python-headless | 5.1.0.95 |
| PyTorch | 2.15.0 |
| torchvision | 0.29.2 |
| timm | 1.0.31 |
| einops | 0.8.2 |

Dependency checks passed and a basic PyTorch CPU tensor operation succeeded. No restricted files or pretrained weights were used in this independent QA run.

## Known technical limitations

Additional synthetic checks outside the 54-test suite reproduced the following behavior:

- A subdirectory with the same stem as a valid image can make the importer report an ambiguous pair. Use the expected flat directories and review exclusions.
- A self-referencing symbolic link can raise `RuntimeError` and stop import or the standalone split audit on Python 3.12.16. Use regular dataset files in the expected directories.
- The standalone audit passed when all three synthetic records were in `train`. Regeneration with `validate_frozen` rejected that modified manifest. Check for nonempty splits and validate against independently trusted inputs before freezing.
- On the tested system, the export directory had mode `0755` and its files had mode `0644`. Store exports under an approved parent with restricted access.
- Exact SHA-256 matching can miss recoded, resized, or cropped copies. The importer accepts structurally valid empty masks and relies on the supplied polarity and coordinate-frame convention. Anatomical correctness requires separate review.

## Outstanding verification requirements

The separate QA package could not be located in the GitHub branches, reachable history, pull request discussions, Actions artifacts, or releases inspected for Issue #15. Its exact location and access terms remain unconfirmed. Aggregate team findings are recorded in the pull requests. [Issue #15](https://github.com/Dilyannn/project-ERUA-NBU/issues/15) · [PR #24](https://github.com/Dilyannn/project-ERUA-NBU/pull/24) · [PR #26](https://github.com/Dilyannn/project-ERUA-NBU/pull/26)

- **Dataset authorization pending.** Obtain written institutional and team authorization and confirm image and mask terms and approved storage.
- **Archive authenticity pending.** Confirm the release and source, obtain trusted checksums where available, and record the measured archive and checkpoint SHA-256 values.
- **Subject mapping pending.** Obtain the QA package and trusted mapping. Confirm original identifiers across eyes and sessions, including the three unusual frame indices, before freezing the split.
- **Independent real-data verification pending.** Repeat the inventory and original-folder audit, review the empty reference with the annotation custodian, agree on any further visual QA, and reproduce the diagnostic smoke test and scoring.
- **Checkpoint training independence pending.** Request checkpoint-specific training and validation provenance, then check potential image and subject overlap through authorized channels. Keep evaluation diagnostic until independence is established.

## Scientific references

- Hugo Proença, Sílvio Filipe, Ricardo Santos, João Oliveira, and Luís A. Alexandre. *The UBIRIS.v2: A Database of Visible Wavelength Iris Images Captured On-the-Move and At-a-Distance*. IEEE TPAMI, 32(8), 1529–1535, 2010. DOI `10.1109/TPAMI.2009.66`. [IEEE](https://ieeexplore.ieee.org/document/4815254)
- Haiqing Li, Caiyong Wang, Guangzhe Zhao, Zhaofeng He, Yunlong Wang, and Zhenan Sun. *Sclera-TransFuse: Fusing Swin Transformer and CNN for Accurate Sclera Segmentation*. IEEE IJCB, 1–8, 2023. DOI `10.1109/IJCB57857.2023.10448814`. [IEEE](https://ieeexplore.ieee.org/document/10448814)
- Caiyong Wang, Haiqing Li, Yixin Zhang, Guangzhe Zhao, Yunlong Wang, and Zhenan Sun. *Sclera-TransFuse: Fusing Vision Transformer and CNN for Accurate Sclera Segmentation and Recognition*. IEEE TBIOM, 6(4), 575–590, 2024. DOI `10.1109/TBIOM.2024.3415484`. [IEEE](https://doi.org/10.1109/TBIOM.2024.3415484)
- Diego R. Lucio, Rayson Laroca, Evair Severo, Alceu S. Britto Jr., and David Menotti. *Fully Convolutional Networks and Generative Adversarial Networks Applied to Sclera Segmentation*. IEEE BTAS, 1–7, 2018. DOI `10.1109/BTAS.2018.8698597`. [Author manuscript](https://arxiv.org/abs/1806.08722) · [UFPR research group](https://web.inf.ufpr.br/vri/databases/sclera-segmentation-annotations/)

The UBIRIS.v2 paper and the Lucio et al. paper were available as author copies. The IEEE BTAS publisher copy could not be inspected, so its DOI and venue were checked against UFPR's citation. The full IEEE texts of both Sclera-TransFuse papers could not be inspected. The publication details were checked against author materials and an author-coauthored bibliography. The experimental descriptions here rely on the accessible poster, journal abstract, and implementation. Those sources do not establish the training history of the released checkpoint. [Author bibliography](https://academia.caiyong.wang/Publications/) · [MICCAI bibliography, references 10–11](https://papers.miccai.org/miccai-2025/paper/2215_paper.pdf) · [IJCB poster](https://wylcasia.github.io/papers/IJCB2023-2-poster.pdf) · [Journal abstract](https://nlpr-sir.github.io/publication/wang-tbiom-2024-1/)

## Reproducibility checklist

- [ ] Confirm written access and mask-use authorization. Record the approved archive source and release and the exact checkpoint. Compare their SHA-256 hashes with trusted values where available.
- [ ] Record the integration and upstream commits, operating system, Python and dependency versions, and inference device.
- [ ] Record importer settings and aggregate inventory and original-folder audit counts. Resolve reference QA and verify the private subject mapping before freezing.
- [ ] Save the split algorithm, seed, ratios, snapshot SHA-256, and aggregate audit results. Regenerate and validate the snapshot with independent trusted inputs.
- [ ] Record the smoke-test selection, RGB crop boxes, settings, and mask scoring tolerance in private storage. Use synthetic or validation samples for development and freeze decisions before opening test.
- [ ] Verify source-resolution outputs, scoring coverage, and unchanged input hashes. Record test results and unresolved real-data or training-independence checks alongside any authorized aggregate scores.
