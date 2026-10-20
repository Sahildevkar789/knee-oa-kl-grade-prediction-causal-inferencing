# Knee OA KL-Grade Prediction — Project Context

BE major project: predict Kellgren-Lawrence (KL) grade (0-4) for knee osteoarthritis
severity from X-rays, with explainability (XAI) and causal inference layered on top.
Causal inference is PARKED for later. Current focus: the KL-grade classifier.

## Hardware
- Local Windows laptop, RTX 3050 (6GB VRAM). Keep training memory-optimized:
  mixed precision (torch.amp), modest batch sizes, save the best checkpoint.
- Classifier: batch size 32 at 224x224 (drop to 16 on CUDA out-of-memory).
- On Windows, if DataLoader errors mention workers/pickling/paging file, use
  `--workers 0`.

## Folder layout (run everything from the project root)
- `dataset/{train,val,test}/{0..4}/*.png` — OAI images (Kaggle split). The scripts
  read these folders directly.
- `data/` — `image_manifest.csv`, `clinical_features.csv` (see below).
- `unified_oai_dataset.csv` — raw merged OAI clinical + image table.
- `segmentation_data/` — CGMH knee segmentation dataset (400 image/mask pairs).
- `notebooks/` — Colab segmentation notebook, kept as a record only. Do NOT rerun
  or convert it; segmentation is finished.
- `outputs/checkpoints/` — `unet_best.pt`, `unetpp_best.pt` (trained in Colab,
  copied from Google Drive); `kl_final_best.pt` + `kl_final_config.json` (the locked
  classifier, see below); other `kl_*_best.pt` from the backbone/ablation search.
- `outputs/results/` — training logs, test reports, confusion matrices,
  `kl_comparison.csv`, `backbone_comparison.csv`, `ablation.csv`, `final_test_report.csv`,
  `significance_final_vs_baseline.csv`, `classifier_summary.md`, `overnight_log.md`,
  `cbam_maps/`. `outputs/predictions/` — single-image prediction figures.
  `outputs/preprocessed/` — crops + debug images from `preprocess_xray.py`.
- `scripts/prepare_dataset.py` — builds the two CSVs in `data/`.
- `scripts/kl_common.py` — shared dataset, transforms, models, eval helper.
- `scripts/train_classifier.py`, `scripts/evaluate_classifier.py`, `scripts/predict.py`.
- `scripts/preprocess_xray.py` — auto-crops an arbitrary X-ray to a 224x224 joint
  crop (see "Preprocessing" section below).

## Datasets — decisions already made, don't redo this analysis
Source: OAI (Osteoarthritis Initiative), originally a Kaggle-derived export.

- **Images**: `dataset/train`, `dataset/val`, `dataset/test` (grade 0-4 subfolders).
  **`auto_test` is dropped** — verified near-duplicate of `test` (811 of 828 test
  patients also appear there under different filenames), not new data.
- **Clinical data**: originally 3 files (`ALLClinical.csv`, `final_project_master.csv`,
  `unified_oai_dataset.csv`). **Only `unified_oai_dataset.csv` is used** — strict
  superset of the other two.
- **Verified**: train/val/test have ZERO patient overlap (checked by patient ID).
- **Class imbalance**: real and consistent across splits (stratified). Train
  distribution: 0=39.6%, 1=18.1%, 2=26.2%, 3=13.1%, 4=3.0%. Handled with
  **class-weighted cross-entropy** (inverse frequency). No weighted sampler unless
  KL3/KL4 recall stays weak after weighted loss alone. No SMOTE (tabular only).

### Generated files in `data/` (via `scripts/prepare_dataset.py`, 2026-09-22)
- `image_manifest.csv` — ID, SIDE_CODE, SPLIT, IMAGE_NAME, IMAGE_PATH, KL_GRADE.
  8,260 rows (train 5,778 / val 826 / test 1,656). One file
  (`dataset/test/4/9215922R.png`) is missing from disk — pre-existing gap in the
  source data; harmless (1 of 8,260). The classifier scripts read folders directly,
  so the test set they use has 1,655 images.
- `clinical_features.csv` — filtered clinical columns for the causal phase (parked).
  Currently holds age (`V00AGE`), BMI (`P01BMI`), WOMAC pain/function R/L,
  family history (`P01FAMHR`/`P01FAMKR`), lateral JSN R/L, osteophyte R/L.
  **Correction:** the original selection was 14 clinical columns, including BOTH
  lateral and medial JSN (`P01SVRKJSL`, `P01SVRKJSM`, `P01SVLKJSL`, `P01SVLKJSM`);
  the "12 columns" figure in the old notes was wrong. Add the two medial JSN columns
  back when the causal phase resumes (medial JSN is clinically the more important one).

## Segmentation — DONE (trained in Colab)
- Ground truth: **CGMH Knee Segmentation Database**
  (github.com/yaufan/Knee_Segmentation_Database) — 400 AP knee X-rays with
  femur+tibia masks (BoneFinder), MIT licensed. Different cohort from OAI.
- Setup (all models): 320 train / 80 val, seed 42, 256x256, flip/rotate ±10°/
  brightness-contrast augmentation, T4 GPU, mixed precision. U-Net, U-Net++ and
  DeepLabV3+ used a ResNet34 ImageNet encoder, Dice loss, Adam 1e-4, batch 8.
  Mask R-CNN: ResNet50-FPN COCO-pretrained, SGD 0.005.
- Pixel-level results on the 80 val images:
  - U-Net (40 epochs):      P 0.9727, R 0.9728, F1 0.9727, IoU 0.9469
  - U-Net++ (40 epochs):    P 0.9756, R 0.9720, F1 0.9738, IoU 0.9490  <- selected
  - DeepLabV3+ (20 epochs): P 0.9743, R 0.9500, F1 0.9620, IoU 0.9268
  - Mask R-CNN (20 epochs): P 0.9571, R 0.9644, F1 0.9607, IoU 0.9244
- **Carrying forward U-Net and U-Net++ only.** U-Net++ leads U-Net by only ~0.1% F1.
- Known issue: on OAI images the masks are patchy. CGMH shows the full leg with the
  knee small in the middle; OAI images are tight joint close-ups (scale mismatch).
  A retrain on OAI-style crops of CGMH was planned but NOT done.
- Because OAI images are already cropped to the joint, mask-based cropping is not
  needed before classification. Masks are used as an optional 4th input channel
  (see below) and later for Grad-CAM validation.

## Classifier — LOCKED as of 2026-10-21 (`kl_final`)
Full history: backbone comparison → warmup check → 5-axis ablation → final model → test +
bootstrap → wired into `predict.py`. Done as one unattended overnight run; every decision
and its reason is in `outputs/results/overnight_log.md` (including a real bug caught and
fixed mid-run: `coral_pos_weights` initially ignored `--class-weights` entirely). Full
write-up with tables: `outputs/results/classifier_summary.md`.

### Final model: `kl_final`
**Pipeline**: grayscale → CLAHE (clipLimit 2.0, 8x8 tiles) → convnext_tiny (ImageNet
pretrained, 2-epoch linear LR warmup then cosine) → CBAM (channel+spatial attention) on the
final feature map → CORAL ordinal head (4 binary "grade>k" outputs, inverse-frequency
pos_weight per threshold) → predicted grade = count of thresholds with sigmoid > 0.5.
Checkpoint: `outputs/checkpoints/kl_final_best.pt` (seed 42, the better of 2 seed runs at
this config). Full option list: `outputs/checkpoints/kl_final_config.json`.

**Test results** (1,655 images, evaluated once): Accuracy **0.6937**, QWK **0.8570**,
Macro F1 **0.6926**, MAE **0.3396** — vs the original `kl_baseline` (EfficientNet-B0,
image-only): Accuracy 0.6284, QWK 0.8246, Macro F1 0.6746, MAE 0.4036. Paired bootstrap
(2,000 resamples) confirms both the QWK gain (+0.0324, 95% CI [0.017, 0.048]) and accuracy
gain (+0.0653, 95% CI [0.040, 0.092]) are statistically significant.
`outputs/results/final_test_report.csv`, `significance_final_vs_baseline.csv`.

**How it was built** (val QWK only, test set untouched until the final eval above):
1. **Backbone**: 4 ImageNet backbones, image-only, 3 seeds each for the top contenders.
   convnext_tiny won on mean QWK (0.8187±0.011 across seeds 42/7/123) vs efficientnet_b0's
   0.8078±0.003 — note convnext_tiny's variance is ~3.5x higher, so its single-seed Step 1
   number was partly a favorable draw.
2. **Warmup**: 2 epochs kept (0.8315 vs 0.8312 without — a tie, but it visibly smoothed the
   early-epoch instability convnext_tiny otherwise shows, e.g. epoch-3 QWK dip 0.43→0.59).
3. **Ablation** (2 seeds each, sequential, each change built on the prior best): **+CLAHE**
   adopted (0.8286→0.8340, and variance shrank too). **+CBAM** adopted (→0.8368). **+CORAL**
   adopted (→0.8416, biggest single gain, best KL0 recall of any config at 0.80). **class-
   weights sqrt** rejected (→0.8346, gutted KL3 recall 0.79→0.66 for little KL4 gain). **TTA**
   (horizontal-flip averaging) rejected (→0.8363) — plausibly because flipping a knee X-ray
   swaps medial/lateral anatomy rather than being a harmless symmetry.
4. CBAM attention maps for 10 val images (2/grade): `outputs/results/cbam_maps/`.

KL1 ("doubtful OA") is still the weakest class for both models — a known, clinically
subjective boundary even between radiologists, not something either architecture resolved.

### Superseded: original 3-run comparison (2026-09-25)
Before the work above, `kl_baseline`/`kl_unet`/`kl_unetpp` (all EfficientNet-B0, image vs
image+U-Net-mask vs image+U-Net++-mask) were compared on QWK: baseline 0.8246, unet 0.8130,
unetpp 0.8144 — baseline won, consistent with the Segmentation section's patchy-mask caveat
(those two still use the original CGMH-trained, full-leg-scale U-Net/U-Net++ checkpoints,
which is why `kl_final` doesn't use a mask channel at all). These checkpoints and results
are kept on disk (`kl_{baseline,unet,unetpp}_best.pt` and their `_test_*` files) and still
work via `predict.py --run baseline|unet|unetpp` — only the *default* run changed to `final`.

## Preprocessing for arbitrary input images — DONE (2026-10-18)
`scripts/preprocess_xray.py` converts any knee X-ray (not just pre-cropped OAI-style
images) into one or more 224x224 crops matching the training data, so `predict.py`
can be pointed at a raw/uncropped X-ray from anywhere.

- `prepare_xray(image_path, seg_model=None, device=None) -> list of (label, 224x224x3 RGB)`.
  Loads a pre-loaded U-Net++ model if none is passed in.
- Pipeline: load (PNG/JPG/BMP/TIFF/DICOM via pydicom, with window/rescale + MONOCHROME1
  handling) → grayscale → trim uniform white/black borders → split into left/right
  halves if the image is much wider than tall (two knees) → find the joint → crop →
  resize to 224x224.
- **Joint-finding**: runs the U-Net++ checkpoint (it was trained on CGMH's full-leg
  images, so it works well on uncropped input - this is the point of reusing it here,
  separate from its patchy performance as a classifier mask channel noted above) to
  get a bone-probability map, then locates the row where bone coverage dips between
  two bone masses (femur above, tibia below). **Must normalize the model's input with
  the same ImageNet MEAN/STD used everywhere else in this project** - feeding it raw
  [0,1] pixels produces an unusable, near-empty mask even though the checkpoint itself
  is fine; this cost real debugging time, don't reintroduce it.
- Crop size = `knee_width_at_joint * SCALE_FACTOR` (1.15, calibrated by eye against
  real `dataset/test` crops - bones should fill nearly the full 224x224 frame, matching
  training data, not leave a visible margin).
- **Already-cropped detection**: if the input already looks like a dataset-style crop
  (square, joint centered, bones filling the width, AND almost nothing was trimmed off
  as border - this last check matters: a full-leg image can *coincidentally* end up
  near-square and bone-filling after its letterbox bars are trimmed, so aspect ratio
  alone isn't enough to tell it apart from a real tight crop), it's resized as-is
  instead of re-cropped.
- **Fallback chain**: mask → intensity (darkest horizontal band in the central region,
  only used if there's a real dip - a flat/featureless image doesn't count as found) →
  manual click (opens a window, you click the joint center). The click fallback is
  untested here since triggering it needs an interactive display; the mask+intensity
  chain was enough for every test image, including two low-quality screenshots that
  previously gave bad predictions (see below).
- **DICOM loading is implemented but untested** - no `.dcm` sample file was available.
  Verify against a real file before relying on it.
- Verified on: 5 full-leg CGMH images (`segmentation_data/Data/Image/`), 3 real
  `dataset/test` crops (correctly detected as already-cropped and passed through), a
  synthetic bilateral image (two real crops stitched side-by-side - split correctly),
  and the two bad screenshot images from `Downloads/` that misled the classifier
  earlier in the project (`grade1.png`, `kl grade2.png` - mostly blank canvas with a
  tiny embedded X-ray). Border-trimming alone fixes those: `grade1.png`'s prediction
  changed from KL3 at 30.7% confidence (wrong) to KL1 at 43.0% (correct) once run
  through `--auto-crop`. Comparison figure: `outputs/results/preprocess_check.png`.
- **`predict.py --auto-crop`** (works in both direct-path and file-browser modes):
  runs `prepare_xray` first, classifies each resulting knee separately (bilateral
  images produce multiple results), and shows the *cropped* image in the result
  window/figure, not the original.

## XAI (after the classifier)
- Grad-CAM on the best classifier; check attention overlaps the joint region using
  the U-Net++ masks.

## Causal inference — PARKED, do not build yet
- Use `clinical_features.csv` (age, BMI, WOMAC, family history) plus imaging
  features. OAI already includes radiologist-graded JSN/osteophyte scores, usable
  directly or to validate mask-derived measurements.
- Needs ONE scoped causal question (e.g. "does BMI causally affect KL grade,
  independent of age and family history"). Tools: DoWhy or CausalNex.

## Environment setup
- venv at `venv/` uses **Python 3.12**, not the system default 3.14 — PyTorch CUDA
  wheels aren't built for 3.14. Created with `py -3.12 -m venv venv`.
- Installed: `torch==2.14.0+cu126` (CUDA confirmed on the RTX 3050), numpy, pandas,
  pillow, scikit-learn, matplotlib. **Do not reinstall torch.**
- `requirements.txt` installed (segmentation-models-pytorch, albumentations,
  opencv-python, scipy, plus pandas/scikit-learn/matplotlib already present) —
  confirmed torch stayed the CUDA build afterwards.
- Git installed via `winget install --id Git.Git`.
- Network is flaky: large downloads can stall at 0 B/s. If there's no byte progress
  for a while, kill and retry. `git config http.postBuffer 524288000` plus a
  shallow `--depth 1` clone fixed the dataset clone.

## Commands
python scripts/train_classifier.py --backbone efficientnet_b0|resnet50|densenet121|convnext_tiny \
    [--mask-source none|unet|unetpp] [--warmup-epochs N] [--clahe] [--attention none|cbam] \
    [--loss ce|coral] [--class-weights inverse|sqrt|none] [--seed N] [--ckpt-name NAME]
python scripts/evaluate_classifier.py [--ckpt-name NAME | --mask-source none|unet|unetpp] \
    [--backbone ...] [--attention ...] [--loss ...] [--clahe] [--tta] [--split val|test]
python scripts/predict.py path/to/xray.png --run final|baseline|unet|unetpp [--show] [--auto-crop]
python scripts/preprocess_xray.py path/to/xray.png --out-dir outputs/preprocessed
(all new flags default to the original behavior; `--run final` is predict.py's new default)

## Current status
Done: dataset cleanup; Python 3.12 venv with PyTorch+CUDA; CGMH segmentation (4 models,
U-Net/U-Net++ selected); `preprocess_xray.py` for arbitrary-input auto-cropping; and the
full classifier optimization pass described above — `kl_final` (convnext_tiny + CLAHE +
CBAM + CORAL) is now the **locked, default model**, significantly beating the original
EfficientNet-B0 baseline on the test set (QWK 0.8570 vs 0.8246, bootstrap-confirmed
significant). Wired into `predict.py` as the default `--run final`.

Next: Grad-CAM-style XAI on `kl_final` (note: CORAL's 4-threshold head means standard
Grad-CAM needs adapting — e.g. run it per threshold output, or on the derived per-grade
probability), validated against the U-Net++ masks. Then causal inference (still parked,
needs one scoped question). Open follow-ups, not urgent: validate DICOM loading in
`preprocess_xray.py` against a real file (still untested), get a real bilateral X-ray to
test the split logic (only synthetically tested), and consider more epochs for CORAL
specifically (its seed-42 ablation run was still improving at the epoch budget's edge).
