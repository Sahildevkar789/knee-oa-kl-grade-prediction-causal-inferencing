# Knee OA KL-Grade Prediction — Project Context

BE major project: predict Kellgren-Lawrence (KL) grade (0-4) for knee osteoarthritis
severity from X-rays, with explainability (XAI) and causal inference layered on top.
**All phases done as of 2026-10-21**: classifier locked (`kl_final`), two hybrid
architectures tried and not adopted, JSW measurement (not validated), XAI (Grad-CAM/SHAP),
and one scoped causal question answered (obesity → KL grade). Full write-up:
`outputs/results/final_results.md`. A web app is deferred to a later stage; the current
interface is `predict.py`'s terminal demo (Grad-CAM + JSW panels, see Commands below).

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
  classifier, see below); `kl_hybrid1_best.pt`/`kl_hybrid2_best.pt` + their config
  JSONs (tried, not adopted - see Classifier section); other `kl_*_best.pt` from the
  backbone/ablation search.
- `outputs/results/` — training logs, test reports, confusion matrices,
  `kl_comparison.csv`, `backbone_comparison.csv`, `ablation.csv`, `final_test_report.csv`,
  `significance_*_vs_*.csv`, `classifier_summary.md`, `overnight_log.md` (classifier-lock
  decision trail), `final_build_log.md` (Phases 1-7 decision trail), `final_results.md`
  (the full write-up), `cbam_maps/`, `xai/` (Grad-CAM/SHAP outputs), `causal/` (DAG,
  estimates, refutation results). `outputs/predictions/` — single-image prediction figures.
  `outputs/preprocessed/` — crops + debug images from `preprocess_xray.py`.
- `data/jsw_features.csv` — per-image joint space width measurements (see JSW section).
- `scripts/prepare_dataset.py` — builds the two CSVs in `data/`.
- `scripts/kl_common.py` — shared dataset, transforms, models, eval helper.
- `scripts/train_classifier.py`, `scripts/evaluate_classifier.py`, `scripts/predict.py`.
- `scripts/preprocess_xray.py` — auto-crops an arbitrary X-ray to a 224x224 joint
  crop (see "Preprocessing" section below).
- `scripts/train_hybrid2.py` / `scripts/evaluate_hybrid2.py` — Hybrid 2 (image+tabular
  fusion) training/eval, kept separate from train_classifier.py since its two-input
  (image, tabular) forward pass doesn't fit the shared single-image loop.
- `scripts/measure_jsw.py` — intensity-based joint space width measurement.
- `scripts/run_xai.py`, `scripts/run_shap_hybrid2.py`, `scripts/run_causal.py` — one-off
  analysis scripts for Phases 4/5, kept under `scripts/` for the record (not reusable CLIs).
- `scripts/causal_model.py` — fits (or loads, from `outputs/results/causal/causal_query_models.json`)
  the per-factor causal models (age total/direct/indirect-via-BMI, BMI, family history, obesity
  by age group) behind `causal_query.py` and `predict.py`'s causal risk panel.
- `scripts/causal_query.py` — interactive causal "what-if" tool; see Commands below.
- `scripts/bootstrap_compare.py` — paired bootstrap significance test between any two
  `*_test_predictions.csv` files; reusable.

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
- `clinical_features.csv` — filtered clinical columns, used by Hybrid 2 (age/BMI) and
  the causal analysis (age/BMI/family history). Holds age (`V00AGE`), BMI (`P01BMI`),
  WOMAC pain/function R/L, family history (`P01FAMHR`/`P01FAMKR`), lateral JSN R/L,
  osteophyte R/L. **Correction:** the original selection was 14 clinical columns,
  including BOTH lateral and medial JSN (`P01SVRKJSL`, `P01SVRKJSM`, `P01SVLKJSL`,
  `P01SVLKJSM`); the "12 columns" figure in the old notes was wrong.
  **No sex/gender column exists anywhere in `unified_oai_dataset.csv`** (checked
  exhaustively across all 1,193 columns) - this affected Hybrid 2's feature set and the
  causal adjustment set; both proceed with age/BMI (+family history for causal) only,
  logged rather than worked around. WOMAC/JSN/osteophyte columns are present but **must
  never be used as model inputs or causal adjustment covariates** (hard rule - they're
  consequences of OA, not causes/predictors to condition on); they're only used to
  *validate* JSW in Phase 2.

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

### Hybrid architectures tried — NOT adopted (2026-10-21/22)
Two more architectures were built on top of `kl_final` and compared the same way (val QWK,
2 seeds, test set touched once at the end). Full write-up: `outputs/results/final_results.md`
section 5; decision trail: `outputs/results/final_build_log.md` Phases 1 and 3.

- **Hybrid 1** (`kl_hybrid1_best.pt`): `kl_final`'s ConvNeXt-Tiny+CBAM feature map → 2-layer
  Transformer encoder (CLS token, learnable positional embeddings, 4 heads) → CORAL head.
  Mean val QWK 0.8320 (both seeds peaked at epoch 1 then degraded - a repeatable instability
  pattern, not seed noise). Test QWK 0.8620 (numerically above kl_final's 0.8570) but the
  bootstrap difference is NOT significant (95% CI [-0.002, 0.013]).
- **Hybrid 2** (`kl_hybrid2_best.pt`): `kl_final`'s pooled image features + a small MLP over
  age/BMI (sex requested but unavailable, see above) → CORAL head. Mean val QWK 0.8170 (same
  early-peak-then-degrade pattern). Test QWK 0.8598, also not significantly different from
  kl_final (95% CI [-0.007, 0.013]).

Both lost on validation QWK, the pre-registered selection criterion - the higher test numbers
don't change that, since test is for confirmation, not selection. **`kl_final` stays the
locked model.** `scripts/train_hybrid2.py`/`evaluate_hybrid2.py` exist if Hybrid 2 is revisited
(e.g. with a lower LR or a frozen-backbone warmup to address the early-epoch instability).

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

## Joint Space Width (JSW) — measured, NOT validated (2026-10-21)
`scripts/measure_jsw.py`: intensity-based JSW on CLAHE-enhanced images - per-column
femur/tibia edge detection with a local joint-line refinement (anchoring every column to one
global row made most columns spuriously read zero - the joint line isn't perfectly
horizontal), medial/lateral split skipping the central ~20% (tibial spines) plus a matching
outer-edge margin (condyle curvature). Ran on all 8,259 images → `data/jsw_features.csv`.

Validated against radiologist JSN grades (match on patient ID + knee side,
`unified_oai_dataset.csv`'s `P01SV{R,L}KJS{L,M}` columns): medial_min_jsw vs matching JSN
rho=-0.179 - correct direction, clean monotonic box-plot trend, but below the -0.3 validation
threshold. **Consequence**: JSW excluded from Hybrid 2's features and from the causal
mediation analysis (both per the pre-registered fallback rules). Details:
`outputs/results/final_results.md` section 6, decision trail in `final_build_log.md` Phase 2.

## XAI — DONE (2026-10-21/22)
Grad-CAM (last ConvNeXt stage, target = sum of CORAL's 4 threshold logits - no single
softmax class score to backprop from) on `kl_final` and Hybrid 1, 15 test images (3/grade,
5 wrong predictions). Quantitative check: fraction of Grad-CAM mass inside a joint-line band
(±15% of image height, using the JSW joint-row detector) - KL4 images show much lower in-band
mass (0.24 vs 0.49-0.60 for KL0-3), traced to the model correctly attending to bilateral
femoral osteophytes sitting just above the band for severe cases, not a failure. CBAM
attention maps saved alongside. SHAP on Hybrid 2's tabular features (per-instance multi-modal
setup, 60 test instances): age matters somewhat more than BMI, both push toward higher
predicted grades. All in `outputs/results/xai/`; write-up in `final_results.md` section 7.
`grad_cam()` lives in `kl_common.py`, shared between `run_xai.py` and `predict.py`.

## Causal inference — DONE, one scoped question answered (2026-10-22)
**Question**: does obesity (BMI≥30) causally affect KL grade, adjusting for age and family
history of knee replacement (`P01FAMKR`)? Sex omitted (unavailable, see above). WOMAC/JSN/
osteophyte never used as adjustment covariates (hard rule - they're OA consequences, not
causes). `scripts/run_causal.py` (DoWhy): DAG saved to `outputs/results/causal/dag.png`.

Primary analysis (one knee per patient, random seed 42, n=4,071): obesity associated with
**+0.48 to +0.58 higher KL grade points** (propensity score matching and linear regression
agree), **+19-25 percentage points** higher probability of KL≥2. All three refuters pass
(placebo collapses to ~0, random common cause and data subset don't move the estimate).
Sensitivity analysis (both knees instead of one) found a genuine methodological issue, not
just confirmed robustness: propensity score matching collapses toward zero because both knees
of the same patient share identical covariates, causing degenerate same-patient matches -
concrete evidence for why the primary analysis needs the one-knee restriction. Linear
regression stays stable either way. Mediation analysis (BMI → JSW → KL grade) skipped - JSW
not validated. **Stated under this DAG's assumptions only**; unmeasured confounders (physical
activity, prior injury, occupation, genetics beyond family history) are a real limitation.
Full write-up: `final_results.md` section 8; decision trail: `final_build_log.md` Phase 5.

## Environment setup
- venv at `venv/` uses **Python 3.12**, not the system default 3.14 — PyTorch CUDA
  wheels aren't built for 3.14. Created with `py -3.12 -m venv venv`.
- Installed: `torch==2.14.0+cu126` (CUDA confirmed on the RTX 3050), numpy, pandas,
  pillow, scikit-learn, matplotlib. **Do not reinstall torch.**
- `requirements.txt` installed (segmentation-models-pytorch, albumentations,
  opencv-python, scipy, pydicom, shap, dowhy, plus pandas/scikit-learn/matplotlib
  already present) — confirmed torch stayed the CUDA build each time something new
  was added (shap and dowhy both have their own dependency trees; neither touched torch).
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
python scripts/predict.py path/to/xray.png --run final|baseline|unet|unetpp [--show] [--auto-crop] \
    [--age N] [--bmi N] [--family-history yes|no]
python scripts/preprocess_xray.py path/to/xray.png --out-dir outputs/preprocessed
python scripts/measure_jsw.py --out data/jsw_features.csv
python scripts/train_hybrid2.py --seed 42|7
python scripts/evaluate_hybrid2.py --ckpt-name kl_hybrid2 --split val|test
python scripts/bootstrap_compare.py <preds_a.csv> <preds_b.csv> <label_a> <label_b> <out.csv>
python scripts/causal_query.py [--age N] [--bmi N] [--family-history yes|no] [--refit]
(all new flags default to the original behavior; `--run final` is predict.py's new default)

### Terminal demo (Phase 6) - predict.py
python scripts/predict.py                              # file-browser mode, repeats until Cancel
python scripts/predict.py path/to/xray.png --show       # single file, pop-up window
python scripts/predict.py path/to/xray.png --auto-crop  # arbitrary/uncropped input, crops first
python scripts/predict.py path/to/xray.png --age 65 --bmi 32   # see note below
(--age/--bmi are accepted but currently inert *for model choice*: Hybrid 2 never beat kl_final
on val (0.8170 vs 0.8416), so kl_final is used regardless and a note explains why - this is
intentional, not a bug, and documented in HYBRID2_BEAT_FINAL at the top of predict.py. --sex is
accepted for forward-compatibility only; no model uses it, since no sex column exists in the
dataset. The result window/figure has 4 panels for `--run final`: input X-ray, Grad-CAM, bar
chart, causal risk panel (3 panels - no Grad-CAM - for `--run baseline|unet|unetpp`, since
Grad-CAM/JSW are specific to the CORAL-head final model). `--show-jsw` adds a 5th panel, labeled
"JSW (experimental, not validated)" (and the matching terminal line) - off by default since
Phase 2 found JSW didn't meet the validation threshold (rho=-0.179 vs -0.3). `--no-xai` hides
Grad-CAM instead, back to the 3-panel layout:
```
python scripts/predict.py path/to/xray.png                  # default: input, Grad-CAM, bar, causal (4 panels)
python scripts/predict.py path/to/xray.png --show-jsw        # + JSW panel (5 panels)
python scripts/predict.py path/to/xray.png --no-xai           # no Grad-CAM: input, bar, causal (3 panels)
```
**A web app is deferred to a later stage** - this terminal demo is the interface for now.

### Causal risk panel (added 2026-10-21/22) - asks for age/BMI/family history
After an image is chosen (file-browser or single-file mode), predict.py prints "Enter patient
details (press Enter to skip):" then asks for age, BMI, and family history of knee replacement
(y/n) - press Enter to skip any of them. `--age`, `--bmi`, `--family-history yes|no` skip their
prompt individually (useful for scripting/non-interactive runs - EOF on stdin is also treated as
"skip", so it degrades gracefully without a terminal attached; if all three are given on the
command line, the header/prompts are skipped entirely). **These values never reach the
classifier** - the predicted KL grade always comes from the image alone via `kl_final` (or
whichever `--run`); they only feed the causal risk panel, appended as one more panel on every
result figure and printed in the terminal:
```
python scripts/predict.py path/to/xray.png --age 65 --bmi 33 --family-history yes   # obese
python scripts/predict.py path/to/xray.png --age 55 --bmi 24 --family-history no    # not obese
python scripts/predict.py path/to/xray.png                                          # prompts; Enter to skip all
```
Example terminal output (age/BMI/family history given on the command line, so no prompt):
```
Image: path/to/xray.png
Predicted KL grade: 2 - Minimal (67.3% confidence)
  KL0:  13.1%
  KL1:  19.5%
  KL2:  67.3%
  KL3:   0.0%
  KL4:   0.0%

  Causal risk panel (population-level estimates, not derived from this image):
    This patient: age 65, BMI 33.0 -> obese (threshold 30), family history yes.
    Expected KL grade: 1.75 (reference age 50/BMI 25/no family history: 0.70).
    Age contribution (65 vs 50): total +0.34 (95% CI 0.28 to 0.40) = direct +0.38 (95% CI 0.32 to 0.44) + via BMI -0.04 (95% CI -0.06 to -0.02)
    BMI contribution (33.0 vs 25): +0.53 (95% CI 0.47 to 0.59)
    Family history contribution: +0.14 (95% CI 0.04 to 0.24)
    Obesity effect for this age group (65+): +0.41 KL grade points (95% CI 0.28 to 0.55)
    Caveat: These are population-level causal estimates from OAI data under this DAG's assumptions, not individual predictions.
Saved result image to outputs/predictions/xray_prediction.png
```
If age or BMI is skipped, the panel shows just "Age/BMI not given - need both for a per-factor
breakdown." plus the caveat. If family history is skipped but age/BMI are given, falls back to
the population's most common value (no) and says so. The underlying effects (age total/direct/
indirect-via-BMI, BMI, family history, obesity-by-age-group) come from `causal_model.py`,
cached at `outputs/results/causal/causal_query_models.json` - same source as the interactive
tool below, so the two always agree.

If run through this assistant's sandboxed terminal (no real console attached), the prompts
auto-skip on EOF instantly instead of waiting - that's a property of that sandbox, not of
predict.py; run it in your own terminal to actually be prompted and type an answer.

### Causal "what-if" tool (added 2026-10-22) - scripts/causal_query.py
Interactive per-factor causal exploration, independent of any image - the same models behind
predict.py's causal risk panel above (`causal_model.py`, cached at
`outputs/results/causal/causal_query_models.json`; same data/DAG spirit as Phase 5: one knee
per patient, seed 42, never adjusting for WOMAC/JSN/osteophyte grades). Full methodology and
results table: `outputs/results/final_results.md` section 9.
```
python scripts/causal_query.py                                        # prompts for age/BMI/family history, loops
python scripts/causal_query.py --age 65 --bmi 33 --family-history yes # answers once, non-interactive
python scripts/causal_query.py --refit                                # refits all models from data instead of the cache
```
For each profile it prints: the expected KL grade (vs. the reference profile age 50/BMI 25/no
family history); each factor's contribution with a 95% CI (age split into total/direct/
indirect-via-BMI); the obesity effect for that patient's age group (<55/55-65/65+); then asks
"Change BMI to?" to see how the expected KL grade changes (blank to stop, loops). Blank age or
BMI at the main prompt quits. Every output repeats the caveat that these are population-level
estimates under the DAG's assumptions, not individual predictions.

**Out-of-range age/BMI is refused, not computed** (added 2026-10-23, same in `predict.py`'s
causal panel - they share `causal_model.out_of_range_messages()`): OAI only enrolled ages
45-79 and BMI 16.9-48.7, so this is a straight line fit to that range - extrapolating it (e.g.
to a child) isn't a cautious estimate, it's nonsense (a real example caught in testing: age 11
gave an expected KL grade of -0.62, below the valid 0-4 range). Querying age 11 or BMI 55
prints e.g. "Age 11 is outside the OAI study population (45-79). The causal model can't make
estimates for this profile." and asks again (or, non-interactively via `--age`/`--bmi`, just
exits) instead of showing a number. As a second line of defense, `expected_kl()` also clips
any displayed KL grade to 0-4 and says so if clipping changed the number (e.g. "0.00 (clipped
from raw model estimate -0.62; KL grades run 0-4)") - this matters for the what-if's "Change
BMI to?" step, which checks the new BMI against the same range before computing anything.
When a factor's value equals the reference (e.g. age 50 vs. the reference age 50), its
contribution line reads "same as reference (no contribution)" rather than a value that could
render as "+0.00" or the cosmetically odd "+-0.00" (a negative-coefficient effect times a zero
delta is IEEE754 negative zero).

## Current status
**All planned phases are done.** Dataset cleanup; Python 3.12 venv with PyTorch+CUDA; CGMH
segmentation (4 models, U-Net/U-Net++ selected); `preprocess_xray.py` for arbitrary-input
auto-cropping; classifier backbone search → ablation → `kl_final` locked (convnext_tiny +
CLAHE + CBAM + CORAL, test QWK 0.8570 vs baseline's 0.8246, bootstrap-significant); two hybrid
architectures tried and not adopted (Transformer head, image+tabular fusion - neither beat
`kl_final` on validation); JSW measured but not validated (rho=-0.179 vs -0.3 threshold);
XAI done (Grad-CAM, quantitative joint-region check, CBAM maps, SHAP); one scoped causal
question answered (obesity → KL grade, +0.48 to +0.58 points under the DAG's assumptions,
refutation-robust); `predict.py` extended into a terminal demo (Grad-CAM + JSW panels,
default `--run final`). Full write-up: `outputs/results/final_results.md`.

Open follow-ups, none urgent, none blocking: validate DICOM loading in `preprocess_xray.py`
against a real file (still untested); get a real bilateral X-ray to test the split logic
(only synthetically tested); consider more epochs or a lower/frozen-backbone LR for the
hybrid architectures specifically (both showed the same early-epoch-peak-then-degrade
instability, worth a second look if hybrids are revisited); a web app, deferred by design
(Phase 6 was explicitly terminal-only "for now").

## KneeNeXt (experimental)

Built under a strict add-only rule (2026-10-26): no existing file was edited. All new code is
in `scripts/kneenext.py` (architecture), `scripts/train_kneenext.py` (two-stage trainer +
cumulative ablation driver), `scripts/evaluate_kneenext.py` (one-time test evaluation +
bootstrap + Grad-CAM), `scripts/predict_kneenext.py` (demo, same behavior as `predict.py` but
using KneeNeXt). All new outputs are under `outputs/kneenext/` only - `predict.py` and
`kl_final` are unchanged and remain the default.

**What it is**: `kl_final`'s ConvNeXt-Tiny+CBAM, loaded read-only from `kl_final_best.pt`,
with four optional modules (each gated by a flag, so they ablate independently and
cumulatively): a **Joint-Zoom branch** (a second, cropped-and-zoomed view of the joint line,
through the same shared backbone), **multi-scale fusion** (Stage 3 features folded into Stage
4 before pooling), **Joint-Guided CBAM** (spatial attention nudged by a learnable-weight
Gaussian prior around the joint line), and an **asymmetry module** (`|left - right|` of the
global view's pooled attended features, medial/lateral compartment asymmetry as an explicit
signal). Full architecture diagram, each module's clinical motivation, and the complete
ablation/results tables: `outputs/kneenext/results/kneenext_summary.md`.

**Ablation result** (validation QWK, 2 seeds each): each module added individually on top of
the running best did NOT beat it (Joint-Zoom alone -0.0003, multi-scale fusion -0.0028,
Joint-Guided CBAM -0.0007) - only the full four-module combination beat `control` (kl_final
retrained with KneeNeXt's two-stage schedule), by +0.0056 mean val QWK (0.8446 vs 0.8390),
a synergistic rather than additive effect.

**Test result** (evaluated once): KneeNeXt (full config) - accuracy 0.6991, QWK 0.8609, macro
F1 0.6940, MAE 0.3353 - numerically better than `kl_final` (0.6937 / 0.8570 / 0.6926 / 0.3396)
on every metric, but the paired bootstrap difference is NOT statistically significant (QWK
diff +0.0040, 95% CI [-0.0021, 0.0102]) - same pattern as Hybrid 1/Hybrid 2 before it. Vs the
original EfficientNet-B0 baseline, KneeNeXt IS significantly better (QWK diff +0.0364, 95% CI
[0.0210, 0.0512]). **`kl_final` remains the locked model** - KneeNeXt is an experimental,
additive result, not a replacement.

Demo: `python scripts/predict_kneenext.py` (file browser) or
`python scripts/predict_kneenext.py path/to/xray.png --show [--age N] [--bmi N]
[--family-history yes|no]` - same patient-detail prompts and causal risk panel as
`predict.py`, plus a zoom-view panel specific to KneeNeXt's architecture.

Note on git: the 10 individual per-config/per-seed ablation checkpoints (~1.1GB total) are
kept locally under `outputs/kneenext/checkpoints/` but were not pushed to GitHub - only
`kneenext_best.pt` (the winner) was. Normally this project excludes non-winning checkpoints
via `.gitignore`/`.gitattributes`, but adding new patterns there would itself be an edit to
an existing file, which this task's add-only rule forbade - so they're simply left untracked
rather than committed, achieving the same "don't bloat the repo" outcome without touching
those files.
