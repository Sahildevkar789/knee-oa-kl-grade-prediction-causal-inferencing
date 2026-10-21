# Knee OA KL-Grade Prediction — Final Results

Full project write-up: dataset, segmentation, classifier development (backbone search →
ablation → two hybrid architectures), JSW measurement, XAI, and causal inference. See
`overnight_log.md` (classifier lock) and `final_build_log.md` (everything in this document)
for the complete decision-by-decision record, including bugs found and fixed along the way.

## 1. Dataset

Source: OAI (Osteoarthritis Initiative), via a Kaggle-derived export, `unified_oai_dataset.csv`.
8,260 image rows (train 5,778 / val 826 / test 1,656; one test image missing on disk, a
pre-existing gap, so the classifier scripts see 1,655 test images) across 4,130 unique
patients, zero patient overlap between splits. Grade 0-4 class distribution is imbalanced and
consistent across splits (train: 39.6% / 18.1% / 26.2% / 13.1% / 3.0%), handled throughout
with class-weighted loss (inverse-frequency for CE, per-threshold pos_weight for CORAL).
Images are pre-cropped to the joint region (OAI/Kaggle export), 224×224 grayscale.

## 2. Segmentation (done in Colab, prior to this build)

Ground truth: CGMH Knee Segmentation Database (400 AP X-rays, femur+tibia masks, MIT
licensed) — a different patient cohort from OAI, full-leg framing rather than OAI's tight
joint crops. Four models compared on the 80-image validation split:

| Model | Precision | Recall | F1 (Dice) | IoU |
|---|---|---|---|---|
| U-Net (40 epochs) | 0.9727 | 0.9728 | 0.9727 | 0.9469 |
| **U-Net++ (40 epochs)** | 0.9756 | 0.9720 | 0.9738 | 0.9490 |
| DeepLabV3+ (20 epochs) | 0.9743 | 0.9500 | 0.9620 | 0.9268 |
| Mask R-CNN (20 epochs) | 0.9571 | 0.9644 | 0.9607 | 0.9244 |

U-Net++ selected (narrow margin over U-Net, ~0.1% F1). Known limitation: on OAI's tight joint
crops the resulting masks are patchy (scale mismatch vs. CGMH's full-leg training images) — a
retrain on OAI-scale crops was planned but not done. This is why the locked classifier
(`kl_final`) doesn't use a segmentation mask as an input channel at all; the U-Net++ checkpoint
is instead reused in `preprocess_xray.py` for auto-cropping arbitrary full-leg inputs, where
its full-leg training actually helps rather than hurts.

## 3. Classifier: backbone comparison

4 ImageNet-pretrained backbones, image-only, original recipe (class-weighted CE, AdamW
3e-4 + cosine, early stopping on val QWK). Top two re-run with 3 seeds for stability:

| Backbone | Mean val QWK | Std (3 seeds) | Params |
|---|---|---|---|
| efficientnet_b0 | 0.8078 | ±0.0031 | 4.0M |
| resnet50 (1 seed) | 0.8179 | — | 23.5M |
| densenet121 (1 seed) | 0.8228 | — | 7.0M |
| **convnext_tiny** | **0.8187** | ±0.0110 | 27.8M |

convnext_tiny won the seed-robust comparison despite ~3.5× efficientnet_b0's variance — its
single-seed number was partly a favorable draw. 2-epoch LR warmup kept (0.8315 vs 0.8312
without, a tie on QWK but it visibly reduced early-training instability).

## 4. Classifier: ablation (sequential, 2 seeds each, mean val QWK decides adoption)

| Configuration | Val QWK (mean) | Adopted |
|---|---|---|
| Step 1 setup (convnext_tiny + warmup) | 0.8286 | starting row |
| + CLAHE | 0.8340 | **yes** |
| + CBAM | 0.8368 | **yes** |
| + CORAL (ordinal head) | **0.8416** | **yes** |
| + class-weights sqrt | 0.8346 | no (hurt KL3 recall badly) |
| + TTA (horizontal flip) | 0.8363 | no (flip breaks medial/lateral anatomy) |

**Locked pipeline**: grayscale → CLAHE → ConvNeXt-Tiny (pretrained) → CBAM attention on the
final feature map → CORAL ordinal head (4 binary "grade>k" outputs) → predicted grade = count
of thresholds exceeded. Checkpoint: `kl_final_best.pt`.

## 5. Final model comparison (test set, 1,655 images, evaluated once per model)

| Model | Accuracy | QWK | Macro F1 | MAE | KL0 R | KL1 R | KL2 R | KL3 R | KL4 R |
|---|---|---|---|---|---|---|---|---|---|
| EfficientNet-B0 baseline | 0.6284 | 0.8246 | 0.6746 | 0.4036 | 0.621 | 0.557 | 0.584 | 0.771 | 0.900 |
| **kl_final** (locked) | **0.6937** | **0.8570** | **0.6926** | **0.3396** | 0.803 | 0.392 | 0.676 | 0.780 | 0.860 |
| Hybrid 1 (+ Transformer) | 0.7082 | 0.8620 | 0.6878 | 0.3281 | 0.889 | 0.291 | 0.689 | 0.749 | 0.860 |
| Hybrid 2 (+ age/BMI) | 0.6973 | 0.8598 | 0.6929 | 0.3384 | 0.858 | 0.355 | 0.615 | 0.812 | 0.900 |

**Bootstrap significance (2,000 resamples each):**

| Comparison | QWK diff | 95% CI | Significant |
|---|---|---|---|
| kl_final vs. baseline | +0.0324 | [0.017, 0.048] | **Yes** |
| Hybrid 1 vs. kl_final | +0.0050 | [-0.002, 0.013] | No |
| Hybrid 2 vs. kl_final | +0.0029 | [-0.007, 0.013] | No |

**kl_final is the locked model.** Both hybrids score numerically higher on test but neither
beats kl_final *significantly*, and critically, neither beat it on the pre-registered
validation selection criterion (Hybrid 1: val QWK 0.8320 vs kl_final's 0.8416; Hybrid 2:
0.8170) — both hybrids' best epoch came in the first 1-2 epochs before degrading, a repeatable
pattern (not seed noise) consistent with a newly-initialized head destabilizing an
already-converged fine-tuned backbone. Model selection used validation throughout; the test
numbers above confirm rather than override that choice.

KL1 ("doubtful OA") is the weakest class for every model (29-56% recall) — a known,
clinically subjective boundary even between radiologists, not resolved by any architecture
tried here.

## 6. Joint Space Width (JSW) — not validated

Intensity-based measurement on CLAHE-enhanced images (`measure_jsw.py`): per-column femur/tibia
edge detection with local joint-line refinement, medial/lateral split. Validated against
radiologist JSN grades (`unified_oai_dataset.csv`, matched on patient ID + knee side):

| Feature | rho vs. matching JSN | rho vs. KL grade |
|---|---|---|
| medial_min_jsw | -0.179 | -0.198 |
| lateral_min_jsw | -0.146 | -0.203 |

Direction is correct and the box-plot trend is cleanly monotonic by both JSN and KL grade, but
the medial_min_jsw correlation (-0.179) falls short of the -0.3 threshold set for validation.
**Consequence**: JSW was excluded from Hybrid 2's tabular features, and the planned causal
mediation analysis (BMI → JSW → KL grade) was skipped. This is an honest result for a simple,
automated, single-landmark method without manual calibration — directionally and visually
sound, just not strong enough on its own.

## 7. XAI findings

**Grad-CAM** (last ConvNeXt stage, target = sum of CORAL's 4 threshold logits) on kl_final and
Hybrid 1, 15 test images. Quantitative check: fraction of Grad-CAM mass inside a ±15%
joint-line band. KL4 images show notably lower in-band mass (0.24) than KL0-3 (0.49-0.60) —
visual inspection shows the model correctly attending to bilateral femoral osteophytes sitting
just above the band for severe cases, a clinically sound signal the band metric doesn't fully
capture, not a model failure. Correct predictions relied somewhat less on the narrow joint-line
band than incorrect ones (0.48 vs 0.55) on this small 15-image sample — a direction worth a
larger check, not a firm conclusion.

**SHAP** on Hybrid 2's tabular features (60 test instances, per-instance multi-modal setup):
age matters somewhat more than BMI (mean |SHAP| 0.0025 vs 0.0015), both pushing toward higher
predicted grades with higher values — directionally consistent with known OA risk factors,
though small relative to the image branch's influence.

## 8. Causal inference (DoWhy)

**Question**: does obesity (BMI≥30) causally affect KL grade, adjusting for age and family
history of knee replacement? (Sex omitted — no sex/gender column exists anywhere in the
dataset.) Primary analysis: one knee per patient (random, seed 42), n=4,071.

| Outcome | Method | ATE |
|---|---|---|
| KL grade (0-4) | propensity score matching | +0.583 |
| KL grade (0-4) | linear regression | +0.485 |
| KL≥2 (binary OA) | propensity score matching | +0.246 |
| KL≥2 (binary OA) | linear regression | +0.190 |

All three refuters pass (placebo treatment collapses the effect to ~0, random common cause and
data subset don't move it). **Sensitivity analysis using both knees instead of one** is a
genuine finding, not just a robustness check: propensity score matching collapses to near-zero
(-0.044) because both knees of the same patient share identical covariates, causing the matcher
to degenerately pair a patient's own knees together — concrete evidence for why the one-knee
design matters. Linear regression stays stable (0.507) either way.

**Stated under this DAG's assumptions** (no unmeasured confounding beyond age and family
history): obesity is associated with roughly **+0.48 to +0.58 higher KL grade points**,
robust to the refutation checks. Not a assumption-free claim — real unmeasured confounders
likely exist (physical activity level, prior knee injury/surgery, occupation/physical labor
history, genetic factors beyond family history of knee replacement).

## 9. Limitations

- **Segmentation domain shift**: U-Net/U-Net++ trained on CGMH's full-leg X-rays produce
  patchy masks on OAI's tight joint crops (different cohort, different framing/scale);
  unaddressed, which is part of why kl_final doesn't use a mask channel.
- **Pre-cropped images**: OAI/Kaggle images are already cropped to the joint region, so this
  pipeline doesn't address whole-X-ray localization in a clinical deployment sense (handled
  separately by `preprocess_xray.py` for arbitrary inputs, but that's a bolt-on, not something
  the classifier itself was trained to need).
- **Single dataset**: everything — classifier, JSW validation, causal analysis — is OAI only;
  no external-cohort generalization check.
- **Observational causal data**: the causal estimate is only as good as the DAG's
  assumptions; it is not a randomized-trial-grade causal claim, and the stated unmeasured
  confounders (activity, injury history, occupation) could meaningfully shift it.
- **JSW not validated**: the automated measurement is a weaker signal than true clinical JSW
  tools, so Hybrid 2 and the mediation analysis didn't use it.
- **No sex/gender data**: a clinically relevant covariate that simply isn't present in this
  export of the dataset, affecting both Hybrid 2's feature set and the causal adjustment set.
