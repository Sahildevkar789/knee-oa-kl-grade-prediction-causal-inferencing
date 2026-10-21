# Final build log

Started 2026-10-21. Working through the 7-phase plan unattended: Hybrid 1 (CNN+Transformer),
JSW measurement, Hybrid 2 (image+tabular fusion), XAI (Grad-CAM/SHAP), causal inference
(DoWhy), terminal demo, final write-up. Every decision and its reason is logged here as it
happens. Model selection uses validation only; test set touched once per phase, at the end.
Committing after each phase.

**Data-leakage hard rules acknowledged and will be enforced**: never use P01SV*JS* (JSN),
P01SV*OST (osteophyte), any KL-derived column, or WOMAC as model inputs. WOMAC excluded from
causal adjustment too (it's a disease outcome, not a cause). Clinical-to-image joins match on
patient ID **and** knee side. Missing numeric values imputed with training-set median only,
count of imputed values recorded.

## Phase 2: Joint Space Width (JSW)
Wrote `scripts/measure_jsw.py`: per image, apply CLAHE -> grayscale -> find the joint line
(darkest horizontal band, central 25-75% of height) -> Otsu threshold -> for each column (per
compartment, see below) refine a LOCAL joint-line estimate within ±10% of image height around
the global row (see bug below for why), then scan up/down from there for the femur's lower
edge and tibia's upper edge; gap = JSW, normalized by joint width (bone span at the joint row).
Skips the central 20% of joint width (tibial spines) AND a matching 20% margin at each
compartment's outer edge (condyle curvature away from the vertical scan direction - found to
matter just as much as the center, see bug below). Medial/lateral compartments assigned from
knee side using a derived AP-view convention (right knee: medial = image-right half; left
knee: medial = image-left half) - **verified correct** by the Spearman convention check below
(as-labeled medial rho=-0.179 beats the swapped check's rho=-0.131).

**Two real bugs found and fixed via diagnosis, not guessing:**
1. Taking a literal per-spec reading (skip only the central 20%, min/mean over everything
   else) made the compartment "min" dominated by edge artifacts: zero-reading columns near a
   compartment's outer boundary (bone curving away, scan genuinely finds no edge) were
   indistinguishable from genuine bone-on-bone narrowing. Fixed by having `column_jsw` return
   `None` (not 0) on a true measurement failure, excluding `None`s from min/mean (only falling
   back to a literal 0 if every column in a compartment failed), and adding a symmetric outer-
   edge trim matching the existing central-spine trim.
2. Even after that fix, a full-dataset check showed `medial_min_jsw`'s median AND 75th
   percentile were still exactly 0, and the mean-by-KL-grade trend wasn't monotonic - far worse
   than a 3-image spot check suggested. Root cause: anchoring every column's up/down scan to a
   single *global* joint row doesn't tolerate the joint line's natural tilt (limb positioning,
   X-ray angle) - a column whose true local gap sits even slightly off that row reads as
   already-touching bone. Fixed by having each column first find its own local darkest point
   within a search radius of the global row, then scanning from there. This dropped the
   zero-fraction from >75% to ~33-37% on a 200-image resample and produced a clean, monotonic
   KL4-is-lowest trend - confirmed by running the fixed version on the full 8,259-image set
   before moving to validation, rather than trusting the 3-image spot check.

**Ran on all train/val/test images** (`data/jsw_features.csv`, 8,259 rows - the same 1 missing
file as everywhere else in this project). Columns: joint_width_px, left/right_min/mean_jsw
(raw, image-relative), medial/lateral_min/mean_jsw (side-assigned, normalized by joint width).

**Validation** (join on patient ID + knee side, matching
`unified_oai_dataset.csv`'s `P01SV{R,L}KJS{L,M}` columns - categorical strings like
`"1: 1 (narrowed, OARSI grade 1-2)"`, parsed to their leading numeric code): matched
8,258/8,259 rows. Spearman correlations (n~8258, all p<1e-30, i.e. the correlations are real,
just not strong):

| Feature | vs matching JSN | rho | vs KL grade | rho |
|---|---|---|---|---|
| medial_min_jsw | jsn_medial | -0.1792 | KL_GRADE | -0.1977 |
| medial_mean_jsw | jsn_medial | -0.1782 | KL_GRADE | -0.1839 |
| lateral_min_jsw | jsn_lateral | -0.1455 | KL_GRADE | -0.2033 |
| lateral_mean_jsw | jsn_lateral | -0.1606 | KL_GRADE | -0.1610 |

Box plots (`outputs/results/jsw_boxplots.png`) show a clean, consistent, monotonically
decreasing trend across all four JSW features by both JSN grade and KL grade - the direction
and shape are exactly right, the correlation is just moderate rather than strong. 10 example
images with the joint line and measured columns drawn on
(`outputs/results/jsw_examples/`) confirm this visually: a KL0 example shows the joint-line
markers sitting cleanly in an open joint space, a KL4 example shows them sitting right at a
visibly collapsed/bone-on-bone region on one side with a thin residual gap on the other -
matches the measured left_min=0.000 / right_min=0.031 exactly.

**Verdict: medial_min_jsw vs medial JSN rho = -0.1792, which does NOT clear the -0.3
threshold -> JSW is NOT VALIDATED.** This is an honest result for a simple, automated,
single-global-landmark intensity method without manual calibration - weaker than a true
clinical JSW measurement tool would achieve, but directionally and visually sound. Per the
brief's explicit fallback: **Phase 3's tabular features will use age/BMI/sex only**, logged
here and will be logged again when Phase 3 builds the tabular model. Phase 5's causal mediation
analysis (BMI -> medial min JSW -> KL grade), which was conditional on validation, will be
skipped and logged as such when reached.

## Phase 3: Hybrid 2 (image + tabular fusion)
**Data limitation found and logged, not guessed around**: the brief says "find the sex column
in unified_oai_dataset.csv and add it to clinical_features.csv," but an exhaustive
case-insensitive search of all 1,193 columns found no sex/gender field at all (`V00isexmdt`
is the only "sex"-containing name, and it's a visit date, not sex). Proceeding with **age + BMI
only** as Hybrid 2's tabular features, per "don't stop to ask, log it and continue."
JSW is also excluded, per Phase 2's not-validated verdict (already logged there).

**Data handling** (hard rules followed): BMI has 6 missing values in train, 0 in val/test -
imputed with the **training-set median only** (28.20), applied unchanged to val/test (and
logged each time). Tabular features standardized with **training-set mean/std only**
(age: mean 61.0, std 9.24; BMI: mean 28.58, std 4.88). `clinical_features.csv` lists one test
row whose image file doesn't exist on disk (the same pre-existing gap noted everywhere else in
this project) - added the same existence filter `list_split()` already uses elsewhere, so
Hybrid 2's test set is the same 1,655 images as every other model's.

**Architecture**: `ConvNeXtCBAMFusion` (`kl_common.py`) - same ConvNeXt-Tiny+CBAM trunk as
kl_final, globally pooled (768-dim) then projected to 128 dims, concatenated with a 2-layer
MLP (32 hidden units) over the 2 standardized tabular features, then a CORAL head. Built as a
dedicated script (`scripts/train_hybrid2.py` + `scripts/evaluate_hybrid2.py`) rather than
bolting a two-input forward pass onto the shared single-image training loop, since the
input signature is genuinely different - kept the blast radius on the well-tested existing
scripts at zero. Initialized features/CBAM from kl_final_best.pt (181 tensors, same transfer
mechanism as Hybrid 1). Smoke-tested (1 epoch, val QWK 0.8222) before committing to full runs.

Trained seeds 42 and 7 (clahe+warmup2, same recipe as kl_final and Hybrid 1):
- seed42: best val QWK 0.8260 at epoch 1, then degraded (same early-epoch-peak pattern as
  Hybrid 1 - a new randomly-initialized head destabilizing a converged fine-tuned backbone).
- seed7: best val QWK 0.8080 at epoch 2, also degraded afterward.
- **Mean val QWK = 0.8170**, below kl_final's 0.8416 reference.

Better seed (42, 0.8260) saved as `outputs/checkpoints/kl_hybrid2_best.pt` (used a seed-
specific filename for seed 7 from the start this time, avoiding Hybrid 1's accidental
same-name overwrite). `outputs/checkpoints/kl_hybrid2_config.json` records every option,
including that this model needs age+BMI at prediction time.

**Test result**: Accuracy 0.6973, QWK **0.8598**, Macro F1 0.6929, MAE 0.3384 - again
numerically beats kl_final (0.6937/0.8570/0.6926/0.3396) despite losing on val, same pattern
as Hybrid 1. Paired bootstrap (2,000 resamples) vs kl_final: QWK mean diff +0.0029, 95% CI
[-0.0072, 0.0132] - **not significant**. Accuracy mean diff +0.0036, 95% CI [-0.0145, 0.0218] -
also not significant. (Bootstrap script needed a path-separator normalization fix - kl_final's
predictions used backslash paths from glob, kl_hybrid2's used forward-slash paths from
clinical_features.csv's IMAGE_PATH column; both reference the same files, just written
differently. Fixed in `bootstrap_compare.py` once, benefits all future comparisons.)

**Conclusion**: Hybrid 2 does not beat kl_final either, for the same reason as Hybrid 1 -
test-set numbers are close but not statistically distinguishable, and it lost on the
pre-registered validation selection criterion. **kl_final remains the project's best
classifier going into Phase 4.**

## Phase 1: Hybrid 1 (CNN + Transformer)
Added `--head {mlp,transformer}` to train_classifier.py/kl_common.py (default mlp = unchanged
behavior). `transformer` builds `ConvNeXtCBAMTransformer`: ConvNeXt-Tiny+CBAM feature map
(768x7x7) -> 1x1 conv projection to 256 dims -> flatten to 49 tokens -> prepend learnable CLS
token -> add learnable positional embeddings (50 positions) -> 2-layer TransformerEncoder
(4 heads, dim_feedforward=1024 [=4x model dim, standard convention - not specified in the
brief, documenting the choice], dropout 0.1) -> CLS output -> CORAL head (Linear(256,4)).
`build_hybrid1()` initializes `features.*`/`cbam.*` from kl_final_best.pt by copying any
state_dict keys with matching name+shape; new layers (projection/CLS/pos-embed/transformer/
head) keep PyTorch's default random init. Forces backbone=convnext_tiny/attention=cbam/
loss=coral when `--head transformer` is passed (logs a note if conflicting flags were given).
Smoke-tested (1 epoch): 181 tensors correctly transferred, val QWK 0.8303 already close to
kl_final's full performance after just 1 epoch - confirms the transfer worked.

Trained seeds 42 and 7 (clahe+warmup2, matching kl_final's recipe exactly, as instructed):
- seed42: best val QWK 0.8317 **at epoch 1** - degraded in every subsequent epoch (down to
  0.73-0.79 by epochs 5-7) before early-stopping triggered.
- seed7: same pattern, best 0.8323 at epoch 1, also degrading afterward.
- **Mean val QWK = 0.8320**, below kl_final's reference 2-seed mean of 0.8416.
This is a real, repeatable (both seeds) finding: the randomly-initialized transformer head,
sitting on top of an already well-converged pretrained backbone, destabilizes fine-tuning
after the first epoch rather than improving on it - plausibly because the backbone's
gradients get dominated by the untrained transformer's large initial gradients. Not explored
further (e.g. freezing the backbone initially, lower LR for the transformer) since the brief
specified the same training recipe as kl_final throughout.

Better seed (7, 0.8323) saved as `outputs/checkpoints/kl_hybrid1_best.pt` per the instructions
(regardless of the val outcome vs kl_final, since the brief says to evaluate on test either
way). Extended evaluate_classifier.py with `--head transformer` to support loading it.

**Test result**: Accuracy 0.7082, QWK **0.8620**, Macro F1 0.6878, MAE 0.3281 - numerically
*beats* kl_final (0.6937 / 0.8570 / 0.6926 / 0.3396) despite losing on val. Paired bootstrap
(2,000 resamples) vs kl_final: QWK mean diff +0.0050, 95% CI [-0.0022, 0.0125] - **not
significant** (CI includes 0). Accuracy mean diff +0.0144, 95% CI [0.0000, 0.0278] - also
not significant (lower bound touches 0). `outputs/results/significance_hybrid1_vs_final.csv`.

**Process note**: `kl_hybrid1_best.pt` is train_classifier.py's default (seed-42) output name,
so copying seed7's checkpoint onto it to finalize "the better seed" overwrote and lost the
seed42 checkpoint file itself (not a violation of the "never overwrite existing results" rule
- this was this phase's own in-progress artifact, not a prior protected result, and seed42's
full metrics are preserved in its training_log.csv regardless). `kl_hybrid1_seed7_best.pt` is
now a redundant duplicate of `kl_hybrid1_best.pt` and won't be committed.

**Conclusion**: Hybrid 1's test-set edge is not statistically distinguishable from noise, and
it lost on the pre-registered validation-based selection criterion (which is what model
selection is supposed to use - the test set was only ever meant to confirm, not decide).
Treating this as "did not beat kl_final" for Phase 7's final comparison, while reporting the
numbers honestly (it's close, and the instability-after-epoch-1 pattern is worth investigating
later if the transformer-head idea is revisited, e.g. with backbone freezing or a transformer-
specific lower LR).

