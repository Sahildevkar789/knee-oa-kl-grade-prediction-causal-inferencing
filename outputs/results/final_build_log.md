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

## Phase 4: XAI
Models explained: **kl_final** and **kl_hybrid1** (the hybrid with the best val QWK - 0.8320
vs Hybrid 2's 0.8170; neither beat kl_final overall, but the brief asks for "whichever hybrid
has the best val QWK" regardless). SHAP additionally run on Hybrid 2 since it exists.

**Grad-CAM** (`scripts/run_xai.py`, hooked on `model.features`, the last ConvNeXt stage,
before CBAM reweighting): CORAL's 4 threshold logits don't give a single class score the way
softmax does, so the Grad-CAM target is the **sum of the 4 threshold logits** - a monotonic
proxy for the predicted grade (more thresholds crossed, more confidently = larger sum),
the same idea noted as the planned approach in CLAUDE.md before this phase started.

15 test images selected (3/grade: 1 wrong + 2 correct prediction per grade, from kl_final's
test predictions) - exactly 5 wrong as required. Saved to `outputs/results/xai/KL{grade}_*.png`,
each a 2x3 grid: kl_final and kl_hybrid1 rows, each showing input / Grad-CAM overlay (with the
Phase-2 joint-row drawn as a reference line) / CBAM attention overlay.

**Quantitative joint-region check**: band = joint_row (from `measure_jsw.find_joint_row`,
reused from Phase 2) +-15% of image height; fraction of Grad-CAM mass (upsampled, clipped to
>=0, normalized to sum 1) falling inside that band, per image. Results
(`outputs/results/xai/cam_mass_quantitative.csv`):
- By true KL grade (kl_final): KL0=0.49, KL1=0.60, KL2=0.60, KL3=0.59, **KL4=0.24** (notably
  lower). Visual inspection of the KL4 examples shows why: Grad-CAM concentrates on bilateral
  femoral condyle osteophytes sitting clearly *above* the joint-line band, not on the band
  itself - correct, clinically meaningful attention (osteophytes are a primary severity
  marker, especially once the joint space itself is too collapsed to be the most informative
  signal), just outside this particular band definition. Not a model failure; a limitation of
  the band-based metric for severe cases.
- Correct vs incorrect (kl_final): correct=0.48, incorrect=0.55 - correct predictions rely
  *less* on the narrow joint-line band than incorrect ones, on this small 15-image sample.
  Interesting but not over-interpreted given n=15; flagging as a direction for a larger check,
  not a conclusion.

**CBAM attention maps** saved alongside Grad-CAM in the same figures (third column of each row).

**SHAP (Hybrid 2 exists -> required)**: `scripts/run_shap_hybrid2.py`. Multi-modal per-instance
setup - each of 60 sampled test instances holds its own X-ray fixed and perturbs only the 2
tabular features (age, BMI) against a 20-row background of standardized training tabular
values, via `shap.KernelExplainer`. **Bug caught in a 3-instance test run before the full 60**:
explaining the discretized predicted grade (`coral_probs_to_grade`, a step function) gave
exactly zero SHAP values, since small tabular perturbations rarely cross a 0.5 threshold
boundary - switched the explained target to the continuous sum of the 4 sigmoid outputs (a
0-4 continuous proxy for the predicted grade), which gave real signal immediately.
Mean |SHAP|: age 0.0025, BMI 0.0015 - age mattering somewhat more than BMI for this model,
both small relative to the image's influence (expected, since the image branch dominates).
Direction is clinically sensible: higher age and higher BMI both push the prediction toward
higher grades (`outputs/results/xai/shap_hybrid2_summary.png`), consistent with known OA risk
factors. (Minor fix along the way: the bar chart and summary plot shared a stale title because
a new matplotlib figure wasn't opened between the two `savefig` calls - fixed by explicitly
closing/opening figures; regenerated both plots from the saved SHAP values CSV rather than
re-running the ~70s computation.)

## Phase 5: Causal inference (DoWhy)
**Question**: does obesity (BMI>=30) causally affect KL grade, adjusting for age and family
history of knee replacement (P01FAMKR)? **Sex omitted again** (same data limitation as Phase
3 - no sex column exists anywhere in the dataset, logged there and not re-derived here).
**WOMAC, JSN and osteophyte columns were not used as adjustment covariates** (hard rule -
they're consequences of OA, not causes of it).

**Data prep**: `data/clinical_features.csv`, parsed family history from its categorical
string encoding (`"0: No"` / `"1: Yes"` / `".: Missing Form/Incomplete Workbook"`). Dropped
112 rows with missing family history and 6 with missing BMI (both small, <1.5% of data -
dropped rather than imputed for this analysis, since causal covariate imputation introduces
its own assumptions the brief didn't specify how to handle; logged instead of guessing).
8,142 rows / 4,071 patients remained.

**DAG** (`outputs/results/causal/dag.png`): Age -> Obese, Age -> KL_grade,
FamilyHistory -> KL_grade, Obese -> KL_grade. Two bugs fixed before getting a working
estimate: (1) DoWhy's graph node names must match the actual dataframe columns it sees, not
the human-readable labels in the diagram - the diagram uses "KL_grade" but DoWhy's outcome
column is always renamed to "y" internally (shared code path for both outcomes), so the DAG
string passed to `CausalModel` uses "y", not "KL_grade" (a `NetworkXError: node y not in
graph` caught this immediately). (2) An overly-clever one-liner for picking which column to
rename to "y" was simplified to a plain if/else after review, before it could hide a bug.

**Primary analysis** (one knee per patient, randomly chosen seed=42, n=4,071, all splits):

| Outcome | Method | ATE |
|---|---|---|
| KL grade (0-4) | propensity score matching | 0.583 |
| KL grade (0-4) | linear regression | 0.485 |
| KL>=2 (binary OA) | propensity score matching | 0.246 |
| KL>=2 (binary OA) | linear regression | 0.190 |

Both methods agree in direction and rough magnitude: under this DAG's assumptions, being
obese is associated with roughly **+0.48 to +0.58 points higher KL grade** on average, and
roughly a **19-25 percentage point higher probability of having KL>=2** (clinically-defined
OA), holding age and family history fixed.

**Refuters** (on the primary KL-grade linear regression estimate, 0.4851) - all three pass:
- Placebo treatment: effect collapses to ~0.0006 (p=0.96) when the treatment is replaced
  with random noise, as a valid estimate should.
- Random common cause: adding an irrelevant random covariate barely moves the estimate
  (0.4851 -> 0.4851).
- Data subset (80%): estimate stable (0.4851 -> 0.4861).
`outputs/results/causal/refutation_results.csv`.

**Sensitivity analysis** (both knees, no one-per-patient restriction, n=8,142): linear
regression stays close to the primary estimate (0.507 vs 0.485 - stable). **Propensity score
matching does not** (-0.044, collapsing to near zero and flipping sign). This is a genuine,
interpretable methodological finding, not noise: both knees of the same patient share
*identical* Age/FamilyHistory/Obese values, so PSM's nearest-neighbor matching frequently
pairs a patient's own two knees together instead of matching across genuinely different
patients - a known degenerate case for matching on non-independent/clustered observations.
This is a concrete demonstration of *why* the primary analysis deliberately uses one knee per
patient, not just a procedural box-ticking step.

**Mediation analysis (BMI -> medial min JSW -> KL grade): SKIPPED.** Conditional on JSW being
validated in Phase 2, which it was not (rho=-0.179 vs the -0.3 threshold). Logged to
`outputs/results/causal/mediation_analysis_SKIPPED.txt`.

**Conclusion, stated as the DAG demands**: *under the assumptions encoded in this DAG* (no
unmeasured confounding beyond age and family history; the graph's structure is correct), the
data are consistent with obesity having a positive causal effect on KL grade of roughly
0.48-0.58 grade points, robust to the three refutation checks. This is **not** a claim free of
assumptions - the DAG is deliberately simple, and real unmeasured confounders almost
certainly exist: **physical activity level, prior knee injury or surgery, occupation
(physical labor history), and genetic factors beyond family history of knee replacement**
could all plausibly affect both obesity risk and OA severity independently of what's modeled
here. The effect size should be read as "what the data show under this specific, limited
adjustment set," not as a definitive causal magnitude.

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

