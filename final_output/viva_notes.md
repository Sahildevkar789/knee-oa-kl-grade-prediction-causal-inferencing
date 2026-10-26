# Viva Notes

## One-paragraph summary

This project predicts Kellgren-Lawrence (KL) knee osteoarthritis grade (0-4) from X-rays
using the OAI dataset. A ConvNeXt-Tiny backbone with CBAM attention and a CORAL ordinal head
(`kl_final`) was selected through a systematic backbone search and 5-step ablation, reaching
test accuracy 0.6937 and QWK 0.8570 — a statistically significant improvement over an
EfficientNet-B0 baseline (0.6284 / 0.8246). Two architectural extensions were then tried: two
"hybrid" variants (transformer head; tabular fusion) and a knee-specific modification called
KneeNeXt (joint-zoom view, multi-scale feature fusion, joint-guided attention, left/right
asymmetry) — only KneeNeXt's full four-module combination beat its own retrained control on
validation (0.8446 vs. 0.8390 QWK). A greedy probability ensemble of `kl_final`'s seed-7 run
and KneeNeXt's seed-7 run, with tuned decision thresholds, reached test accuracy 0.7172 —
above the 70% target and the first configuration in this project to beat `kl_final` with
statistical significance, though only on accuracy, not QWK. Explainability work (Grad-CAM,
Grad-CAM++, Integrated Gradients, occlusion, deletion-curve faithfulness testing, a parameter-
randomization sanity check, nearest-neighbor case retrieval, and calibration/uncertainty
analysis) and causal inference work (DoWhy: obesity's effect on KL grade, its sensitivity to
hidden confounders via E-value, a knee-injury association, EconML CausalForestDML for
individual-level effects, and a BMI dose-response curve) were layered on top, each evaluated
honestly — including methods that did not work as hoped (JSW measurement, segmentation masks
as an input channel, Grad-CAM's faithfulness) rather than only reporting the successes.

## What to claim / what not to claim

**Claim:**
- `kl_final` beats an EfficientNet-B0 baseline with statistical significance (QWK and
  accuracy both, 2,000-resample paired bootstrap).
- The final ensemble beats `kl_final` on accuracy with statistical significance, and clears
  70% test accuracy (71.72%).
- The obesity → KL grade causal estimate is robust to the sensitivity checks performed
  (E-value 2.345; a DoWhy hidden-confounder simulation never flipped it negative up to the
  tested strength).
- Integrated Gradients passed a quantitative faithfulness test (deletion curves); the
  parameter-randomization sanity check passed for Grad-CAM.
- An entropy-based uncertainty flag meaningfully separates more-accurate from less-accurate
  predictions on the test set.

**Do NOT claim:**
- That the ensemble is a QWK improvement over `kl_final` — it is not, statistically (tied).
- That KneeNeXt or the ensemble have been validated on any data outside OAI.
- That the knee injury finding is causal — it is an association from self-reported,
  timing-uncertain survey data.
- That Grad-CAM's explanations are faithful by the deletion-curve test — they were not, in
  this specific test, even though the sanity check passed (those are different properties).
- That segmentation masks or JSW measurements improved anything — both were tried and
  dropped based on the evidence.
- That this system is validated for clinical use — it is a research prototype on a single
  cohort.

## The math

**Ensemble**: for each model *m* in the selected set (here `kl_final` seed 7 and KneeNeXt
seed 7), the CORAL head outputs 4 cumulative probabilities `p_m[k] = P(grade > k)` for
k=0,1,2,3. The ensemble averages these elementwise:
`p_ensemble[k] = mean over m of p_m[k]`.
The predicted grade is the count of thresholds exceeded, but with **tuned**, not flat-0.5,
per-threshold cutoffs `t[k]` (chosen by grid search, validated with 5-fold cross-validation
on the validation set, adopted only because the cross-validated gain was positive):
`grade = sum over k of 1[p_ensemble[k] > t[k]]`, with `t = [0.70, 0.45, 0.50, 0.55]`.

**KneeNeXt**, on top of `kl_final`'s ConvNeXt-Tiny + CBAM backbone (shared weights across
both views):
- *Joint-Zoom branch*: a second 224×224 view, cropped to a band of ±25% of the image height
  around the detected joint line, passed through the **same** backbone weights as the global
  view.
- *Multi-scale fusion*: Stage 3 features (384×14×14) are projected to 768 channels with a
  1×1 convolution, downsampled to 7×7, and added elementwise to Stage 4 (768×7×7), for both
  views: `fused = stage4 + downsample(conv1x1(stage3))`.
- *Joint-Guided CBAM*: standard CBAM channel attention, unchanged; spatial attention becomes
  `sigmoid(learned_spatial_map + alpha * joint_prior)`, where `joint_prior` is a Gaussian
  band around the joint line (centered, for the zoom view) and `alpha` is a learnable scalar
  initialized at 1.0 (learned to roughly 1.15-1.18 in practice).
- *Asymmetry feature*: the global view's CBAM-attended 768×7×7 feature map is split into left
  and right halves (dropping the central column), each globally pooled to a 768-dim vector,
  and `|pooled_left - pooled_right|` is concatenated onto the pooled global and zoom features
  before the head.
- *Head*: `concat([global_pooled, zoom_pooled, asymmetry]) -> LayerNorm -> Dropout(0.3) ->
  Linear(->4) CORAL head`, same ordinal decoding as `kl_final`.

## 15 likely examiner questions

1. **Why QWK and not just accuracy?** KL grading is ordinal — predicting grade 3 when the
   truth is 4 is a smaller error than predicting grade 0. QWK penalizes distant errors more
   than adjacent ones; accuracy treats every misclassification the same.

2. **Why did the ensemble beat `kl_final` on accuracy but not QWK?** The tuned decision
   thresholds shift where the model draws grade boundaries, which can increase the count of
   exactly-correct predictions (accuracy) without proportionally reducing the *size* of
   errors when it is wrong (QWK) — and it measurably cost KL1 recall (0.3918 → 0.2432) while
   helping elsewhere.

3. **Is KneeNeXt a new algorithm?** No — it is `kl_final`'s backbone with four additional,
   domain-informed modules. We say this explicitly rather than overselling the contribution.

4. **Why did adding each KneeNeXt module individually not help?** Only the full combination
   of all four beat the retrained control (+0.0056 mean val QWK); each module alone, added
   to the running best, did not. This suggests a synergistic rather than additive effect —
   reported honestly as an observed result, not a designed-in guarantee.

5. **What does the E-value of 2.345 actually mean?** An unmeasured confounder would need to
   be associated with *both* obesity and KL grade by at least a 2.34-fold risk ratio each
   (beyond age and family history, already adjusted for) to fully explain away the obesity
   effect. Weaker confounding could not.

6. **Why is the knee injury result only an "association"?** It comes from a self-reported
   OAI questionnaire column; we cannot establish from this data whether the injury occurred
   before or after the knee changes being measured, so causal language is avoided.

7. **Why did Grad-CAM fail the deletion test but pass the sanity check?** These test
   different things. The sanity check asks "does the map depend on learned weights at all"
   (yes — it passed). The deletion test asks "are the specific regions it highlights the ones
   that actually matter most" (less clearly so, at the patch granularity tested) — a 7×7
   feature map upsampled to 224×224 is inherently coarse.

8. **Why didn't segmentation masks help the classifier?** The U-Net/U-Net++ models were
   trained on CGMH's full-leg X-rays; applied to OAI's already-tight joint crops, they
   produce patchy, less meaningful masks — a known domain-shift issue, not a modeling error
   in the classifier itself.

9. **Why wasn't JSW (joint space width) used in the final models?** Its correlation with
   radiologist JSN grades (ρ=-0.179) fell short of our own pre-registered validation
   threshold (-0.3). We excluded it from downstream use rather than lower the bar.

10. **How was data leakage avoided?** Radiologist JSN/osteophyte grades and WOMAC scores
    were never used as model inputs or causal adjustment covariates — they are consequences
    of OA, not causes/predictors, and using them would leak the outcome into the inputs.

11. **How was model selection kept honest?** Every model-selection decision (backbone,
    ablation, hybrids, KneeNeXt's ablation, the ensemble's greedy selection and threshold
    tuning) used validation data only; each model touched the test set exactly once, at the
    end, for confirmation, not for choosing between configurations.

12. **Why one knee per patient for the causal analysis?** Both knees of the same patient
    share nearly identical covariates (age, BMI, family history), which caused propensity
    score matching to degenerate (match a patient's knees to each other) when both knees were
    used — a concrete, demonstrated reason for the one-knee restriction, not just caution.

13. **What would you do with more time/data?** External validation on a second cohort;
    retrain the segmentation models on OAI-scale crops; revisit JSW with a more precise
    (e.g. landmark-based) measurement method; explore why the hybrid architectures showed an
    early-epoch-peak-then-degrade pattern.

14. **Is this ready for clinical use?** No — single dataset, no external validation, modest
    calibration (ECE 0.1514 untouched by temperature scaling), and the weakest class (KL1,
    "doubtful OA") remains genuinely hard even for the best configuration, consistent with it
    being a known, clinically subjective boundary.

15. **What's the single most defensible result in this project?** `kl_final` beating the
    EfficientNet-B0 baseline with statistical significance on both accuracy and QWK — the
    rest of the project (hybrids, KneeNeXt, ensemble, causal/XAI extensions) builds on that
    foundation with mixed, honestly-reported results rather than uniform success.
