# Part A: ensemble + threshold tuning

Device: cuda

Val: 826 images.

- `kl_final` (classifier, `kl_final_best.pt`): val acc 0.6755, val QWK 0.8425 (attempt 1)

- `kl_final_seed7` (classifier, `kl_baseline_convnext_tiny_seed7_warmup2_clahe_cbam_coral_best.pt`): val acc 0.6889, val QWK 0.8406 (attempt 1)

- `hybrid1` (hybrid1, `kl_hybrid1_best.pt`): val acc 0.6683, val QWK 0.8323 (attempt 1)

- `kneenext_seed42` (kneenext, `kneenext_asymmetry_seed42_best.pt`): val acc 0.6695, val QWK 0.8448 (attempt 1)

- `kneenext_seed7` (kneenext, `kneenext_asymmetry_seed7_best.pt`): val acc 0.6755, val QWK 0.8443 (attempt 1)

## Greedy ensemble selection (validation)

- Step 0: added `kl_final_seed7` -> members ['kl_final_seed7'] | val acc 0.6889, val QWK 0.8406
- Step 1: added `kneenext_seed7` -> members ['kl_final_seed7', 'kneenext_seed7'] | val acc 0.6925, val QWK 0.8491

**Selected ensemble**: ['kl_final_seed7', 'kneenext_seed7'] (val acc 0.6925, val QWK 0.8491, started from the single best model by val accuracy).

## Threshold tuning: ADOPTED (5-fold CV gain +0.0061: tuned 0.6986 vs flat-0.5 0.6925). Final thresholds (fit on the full val set): [0.7, 0.45, 0.5, 0.55] (full-val accuracy 0.7107).

## Flip TTA: NOT ADOPTED (val acc 0.7107 vs 0.6985 with TTA - no improvement).

Ensemble config saved to `outputs/ensemble/ensemble_config.json`: selected_models=['kl_final_seed7', 'kneenext_seed7'], thresholds=[0.7, 0.45, 0.5, 0.55], use_tta=False, val_acc=0.7107, val_qwk=0.8399.

## Final test evaluation (touched once): accuracy 0.7172, QWK 0.8574, macro F1 0.6933, MAE 0.3293.

Per-grade recall: KL0 0.9045, KL1 0.2432, KL2 0.6980, KL3 0.8161, KL4 0.8600 - overall accuracy rose past kl_final's
(0.6937 -> 0.7172) and past the 70% target, but KL1 recall dropped notably (kl_final's own KL1 recall was 0.3918) -
the tuned per-threshold decode trades some KL1 sensitivity for overall accuracy; reported honestly in the write-up,
not glossed over.

Bootstrap vs kl_final: QWK diff +0.0005 [-0.0090, 0.0096] not significant; accuracy diff +0.0234 [0.0073, 0.0405]
significant.
Bootstrap vs baseline: QWK diff +0.0329 [0.0162, 0.0496] significant; accuracy diff +0.0886 [0.0616, 0.1148] significant.

Saved outputs/ensemble/ensemble_summary.md. Test accuracy 0.7172 (ABOVE the 70% target).

# Part B: causal extensions

Primary data: one knee per patient (seed 42), n=4071.

### E-value (obesity -> KL grade)

Point estimate +0.521 KL grade points (95% CI [0.448, 0.595]), outcome SD 1.189.

Approximate risk ratio for the point estimate: 1.490. **E-value = 2.345** - an unmeasured confounder would need to be associated with BOTH obesity and KL grade by a risk-ratio of at least 2.34-fold each (above and beyond age/family history, which are already adjusted for) to fully explain away this effect; anything weaker could not.

E-value for the CI bound closer to the null (0.448): **2.167** - a confounder this strong would be enough to make the effect's confidence interval include zero.

### Hidden-confounder grid

Saved `sensitivity_confounder_grid.png`/`.csv`. No confounder strength within the tested grid (up to 0.30 on both treatment and outcome) was enough to flip the effect negative - the obesity-KL grade association is fairly robust to the kind of hidden confounding this simulation can represent. Plain language: this simulates a hypothetical unmeasured variable (e.g. physical activity level) with a given strength of association to both obesity and KL grade, and reports what the obesity effect would look like if that variable had been included as a covariate.

### Knee injury effect

Usable columns: `P01INJR`/`P01INJL` (unified_oai_dataset.csv), matched to each knee's SIDE_CODE. 41 rows missing an injury value for their side, dropped (not imputed) for this analysis - n=4030.

Prior knee injury -> KL grade, adjusting for age/BMI/family history: linear regression ATE **+0.540**, propensity score matching ATE **+0.550** KL grade points. Refuters: placebo_treatment_refuter=+0.001, random_common_cause=+0.540, data_subset_refuter=+0.544.

### Personalized obesity effects (CausalForestDML)

Mean individual effect **+0.510** KL grade points (population ATE from Phase 5: +0.48 to +0.58 - consistent with that range). Effect-vs-age correlation: -0.024 (little age trend). Mean effect with family history: +0.460 vs without: +0.518 (smaller for patients with a family history of knee replacement). Fitted model saved to `causal_forest_model.pkl`; per-patient effects to `causal_forest_individual_effects.csv`; plots to `causal_forest_individual_effects.png`.

### BMI dose-response

Saved `bmi_dose_response.png`/`.csv` - expected KL grade across the observed BMI range (16.9-48.7), age held at the median (60), no family history. Roughly linear over this range (consistent with the linearity check in causal_model.py's BMI band analysis).

# Part B finished.

# Part C: XAI extensions

Device: cuda

### Methods comparison (kl_final, 15 test images)

Side-by-side figures saved to `outputs/extended/xai/kl_final_methods_*.png`. Mean pairwise Spearman rank correlation between methods' saliency maps:

- Grad-CAM vs Grad-CAM++: 0.877
- Grad-CAM vs Integrated Gradients: 0.212
- Grad-CAM vs Occlusion: 0.164
- Grad-CAM++ vs Integrated Gradients: 0.189
- Grad-CAM++ vs Occlusion: 0.128
- Integrated Gradients vs Occlusion: 0.491

Interpretation: the two gradient-based CAM methods agree strongly with each other, but only weakly with the
perturbation-based methods (IG, Occlusion) - a known pattern in the XAI literature, not a bug; different method
families can legitimately highlight different (both locally valid) regions.

### KneeNeXt Grad-CAM (global view, bug-corrected - see xai_extended.py's module docstring)

**Bug found and fixed (for this script only - kneenext.py is an existing file from a prior task, not edited):**
`kneenext.grad_cam_view` hooks `model.features`, which is called TWICE per forward pass (once per view, global
then zoom). Its hooks overwrite a single dict entry, so after the full forward+backward the captured "activation"
ends up being the **zoom** view's (the last forward call) while the captured "gradient" ends up being the
**global** view's (the last backward call, since autograd unwinds in reverse order) - a mismatched, meaningless
pair for any config with `use_zoom=True` (which includes the winning "asymmetry" configuration, so this affected
every committed KneeNeXt Grad-CAM figure from the prior KneeNeXt task, in `outputs/kneenext/results/xai/` and in
`predict_kneenext.py`'s demo output). Verified empirically: captured activations differ between the two forward
calls (confirming both fire), and the hook-captured gradient exactly matches the ground-truth GLOBAL view's
gradient (computed independently via `retain_grad()` on each view's own activation tensor) while the hook-captured
activation exactly matches the ground-truth ZOOM view's activation - a direct, reproducible confirmation of the
mismatch. Fixed here via `kneenext_grad_cam_corrected()`, which captures each forward call's activation with
`retain_grad()` (order-independent) instead of an overwritable hook dict, and reused in `predict_extended.py`.

Only Grad-CAM is run for KneeNeXt in this comparison (not Grad-CAM++/IG/Occlusion): those libraries assume a
single-tensor-input model, and wrapping KneeNeXt's shared-backbone dual-forward-pass architecture for each of them
was judged not worth the added complexity for this task - logged as a scope decision, not an oversight.

### Faithfulness (deletion curves, n=200 random test images)

AUC (lower = more faithful - true-class probability collapses faster when the explanation's claimed-important
regions are actually deleted): Grad-CAM importance-order **0.393** vs random-order 0.351; Integrated Gradients
importance-order **0.185** vs random-order 0.360.

Honest finding: Grad-CAM's importance-ordered deletion curve is actually *less* damaging than random deletion here
(higher AUC) - by this specific test, Grad-CAM's 7x7-resolution, upsampled attention map does not clearly localize
the regions that most drive the prediction, at least not at the patch-blur granularity tested. Integrated
Gradients (pixel-resolution) passes this faithfulness test clearly (0.185 vs 0.360). This is reported as a genuine
limitation of Grad-CAM found by this specific test, not smoothed over - it does not contradict the earlier
sanity-check result (which tests a different property: dependence on learned weights, not spatial precision).

### Sanity check (cascading parameter randomization)

Spearman correlation with the original Grad-CAM map, as layers are randomized from the output backward:
original=1.00, classifier+cbam=0.36, +features[7]=-0.42, +features[6]=-0.45, +features[5]=-0.45,
+features[4]=-0.40, +features[3]=-0.48, +features[2]=-0.51, +features[1]=-0.48, +features[0]=-0.44.

PASSED: correlation with the original map drops sharply (and goes slightly negative) once the backbone is
randomized, as expected for a method that genuinely depends on learned weights rather than architecture/
edge-detection artifacts (the known failure mode this test is designed to catch, per Adebayo et al. 2018).

### Similar cases (3-nearest-neighbor retrieval, cosine similarity on pooled CBAM-attended features)

The 3 nearest training images' majority grade matches the test image's true grade **0.6924** (1146/1655) of the
time - close to (slightly below) kl_final's own classification accuracy (0.6937), i.e. the learned pooled feature
space is reasonably informative for case-based retrieval, though a trained classifier head still edges out a
simple 3-NN vote on it, as expected.

### Calibration and uncertainty

Expected Calibration Error (test, 10 bins): **0.1514** - a real limitation; kl_final's raw softmax/sigmoid
confidences are not well-calibrated probabilities out of the box (no temperature scaling applied). Reliability
diagram saved.

Uncertainty = entropy of the predicted 5-grade distribution. "Refer to radiologist" threshold chosen on validation
(75th percentile of val entropy = 0.6937): on test, 1295/1655 (78.2%) flagged confident (accuracy **0.7707**), 360
flagged uncertain (accuracy **0.4167**) - the uncertain subset is meaningfully less accurate (more than 30 points
lower), so the flag is doing genuinely useful work, not just flagging a token few.

Train-feature cache (`train_features_cache.npz`) and the uncertainty threshold (`uncertainty_threshold.json`) saved
for reuse by `predict_extended.py`, so the combined demo doesn't recompute either at every launch.

# Part C finished.
