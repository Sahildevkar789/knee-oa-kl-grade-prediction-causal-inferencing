# Extended work (experimental) — Summary

Three parts built under a strict **add-only** rule: no existing script, checkpoint, result,
log, or data file was edited, overwritten, renamed, moved, or deleted; all existing models
and clinical data were loaded read-only. New code: `scripts/ensemble_kl.py`,
`scripts/causal_extended.py`, `scripts/xai_extended.py`, `scripts/predict_extended.py`. New
outputs: `outputs/ensemble/`, `outputs/extended/`. Decision-by-decision log (including the
real bug found in Part C):  `outputs/extended/extended_log.md`.

No classifiers were retrained anywhere in this work. Model selection throughout used
validation only; the test set was touched exactly once, at the end of Part A.

## Part A — Ensemble (push accuracy above 70%)

Collected validation CORAL probabilities from every available trained model: `kl_final`
(seed 42), its seed-7 sibling from the backbone ablation, Hybrid 1, and KneeNeXt (both
seeds, its winning "asymmetry" configuration).

| Step | Added | Members | Val acc | Val QWK |
|---|---|---|---|---|
| 0 | kl_final_seed7 | [kl_final_seed7] | 0.6889 | 0.8406 |
| 1 | kneenext_seed7 | [kl_final_seed7, kneenext_seed7] | 0.6925 | 0.8491 |

Greedy selection started from `kl_final_seed7` (the single best model by val accuracy, not
`kl_final` itself) and added `kneenext_seed7`; nothing else improved accuracy without
dropping QWK by more than 0.005. **Threshold tuning adopted** (5-fold CV gain +0.0061 over
flat 0.5; final thresholds [0.70, 0.45, 0.50, 0.55], full-val accuracy 0.7107). **Flip TTA
not adopted** (0.7107 → 0.6985 with TTA — made it worse).

**Test set (evaluated once)**: Accuracy **0.7172**, QWK **0.8574**, Macro F1 **0.6933**, MAE
**0.3293** — **above the 70% target**. Honest caveat: per-grade recall shows the tuned
thresholds traded away some KL1 sensitivity (0.2432, vs. `kl_final`'s own 0.3918) for the
overall accuracy gain — not a free lunch, reported plainly rather than only highlighting the
headline number.

| Comparison | QWK diff | 95% CI | Significant | Acc diff | 95% CI | Significant |
|---|---|---|---|---|---|---|
| Ensemble vs. kl_final | +0.0005 | [−0.0090, 0.0096] | No | +0.0234 | [0.0073, 0.0405] | **Yes** |
| Ensemble vs. baseline | +0.0329 | [0.0162, 0.0496] | **Yes** | +0.0886 | [0.0616, 0.1148] | **Yes** |

The ensemble's accuracy gain over `kl_final` is statistically significant; its QWK gain is
not (essentially tied). Full detail: `outputs/ensemble/ensemble_summary.md`.

## Part B — Causal extensions

Same one-knee-per-patient data (seed 42, n=4,071) and DAG discipline as Phase 5: never
adjusted for WOMAC, JSN, osteophyte grades, or physical activity (PASE) — all are affected by
OA itself, not causes of it.

**Sensitivity to hidden confounders**: E-value for the obesity→KL grade effect (+0.521, 95%
CI [0.448, 0.595]) is **2.345** — an unmeasured confounder would need to be associated with
*both* obesity and KL grade by at least a 2.34-fold risk ratio each, beyond age/family
history, to fully explain the effect away; the CI's near-null bound has E-value 2.167. A grid
simulation (`sensitivity_confounder_grid.png`) over confounder strengths up to 0.30 on both
treatment and outcome never flipped the effect negative — reasonably robust to this kind of
simulated hidden confounding.

**Knee injury as a second cause**: `P01INJR`/`P01INJL` (unified_oai_dataset.csv, matched by
knee side) are usable injury-history columns (41/4071 missing, dropped). Prior knee injury →
KL grade, adjusted for age/BMI/family history: linear regression **+0.540**, propensity score
matching **+0.550** KL grade points — similar magnitude to obesity's effect. All three
refuters pass (placebo ≈ 0, others stable).

**Personalized obesity effects** (EconML `CausalForestDML`, X=age+family history): mean
individual effect **+0.510** (consistent with Phase 5's population range of +0.48 to +0.58).
Little age trend (effect-vs-age correlation −0.024). Patients with a family history of knee
replacement show a *smaller* mean individual effect (+0.460) than those without (+0.518) —
a modest, not strongly conclusive, difference. Fitted model saved
(`causal_forest_model.pkl`) and reused by the demo for per-patient estimates.

**BMI dose-response**: expected KL grade vs. continuous BMI (age at median, no family
history) is close to linear across the observed range (16.9–48.7), consistent with the
band-linearity check already done in `causal_model.py`. Plot + 95% CI band:
`bmi_dose_response.png`.

Full detail and all figures: `outputs/extended/causal/`.

## Part C — XAI extensions

**A real bug was found and fixed** (for this script only — `kneenext.py` is an existing file
from the prior task, not edited): `kneenext.grad_cam_view` hooks `model.features`, which runs
*twice* per forward pass (once per view). Its hooks overwrite a single variable, so the
captured "activation" ends up being the **zoom** view's (the last forward call) paired with
the **global** view's gradient (the last backward call, since autograd unwinds in reverse
order) — a mismatched, meaningless CAM for any KneeNeXt configuration with `use_zoom=True`,
which includes the winning "asymmetry" config. **This affected every committed KneeNeXt
Grad-CAM figure from the prior task** (`outputs/kneenext/results/xai/` and
`predict_kneenext.py`'s demo). Verified empirically by comparing the hook's captured
activation/gradient against ground truth computed independently via `retain_grad()` on each
view's own tensor — an exact match confirms the mismatch, not a guess. Fixed here via
`kneenext_grad_cam_corrected()`, reused by `predict_extended.py`; `kneenext.py` itself is
untouched, per this task's add-only rule.

**Methods comparison** (kl_final, 15 test images, 5 wrong): Grad-CAM and Grad-CAM++ agree
strongly (mean Spearman ρ=0.877), but both correlate only weakly with Integrated Gradients
(ρ=0.21, 0.19) and Occlusion (ρ=0.16, 0.13) — gradient-based and perturbation-based methods
emphasize different regions, a known pattern in the XAI literature. For KneeNeXt, only the
(bug-fixed) Grad-CAM was run — wrapping the other three methods for its dual-input
architecture was judged not worth the added complexity here, logged as a scope decision.

**Faithfulness (deletion curves, n=200 random test images)**: Integrated Gradients clearly
passes this test — deleting its claimed-important regions first collapses the true-class
probability much faster than random deletion (AUC 0.185 vs. 0.360). **Grad-CAM does not** —
its importance-ordered deletion curve has a *higher* AUC than random (0.393 vs. 0.351),
meaning its 7×7-resolution, upsampled attention map doesn't clearly localize the regions that
most drive the prediction at the patch granularity tested. Reported as a genuine limitation,
not smoothed over — it doesn't contradict the sanity check below, which tests a different
property.

**Sanity check** (cascading parameter randomization, Adebayo et al. 2018 style): **passed** —
correlation with the original Grad-CAM map drops sharply (to around −0.44 to −0.51) once the
head/CBAM and then the backbone are progressively randomized, confirming the map genuinely
depends on learned weights rather than architecture artifacts.

**Similar cases**: 3-nearest-neighbor majority grade (cosine similarity on pooled
CBAM-attended features) matches the true grade **69.24%** of the time on the test set — close
to (slightly below) `kl_final`'s own 69.37% classification accuracy.

**Calibration and uncertainty**: Expected Calibration Error **0.1514** on raw test
probabilities (no temperature scaling) — a real, reported limitation. An entropy-based
uncertainty flag, with its threshold chosen on validation (75th percentile), splits the test
set into a confident 78.2% (accuracy 77.1%) and an uncertain 21.8% (accuracy 41.7%) — more
than 30 points lower, so the flag is doing genuinely useful triage work.

Full detail and all figures: `outputs/extended/xai/`.

## Combined demo: predict_extended.py

Same behavior as `predict.py` (file browser, patient-detail prompts, Grad-CAM, causal panel),
plus: uses the Part A ensemble if it beats `kl_final`'s solo validation accuracy (checked at
startup, not just assumed — it does: 0.7107 vs. 0.6755), states which model(s) were used,
shows Grad-CAM averaged over those models, shows the 3 most similar training cases with their
grades, flags uncertain predictions ("refer to radiologist") using Part C's
validation-chosen threshold, and extends the causal panel with this patient's individual
obesity effect (Part B's causal forest) and the population knee-injury finding. `predict.py`
and its default model (`kl_final`) are unchanged. Tested on one image per KL grade without
the GUI — all five ran correctly, including one genuinely ambiguous case (true KL2, predicted
KL0) that the uncertainty flag correctly caught and labeled "REFER TO RADIOLOGIST."

## Final comparison table

| Model | Accuracy | QWK | Macro F1 | MAE |
|---|---|---|---|---|
| EfficientNet-B0 baseline | 0.6284 | 0.8246 | 0.6746 | 0.4036 |
| kl_final (locked) | 0.6937 | 0.8570 | 0.6926 | 0.3396 |
| KneeNeXt (full config) | 0.6991 | 0.8609 | 0.6940 | 0.3353 |
| **Ensemble (kl_final_seed7 + kneenext_seed7, tuned thresholds)** | **0.7172** | **0.8574** | **0.6933** | **0.3293** |

**Bootstrap significance (2,000 resamples each, vs. kl_final):**

| Model | QWK diff | 95% CI | Significant | Acc diff | 95% CI | Significant |
|---|---|---|---|---|---|---|
| KneeNeXt | +0.0040 | [−0.0021, 0.0102] | No | +0.0053 | [−0.0060, 0.0169] | No |
| Ensemble | +0.0005 | [−0.0090, 0.0096] | No | +0.0234 | [0.0073, 0.0405] | **Yes** |

The ensemble is the first configuration in this whole project line to beat `kl_final` on
accuracy with statistical significance, and it clears the 70% accuracy target. Its QWK is
essentially tied with `kl_final`'s, and the accuracy gain comes with a real trade-off (weaker
KL1 recall) — reported as a genuine, qualified improvement, not an unqualified win.
`kl_final` remains the project's locked model; the ensemble and KneeNeXt are experimental,
additive results documented alongside it.
