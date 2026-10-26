# The Final System — In Simple Language

This document explains what the finished pipeline actually does, why each piece was chosen,
how to run it, and — just as importantly — what it does *not* do well. Every number here
comes from the project's existing results files (see `final_output/tables/` and
`final_output/packaging_log.md` for exactly where each one was read from).

## The pipeline, step by step

```
Input X-ray
    -> CLAHE (contrast enhancement, same preprocessing as every model in this project)
    -> Ensemble: kl_final (seed 7) + KneeNeXt (seed 7), averaged CORAL probabilities,
       tuned decision thresholds
    -> Uncertainty flag (entropy of the predicted distribution; "refer to radiologist"
       if above a threshold chosen on validation data)
    -> XAI: Integrated Gradients (kl_final) + bug-fixed Grad-CAM (KneeNeXt) + 3 most
       similar training cases
    -> Causal risk panel: population findings (obesity, BMI, family history) + this
       patient's individual obesity effect (causal forest) + knee injury association
```

**Why an ensemble of exactly these two models?** A greedy selection procedure, run on the
validation set only, tried every available trained model (`kl_final` seed 42, its seed-7
sibling, Hybrid 1, KneeNeXt both seeds) and kept adding whichever one most improved accuracy
without hurting QWK by more than 0.005. It started from `kl_final`'s seed-7 sibling (the best
single model on validation accuracy, not `kl_final` itself) and added KneeNeXt's seed-7 run.
Nothing else helped. The 4 CORAL decision thresholds were then tuned (5-fold cross-validated
on validation, kept because the gain generalized) instead of using the default 0.5 for all
four. Horizontal-flip test-time augmentation was tried and rejected — it made things worse.

**Why CLAHE?** It was the first adopted step in `kl_final`'s own ablation (raised val QWK from
0.8286 to 0.8340) and every model downstream reuses it for consistency.

**Why an uncertainty flag?** On the test set, the 21.8% of images flagged "uncertain" (by an
entropy threshold chosen on validation data, not test data) have 41.67% accuracy, vs. 77.07%
for the "confident" 78.2% — a genuinely useful triage signal for a radiologist's attention.

**Why Integrated Gradients for kl_final and a separately-fixed Grad-CAM for KneeNeXt?** A real
bug was found in the prior KneeNeXt work's Grad-CAM implementation (it averaged the wrong
view's activation with the wrong view's gradient whenever KneeNeXt used both its global and
zoomed views — which it always does in its winning configuration). The fix lives in
`xai_extended.py`'s `kneenext_grad_cam_corrected()`; every KneeNeXt Grad-CAM image in this
final package uses only that corrected method — no old KneeNeXt Grad-CAM figure was reused.

## What each component does and why it exists

| Component | What it does | Why it's here |
|---|---|---|
| CLAHE | Contrast-enhances the X-ray | Adopted in `kl_final`'s ablation; keeps every model's input consistent |
| kl_final (ConvNeXt-Tiny + CBAM + CORAL) | The locked, single-view classifier | The project's original, bootstrap-significant improvement over the EfficientNet-B0 baseline |
| KneeNeXt | A knee-specific modification of `kl_final`'s backbone (joint-zoom view, multi-scale fusion, joint-guided attention, left/right asymmetry feature) — **not a new algorithm**, an architecture modification | Only the full 4-module combination beat `kl_final`'s retrained control on validation; included in the ensemble because it measurably helps there |
| Threshold-tuned ensemble | Averages the two models' CORAL probabilities, decodes with 4 tuned thresholds instead of flat 0.5 | The only configuration in this project line to beat `kl_final` on test accuracy with statistical significance |
| Uncertainty flag | Entropy of the predicted grade distribution, thresholded on validation data | Separates a meaningfully more error-prone subset of predictions for human review |
| Integrated Gradients / Grad-CAM | Pixel- and region-level explanations of what the model looked at | IG passed a faithfulness test that Grad-CAM did not (see Limitations) |
| 3 similar training cases | Nearest neighbors in `kl_final`'s pooled feature space, by cosine similarity | A case-based second opinion a radiologist can sanity-check against |
| Causal risk panel | Population and (where inputs are given) individual obesity effect, BMI what-if, family history, knee injury association | Answers "why might this patient be at higher risk," separately from "what grade is this image" |

## How to run it

```
python scripts/predict_extended.py                              # file browser
python scripts/predict_extended.py path/to/xray.png --show [--age N] [--bmi N] [--family-history yes|no]
```

`predict.py` (the original, `kl_final`-only demo) still works exactly as before and is
unchanged — confirmed by re-running it during this packaging step.

To regenerate every table and figure in this folder from the existing results files:
```
python scripts/make_final_figures.py
```

If a UI (`ui-integration` branch) has been built, see `ui/README.md` for its own run
command; it calls into the same `predict_extended.py` code via `ui/model_bridge.py`, not a
separate copy of the model logic.

## Limitations and honest notes

- **The ensemble ties `kl_final` on QWK** (0.8574 vs. 0.8570, difference +0.0005, 95% CI
  [-0.0090, 0.0096] — not significant). **The statistically significant gain is in accuracy**
  (+0.0234, 95% CI [0.0073, 0.0405]), not ranking quality.
- **KL1 recall dropped** from `kl_final`'s 0.3918 to the ensemble's 0.2432 — the tuned
  thresholds traded away some doubtful-OA sensitivity for overall accuracy. Reported
  plainly, not hidden behind the headline accuracy number.
- **Grad-CAM failed the deletion-curve faithfulness test**; Integrated Gradients passed it
  clearly. Deleting Grad-CAM's claimed-important regions first was *less* damaging to the
  model's confidence than random deletion (AUC 0.393 vs. 0.351) — its low-resolution,
  upsampled attention map doesn't precisely localize what drives the prediction, at least
  not at the patch granularity tested. IG's importance-ordered deletion collapsed confidence
  much faster than random (AUC 0.185 vs. 0.360).
- **The knee injury result is an association, not a demonstrated cause.** It comes from
  self-reported injury history in the OAI questionnaire data, with no way to confirm from
  our data whether the injury happened before or after the knee changes being measured —
  reverse causation and recall bias are real possibilities we cannot rule out.
- **KneeNeXt is a knee-specific architecture modification of `kl_final`, not a new
  algorithm** — it reuses the same ConvNeXt-Tiny backbone, CBAM attention, and CORAL head,
  adding four domain-informed modules on top.
- **Segmentation masks did not improve the classifier**: the U-Net/U-Net++ mask-channel
  variants scored lower on test QWK (0.8130 / 0.8144) than the plain image-only baseline
  (0.8246) — consistent with the segmentation models' known patchy performance on OAI's
  tight joint crops (they were trained on CGMH's full-leg images).
- **JSW (joint space width) validation failed**: the automated measurement's correlation
  with radiologist JSN grades (ρ=-0.179) fell short of the pre-registered -0.3 threshold, so
  it was excluded from every downstream model and analysis that depended on it passing.
- **Single dataset, no external validation**: every number in this project — classifier,
  causal findings, XAI findings — comes from the OAI cohort alone. Nothing here has been
  checked against an independent dataset or clinical population.
