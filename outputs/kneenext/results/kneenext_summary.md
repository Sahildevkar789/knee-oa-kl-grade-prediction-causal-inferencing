# KneeNeXt (experimental) — Summary

Experimental knee-specific modification of `kl_final`'s ConvNeXt-Tiny classifier, built and
evaluated under a strict **add-only** rule: no existing script, checkpoint, result, log, or
data file was edited, overwritten, renamed, moved, or deleted. All new code lives in
`scripts/kneenext.py`, `scripts/train_kneenext.py`, `scripts/evaluate_kneenext.py`,
`scripts/predict_kneenext.py`; all new outputs live under `outputs/kneenext/`. Full
decision-by-decision trail (including the two real bugs found and fixed during development):
`outputs/kneenext/results/kneenext_log.md`.

## 1. Architecture

`outputs/kneenext/results/kneenext_architecture.png` (two-row diagram — top row: global view,
bottom row: zoom view, both sharing one ConvNeXt-Tiny backbone, converging at the head).
*Note: `Classifier_Architecture.png`, named in the brief as the style to match, was searched
for across the whole repository and not found anywhere, so this uses its own clean two-row
convention instead.*

KneeNeXt starts from `kl_final`'s trained ConvNeXt-Tiny + CBAM weights (loaded read-only from
`outputs/checkpoints/kl_final_best.pt`) and adds four optional modules, each gated by a flag
so they can be ablated independently and cumulatively:

| Module | What it does | Clinical motivation |
|---|---|---|
| **Joint-Zoom branch** | A second, cropped-and-zoomed view (±25% of image height around the detected joint line, resized to 224×224) passed through the *same shared* ConvNeXt-Tiny weights as the global view. | A radiologist grading KL severity looks specifically at the joint space — narrowing, osteophytes, sclerosis. Resizing the whole 224×224 knee down for a single global view compresses that region to a handful of pixels; a dedicated zoomed view gives the network the joint space at genuinely higher effective resolution, not just wherever global pooling happens to notice it. |
| **Multi-scale fusion** | Stage 3 (384×14×14) is projected to 768 channels (1×1 conv) and downsampled to 7×7, then added to Stage 4 (768×7×7), for both views. | Joint-space narrowing can be a fine, thin-structure change. Stage 4 alone has already downsampled through two more stride-2 stages past Stage 3, which risks blurring out exactly this kind of detail before the attention/pooling stage ever sees it. |
| **Joint-Guided CBAM** | CBAM's spatial attention becomes `sigmoid(learned_map + α·joint_prior)`, where `joint_prior` is a Gaussian band around the joint line (centered, for the zoom view) and α is a learnable scalar initialized at 1.0. | Generic CBAM has to discover "the joint line matters" purely from label supervision, with no guarantee it ever really does. Encoding "look here first" directly as a prior — but with a *learnable* weight, not a hard mask — lets the network keep that inductive bias while still being free to downweight it if the data disagrees. |
| **Asymmetry module** | The global view's CBAM-attended feature map is split into left/right halves (dropping the central column — the tibial spines/intercondylar notch, the same anatomically-ambiguous region `measure_jsw.py` already skips for its compartment split), each half pooled, and `|left − right|` is concatenated onto the pooled global/zoom features. | KL grading is sensitive to *asymmetric* wear — e.g. isolated medial-compartment narrowing with an untouched lateral compartment is a real, specific pattern. A single symmetric pooled vector structurally cannot represent "the two sides differ"; this module makes that difference an explicit, separate input to the head. |

Head: `concat([global_pooled, zoom_pooled, |asymmetry|]) → LayerNorm → Dropout(0.3) → CORAL
head (4 thresholds, class-weighted)` — same CORAL ordinal decoding as `kl_final`
(predicted grade = count of thresholds with sigmoid > 0.5).

With every flag off ("control"), KneeNeXt is architecturally identical to `kl_final` (single
view, plain CBAM on Stage 4 only) — used as the ablation's baseline, retrained with KneeNeXt's
own two-stage schedule so the schedule itself isn't a confound when judging the other modules.

**Training**: two-stage, both stages seeded (42 and 7) for every configuration. Stage 1 —
ConvNeXt-Tiny backbone frozen, new modules trained for up to 6 epochs at lr 1e-3. Stage 2 —
everything unfrozen (backbone lr 5e-5, new modules lr 3e-4), 1 warmup epoch then cosine,
early stopping on validation QWK (patience 6, up to 25 epochs). Same data split, CLAHE, and
geometric/color augmentation as `kl_final` — augmentation is applied once to the shared
pre-split CLAHE image, so both the global and zoom views see an identical flip/rotation/
brightness draw, not two independent random ones. Batch size 32 throughout (no OOM
encountered; the fallback-to-16 retry path exists but was never triggered). `--workers 0`.

**Joint-line detector**: copied from `measure_jsw.find_joint_row`'s core algorithm (darkest
horizontal band, central 50% of image height) and extended with a failure/fallback check (a
z-score test on how much darker that row actually is than its surroundings) since, unlike a
validation-only JSW measurement, a bad crop here would silently train the zoom branch on
noise. Falls back to the vertical center on failure. Measured failure rate on the full
train+val set: **0.02%** (2 images out of ~6,600). Visual check figure (4 images per grade,
20 total): `outputs/kneenext/results/kneenext_joint_zoom_check.png` — the detected joint line
and crop band line up correctly with the visible joint space across all 5 grades.

## 2. Ablation (validation set, 2 seeds each; test set untouched until section 3)

| Config | Mean val Acc | Mean val QWK | Mean val MacroF1 | Mean val MAE | Params | Train time (min) | Adopted |
|---|---|---|---|---|---|---|---|
| control (kl_final, 2-stage retrain) | 0.6743 ± 0.0061 | 0.8390 ± 0.0009 | 0.6913 ± 0.0045 | 0.3692 ± 0.0012 | 27.90M | 13.8 | **yes** |
| + Joint-Zoom | 0.6804 ± 0.0109 | 0.8387 ± 0.0012 | 0.6827 ± 0.0058 | 0.3680 ± 0.0048 | 27.90M | 19.0 | no |
| + Multi-scale fusion | 0.6592 ± 0.0006 | 0.8362 ± 0.0002 | 0.6887 ± 0.0034 | 0.3783 ± 0.0006 | 28.20M | 18.1 | no |
| + Joint-Guided CBAM | 0.6707 ± 0.0085 | 0.8383 ± 0.0002 | 0.6884 ± 0.0093 | 0.3717 ± 0.0036 | 28.20M | 23.5 | no |
| + Asymmetry (**full KneeNeXt**) | 0.6725 ± 0.0030 | **0.8446 ± 0.0002** | 0.6817 ± 0.0027 | 0.3662 ± 0.0030 | 28.20M | 18.6 | **yes** |

Full table with per-grade recall: `outputs/kneenext/results/kneenext_ablation.csv`. Learned
Joint-Guided CBAM α: ~1.15–1.18 across the two configs that use it (grew somewhat from its
1.0 init — the prior's influence increased during training rather than being learned away).

**Each intermediate module, added individually on top of the running best, did not beat it.**
Joint-Zoom alone was a statistical tie (−0.0003 QWK); Multi-scale fusion actively hurt
(−0.0028); Joint-Guided CBAM was a near-tie (−0.0007). Per the brief's literal "add modules
cumulatively" design, each config always builds on the *full* cumulative stack regardless of
whether the previous step individually cleared the bar (not a greedy "skip if it didn't
help" search) — so "asymmetry" is genuinely the full four-module KneeNeXt, not an
accidentally-pruned subset. **Only the full combination — all four modules together —
beat `control`**, by +0.0056 mean val QWK, suggesting a synergistic rather than additive
effect: the asymmetry signal may only become informative once the zoom view, multi-scale
features, and joint-guided attention are all already in place to feed it a meaningful
left/right-attended feature map. This is reported as an interesting, honest finding, not a
designed-in outcome — the ablation could equally have shown no configuration beating control.

## 3. Final test-set comparison (evaluated once, on KneeNeXt's winning config only)

| Model | Accuracy | QWK | Macro F1 | MAE |
|---|---|---|---|---|
| EfficientNet-B0 baseline | 0.6284 | 0.8246 | 0.6746 | 0.4036 |
| kl_final (locked) | 0.6937 | 0.8570 | 0.6926 | 0.3396 |
| **KneeNeXt (asymmetry = full KneeNeXt)** | **0.6991** | **0.8609** | **0.6940** | **0.3353** |

Per-grade test report: `outputs/kneenext/results/kneenext_test_report.csv`. Confusion matrix:
`outputs/kneenext/results/kneenext_test_confusion_matrix.png`. Per-image predictions:
`outputs/kneenext/results/kneenext_test_predictions.csv`.

**Paired bootstrap (2,000 resamples, via the existing `bootstrap_compare.py`, unmodified):**

| Comparison | QWK diff | 95% CI | Significant | Accuracy diff | 95% CI | Significant |
|---|---|---|---|---|---|---|
| KneeNeXt vs. kl_final | +0.0040 | [−0.0021, 0.0102] | **No** | +0.0053 | [−0.0060, 0.0169] | **No** |
| KneeNeXt vs. baseline | +0.0364 | [0.0210, 0.0512] | **Yes** | +0.0706 | [0.0453, 0.0961] | **Yes** |

KneeNeXt scores numerically higher than `kl_final` on every metric, continuing the same
pattern as Hybrid 1 and Hybrid 2 before it (both also numerically ahead of `kl_final` on
test, neither significantly so) — but model selection here used validation QWK throughout
(0.8446 vs. control's 0.8390), and KneeNeXt did win that comparison, unlike the two earlier
hybrids. **`kl_final` is not being proposed as replaced** — KneeNeXt remains an experimental,
additive result, significantly better than the original EfficientNet-B0 baseline but not
distinguishable from `kl_final` on this test set at 95% confidence.

## 4. XAI

Grad-CAM (global view, last ConvNeXt stage) + the zoom-view crop for 15 test images (3/grade,
5 wrong predictions): `outputs/kneenext/results/xai/`. Visual pattern consistent with
`kl_final`'s own Phase 4 findings — correct/moderate-grade predictions show attention tightly
localized on the joint space itself (e.g. `KL2_9003316R.png`), while severe (KL4) cases often
show attention shifted to marginal osteophytes beside the now-collapsed joint space rather
than the joint line band itself (e.g. `KL4_9101066R.png`) — the same clinically-sound
"looking at osteophytes instead of a joint space that barely exists anymore" pattern already
documented for `kl_final`, not a new failure mode introduced by KneeNeXt.

## 5. Where everything is

- Scripts: `scripts/kneenext.py`, `scripts/train_kneenext.py`, `scripts/evaluate_kneenext.py`,
  `scripts/predict_kneenext.py` (demo — same behavior as `predict.py`: file browser, patient-
  detail prompts, Grad-CAM, causal risk panel — plus KneeNeXt's zoom view panel).
- Checkpoints: `outputs/kneenext/checkpoints/kneenext_best.pt` (+ `kneenext_best_config.json`
  naming the winning config/seed) pushed to GitHub. The 10 individual per-config/per-seed
  ablation checkpoints (~1.1GB total) stay local only — editing `.gitignore`/`.gitattributes`
  to exclude them (the pattern used elsewhere in this project for non-winning checkpoints)
  would itself be an edit to an existing file, which the add-only rule for this task forbids;
  they're simply left untracked rather than committed, so the repo isn't bloated with ~1.1GB
  of non-winning binaries.
- Decision log: `outputs/kneenext/results/kneenext_log.md`.
