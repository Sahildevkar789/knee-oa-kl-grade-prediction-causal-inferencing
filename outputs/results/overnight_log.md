# Overnight run log

Started 2026-10-20. Working through the unattended plan: finish Step 1 (warmup decision),
Step 2 (ablation), Step 3 (final model), Step 4 (test + bootstrap), Step 5 (predict.py),
Step 6 (write-up). Every decision and any failures are logged here as they happen.

## Step 1: warmup decision
- Backbone picked: **convnext_tiny** (mean val QWK across seeds 42/7/123: 0.8187 ± 0.011,
  vs efficientnet_b0's 0.8078 ± 0.003). Logged here since the user's message already
  settled this; carrying it forward as the Step 1 setup.
- Warmup run (2 epochs, seed 42) finished: best val QWK **0.8315** (epoch 7), vs **0.8312**
  without warmup. Difference is +0.0003 (warmup is marginally *better*, well within the
  "no more than 0.01 below" keep-threshold either way).
  **Decision: KEEP warmup (2 epochs).**
- Collapse check: without warmup, epoch 1-3 QWK was 0.5061 -> 0.6785 -> 0.4325 (a real dip
  at epoch 3). With warmup, epoch 1-3 QWK was 0.7301 -> 0.7265 -> 0.5919 - epoch 1 starts far
  higher (0.51 -> 0.73) and the epoch-3 dip is much shallower (0.43 -> 0.59). Warmup clearly
  reduced the early instability, though a smaller dip still happens at epoch 3.
- **Step 1 setup carried into Step 2**: backbone=convnext_tiny, warmup_epochs=2, loss=ce,
  class_weights=inverse, no CLAHE/CBAM/TTA. This is the ablation's starting/baseline row.

## Step 2: ablation plan
Implemented all 5 options as CLI flags in train_classifier.py/kl_common.py (old behavior
stays the default in every case): `--clahe`, `--attention cbam` (convnext_tiny-only, wraps
the final feature map before pooling), `--loss coral` (4 binary "grade>k" outputs, per-
threshold pos_weight, predicted grade = count of thresholds exceeded), `--class-weights
{inverse,sqrt,none}`, and TTA (implemented as an eval-time flag in run_eval_probs, applied
post-hoc - not a training-time option, since it only ever changes evaluation).
Smoke-tested each individually (1 epoch) before committing GPU time: CBAM added exactly the
expected param count (+73,826), CORAL's head is exactly 769 params smaller than the 5-way
head (Linear(768,4) vs Linear(768,5)), sqrt weights matched sqrt(inverse) by hand-check.
All smoke-test artifacts removed before the real runs.

**Ablation order** (as listed in the plan): clahe -> cbam -> coral -> class-weights sqrt,
each tested on top of the current best config so far, 2 seeds (42, 7) each, comparing mean
val QWK. TTA is evaluated last, on the best model from steps 1-4, with/without TTA on val
(no retraining needed for TTA).

**Starting row** (Step 1 setup, convnext_tiny+warmup2): seed42=0.8315, seed7=0.8257.
Current best mean val QWK = 0.8286.

### Ablation axis 1: --clahe
seed42=0.8332, seed7=0.8347, mean=**0.8340**. Beats 0.8286.
**Decision: ADOPT CLAHE.** Current best config: convnext_tiny+warmup2+clahe, mean=0.8340.

### Ablation axis 2: --attention cbam (on top of clahe)
seed42=0.8433, seed7=0.8302, mean=**0.8368**. Beats 0.8340.
**Decision: ADOPT CBAM.** Current best config: convnext_tiny+warmup2+clahe+cbam, mean=0.8368.
Note: seed7's CBAM result (0.8302) was slightly below clahe-alone's seed7 (0.8347) - the mean
still wins per the stated rule, but this is a reminder CBAM's seed-to-seed variance is real;
following the mechanical "compare means" rule as instructed rather than overriding it.
Attention maps for 10 val images (2/grade) saved to outputs/results/cbam_maps/ - qualitatively
reasonable (covers the bone/joint region broadly) but coarse, as expected from a 7x7 feature
map upsampled to 224x224; not pixel-precise like Grad-CAM would be.

### Ablation axis 3: --loss coral (on top of clahe+cbam)
seed42=0.8425, seed7=0.8406, mean=**0.8416**. Beats 0.8368.
**Decision: ADOPT CORAL.** Current best config: convnext_tiny+warmup2+clahe+cbam+coral, mean=0.8416.
Note: seed42's run hit the best QWK at the final epoch (25/25) - still improving when training
stopped due to the fixed epoch budget, not early-stopping. Kept epochs=25 as instructed
("keep all existing settings unless a step says otherwise"); flagging this as a place where
more epochs might help further, for future reference, not something to change here.

### Bug caught before wasting a run: --class-weights sqrt + --loss coral
`coral_pos_weights()` computed its per-threshold BCE pos_weight directly from the label
distribution and never looked at `--class-weights` at all - so the first "+class-weights sqrt"
run (on top of clahe+cbam+coral) would have been silently identical to plain coral, making
that ablation step meaningless. Killed it immediately (no partial checkpoint existed yet),
added a `mode` parameter to `coral_pos_weights` (inverse/sqrt/none, same semantics as
`compute_class_weights`), wired `args.class_weights` through to it in train_classifier.py,
and verified the three modes now actually produce different weight tensors before re-running.

### Ablation axis 4: --class-weights sqrt (on top of clahe+cbam+coral)
seed42=0.8346, seed7=0.8346 (coincidentally identical), mean=**0.8346**. Below 0.8416.
**Decision: NOT ADOPTED.** Current best config stays convnext_tiny+warmup2+clahe+cbam+coral
(inverse class weights), mean=0.8416. This is the config going into the TTA check and Step 3.

### TTA check (on the best config's two seed checkpoints, val set)
seed42: no-TTA 0.8425 -> TTA 0.8340 (worse). seed7: no-TTA 0.8406 -> TTA 0.8386 (worse).
Mean no-TTA=0.8416 vs mean TTA=0.8363.
**Decision: TTA NOT ADOPTED.** Plausible reason: horizontal flip swaps medial/lateral knee
anatomy (unlike typical photos, left/right isn't an arbitrary symmetry for a knee X-ray), so
flip-averaging likely blends in a genuinely different anatomical view rather than a harmless
augmentation - consistent with it hurting rather than helping here.

`outputs/results/ablation.csv` written with mean+-std for all 6 rows (val acc/QWK/macroF1/MAE,
KL0 recall, KL1 precision, KL3 recall, KL4 recall, params, train time, adopted).

## Step 2 final answer
**Adopted: clahe + cbam + coral (inverse class weights, no TTA)**, on top of the Step 1 setup
(convnext_tiny, warmup_epochs=2). Final ablation mean val QWK = 0.8416, up from the Step 1
starting mean of 0.8286.

## Step 3: final model
Best config's two seeds: 42=0.8425, 7=0.8406 -> seed 42 used.
Copied `kl_baseline_convnext_tiny_warmup2_clahe_cbam_coral_best.pt` -> `outputs/checkpoints/kl_final_best.pt`.
Wrote `outputs/checkpoints/kl_final_config.json` describing every option (backbone, attention,
loss, class_weights, clahe, tta, warmup_epochs, seed, hyperparams, and the full ablation history).

## Step 4: test evaluation (test set touched for the first time here)
kl_final on test (1655 images, TTA not applied since it wasn't adopted):
Accuracy 0.6937, QWK **0.8570**, Macro F1 0.6926, MAE 0.3396.
vs kl_baseline (EfficientNet-B0): Accuracy 0.6284, QWK 0.8246, Macro F1 0.6746, MAE 0.4036.
Saved `outputs/results/final_test_report.csv` (head-to-head table incl. per-grade P/R/F1),
`kl_final_test_predictions.csv`, `kl_final_test_confusion_matrix.png`.

**Paired bootstrap (2,000 resamples, seed 42) vs kl_baseline:**
- QWK: mean diff +0.0324, 95% CI [0.0174, 0.0478] - **significant** (CI excludes 0)
- Accuracy: mean diff +0.0653, 95% CI [0.0399, 0.0918] - **significant**
Saved to `outputs/results/significance_final_vs_baseline.csv`.

## Step 5: wiring kl_final into predict.py
Added `--run final` (new default) to predict.py: loads kl_final_best.pt (convnext_tiny,
out_dim=4, attention=cbam), applies CLAHE before classification, decodes via
coral_probs_to_grade (count of thresholds exceeded) instead of argmax, and converts the 4
threshold probabilities into a 5-length display distribution (p0=1-P(>0), pk=P(>k-1)-P(>k),
p4=P(>3), clipped+renormalized) so the existing bar-chart UI works unchanged. TTA not wired
in since it wasn't adopted. `--auto-crop` composes correctly with `--run final` (tested).
Old `--run baseline|unet|unetpp` verified byte-identical to their pre-change behavior
(same 99.0%/99.8% KL4 confidence as before on the same test image).

Ran on 5 test images (one per grade), non-interactively, outputs saved to outputs/predictions/:
KL0 img -> predicted KL0 (94.2%, correct); KL1 img -> predicted KL0 (89.4%, wrong - adjacent-
grade confusion, a known weak spot); KL2 img -> predicted KL0 (50.3%, wrong, low confidence);
KL3 img -> predicted KL3 (99.3%, correct); KL4 img -> predicted KL4 (99.6%, correct).
3/5 correct on this tiny sample - consistent with ~69% overall test accuracy, not a new issue.

## Step 6: write-up
Wrote `outputs/results/classifier_summary.md` (backbone table w/ seed means, full ablation
table, adopted/rejected reasoning, one-line final pipeline, test results, bootstrap result,
confusion matrix embed). Updated CLAUDE.md's Classifier section (full rewrite describing
kl_final, kept the original 3-run comparison as a "superseded" subsection since those
checkpoints/results are still on disk and still usable via predict.py), folder layout, and
Current status.

## Retries / failures
None of the training or evaluation runs failed outright (no 3-retry case was needed). The
one real problem (coral_pos_weights ignoring --class-weights) was caught via inspection
before wasting a training run, not via a failed/retried run - see the "Bug caught" entry
above. No other errors to report.

## All done.
Final config: convnext_tiny + CLAHE + CBAM + CORAL (inverse weights, warmup=2, no TTA),
seed 42. Test QWK 0.8570 vs baseline 0.8246 (+0.0324, bootstrap-significant, 95% CI
[0.017, 0.048]). Test accuracy 0.6937 vs 0.6284 (+0.0653, significant, CI [0.040, 0.092]).

**Caveat logged per the "never overwrite existing results" rule**: while regression-testing
the rewritten `evaluate_classifier.py` against the original locked `kl_baseline_best.pt`, it
re-wrote `kl_baseline_test_predictions.csv` and `kl_baseline_test_report.csv` (byte-identical
content - same checkpoint, same test set, nothing changed) and additionally created a new,
differently-named confusion matrix file (`kl_baseline_test_confusion_matrix.png`, from the
new `{name}_{split}_confusion_matrix.png` pattern) alongside the original
`kl_baseline_confusion_matrix.png`, which was left untouched. No information was lost, but
noting this as the one case where a locked result's filename technically got rewritten.
Will not re-run evaluate_classifier.py against the original locked checkpoints again for the
rest of this task to avoid repeating it.
