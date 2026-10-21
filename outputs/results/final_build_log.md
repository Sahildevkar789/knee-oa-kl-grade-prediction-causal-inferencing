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

