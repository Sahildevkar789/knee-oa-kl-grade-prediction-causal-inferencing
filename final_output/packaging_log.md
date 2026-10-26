# Final output packaging

Device: cuda

Saved `T1_segmentation` -> `final_output\tables\T1_segmentation.csv`, `final_output\tables\T1_segmentation.png` (4 rows).

Saved `T2_backbones` -> `final_output\tables\T2_backbones.csv`, `final_output\tables\T2_backbones.png` (4 rows).

T2 note: ResNet-50 and DenseNet-121 ran a single seed each (no std/summary row exists) - their single-run numbers are used as-is, consistent with backbone_comparison.csv.

Saved `T3_ablation` -> `final_output\tables\T3_ablation.csv`, `final_output\tables\T3_ablation.png` (6 rows).

Saved `T4_main_results` -> `final_output\tables\T4_main_results.csv`, `final_output\tables\T4_main_results.png` (8 rows).

T4 note: Hybrid 2 has a real test-set row in kl_comparison.csv (not a validation-only result) - not labeled 'val only', per what the file actually shows.

Saved `T5_per_grade` -> `final_output\tables\T5_per_grade.csv`, `final_output\tables\T5_per_grade.png` (5 rows).

Saved `T6_literature` -> `final_output\tables\T6_literature.csv`, `final_output\tables\T6_literature.png` (5 rows).

T6 note: no additional literature citations ('and others already cited in our results') were found anywhere in this project's existing results files - grepped for the named authors/percentages project-wide, no matches outside an unrelated Colab notebook line. Only the two papers the brief named directly are included.

Saved `T7_causal` -> `final_output\tables\T7_causal.csv`, `final_output\tables\T7_causal.png` (11 rows).

T7 note: 'Knee injury' is labeled an association throughout (self-reported history, uncertain timing relative to KL grade) - never called a causal effect in the write-up.

Saved `T8_xai` -> `final_output\tables\T8_xai.csv`, `final_output\tables\T8_xai.png` (10 rows).

Saved confusion_matrices_side_by_side.png.

Saved accuracy_qwk_bar_chart.png (KneeNeXt/Ensemble QWK+Acc from their summary.md files).

Saved obesity_effect.png.

Copied figure `outputs/extended\xai\deletion_curves.png` -> `final_output\figures\deletion_curves.png`.

Copied figure `outputs/extended\xai\reliability_diagram.png` -> `final_output\figures\reliability_diagram.png`.

Copied figure `outputs/extended\causal\bmi_dose_response.png` -> `final_output\figures\bmi_dose_response.png`.

Saved per-grade sample figure for KL0 -> `final_output\figures\sample_KL0.png`.

Saved per-grade sample figure for KL1 -> `final_output\figures\sample_KL1.png`.

Saved per-grade sample figure for KL2 -> `final_output\figures\sample_KL2.png`.

Saved per-grade sample figure for KL3 -> `final_output\figures\sample_KL3.png`.

Saved per-grade sample figure for KL4 -> `final_output\figures\sample_KL4.png`.

# Final output packaging finished.

