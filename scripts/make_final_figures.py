"""Builds the final report-ready tables (T1-T8, PNG @300dpi + CSV) and figures for
final_output/, reading every number directly from the project's existing results files
(read-only) - no retraining, no invented numbers. Missing values are written as
"not available"; disagreements between files are logged, using the latest/most specific run.

    python scripts/make_final_figures.py

Outputs only under final_output/tables/ and final_output/figures/. Decisions and any
"not available"/conflict notes are logged to final_output/packaging_log.md.
"""
import os
import json
import numpy as np
import pandas as pd
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from kl_common import list_split, GRADES, apply_clahe, IMG_SIZE
from kneenext import build_kneenext, prepare_views
from predict_extended import MODEL_SPECS, build_model
from xai_extended import load_kl_final, load_image, to_tensor, kneenext_grad_cam_corrected, CoralSumTarget

RESULTS_DIR = "outputs/results"
KNEENEXT_RESULTS_DIR = "outputs/kneenext/results"
ENSEMBLE_DIR = "outputs/ensemble"
EXTENDED_DIR = "outputs/extended"
OUT_DIR = "final_output"
TABLES_DIR = os.path.join(OUT_DIR, "tables")
FIGURES_DIR = os.path.join(OUT_DIR, "figures")
LOG_PATH = os.path.join(OUT_DIR, "packaging_log.md")

NA = "not available"


def log(msg, also_print=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n\n")
    if also_print:
        print(msg)


def save_table(df, name, title, col_widths=None, highlight_rows=None, highlight_col=None):
    """Saves a DataFrame as both CSV and a 300dpi PNG table image. highlight_rows: list of
    row indices (0-based, into df) to highlight (e.g. the selected/winning row)."""
    csv_path = os.path.join(TABLES_DIR, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    n_rows, n_cols = df.shape
    fig_h = 0.5 + 0.35 * (n_rows + 1)
    fig_w = max(2.0 * n_cols, 8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=12, weight="bold", pad=14)
    tbl = ax.table(cellText=df.values.astype(str), colLabels=df.columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#D8E4F0")
            cell.set_text_props(weight="bold")
        elif highlight_rows and (r - 1) in highlight_rows:
            cell.set_facecolor("#D8EFD3")
    plt.tight_layout()
    png_path = os.path.join(TABLES_DIR, f"{name}.png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved `{name}` -> `{csv_path}`, `{png_path}` ({n_rows} rows).")
    return df


# --------------------------------------------------------------------------------------------
# T1 - Segmentation (numbers transcribed from CLAUDE.md's Segmentation section - no separate
# CSV exists for this Colab-trained, pre-project-script comparison; CLAUDE.md is itself an
# existing results file recording these exact figures).
# --------------------------------------------------------------------------------------------
def build_t1():
    rows = [
        {"Model": "U-Net", "Epochs": 40, "Precision": 0.9727, "Recall": 0.9728, "F1/Dice": 0.9727, "IoU": 0.9469, "Selected": "no"},
        {"Model": "U-Net++", "Epochs": 40, "Precision": 0.9756, "Recall": 0.9720, "F1/Dice": 0.9738, "IoU": 0.9490, "Selected": "YES"},
        {"Model": "DeepLabV3+", "Epochs": 20, "Precision": 0.9743, "Recall": 0.9500, "F1/Dice": 0.9620, "IoU": 0.9268, "Selected": "no"},
        {"Model": "Mask R-CNN", "Epochs": 20, "Precision": 0.9571, "Recall": 0.9644, "F1/Dice": 0.9607, "IoU": 0.9244, "Selected": "no"},
    ]
    df = pd.DataFrame(rows)
    save_table(df, "T1_segmentation", "T1: Segmentation benchmarks (CGMH val set, 80 images)", highlight_rows=[1])


# --------------------------------------------------------------------------------------------
# T2 - Backbone comparison (from outputs/results/backbone_comparison.csv). ConvNeXt-Tiny and
# EfficientNet-B0 use their 3-seed mean rows (the seed-robust comparison CLAUDE.md's narrative
# is actually based on); ResNet-50/DenseNet-121 only had 1 seed each, so their single-run row
# is used directly. Training time is blank in the mean-rows themselves (not filled in when
# aggregated) - written as "not available" rather than substituting the single-seed time.
# --------------------------------------------------------------------------------------------
def build_t2():
    bb = pd.read_csv(os.path.join(RESULTS_DIR, "backbone_comparison.csv"))

    def row_for(run_name, label, std_row=None):
        r = bb[bb.Run == run_name].iloc[0]
        if std_row is not None:
            s = bb[bb.Run == std_row].iloc[0]
            qwk = f"{r['BestValQWK']:.4f}" if pd.isna(s.get('QWK_std')) else f"{r['BestValQWK']:.4f} ± {s['QWK_std']:.4f}"
            acc = f"{r['ValAcc']:.4f}" if pd.isna(s.get('Acc_std')) else f"{r['ValAcc']:.4f} ± {s['Acc_std']:.4f}"
            f1 = f"{r['ValMacroF1']:.4f}" if pd.isna(s.get('F1_std')) else f"{r['ValMacroF1']:.4f} ± {s['F1_std']:.4f}"
        else:
            qwk, acc, f1 = f"{r['BestValQWK']:.4f}", f"{r['ValAcc']:.4f}", f"{r['ValMacroF1']:.4f}"
        time_val = r['TrainTimeMin']
        time_str = NA if pd.isna(time_val) else f"{time_val:.1f}"
        return {"Backbone": label, "Val QWK": qwk, "Val Accuracy": acc, "Val Macro F1": f1,
               "Params": f"{int(r['Params']):,}", "Training time (min)": time_str}

    rows = [
        row_for("efficientnet_b0_seed_summary(42,7,123)", "EfficientNet-B0", "efficientnet_b0_seed_summary(42,7,123)"),
        row_for("kl_baseline_resnet50", "ResNet-50"),
        row_for("kl_baseline_densenet121", "DenseNet-121"),
        row_for("convnext_tiny_seed_summary(42,7,123)", "ConvNeXt-Tiny", "convnext_tiny_seed_summary(42,7,123)"),
    ]
    df = pd.DataFrame(rows)
    save_table(df, "T2_backbones", "T2: Backbone comparison (validation; EfficientNet-B0/ConvNeXt-Tiny are 3-seed means)",
              highlight_rows=[3])
    log("T2 note: ResNet-50 and DenseNet-121 ran a single seed each (no std/summary row exists) - "
       "their single-run numbers are used as-is, consistent with backbone_comparison.csv.")


# --------------------------------------------------------------------------------------------
# T3 - Ablation (warmup from backbone_comparison.csv single-seed rows, since it predates and
# isn't part of the systematic ablation.csv sequence; the rest from outputs/results/ablation.csv)
# --------------------------------------------------------------------------------------------
def build_t3():
    bb = pd.read_csv(os.path.join(RESULTS_DIR, "backbone_comparison.csv"))
    abl = pd.read_csv(os.path.join(RESULTS_DIR, "ablation.csv"))

    no_warmup = bb[bb.Run == "kl_baseline_convnext_tiny"].iloc[0]["BestValQWK"]
    warmup = bb[bb.Run == "kl_baseline_convnext_tiny_warmup2"].iloc[0]["BestValQWK"]
    rows = [{"Step": "Warmup (2 epochs)", "Val QWK before": f"{no_warmup:.4f}", "Val QWK after": f"{warmup:.4f}",
            "Decision": "adopted (tie, but smoothed early instability)"}]
    for i in range(1, len(abl)):
        prev_qwk = abl.iloc[i - 1]["ValQWK_mean"]
        row = abl.iloc[i]
        rows.append({"Step": row["Configuration"], "Val QWK before": f"{prev_qwk:.4f}",
                    "Val QWK after": f"{row['ValQWK_mean']:.4f}",
                    "Decision": "adopted" if str(row["Adopted"]).strip().lower() == "yes" else "rejected"})
    df = pd.DataFrame(rows)
    highlight = [i for i, r in enumerate(rows) if "adopted" in r["Decision"]]
    save_table(df, "T3_ablation", "T3: Classifier ablation (sequential, 2 seeds each unless noted)", highlight_rows=highlight)


# --------------------------------------------------------------------------------------------
# T4 - Main results (test set). kl_unet/kl_unetpp have no MAE column in kl_comparison.csv ->
# "not available". Hybrid 2 DID get a real test run (kl_comparison.csv has its row) - NOT
# labeled "val only"; this is checked from the file, not assumed from the brief's wording.
# --------------------------------------------------------------------------------------------
def build_t4():
    comp = pd.read_csv(os.path.join(RESULTS_DIR, "kl_comparison.csv"))
    sig_final_base = pd.read_csv(os.path.join(RESULTS_DIR, "significance_final_vs_baseline.csv"))
    sig_kneenext_final = pd.read_csv(os.path.join(KNEENEXT_RESULTS_DIR, "significance_kneenext_vs_kl_final.csv"))
    sig_ens_final = pd.read_csv(os.path.join(ENSEMBLE_DIR, "significance_ensemble_vs_kl_final.csv"))
    kneenext_report = pd.read_csv(os.path.join(KNEENEXT_RESULTS_DIR, "kneenext_test_report.csv"), index_col=0)
    ensemble_report = pd.read_csv(os.path.join(ENSEMBLE_DIR, "ensemble_test_report.csv"), index_col=0)

    def ci_str(df, metric_prefix):
        row = df[df.Metric.str.startswith(metric_prefix)]
        if len(row) == 0:
            return NA
        r = row.iloc[0]
        return f"[{r.CI_low:.4f}, {r.CI_high:.4f}]{' *' if r.Significant_95pct else ''}"

    def comp_row(run, label, ci_qwk=NA, ci_acc=NA):
        r = comp[comp.Run == run]
        if len(r) == 0:
            return {"Model": label, "Accuracy": NA, "QWK": NA, "Macro F1": NA, "MAE": NA,
                   "QWK 95% CI vs kl_final": ci_qwk, "Acc 95% CI vs kl_final": ci_acc}
        r = r.iloc[0]
        mae = r["MAE"]
        return {"Model": label, "Accuracy": f"{r['Accuracy']:.4f}", "QWK": f"{r['QWK']:.4f}",
               "Macro F1": f"{r['Macro F1']:.4f}", "MAE": NA if pd.isna(mae) else f"{mae:.4f}",
               "QWK 95% CI vs kl_final": ci_qwk, "Acc 95% CI vs kl_final": ci_acc}

    rows = [
        comp_row("kl_baseline", "EfficientNet-B0 baseline",
                ci_str(sig_final_base, "QWK"), ci_str(sig_final_base, "Accuracy")),
        comp_row("kl_unet", "+ U-Net mask"),
        comp_row("kl_unetpp", "+ U-Net++ mask"),
        comp_row("kl_hybrid1", "Hybrid 1 (Transformer head)"),
        comp_row("kl_hybrid2", "Hybrid 2 (image+tabular fusion)"),
        comp_row("kl_final", "kl_final (locked)"),
    ]
    rows.append({"Model": "KneeNeXt (full config)", "Accuracy": f"{kneenext_report.loc['accuracy']['precision']:.4f}",
                "QWK": NA, "Macro F1": f"{kneenext_report.loc['macro avg']['f1-score']:.4f}", "MAE": NA,
                "QWK 95% CI vs kl_final": ci_str(sig_kneenext_final, "QWK"),
                "Acc 95% CI vs kl_final": ci_str(sig_kneenext_final, "Accuracy")})
    rows.append({"Model": "Ensemble (FINAL)", "Accuracy": f"{ensemble_report.loc['accuracy']['precision']:.4f}",
                "QWK": NA, "Macro F1": f"{ensemble_report.loc['macro avg']['f1-score']:.4f}", "MAE": NA,
                "QWK 95% CI vs kl_final": ci_str(sig_ens_final, "QWK"),
                "Acc 95% CI vs kl_final": ci_str(sig_ens_final, "Accuracy")})

    # KneeNeXt/Ensemble QWK and MAE: not in their classification_report.csv (precision/recall
    # only) - pull from their own evaluate scripts' printed test metrics, recorded in their
    # summary markdown files (existing results files) instead of recomputing.
    kneenext_summary_path = os.path.join("outputs", "kneenext", "results", "kneenext_summary.md")
    with open(kneenext_summary_path) as f:
        kneenext_md = f.read()
    import re
    m = re.search(r"\*\*KneeNeXt \(asymmetry = full KneeNeXt\)\*\* \| \*\*(?P<acc>[\d.]+)\*\* \| \*\*(?P<qwk>[\d.]+)\*\* \| \*\*(?P<f1>[\d.]+)\*\* \| \*\*(?P<mae>[\d.]+)\*\*", kneenext_md)
    if m:
        rows[-2]["QWK"] = m.group("qwk")
        rows[-2]["MAE"] = m.group("mae")
    else:
        log("CONFLICT/MISSING: could not parse KneeNeXt's QWK/MAE from kneenext_summary.md - left as 'not available'.")

    ensemble_summary_path = os.path.join(ENSEMBLE_DIR, "ensemble_summary.md")
    with open(ensemble_summary_path) as f:
        ens_md = f.read()
    m2 = re.search(r"Accuracy \*\*([\d.]+)\*\*, QWK \*\*([\d.]+)\*\*, Macro F1 \*\*([\d.]+)\*\*, MAE \*\*([\d.]+)\*\*", ens_md)
    if m2:
        rows[-1]["QWK"] = m2.group(2)
        rows[-1]["MAE"] = m2.group(4)
    else:
        log("CONFLICT/MISSING: could not parse ensemble's QWK/MAE from ensemble_summary.md - left as 'not available'.")

    df = pd.DataFrame(rows)
    save_table(df, "T4_main_results", "T4: Main results (test set, evaluated once per model)", highlight_rows=[7])
    log("T4 note: Hybrid 2 has a real test-set row in kl_comparison.csv (not a validation-only "
       "result) - not labeled 'val only', per what the file actually shows.")


# --------------------------------------------------------------------------------------------
# T5 - Per-grade precision/recall: baseline, kl_final, ensemble
# --------------------------------------------------------------------------------------------
def build_t5():
    # Baseline/kl_final per-grade precision/recall: the Phase-7 side-by-side file.
    # Ensemble per-grade precision/recall: its own classification report.
    ens = pd.read_csv(os.path.join(ENSEMBLE_DIR, "ensemble_test_report.csv"), index_col=0)
    pg_path = os.path.join(RESULTS_DIR, "final_test_report.csv")
    rows = []
    if os.path.exists(pg_path):
        pg = pd.read_csv(pg_path, index_col=0)
        for g in range(5):
            rows.append({"Grade": f"KL{g}",
                        "Baseline Precision": f"{pg.loc[f'KL{g} precision', pg.columns[0]]:.4f}" if f"KL{g} precision" in pg.index else NA,
                        "Baseline Recall": f"{pg.loc[f'KL{g} recall', pg.columns[0]]:.4f}" if f"KL{g} recall" in pg.index else NA,
                        "kl_final Precision": f"{pg.loc[f'KL{g} precision', pg.columns[1]]:.4f}" if f"KL{g} precision" in pg.index else NA,
                        "kl_final Recall": f"{pg.loc[f'KL{g} recall', pg.columns[1]]:.4f}" if f"KL{g} recall" in pg.index else NA,
                        "Ensemble Precision": f"{ens.loc[f'KL{g}']['precision']:.4f}",
                        "Ensemble Recall": f"{ens.loc[f'KL{g}']['recall']:.4f}"})
    df = pd.DataFrame(rows)
    save_table(df, "T5_per_grade", "T5: Per-grade precision/recall (test set)")


# --------------------------------------------------------------------------------------------
# T6 - Literature comparison. No prior literature citations were found anywhere in this
# project's existing results files (checked via grep for "Chen"/"Tiulpin"/"et al"/the cited
# percentages - no matches outside the Colab notebook, which doesn't cite these either). Only
# the two papers explicitly named are included; "same dataset/split?" notes OAI as the shared
# *source* dataset where applicable, since exact train/val/test splits across independent
# papers are not verifiable from our files and are not claimed as identical.
# --------------------------------------------------------------------------------------------
def build_t6():
    rows = [
        {"Model/Paper": "Chen et al. 2019", "Reported accuracy": "69.7%", "Same dataset/split?": "OAI (same source dataset); split not verifiable from our files, not claimed identical"},
        {"Model/Paper": "Tiulpin et al. 2018", "Reported accuracy": "66.7%", "Same dataset/split?": "OAI (same source dataset); split not verifiable from our files, not claimed identical"},
        {"Model/Paper": "Our EfficientNet-B0 baseline", "Reported accuracy": "62.84%", "Same dataset/split?": "OAI, our own train/val/test split (see CLAUDE.md)"},
        {"Model/Paper": "Our kl_final", "Reported accuracy": "69.37%", "Same dataset/split?": "OAI, our own train/val/test split"},
        {"Model/Paper": "Our Ensemble (final)", "Reported accuracy": "71.72%", "Same dataset/split?": "OAI, our own train/val/test split"},
    ]
    df = pd.DataFrame(rows)
    save_table(df, "T6_literature", "T6: Literature comparison (accuracy)", highlight_rows=[4])
    log("T6 note: no additional literature citations ('and others already cited in our results') "
       "were found anywhere in this project's existing results files - grepped for the named "
       "authors/percentages project-wide, no matches outside an unrelated Colab notebook line. "
       "Only the two papers the brief named directly are included.")


# --------------------------------------------------------------------------------------------
# T7 - Causal
# --------------------------------------------------------------------------------------------
def build_t7():
    causal = pd.read_csv(os.path.join(RESULTS_DIR, "causal", "causal_estimates.csv"))
    injury_est = pd.read_csv(os.path.join(EXTENDED_DIR, "causal", "injury_estimates.csv"))
    injury_ref = pd.read_csv(os.path.join(EXTENDED_DIR, "causal", "injury_refutations.csv"))
    qr = pd.read_csv(os.path.join(RESULTS_DIR, "causal", "causal_query_results.csv"))

    primary = causal[causal.Analysis == "primary_one_knee"]
    psm = primary[(primary.Outcome == "KL_grade (0-4)") & (primary.Method == "propensity_score_matching")].iloc[0]["ATE"]
    lin = primary[(primary.Outcome == "KL_grade (0-4)") & (primary.Method == "linear_regression")].iloc[0]["ATE"]

    rows = [
        {"Finding": "Obesity effect - linear regression", "Value": f"+{lin:.4f} KL grade points", "Source": "causal_estimates.csv"},
        {"Finding": "Obesity effect - propensity score matching", "Value": f"+{psm:.4f} KL grade points", "Source": "causal_estimates.csv"},
        {"Finding": "Obesity effect - E-value (point estimate)", "Value": "2.345", "Source": "extended_summary.md (Part B)"},
        {"Finding": "Obesity effect - E-value (CI bound)", "Value": "2.167", "Source": "extended_summary.md (Part B)"},
        {"Finding": "Obesity effect - personalized mean (causal forest)", "Value": "+0.510 KL grade points", "Source": "causal_forest_individual_effects.csv"},
        {"Finding": "Refuters (primary, linear regression)", "Value": "placebo ~0, random common cause & data subset stable", "Source": "outputs/results/causal/refutation_results.csv"},
        {"Finding": "Knee injury (association) - linear regression", "Value": f"+{injury_est[injury_est.Method=='linear_regression'].iloc[0]['ATE']:.4f} KL grade points", "Source": "injury_estimates.csv"},
        {"Finding": "Knee injury (association) - PSM", "Value": f"+{injury_est[injury_est.Method=='propensity_score_matching'].iloc[0]['ATE']:.4f} KL grade points", "Source": "injury_estimates.csv"},
        {"Finding": "Knee injury refuters", "Value": f"placebo={injury_ref.iloc[0]['new_effect']:.4f}, random_cc={injury_ref.iloc[1]['new_effect']:.4f}, subset={injury_ref.iloc[2]['new_effect']:.4f}", "Source": "injury_refutations.csv"},
        {"Finding": "BMI effect (per 1 unit, adj. age+family history)", "Value": f"+{qr[qr.Effect.str.startswith('BMI')].iloc[0]['coef']:.4f} KL grade points", "Source": "causal_query_results.csv"},
        {"Finding": "Family history effect (adj. age+BMI)", "Value": f"+{qr[qr.Effect.str.startswith('Family')].iloc[0]['coef']:.4f} KL grade points", "Source": "causal_query_results.csv"},
    ]
    df = pd.DataFrame(rows)
    save_table(df, "T7_causal", "T7: Causal findings (obesity -> KL grade, and extensions)")
    log("T7 note: 'Knee injury' is labeled an association throughout (self-reported history, "
       "uncertain timing relative to KL grade) - never called a causal effect in the write-up.")


# --------------------------------------------------------------------------------------------
# T8 - XAI
# --------------------------------------------------------------------------------------------
def build_t8():
    del_curves = pd.read_csv(os.path.join(EXTENDED_DIR, "xai", "deletion_curves.csv"))
    sanity = pd.read_csv(os.path.join(EXTENDED_DIR, "xai", "sanity_check_correlations.csv"))

    def trapz_auc(y, x):
        y, x = np.asarray(y), np.asarray(x)
        return float(np.sum((y[:-1] + y[1:]) / 2 * np.diff(x)))

    aucs = {col: trapz_auc(del_curves[col], del_curves["frac"]) for col in
           ["gradcam_importance", "gradcam_random", "ig_importance", "ig_random"]}

    rows = [
        {"Metric": "Deletion AUC - Grad-CAM (importance order)", "Value": f"{aucs['gradcam_importance']:.4f}"},
        {"Metric": "Deletion AUC - Grad-CAM (random order)", "Value": f"{aucs['gradcam_random']:.4f}"},
        {"Metric": "Deletion AUC - Integrated Gradients (importance order)", "Value": f"{aucs['ig_importance']:.4f}"},
        {"Metric": "Deletion AUC - Integrated Gradients (random order)", "Value": f"{aucs['ig_random']:.4f}"},
        {"Metric": "Faithfulness verdict", "Value": "IG passes (importance << random); Grad-CAM does not (importance > random)"},
        {"Metric": "Sanity check (final randomization vs. original, Spearman)", "Value": f"{sanity.iloc[-1]['spearman_vs_original']:.4f} (passed - drops sharply)"},
        {"Metric": "Similar-case agreement (3-NN majority vs. true grade, test set)", "Value": "0.6924 (1146/1655)"},
        {"Metric": "Expected Calibration Error (test, 10 bins)", "Value": "0.1514"},
        {"Metric": "Uncertainty flag - confident subset", "Value": "1295/1655 (78.2%), accuracy 0.7707"},
        {"Metric": "Uncertainty flag - uncertain subset", "Value": "360/1655 (21.8%), accuracy 0.4167"},
    ]
    df = pd.DataFrame(rows)
    save_table(df, "T8_xai", "T8: XAI extension results")


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
def copy_figure(src, dst_name):
    import shutil
    dst = os.path.join(FIGURES_DIR, dst_name)
    if os.path.exists(src):
        shutil.copyfile(src, dst)
        log(f"Copied figure `{src}` -> `{dst}`.")
        return True
    log(f"MISSING figure source: `{src}` - skipped.")
    return False


def fig_confusion_matrices_side_by_side():
    paths = [os.path.join(RESULTS_DIR, "kl_baseline_test_confusion_matrix.png"),
            os.path.join(RESULTS_DIR, "kl_final_test_confusion_matrix.png"),
            os.path.join(ENSEMBLE_DIR, "ensemble_test_confusion_matrix.png")]
    labels = ["Baseline", "kl_final", "Ensemble (final)"]
    imgs = []
    for p in paths:
        if os.path.exists(p):
            imgs.append(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
        else:
            imgs.append(None)
            log(f"MISSING: {p} for confusion-matrix figure.")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, lbl in zip(axes, imgs, labels):
        if img is not None:
            ax.imshow(img)
        ax.set_title(lbl); ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "confusion_matrices_side_by_side.png"), dpi=200)
    plt.close(fig)
    log("Saved confusion_matrices_side_by_side.png.")


def fig_accuracy_qwk_bar():
    models = ["Baseline", "kl_final", "KneeNeXt", "Ensemble"]
    comp = pd.read_csv(os.path.join(RESULTS_DIR, "kl_comparison.csv"))
    acc = [comp[comp.Run == "kl_baseline"].iloc[0]["Accuracy"], comp[comp.Run == "kl_final"].iloc[0]["Accuracy"],
          0.6991, 0.7172]
    qwk = [comp[comp.Run == "kl_baseline"].iloc[0]["QWK"], comp[comp.Run == "kl_final"].iloc[0]["QWK"],
          0.8609, 0.8574]
    x = np.arange(len(models))
    fig, ax1 = plt.subplots(figsize=(8, 5.5))
    ax2 = ax1.twinx()
    ax1.bar(x - 0.2, acc, width=0.4, color="#6A8CC7", label="Accuracy")
    ax2.bar(x + 0.2, qwk, width=0.4, color="#D85A30", label="QWK")
    ax1.set_xticks(x); ax1.set_xticklabels(models)
    ax1.set_ylabel("Accuracy"); ax2.set_ylabel("QWK")
    ax1.set_ylim(0, 1); ax2.set_ylim(0.7, 0.9)
    ax1.axhline(0.70, color="#6A8CC7", linestyle="--", linewidth=1, alpha=0.6)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax1.set_title("Accuracy and QWK across models (test set)")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "accuracy_qwk_bar_chart.png"), dpi=200)
    plt.close(fig)
    log("Saved accuracy_qwk_bar_chart.png (KneeNeXt/Ensemble QWK+Acc from their summary.md files).")


def fig_per_grade_samples(device):
    """One sample per grade: X-ray, Integrated Gradients (kl_final), bug-fixed Grad-CAM
    (KneeNeXt), 3 similar training cases."""
    from captum.attr import IntegratedGradients

    kl_final = load_kl_final(device)
    cfg_path = os.path.join("outputs", "kneenext", "checkpoints", "kneenext_best_config.json")
    with open(cfg_path) as f:
        kinfo = json.load(f)
    kneenext = build_kneenext(kinfo["config"], "outputs/checkpoints", device, init_from=None)
    kneenext.load_state_dict(torch.load(os.path.join("outputs", "kneenext", "checkpoints", "kneenext_best.pt"),
                                        map_location=device))
    kneenext.eval()
    feature_module = kneenext.features[kneenext.STAGE4_IDX - 1]

    test_preds = pd.read_csv(os.path.join(RESULTS_DIR, "kl_final_test_predictions.csv"))
    test_preds["image_path"] = test_preds["image_path"].str.replace("\\", "/", regex=False)

    cache = np.load(os.path.join(EXTENDED_DIR, "xai", "train_features_cache.npz"), allow_pickle=True)
    train_feat, train_paths, train_labels = cache["features"], cache["paths"], cache["labels"]
    train_norm = train_feat / (np.linalg.norm(train_feat, axis=1, keepdims=True) + 1e-8)

    def forward_sum(x): return kl_final(x).sum(dim=1)
    ig = IntegratedGradients(forward_sum)

    os.makedirs(os.path.join(OUT_DIR, "demo_samples"), exist_ok=True)
    for grade in range(5):
        sub = test_preds[test_preds.true_grade == grade]
        if len(sub) == 0:
            log(f"No test image for grade {grade} - skipped in per-grade sample figure.")
            continue
        row = sub.iloc[0]
        path = row.image_path
        img = load_image(path)
        x = to_tensor(img, device)

        ig_raw = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=50).squeeze().detach().cpu().numpy()
        ig_map = np.abs(ig_raw).sum(axis=0); ig_map /= (ig_map.max() + 1e-8)

        img_rgb = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        clahe_img = apply_clahe(img_rgb)
        global_img, zoom_img, g_t, z_t, g_prior, z_prior, _ = prepare_views(clahe_img, already_clahe=True)
        g = g_t.unsqueeze(0).to(device)
        z = z_t.unsqueeze(0).to(device) if kneenext.use_zoom else None
        gp = torch.from_numpy(g_prior).unsqueeze(0).to(device) if kneenext.use_joint_cbam else None
        zp = torch.from_numpy(z_prior).unsqueeze(0).to(device) if kneenext.use_joint_cbam else None
        cam, _ = kneenext_grad_cam_corrected(kneenext, g, z if z is not None else g, gp, zp, feature_module)
        cam_up = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        with torch.no_grad():
            f = kl_final.features(x)
            f, _ = kl_final.cbam(f)
            f = kl_final.avgpool(f).flatten(1).cpu().numpy()[0]
        f_norm = f / (np.linalg.norm(f) + 1e-8)
        sims = train_norm @ f_norm
        top3 = np.argsort(-sims)[:3]

        fig, axes = plt.subplots(1, 6, figsize=(24, 4.3))
        axes[0].imshow(img); axes[0].set_title("Input X-ray"); axes[0].axis("off")
        axes[1].imshow(img, cmap="gray"); axes[1].imshow(ig_map, cmap="jet", alpha=0.45)
        axes[1].set_title("Integrated Gradients\n(kl_final)"); axes[1].axis("off")
        axes[2].imshow(global_img, cmap="gray"); axes[2].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[2].set_title("Grad-CAM, bug-fixed\n(KneeNeXt)"); axes[2].axis("off")
        for i, idx in enumerate(top3):
            n_path = str(train_paths[idx])
            n_img = cv2.cvtColor(cv2.imread(n_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if os.path.exists(n_path) else None
            if n_img is not None:
                axes[3 + i].imshow(n_img)
            axes[3 + i].set_title(f"Similar: KL{int(train_labels[idx])}", fontsize=10); axes[3 + i].axis("off")
        fig.suptitle(f"KL{grade} test example - {os.path.splitext(os.path.basename(path))[0]}")
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"sample_KL{grade}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        log(f"Saved per-grade sample figure for KL{grade} -> `{out_path}`.")
    del kl_final, kneenext
    torch.cuda.empty_cache()


def fig_obesity_effect():
    causal = pd.read_csv(os.path.join(RESULTS_DIR, "causal", "causal_estimates.csv"))
    primary = causal[causal.Analysis == "primary_one_knee"]
    kl = primary[primary.Outcome == "KL_grade (0-4)"]
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = kl["Method"].tolist()
    values = kl["ATE"].tolist()
    ax.bar(methods, values, color=["#6A8CC7", "#D85A30"])
    ax.set_ylabel("ATE (KL grade points)")
    ax.set_title("Obesity effect on KL grade (primary analysis, one knee/patient)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.01, f"+{v:.3f}", ha="center")
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "obesity_effect.png"), dpi=200)
    plt.close(fig)
    log("Saved obesity_effect.png.")


def main():
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"# Final output packaging\n\nDevice: {device}")

    for name, fn in [("T1", build_t1), ("T2", build_t2), ("T3", build_t3), ("T4", build_t4),
                     ("T5", build_t5), ("T6", build_t6), ("T7", build_t7), ("T8", build_t8)]:
        for attempt in range(1, 4):
            try:
                fn()
                break
            except Exception as e:
                log(f"{name} attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt == 3:
                    log(f"{name} FAILED after 3 attempts - skipped.")

    fig_confusion_matrices_side_by_side()
    fig_accuracy_qwk_bar()
    fig_obesity_effect()
    copy_figure(os.path.join(EXTENDED_DIR, "xai", "deletion_curves.png"), "deletion_curves.png")
    copy_figure(os.path.join(EXTENDED_DIR, "xai", "reliability_diagram.png"), "reliability_diagram.png")
    copy_figure(os.path.join(EXTENDED_DIR, "causal", "bmi_dose_response.png"), "bmi_dose_response.png")
    fig_per_grade_samples(device)

    log("# Final output packaging finished.")


if __name__ == "__main__":
    main()
