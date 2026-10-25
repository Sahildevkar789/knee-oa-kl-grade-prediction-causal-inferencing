"""Evaluates the winning KneeNeXt configuration (selected on validation only by
train_kneenext.py) on the test set, exactly once. Saves accuracy/QWK/macro F1/MAE, a
per-grade classification report, a confusion matrix image, and per-image predictions - same
format as evaluate_classifier.py's, so it can be fed straight into bootstrap_compare.py (which
is invoked here, unmodified, via subprocess, against kl_final's and the EfficientNet-B0
baseline's existing test predictions - read-only). Also generates Grad-CAM (both views) for
15 test images (3/grade, >=5 wrong).

    python scripts/evaluate_kneenext.py

All outputs go to outputs/kneenext/results/ only; reads outputs/checkpoints/*_test_predictions.csv
(kl_final, baseline) and outputs/checkpoints/kl_final_best.pt read-only.
"""
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)

from kl_common import list_split, GRADES
from kneenext import KneeNextDataset, build_kneenext, run_eval_kneenext, grad_cam_view, prepare_views, IMG_SIZE

KL_RESULTS_DIR = "outputs/results"            # read-only: kl_final/baseline test predictions
OUT_CKPT_DIR = "outputs/kneenext/checkpoints"
OUT_RESULTS_DIR = "outputs/kneenext/results"
OUT_XAI_DIR = os.path.join(OUT_RESULTS_DIR, "xai")


def load_best_model(device):
    with open(os.path.join(OUT_CKPT_DIR, "kneenext_best_config.json")) as f:
        info = json.load(f)
    model = build_kneenext(info["config"], "outputs/checkpoints", device, init_from=None)
    model.load_state_dict(torch.load(os.path.join(OUT_CKPT_DIR, "kneenext_best.pt"), map_location=device))
    model.eval()
    return model, info


def select_images(preds, n_per_grade=3):
    """Same selection rule as run_xai.py's: 1 wrong + 2 correct per grade where available, so
    15 images cover all 5 grades and include >=5 wrong predictions overall."""
    chosen = []
    for g in range(5):
        sub = preds[preds.true_grade == g]
        wrong = sub[sub.pred_grade != g]
        correct = sub[sub.pred_grade == g]
        picks = []
        if len(wrong) > 0:
            picks.append(wrong.iloc[0])
        picks += [correct.iloc[i] for i in range(min(2, len(correct)))]
        while len(picks) < n_per_grade and len(sub) > len(picks):
            extra = sub[~sub.image_path.isin([p.image_path for p in picks])].iloc[0]
            picks.append(extra)
        chosen += picks[:n_per_grade]
    return pd.DataFrame(chosen)


def load_views(path):
    """Loads an image from disk and builds its global+zoom views via kneenext.prepare_views."""
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    global_img, zoom_img, g_t, z_t, g_prior, z_prior, _ = prepare_views(img)
    return global_img, zoom_img, g_t, z_t, g_prior, z_prior


def run_xai(model, preds_df, device):
    os.makedirs(OUT_XAI_DIR, exist_ok=True)
    chosen = select_images(preds_df)
    feature_module = model.features[model.STAGE4_IDX - 1]
    rows = []
    for _, row in chosen.iterrows():
        path = row.image_path
        global_img, zoom_img, g_t, z_t, g_prior, z_prior = load_views(path)
        g = g_t.unsqueeze(0).to(device)
        z = z_t.unsqueeze(0).to(device) if model.use_zoom else None
        gp = torch.from_numpy(g_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
        zp = torch.from_numpy(z_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
        cam, probs = grad_cam_view(model, g, z if z is not None else g, gp, zp, feature_module)

        stem = os.path.splitext(os.path.basename(path))[0]
        grade = int(row.true_grade)
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cam_up = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        axes[0].imshow(global_img); axes[0].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[0].set_title("Global view + Grad-CAM"); axes[0].axis("off")
        axes[1].imshow(zoom_img)
        axes[1].set_title("Zoom view (joint-line crop)"); axes[1].axis("off")
        fig.suptitle(f"KL{grade} (true) / KL{row.pred_grade} (pred) - {stem}")
        plt.tight_layout()
        out_path = os.path.join(OUT_XAI_DIR, f"KL{grade}_{stem}.png")
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        rows.append({"image_path": path, "true_grade": grade, "pred_grade": int(row.pred_grade), "xai_figure": out_path})
    pd.DataFrame(rows).to_csv(os.path.join(OUT_XAI_DIR, "kneenext_xai_index.csv"), index=False)
    print(f"Grad-CAM figures for {len(rows)} test images saved to {OUT_XAI_DIR}/")


def main():
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model, info = load_best_model(device)
    print(f"Evaluating KneeNeXt config '{info['config']}' (seed {info['seed']}, "
         f"val QWK {info['val_qwk']:.4f}) on the test set.")

    test_p, test_y = list_split("dataset", "test")
    test_loader = DataLoader(KneeNextDataset(test_p, test_y, train=False), batch_size=32,
                             shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    pred, true, probs = run_eval_kneenext(model, test_loader, device, use_amp)

    acc = accuracy_score(true, pred)
    qwk = cohen_kappa_score(true, pred, weights="quadratic")
    macro_f1 = f1_score(true, pred, average="macro")
    mae = mean_absolute_error(true, pred)
    report = classification_report(true, pred, target_names=GRADES, digits=4, output_dict=True, zero_division=0)

    print(f"\n=== KneeNeXt ({info['config']}) on {len(true)} test images ===")
    print(f"Accuracy: {acc:.4f} | QWK: {qwk:.4f} | Macro F1: {macro_f1:.4f} | MAE: {mae:.4f}\n")
    print(classification_report(true, pred, target_names=GRADES, digits=4, zero_division=0))
    pd.DataFrame(report).T.round(4).to_csv(os.path.join(OUT_RESULTS_DIR, "kneenext_test_report.csv"))

    prob_cols = {f"prob_thresh{g}gt": probs[:, g] for g in range(probs.shape[1])}
    preds_df = pd.DataFrame({"image_path": test_p, "true_grade": true, "pred_grade": pred, **prob_cols}).round(4)
    preds_path = os.path.join(OUT_RESULTS_DIR, "kneenext_test_predictions.csv")
    preds_df.to_csv(preds_path, index=False)
    print(f"Per-image predictions saved to {preds_path}")

    cm = confusion_matrix(true, pred, labels=range(5))
    pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{cm[r, c]}\n({pct[r, c]:.1f}%)", ha="center", va="center",
                    color="white" if pct[r, c] > 50 else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(GRADES)
    ax.set_yticks(range(5)); ax.set_yticklabels(GRADES)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("Actual grade")
    ax.set_title(f"KneeNeXt ({info['config']}): test confusion matrix")
    plt.tight_layout()
    cm_path = os.path.join(OUT_RESULTS_DIR, "kneenext_test_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved to {cm_path}")

    # Paired bootstrap vs kl_final and vs the EfficientNet-B0 baseline, via the existing
    # bootstrap_compare.py script (unmodified) - their test predictions are read-only inputs.
    for other_name, other_file in [("kl_final", "kl_final_test_predictions.csv"),
                                   ("baseline", "kl_baseline_test_predictions.csv")]:
        other_path = os.path.join(KL_RESULTS_DIR, other_file)
        out_csv = os.path.join(OUT_RESULTS_DIR, f"significance_kneenext_vs_{other_name}.csv")
        if not os.path.exists(other_path):
            print(f"WARNING: {other_path} not found - skipping bootstrap vs {other_name}.")
            continue
        subprocess.run([sys.executable, "scripts/bootstrap_compare.py", preds_path, other_path,
                       "kneenext", other_name, out_csv], check=True)

    run_xai(model, preds_df, device)

    print(f"\nDone. KneeNeXt ({info['config']}): accuracy {acc:.4f}, QWK {qwk:.4f}, "
         f"macro F1 {macro_f1:.4f}, MAE {mae:.4f}.")


if __name__ == "__main__":
    main()
