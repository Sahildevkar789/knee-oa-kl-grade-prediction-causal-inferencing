"""Evaluates the locked Hybrid 2 checkpoint on test. Recomputes the training-set BMI median
and tabular mean/std the same deterministic way train_hybrid2.py does (these don't depend on
seed, only on the train split), then applies them unchanged to the test set.

Example:
    python scripts/evaluate_hybrid2.py --ckpt-name kl_hybrid2
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)

from kl_common import get_transforms, KLDatasetFusion, build_hybrid2, coral_probs_to_grade, GRADES
from train_hybrid2 import load_tabular_split, TABULAR_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-name", default="kl_hybrid2")
    ap.add_argument("--clinical-csv", default="data/clinical_features.csv")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()

    df = pd.read_csv(args.clinical_csv, dtype={"ID": str})
    train_p, train_y, train_tab, bmi_median, _ = load_tabular_split(df, "train")
    eval_p, eval_y, eval_tab, _, n_missing = load_tabular_split(df, args.split, impute_median=bmi_median)
    tab_mean, tab_std = train_tab.mean(axis=0), train_tab.std(axis=0)
    tab_std[tab_std == 0] = 1.0
    eval_tab_std = (eval_tab - tab_mean) / tab_std
    print(f"{args.split}: {len(eval_p)} images, {n_missing} BMI values imputed with training median {bmi_median:.2f}")

    _, eval_tf = get_transforms()
    loader = DataLoader(KLDatasetFusion(eval_p, eval_y, eval_tab_std, eval_tf, use_clahe=True),
                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                        pin_memory=device.type == "cuda")

    model = build_hybrid2(args.ckpt_dir, device, tabular_dim=len(TABULAR_COLS), init_from=None)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{args.ckpt_name}_best.pt"), map_location=device))
    model.eval()

    preds, labels, probs_list = [], [], []
    with torch.no_grad():
        for imgs, tab, y in loader:
            imgs, tab = imgs.to(device), tab.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs, tab)
                probs = torch.sigmoid(logits.float())
            preds.append(coral_probs_to_grade(probs).cpu())
            labels.append(y)
            probs_list.append(probs.cpu())
    tp = torch.cat(preds).numpy()
    ty = torch.cat(labels).numpy()
    tprobs = torch.cat(probs_list).numpy()

    acc = accuracy_score(ty, tp)
    qwk = cohen_kappa_score(ty, tp, weights="quadratic")
    macro_f1 = f1_score(ty, tp, average="macro")
    mae = mean_absolute_error(ty, tp)
    report = classification_report(ty, tp, target_names=GRADES, digits=4, output_dict=True, zero_division=0)

    print(f"=== {args.ckpt_name} on {len(ty)} {args.split} images ===")
    print(f"Accuracy: {acc:.4f} | QWK: {qwk:.4f} | Macro F1: {macro_f1:.4f} | MAE: {mae:.4f}\n")
    print(classification_report(ty, tp, target_names=GRADES, digits=4, zero_division=0))
    pd.DataFrame(report).T.round(4).to_csv(os.path.join(args.results_dir, f"{args.ckpt_name}_{args.split}_report.csv"))

    pred_df = pd.DataFrame({"image_path": eval_p, "true_grade": ty, "pred_grade": tp,
                            **{f"prob_thresh{g}gt": tprobs[:, g] for g in range(tprobs.shape[1])}}).round(4)
    pred_path = os.path.join(args.results_dir, f"{args.ckpt_name}_{args.split}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Per-image predictions saved to {pred_path}")

    cm = confusion_matrix(ty, tp, labels=range(5))
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
    ax.set_title(f"{args.ckpt_name}: {args.split} confusion matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, f"{args.ckpt_name}_{args.split}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    if args.split == "test":
        summary_path = os.path.join(args.results_dir, "kl_comparison.csv")
        row = pd.DataFrame([{"Run": args.ckpt_name, "Accuracy": acc, "QWK": qwk, "Macro F1": macro_f1, "MAE": mae,
                             "Recall KL3": report["KL3"]["recall"], "Recall KL4": report["KL4"]["recall"]}]).round(4)
        if os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            row = pd.concat([old[old["Run"] != args.ckpt_name], row], ignore_index=True)
        row.to_csv(summary_path, index=False)
        print("\nComparison so far:")
        print(row.to_string(index=False))


if __name__ == "__main__":
    main()
