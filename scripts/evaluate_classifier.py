"""Evaluate a trained KL classifier on the validation or test set.

Example:
    python scripts/evaluate_classifier.py --mask-source none
    python scripts/evaluate_classifier.py --ckpt-name kl_baseline_convnext_tiny_warmup2 --backbone convnext_tiny --split val
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # save figures to file, no pop-up window
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, classification_report,
                             confusion_matrix, f1_score, mean_absolute_error)

from kl_common import (list_split, get_transforms, KLDataset, build_classifier,
                       load_seg_model, run_eval_probs, GRADES, BACKBONES, build_hybrid1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-source", choices=["none", "unet", "unetpp"], default="none")
    ap.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    ap.add_argument("--attention", choices=["none", "cbam"], default="none")
    ap.add_argument("--loss", choices=["ce", "coral"], default="ce")
    ap.add_argument("--head", choices=["mlp", "transformer"], default="mlp")
    ap.add_argument("--clahe", action="store_true", help="Must match how the checkpoint was trained")
    ap.add_argument("--tta", action="store_true", help="Average predictions of the image and its horizontal flip")
    ap.add_argument("--split", choices=["val", "test"], default="test")
    ap.add_argument("--ckpt-name", default=None,
                    help="Exact checkpoint stem to load (overrides the mask-source-based default naming)")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    mask_source = None if args.mask_source == "none" else args.mask_source
    # the original, locked efficientnet_b0 runs use this naming; everything else passes --ckpt-name explicitly
    name = args.ckpt_name or (f"kl_{mask_source}" if mask_source else "kl_baseline")

    _, eval_tf = get_transforms()
    eval_p, eval_y = list_split(args.data_root, args.split)
    eval_loader = DataLoader(KLDataset(eval_p, eval_y, eval_tf, use_clahe=args.clahe), batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")

    seg_model = load_seg_model(mask_source, args.ckpt_dir, device) if mask_source else None
    out_dim = 4 if args.loss == "coral" else 5
    attention = args.attention if args.attention != "none" else None
    if args.head == "transformer":
        model = build_hybrid1(args.ckpt_dir, device, init_from=None)   # architecture only, weights loaded below
    else:
        model = build_classifier(args.backbone, 4 if mask_source else 3, pretrained=False,
                                 out_dim=out_dim, attention=attention).to(device)
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f"{name}_best.pt"), map_location=device))

    eval_name = name + ("_tta" if args.tta else "")
    tp, ty, tprobs = run_eval_probs(model, eval_loader, seg_model, device, use_amp,
                                    loss_type=args.loss, tta=args.tta)
    acc = accuracy_score(ty, tp)
    qwk = cohen_kappa_score(ty, tp, weights="quadratic")
    macro_f1 = f1_score(ty, tp, average="macro")
    mae = mean_absolute_error(ty, tp)
    report = classification_report(ty, tp, target_names=GRADES, digits=4, output_dict=True, zero_division=0)

    print(f"=== {eval_name} on {len(ty)} {args.split} images ===")
    print(f"Accuracy: {acc:.4f} | QWK: {qwk:.4f} | Macro F1: {macro_f1:.4f} | MAE: {mae:.4f}\n")
    print(classification_report(ty, tp, target_names=GRADES, digits=4, zero_division=0))
    pd.DataFrame(report).T.round(4).to_csv(os.path.join(args.results_dir, f"{eval_name}_{args.split}_report.csv"))

    # Per-image predictions (eval_loader uses shuffle=False, so tp/ty/tprobs line up with eval_p in order)
    prob_cols = {f"prob_thresh{g}gt": tprobs[:, g] for g in range(tprobs.shape[1])} if args.loss == "coral" \
        else {f"prob_KL{g}": tprobs[:, g] for g in range(5)}
    pred_df = pd.DataFrame({"image_path": eval_p, "true_grade": ty, "pred_grade": tp, **prob_cols}).round(4)
    pred_path = os.path.join(args.results_dir, f"{eval_name}_{args.split}_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"Per-image predictions saved to {pred_path}")

    # 5x5 confusion matrix
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
    ax.set_title(f"{eval_name}: {args.split} confusion matrix")
    plt.tight_layout()
    cm_path = os.path.join(args.results_dir, f"{eval_name}_{args.split}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    print(f"Confusion matrix saved to {cm_path}")

    # Comparison table across runs (one row per run) - only for test-set runs, to keep the
    # original kl_comparison.csv meaning intact (never touched for val-set ablation checks)
    if args.split == "test":
        summary_path = os.path.join(args.results_dir, "kl_comparison.csv")
        row = pd.DataFrame([{"Run": eval_name, "Accuracy": acc, "QWK": qwk, "Macro F1": macro_f1, "MAE": mae,
                             "Recall KL3": report["KL3"]["recall"], "Recall KL4": report["KL4"]["recall"]}]).round(4)
        if os.path.exists(summary_path):
            old = pd.read_csv(summary_path)
            row = pd.concat([old[old["Run"] != eval_name], row], ignore_index=True)
        row.to_csv(summary_path, index=False)
        print("\nComparison so far:")
        print(row.to_string(index=False))


if __name__ == "__main__":
    main()
