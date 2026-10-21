"""Train Hybrid 2: kl_final's image branch fused with a small MLP over tabular features
(age, BMI - sex was requested but no sex/gender column exists anywhere in
unified_oai_dataset.csv, checked exhaustively across all 1,193 columns; logged and
proceeding with age/BMI only, see final_build_log.md Phase 3). JSW features are not
included since Phase 2 found JSW NOT validated (rho=-0.179, threshold was -0.3).

Missing BMI values are imputed with the TRAINING-set median only (count logged). Tabular
features are standardized using TRAINING-set mean/std only, applied unchanged to val/test.

Examples:
    python scripts/train_hybrid2.py --seed 42
    python scripts/train_hybrid2.py --seed 7
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score

from kl_common import (get_transforms, KLDatasetFusion, build_hybrid2,
                       coral_pos_weights, coral_loss, coral_probs_to_grade, apply_clahe, IMG_SIZE)

TABULAR_COLS = ["V00AGE", "P01BMI"]   # sex unavailable in this dataset - see module docstring


def load_tabular_split(df, split, impute_median=None):
    """Returns (paths, labels, tabular_array, impute_median_used). If impute_median is None,
    computes it from this split (must only be called with the TRAIN split) - callers passing
    val/test must supply the training median explicitly."""
    sub = df[df.SPLIT == split].reset_index(drop=True)
    exists = sub["IMAGE_PATH"].apply(os.path.exists)
    if (~exists).any():
        print(f"  ({split}: {(~exists).sum()} listed image(s) missing on disk, skipping - "
             f"known pre-existing gap, see CLAUDE.md)")
    sub = sub[exists].reset_index(drop=True)
    tab = sub[TABULAR_COLS].copy()
    n_missing = tab["P01BMI"].isna().sum()
    if impute_median is None:
        impute_median = tab["P01BMI"].median()
    tab["P01BMI"] = tab["P01BMI"].fillna(impute_median)
    return sub["IMAGE_PATH"].tolist(), sub["KL_GRADE"].tolist(), tab.values.astype(np.float32), impute_median, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinical-csv", default="data/clinical_features.csv")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--ckpt-name", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    name = args.ckpt_name or ("kl_hybrid2" if args.seed == 42 else f"kl_hybrid2_seed{args.seed}")
    print(f"Device: {device} | Run: {name} | Seed: {args.seed} | Tabular features: {TABULAR_COLS}")

    df = pd.read_csv(args.clinical_csv, dtype={"ID": str})
    train_p, train_y, train_tab, bmi_median, n_missing_train = load_tabular_split(df, "train")
    val_p, val_y, val_tab, _, n_missing_val = load_tabular_split(df, "val", impute_median=bmi_median)
    print(f"Train: {len(train_p)} | Val: {len(val_p)}")
    print(f"BMI imputation (training median = {bmi_median:.2f}): {n_missing_train} missing in train, "
         f"{n_missing_val} missing in val (imputed with the SAME training median)")

    # Standardize using TRAINING statistics only
    tab_mean, tab_std = train_tab.mean(axis=0), train_tab.std(axis=0)
    tab_std[tab_std == 0] = 1.0
    train_tab_std = (train_tab - tab_mean) / tab_std
    val_tab_std = (val_tab - tab_mean) / tab_std
    print(f"Tabular standardization - mean: {dict(zip(TABULAR_COLS, tab_mean.round(2)))}, "
         f"std: {dict(zip(TABULAR_COLS, tab_std.round(2)))}")

    train_tf, eval_tf = get_transforms()
    pin = device.type == "cuda"
    train_loader = DataLoader(KLDatasetFusion(train_p, train_y, train_tab_std, train_tf, use_clahe=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(KLDatasetFusion(val_p, val_y, val_tab_std, eval_tf, use_clahe=True),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin)

    counts = np.bincount(train_y, minlength=5)
    print("Train counts per grade:", counts.tolist())

    model = build_hybrid2(args.ckpt_dir, device, tabular_dim=len(TABULAR_COLS))
    n_params = sum(p.numel() for p in model.parameters())
    pos_w = coral_pos_weights(train_y, device, mode="inverse")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                               total_iters=args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                       milestones=[args.warmup_epochs])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def run_eval_fusion(loader):
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for imgs, tab, y in loader:
                imgs, tab = imgs.to(device), tab.to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(imgs, tab)
                    probs = torch.sigmoid(logits.float())
                preds.append(coral_probs_to_grade(probs).cpu())
                labels.append(y)
        return torch.cat(preds).numpy(), torch.cat(labels).numpy()

    best_qwk, best_acc, best_f1, best_epoch, no_improve, history = -1.0, 0.0, 0.0, 0, 0, []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for imgs, tab, y in train_loader:
            imgs, tab, y = imgs.to(device), tab.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(imgs, tab)
                loss = coral_loss(logits, y, pos_w)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
        scheduler.step()

        vp, vy = run_eval_fusion(val_loader)
        val_acc = float((vp == vy).mean())
        val_qwk = float(cohen_kappa_score(vy, vp, weights="quadratic"))
        val_f1 = float(f1_score(vy, vp, average="macro"))
        train_loss = total_loss / len(train_y)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_acc": val_acc,
                        "val_qwk": val_qwk, "val_macro_f1": val_f1})
        print(f"Epoch {epoch}/{args.epochs} | loss: {train_loss:.4f} | val acc: {val_acc:.4f} "
              f"| val QWK: {val_qwk:.4f} | val macro F1: {val_f1:.4f} | {time.time() - start:.0f}s")

        if val_qwk > best_qwk:
            best_qwk, best_acc, best_f1, best_epoch, no_improve = val_qwk, val_acc, val_f1, epoch, 0
            torch.save(model.state_dict(), os.path.join(args.ckpt_dir, f"{name}_best.pt"))
            print("  -> new best, saved")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"No improvement for {args.patience} epochs, stopping early.")
                break
    train_time_s = time.time() - train_start

    pd.DataFrame(history).round(4).to_csv(os.path.join(args.results_dir, f"{name}_training_log.csv"), index=False)
    print(f"Done. Best val QWK: {best_qwk:.4f} (epoch {best_epoch}) | val acc: {best_acc:.4f} | "
          f"val macro F1: {best_f1:.4f}")
    print(f"Params: {n_params:,} | Training time: {train_time_s/60:.1f} min | Checkpoint: "
          f"{os.path.join(args.ckpt_dir, name + '_best.pt')}")

    summary_path = os.path.join(args.results_dir, "backbone_comparison.csv")
    row = pd.DataFrame([{
        "Run": name, "Backbone": "convnext_tiny_fusion", "MaskSource": "none",
        "BestValQWK": best_qwk, "ValAcc": best_acc, "ValMacroF1": best_f1,
        "Params": n_params, "TrainTimeMin": round(train_time_s / 60, 1),
        "BestEpoch": best_epoch, "BatchSize": args.batch_size, "Seed": args.seed,
        "WarmupEpochs": args.warmup_epochs, "CLAHE": True, "Attention": "cbam", "Loss": "coral",
        "ClassWeights": "inverse", "Head": "fusion_tabular",
    }]).round(4)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        row = pd.concat([old[old["Run"] != name], row], ignore_index=True)
    row.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
