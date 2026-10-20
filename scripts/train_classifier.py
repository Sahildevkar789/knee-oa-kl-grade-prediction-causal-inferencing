"""Train the KL-grade classifier.

Examples (run from the project root):
    python scripts/train_classifier.py --mask-source none
    python scripts/train_classifier.py --mask-source unet
    python scripts/train_classifier.py --mask-source unetpp
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score

from kl_common import (list_split, get_transforms, KLDataset, build_classifier,
                       load_seg_model, add_mask_channel, run_eval, run_name, BACKBONES,
                       compute_class_weights, coral_pos_weights, coral_loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask-source", choices=["none", "unet", "unetpp"], default="none")
    ap.add_argument("--backbone", choices=BACKBONES, default="efficientnet_b0")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--results-dir", default="outputs/results")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup-epochs", type=int, default=0,
                    help="Linear LR warmup for this many epochs before the cosine schedule")
    ap.add_argument("--clahe", action="store_true", help="Apply CLAHE to images before training/eval")
    ap.add_argument("--attention", choices=["none", "cbam"], default="none",
                    help="cbam: CBAM block on convnext_tiny's final feature map (other backbones unsupported)")
    ap.add_argument("--loss", choices=["ce", "coral"], default="ce",
                    help="coral: CORAL ordinal head (4 binary 'grade>k' outputs) instead of plain 5-way softmax")
    ap.add_argument("--class-weights", choices=["inverse", "sqrt", "none"], default="inverse")
    ap.add_argument("--ckpt-name", default=None,
                    help="Override the auto-generated run name (used by the ablation orchestration "
                         "to compose option suffixes consistently)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    mask_source = None if args.mask_source == "none" else args.mask_source
    if args.ckpt_name:
        name = args.ckpt_name
    else:
        name = run_name(mask_source, args.backbone)
        if args.seed != 42:
            name += f"_seed{args.seed}"
        if args.warmup_epochs > 0:
            name += f"_warmup{args.warmup_epochs}"
        if args.clahe:
            name += "_clahe"
        if args.attention != "none":
            name += f"_{args.attention}"
        if args.loss != "ce":
            name += f"_{args.loss}"
        if args.class_weights != "inverse":
            name += f"_cw{args.class_weights}"
    print(f"Device: {device} | Run: {name} | Seed: {args.seed}")
    if device.type == "cpu":
        print("WARNING: no GPU found - training will be very slow. Check your PyTorch CUDA install.")

    # Data
    train_tf, eval_tf = get_transforms()
    train_p, train_y = list_split(args.data_root, "train")
    val_p, val_y = list_split(args.data_root, "val")
    print(f"Train: {len(train_p)} | Val: {len(val_p)}")
    pin = device.type == "cuda"
    train_loader = DataLoader(KLDataset(train_p, train_y, train_tf, use_clahe=args.clahe), batch_size=args.batch_size,
                              shuffle=True, num_workers=args.workers, pin_memory=pin)
    val_loader = DataLoader(KLDataset(val_p, val_y, eval_tf, use_clahe=args.clahe), batch_size=args.batch_size,
                            shuffle=False, num_workers=args.workers, pin_memory=pin)

    # Class weights for the imbalance (rarer grade -> larger weight)
    counts = np.bincount(train_y, minlength=5)
    class_w = compute_class_weights(train_y, args.class_weights, device)
    print("Train counts per grade:", counts.tolist())
    print("Class weights:", [round(w, 2) for w in class_w.tolist()])

    # Models
    seg_model = load_seg_model(mask_source, args.ckpt_dir, device) if mask_source else None
    out_dim = 4 if args.loss == "coral" else 5
    attention = args.attention if args.attention != "none" else None
    model = build_classifier(args.backbone, 4 if mask_source else 3, out_dim=out_dim, attention=attention).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    if args.loss == "coral":
        pos_w = coral_pos_weights(train_y, device, mode=args.class_weights)
        loss_fn = lambda logits, y: coral_loss(logits, y, pos_w)
    else:
        loss_fn = nn.CrossEntropyLoss(weight=class_w)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                                                    total_iters=args.warmup_epochs)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.warmup_epochs)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine],
                                                           milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Training loop
    best_qwk, best_acc, best_f1, best_epoch, no_improve, history = -1.0, 0.0, 0.0, 0, 0, []
    train_start = time.time()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total_loss = 0.0
        for imgs, y in train_loader:
            imgs = add_mask_channel(imgs.to(device), seg_model)
            y = y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = loss_fn(model(imgs), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * y.size(0)
        scheduler.step()

        vp, vy = run_eval(model, val_loader, seg_model, device, use_amp, loss_type=args.loss)
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
    print(f"Done. Best val QWK: {best_qwk:.4f} (epoch {best_epoch}) | "
          f"val acc: {best_acc:.4f} | val macro F1: {best_f1:.4f}")
    print(f"Params: {n_params:,} | Training time: {train_time_s/60:.1f} min | Checkpoint: "
          f"{os.path.join(args.ckpt_dir, name + '_best.pt')}")

    # Cross-backbone comparison table (one row per run, keyed by name - reruns overwrite their own row)
    summary_path = os.path.join(args.results_dir, "backbone_comparison.csv")
    row = pd.DataFrame([{
        "Run": name, "Backbone": args.backbone, "MaskSource": args.mask_source,
        "BestValQWK": best_qwk, "ValAcc": best_acc, "ValMacroF1": best_f1,
        "Params": n_params, "TrainTimeMin": round(train_time_s / 60, 1),
        "BestEpoch": best_epoch, "BatchSize": args.batch_size,
        "Seed": args.seed, "WarmupEpochs": args.warmup_epochs,
        "CLAHE": args.clahe, "Attention": args.attention, "Loss": args.loss,
        "ClassWeights": args.class_weights,
    }]).round(4)
    if os.path.exists(summary_path):
        old = pd.read_csv(summary_path)
        row = pd.concat([old[old["Run"] != name], row], ignore_index=True)
    row.to_csv(summary_path, index=False)


if __name__ == "__main__":
    main()
