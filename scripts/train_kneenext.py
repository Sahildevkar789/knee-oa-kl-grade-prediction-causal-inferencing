"""Trains KneeNeXt through its cumulative 5-step ablation (control -> +joint_zoom ->
+multiscale -> +joint_cbam -> +asymmetry), 2 seeds per step, deciding which modules to keep
by mean validation QWK - same selection discipline as kl_final's ablation (val only; the test
set is never touched here). Default mode runs the full sweep; --mode single reruns one
config/seed (for a targeted retry without repeating the whole sweep).

    python scripts/train_kneenext.py                       # full ablation sweep (both seeds)
    python scripts/train_kneenext.py --smoke-test           # tiny subset, 1+1 epochs, pipeline check only
    python scripts/train_kneenext.py --mode single --config joint_cbam --seed 7

All new checkpoints/results go under outputs/kneenext/ only - never touches
outputs/checkpoints/ or outputs/results/ except to READ kl_final_best.pt.
"""
import os
import time
import argparse
import traceback
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score, mean_absolute_error

from kl_common import list_split, coral_pos_weights, coral_loss
from kneenext import KneeNextDataset, build_kneenext, run_eval_kneenext, CONFIGS, detect_joint_row, crop_zoom_view

KL_CKPT_DIR = "outputs/checkpoints"          # read-only: kl_final_best.pt lives here
OUT_CKPT_DIR = "outputs/kneenext/checkpoints"
OUT_RESULTS_DIR = "outputs/kneenext/results"
LOG_PATH = os.path.join(OUT_RESULTS_DIR, "kneenext_log.md")
ABLATION_CSV = os.path.join(OUT_RESULTS_DIR, "kneenext_ablation.csv")


def log(msg, also_print=True):
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n\n")
    if also_print:
        print(msg)


def save_joint_zoom_check_figure(val_paths, val_labels, out_path, n_per_grade=4):
    """Visual check: for n_per_grade images per KL grade, draws the detected joint row and
    crop band on the full (CLAHE, no augmentation) image next to the resulting zoom crop.
    Deterministic (no augmentation) - this is a sanity check of the detector, not of training."""
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from kl_common import apply_clahe

    by_grade = {g: [] for g in range(5)}
    for p, y in zip(val_paths, val_labels):
        if len(by_grade[y]) < n_per_grade:
            by_grade[y].append(p)

    rows = sum(len(v) for v in by_grade.values())
    fig, axes = plt.subplots(rows, 2, figsize=(6, 3 * rows))
    r = 0
    for grade in range(5):
        for p in by_grade[grade]:
            img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            img = apply_clahe(img)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            joint_row, fallback = detect_joint_row(gray)
            zoom = crop_zoom_view(img, joint_row)

            ax = axes[r, 0]
            ax.imshow(img)
            ax.axhline(joint_row, color="yellow", linestyle="--", linewidth=1)
            h = img.shape[0]
            half = int(0.25 * h)
            ax.axhspan(max(0, joint_row - half), min(h, joint_row + half), color="lime", alpha=0.15)
            ax.set_title(f"KL{grade} - full ({'FALLBACK' if fallback else 'detected'})", fontsize=8)
            ax.axis("off")
            axes[r, 1].imshow(zoom); axes[r, 1].set_title("zoom crop", fontsize=8); axes[r, 1].axis("off")
            r += 1
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def compute_fallback_rate(paths, sample_n=None):
    """Full (or sampled) pass over a split to report the joint-detector's real failure rate -
    deterministic, no augmentation, CPU-only (cheap: ~6ms/image on this machine)."""
    import cv2
    from kl_common import apply_clahe
    if sample_n is not None:
        paths = paths[:sample_n]
    n_fail = 0
    for p in paths:
        img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(apply_clahe(img), cv2.COLOR_RGB2GRAY)
        _, fallback = detect_joint_row(gray)
        n_fail += int(fallback)
    return n_fail / len(paths)


def set_param_groups(model, backbone_lr, other_lr):
    backbone_params = list(model.features.parameters())
    backbone_ids = {id(p) for p in backbone_params}
    other_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    return [{"params": backbone_params, "lr": backbone_lr}, {"params": other_params, "lr": other_lr}]


def evaluate_metrics(model, loader, device, use_amp):
    """cohen_kappa_score returns nan if predictions collapse to a single class with no
    overlap against the true labels (possible early in training, especially for a tiny
    smoke-test subset) - nan never beats the -1.0 "nothing saved yet" sentinel in the
    training loop below, which would silently leave NO checkpoint on disk for that run.
    Mapped to 0.0 (a real, comparable QWK value) rather than left to propagate."""
    pred, true, _ = run_eval_kneenext(model, loader, device, use_amp)
    acc = float((pred == true).mean())
    qwk = cohen_kappa_score(true, pred, weights="quadratic")
    qwk = 0.0 if qwk is None or np.isnan(qwk) else float(qwk)
    f1 = float(f1_score(true, pred, average="macro", zero_division=0))
    mae = float(mean_absolute_error(true, pred))
    recall = recall_score(true, pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    return acc, qwk, f1, mae, recall


def _run_single(config, seed, batch_size, train_loader, val_loader, device, use_amp,
                train_y, max_frozen_epochs, max_epochs, patience, run_name):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_kneenext(config, KL_CKPT_DIR, device, init_from="kl_final_best.pt")
    n_params = sum(p.numel() for p in model.parameters())
    pos_w = coral_pos_weights(train_y, device, mode="inverse")
    loss_fn = lambda logits, y: coral_loss(logits, y, pos_w)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    t_start = time.time()

    # Checkpoint is written to disk the first time ANY epoch (in either stage) improves on the
    # running best - not just kept in memory - so a real file always exists after this run,
    # even if stage 2 never strictly beats stage 1's peak (a tie or a regression right after
    # unfreezing is plausible and must not silently leave nothing on disk).
    os.makedirs(OUT_CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_CKPT_DIR, f"{run_name}_best.pt")
    best_qwk, best_epoch, best_metrics = -1.0, 0, None

    # --- Stage 1: freeze the ConvNeXt backbone, train the new modules only ---
    for p in model.features.parameters():
        p.requires_grad = False
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, max_frozen_epochs + 1):
        model.train()
        total_loss = 0.0
        for g, z, gp, zp, y in train_loader:
            g, z, gp, zp, y = g.to(device), z.to(device), gp.to(device), zp.to(device), y.to(device)
            opt1.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(g, z if model.use_zoom else None,
                               gp if model.use_joint_cbam else None, zp if model.use_joint_cbam else None)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt1)
            scaler.update()
            total_loss += loss.item() * y.size(0)
        acc, qwk, f1, mae, recall = evaluate_metrics(model, val_loader, device, use_amp)
        history.append({"stage": 1, "epoch": epoch, "train_loss": total_loss / len(train_y),
                        "val_acc": acc, "val_qwk": qwk, "val_f1": f1, "val_mae": mae})
        print(f"  [{run_name}] stage1 epoch {epoch}/{max_frozen_epochs} | val QWK {qwk:.4f}")
        if qwk > best_qwk:
            best_qwk, best_epoch = qwk, epoch
            best_metrics = {"val_acc": acc, "val_qwk": qwk, "val_f1": f1, "val_mae": mae,
                            "val_recall": recall.tolist()}
            torch.save(model.state_dict(), ckpt_path)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # --- Stage 2: unfreeze everything, differential LR, warmup+cosine, early stop on val QWK ---
    for p in model.parameters():
        p.requires_grad = True
    groups = set_param_groups(model, backbone_lr=5e-5, other_lr=3e-4)
    opt2 = torch.optim.AdamW(groups, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt2, start_factor=0.1, end_factor=1.0, total_iters=1)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(max_epochs - 1, 1))
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt2, schedulers=[warmup, cosine], milestones=[1])
    no_improve = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        for g, z, gp, zp, y in train_loader:
            g, z, gp, zp, y = g.to(device), z.to(device), gp.to(device), zp.to(device), y.to(device)
            opt2.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(g, z if model.use_zoom else None,
                               gp if model.use_joint_cbam else None, zp if model.use_joint_cbam else None)
                loss = loss_fn(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt2)
            scaler.update()
            total_loss += loss.item() * y.size(0)
        scheduler.step()
        acc, qwk, f1, mae, recall = evaluate_metrics(model, val_loader, device, use_amp)
        history.append({"stage": 2, "epoch": epoch, "train_loss": total_loss / len(train_y),
                        "val_acc": acc, "val_qwk": qwk, "val_f1": f1, "val_mae": mae})
        print(f"  [{run_name}] stage2 epoch {epoch}/{max_epochs} | val QWK {qwk:.4f}")
        if qwk > best_qwk:
            best_qwk, no_improve, best_epoch = qwk, 0, max_frozen_epochs + epoch
            best_metrics = {"val_acc": acc, "val_qwk": qwk, "val_f1": f1, "val_mae": mae,
                            "val_recall": recall.tolist()}
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                break
    train_time_s = time.time() - t_start

    alpha = float(model.cbam.alpha.item()) if hasattr(model.cbam, "alpha") else None
    pd.DataFrame(history).round(4).to_csv(os.path.join(OUT_RESULTS_DIR, f"{run_name}_training_log.csv"), index=False)
    return {"n_params": n_params, "train_time_s": train_time_s, "best_epoch": best_epoch,
           "alpha": alpha, **best_metrics}


def run_single_with_retries(config, seed, args, train_p, train_y, val_p, val_y, device, use_amp):
    run_name = f"kneenext_{config}_seed{seed}"
    batch_size = args.batch_size
    for attempt in range(1, 4):
        try:
            train_loader = DataLoader(KneeNextDataset(train_p, train_y, train=True), batch_size=batch_size,
                                      shuffle=True, num_workers=args.workers, pin_memory=device.type == "cuda")
            val_loader = DataLoader(KneeNextDataset(val_p, val_y, train=False), batch_size=batch_size,
                                    shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
            result = _run_single(config, seed, batch_size, train_loader, val_loader, device, use_amp,
                                 train_y, args.max_frozen_epochs, args.max_epochs, args.patience, run_name)
            result["batch_size"] = batch_size
            log(f"- `{run_name}` succeeded on attempt {attempt} (batch size {batch_size}): "
               f"val QWK {result['val_qwk']:.4f}, val acc {result['val_acc']:.4f}, "
               f"{result['train_time_s']/60:.1f} min, {result['n_params']:,} params"
               + (f", alpha={result['alpha']:.3f}" if result["alpha"] is not None else ""))
            return result
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            log(f"- `{run_name}` attempt {attempt}: CUDA OOM at batch size {batch_size} - "
               f"{'halving batch size and retrying' if batch_size > 8 else 'already at minimum batch size'}.")
            batch_size = max(batch_size // 2, 8)
        except Exception as e:
            log(f"- `{run_name}` attempt {attempt} failed: {type(e).__name__}: {e}\n"
               f"```\n{traceback.format_exc()}\n```")
    log(f"- `{run_name}` FAILED after 3 attempts - skipping this config/seed, continuing the sweep.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ablation", "single"], default="ablation")
    ap.add_argument("--config", choices=CONFIGS, default=None, help="--mode single only")
    ap.add_argument("--seed", type=int, default=42, help="--mode single only")
    ap.add_argument("--data-root", default="dataset")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--max-frozen-epochs", type=int, default=6)
    ap.add_argument("--max-epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--smoke-test", action="store_true",
                    help="Tiny subset (40 train/20 val) and 1+1 epochs, to validate the pipeline before a real run")
    args = ap.parse_args()
    if args.smoke_test:
        args.max_frozen_epochs, args.max_epochs, args.patience = 1, 1, 1

    os.makedirs(OUT_CKPT_DIR, exist_ok=True)
    os.makedirs(OUT_RESULTS_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    print(f"Device: {device}")

    train_p, train_y = list_split(args.data_root, "train")
    val_p, val_y = list_split(args.data_root, "val")
    if args.smoke_test:
        train_p, train_y = train_p[:40], train_y[:40]
        val_p, val_y = val_p[:20], val_y[:20]
    print(f"Train: {len(train_p)} | Val: {len(val_p)}")

    if args.mode == "single":
        if args.config is None:
            raise SystemExit("--mode single requires --config")
        result = run_single_with_retries(args.config, args.seed, args, train_p, train_y, val_p, val_y, device, use_amp)
        print(result)
        return

    # --- Full cumulative ablation sweep ---
    log(f"# KneeNeXt ablation run started\n\nDevice: {device} | Train: {len(train_p)} | Val: {len(val_p)}"
       + (" | **SMOKE TEST MODE**" if args.smoke_test else ""))

    fig_path = os.path.join(OUT_RESULTS_DIR, "kneenext_joint_zoom_check.png")
    save_joint_zoom_check_figure(val_p, val_y, fig_path)
    fallback_rate = compute_fallback_rate(val_p + train_p, sample_n=None if not args.smoke_test else 60)
    log(f"Joint-line detector: check figure saved to `{fig_path}`. Failure/fallback rate over "
       f"{'val+train' if not args.smoke_test else 'a smoke-test sample'}: {fallback_rate:.4f} "
       f"({fallback_rate*100:.2f}%) - falls back to the vertical center when the 'darkest row' "
       f"isn't a meaningful dip (z-score < {1.0}).")

    best_qwk_so_far = None
    ablation_rows = []
    for config in CONFIGS:
        seed_results = []
        for seed in (42, 7):
            r = run_single_with_retries(config, seed, args, train_p, train_y, val_p, val_y, device, use_amp)
            if r is not None:
                seed_results.append(r)
        if not seed_results:
            log(f"## Config `{config}`: both seeds failed - not adopted (nothing to compare).")
            continue

        qwks = [r["val_qwk"] for r in seed_results]
        accs = [r["val_acc"] for r in seed_results]
        f1s = [r["val_f1"] for r in seed_results]
        maes = [r["val_mae"] for r in seed_results]
        recalls = np.array([r["val_recall"] for r in seed_results])
        mean_qwk = float(np.mean(qwks))
        adopted = best_qwk_so_far is None or mean_qwk > best_qwk_so_far
        ablation_rows.append({
            "Config": config, "N_seeds": len(seed_results),
            "MeanValAcc": np.mean(accs), "StdValAcc": np.std(accs),
            "MeanValQWK": mean_qwk, "StdValQWK": np.std(qwks),
            "MeanValMacroF1": np.mean(f1s), "StdValMacroF1": np.std(f1s),
            "MeanValMAE": np.mean(maes), "StdValMAE": np.std(maes),
            **{f"Recall_KL{g}": np.mean(recalls[:, g]) for g in range(5)},
            "Params": seed_results[0]["n_params"],
            "TrainTimeMin": np.mean([r["train_time_s"] for r in seed_results]) / 60,
            "Alpha": seed_results[0]["alpha"],
            "Adopted": adopted,
        })
        log(f"## Config `{config}`: mean val QWK {mean_qwk:.4f} (seeds: {[round(q,4) for q in qwks]}) "
           f"vs best-so-far {best_qwk_so_far if best_qwk_so_far is not None else 'N/A (first config)'} "
           f"-> **{'ADOPTED' if adopted else 'NOT ADOPTED'}**")
        if adopted:
            best_qwk_so_far = mean_qwk

    pd.DataFrame(ablation_rows).round(4).to_csv(ABLATION_CSV, index=False)
    log(f"Ablation table saved to `{ABLATION_CSV}`.")

    if ablation_rows:
        import shutil
        import json
        best_row = max(ablation_rows, key=lambda r: r["MeanValQWK"] if r["Adopted"] else -1)
        # Among the adopted config's two seed checkpoints, keep the better one as the overall best.
        best_cfg = best_row["Config"]
        cand = [(s, os.path.join(OUT_CKPT_DIR, f"kneenext_{best_cfg}_seed{s}_best.pt")) for s in (42, 7)]
        cand = [(s, p) for s, p in cand if os.path.exists(p)]
        # Re-read each seed's best QWK from its training log to pick the winner.
        best_seed, best_qwk_val = None, -1
        for s, p in cand:
            tlog = os.path.join(OUT_RESULTS_DIR, f"kneenext_{best_cfg}_seed{s}_training_log.csv")
            if os.path.exists(tlog):
                q = pd.read_csv(tlog)["val_qwk"].max()
                if q > best_qwk_val:
                    best_qwk_val, best_seed = q, s
        if best_seed is not None:
            src = os.path.join(OUT_CKPT_DIR, f"kneenext_{best_cfg}_seed{best_seed}_best.pt")
            dst = os.path.join(OUT_CKPT_DIR, "kneenext_best.pt")
            shutil.copyfile(src, dst)
            with open(os.path.join(OUT_CKPT_DIR, "kneenext_best_config.json"), "w") as f:
                json.dump({"config": best_cfg, "seed": best_seed, "val_qwk": float(best_qwk_val)}, f, indent=2)
            log(f"**Overall best KneeNeXt config: `{best_cfg}` (seed {best_seed}, val QWK {best_qwk_val:.4f})** "
               f"- copied to `{dst}`, flags saved to `kneenext_best_config.json`.")

    log("# KneeNeXt ablation run finished.")


if __name__ == "__main__":
    main()
