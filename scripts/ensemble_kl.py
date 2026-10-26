"""Part A - greedy probability ensemble + threshold tuning + flip-TTA, selected entirely on
validation, over every available trained CORAL model: kl_final (seed 42), its seed-7 sibling
from the backbone ablation, KneeNeXt (both seeds, its winning "asymmetry" configuration), and
Hybrid 1. No retraining - every model's checkpoint is loaded read-only. The test set is
touched exactly once, at the end, to evaluate whichever configuration validation selected.

    python scripts/ensemble_kl.py

Outputs under outputs/ensemble/ only. Decisions logged to outputs/extended/extended_log.md
(shared across Parts A/B/C per the brief).
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
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error,
                             classification_report, confusion_matrix)
from sklearn.model_selection import KFold

from kl_common import list_split, get_transforms, KLDataset, build_classifier, build_hybrid1, run_eval_probs, GRADES
from kneenext import KneeNextDataset, build_kneenext, run_eval_kneenext

KL_CKPT_DIR = "outputs/checkpoints"
KNEENEXT_CKPT_DIR = "outputs/kneenext/checkpoints"
OUT_DIR = "outputs/ensemble"
EXTENDED_DIR = "outputs/extended"
LOG_PATH = os.path.join(EXTENDED_DIR, "extended_log.md")
CONFIG_PATH = os.path.join(OUT_DIR, "ensemble_config.json")

MODEL_SPECS = [
    {"name": "kl_final", "kind": "classifier", "ckpt_dir": KL_CKPT_DIR, "ckpt": "kl_final_best.pt"},
    {"name": "kl_final_seed7", "kind": "classifier", "ckpt_dir": KL_CKPT_DIR,
     "ckpt": "kl_baseline_convnext_tiny_seed7_warmup2_clahe_cbam_coral_best.pt"},
    {"name": "hybrid1", "kind": "hybrid1", "ckpt_dir": KL_CKPT_DIR, "ckpt": "kl_hybrid1_best.pt"},
    {"name": "kneenext_seed42", "kind": "kneenext", "config": "asymmetry", "ckpt_dir": KNEENEXT_CKPT_DIR,
     "ckpt": "kneenext_asymmetry_seed42_best.pt"},
    {"name": "kneenext_seed7", "kind": "kneenext", "config": "asymmetry", "ckpt_dir": KNEENEXT_CKPT_DIR,
     "ckpt": "kneenext_asymmetry_seed7_best.pt"},
]


def log(msg, also_print=True):
    os.makedirs(EXTENDED_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n\n")
    if also_print:
        print(msg)


def build_model(spec, device):
    if spec["kind"] == "classifier":
        m = build_classifier("convnext_tiny", 3, pretrained=False, out_dim=4, attention="cbam").to(device)
    elif spec["kind"] == "hybrid1":
        m = build_hybrid1(spec["ckpt_dir"], device, init_from=None)
    elif spec["kind"] == "kneenext":
        m = build_kneenext(spec["config"], KL_CKPT_DIR, device, init_from=None)
    else:
        raise ValueError(spec["kind"])
    m.load_state_dict(torch.load(os.path.join(spec["ckpt_dir"], spec["ckpt"]), map_location=device))
    m.eval()
    return m


def _kneenext_tta_probs(model, loader, device, use_amp):
    model.eval()
    all_probs = []
    with torch.no_grad():
        for g, z, gp, zp, y in loader:
            g, z, gp, zp = g.to(device), z.to(device), gp.to(device), zp.to(device)
            g_f = torch.flip(g, dims=[3])
            z_f = torch.flip(z, dims=[3]) if model.use_zoom else z
            with torch.amp.autocast("cuda", enabled=use_amp):
                l1 = model(g, z if model.use_zoom else None,
                           gp if model.use_joint_cbam else None, zp if model.use_joint_cbam else None)
                l2 = model(g_f, z_f if model.use_zoom else None,
                           gp if model.use_joint_cbam else None, zp if model.use_joint_cbam else None)
            p = (torch.sigmoid(l1.float()) + torch.sigmoid(l2.float())) / 2
            all_probs.append(p.cpu())
    return torch.cat(all_probs).numpy()


def collect_probs(spec, model, paths, labels, device, use_amp, tta=False):
    """Returns (N,4) cumulative CORAL probabilities P(grade>k) for one model on one split."""
    if spec["kind"] == "kneenext":
        loader = DataLoader(KneeNextDataset(paths, labels, train=False), batch_size=32,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        if tta:
            return _kneenext_tta_probs(model, loader, device, use_amp)
        _, _, probs = run_eval_kneenext(model, loader, device, use_amp)
        return probs
    else:
        _, eval_tf = get_transforms()
        loader = DataLoader(KLDataset(paths, labels, eval_tf, use_clahe=True), batch_size=32,
                            shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
        _, _, probs = run_eval_probs(model, loader, None, device, use_amp, loss_type="coral", tta=tta)
        return probs


def decode(probs, thresholds):
    return (probs > np.array(thresholds)).sum(axis=1)


def metrics(y_true, y_pred):
    return {"acc": float(accuracy_score(y_true, y_pred)),
           "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
           "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
           "mae": float(mean_absolute_error(y_true, y_pred))}


def greedy_select(probs_dict, val_y):
    """Starts with the single best model (by val accuracy), then repeatedly adds whichever
    remaining model most improves val accuracy without dropping val QWK by more than 0.005,
    until nothing qualifies."""
    names = list(probs_dict.keys())
    solo_metrics = {n: metrics(val_y, decode(probs_dict[n], [0.5] * 4)) for n in names}
    start = max(names, key=lambda n: solo_metrics[n]["acc"])
    selected = [start]
    cur_m = solo_metrics[start]
    steps = [{"step": 0, "added": start, "members": list(selected), **cur_m}]

    remaining = [n for n in names if n != start]
    step_i = 1
    while remaining:
        best_cand, best_probs, best_m = None, None, None
        for cand in remaining:
            trial_probs = np.mean([probs_dict[n] for n in selected + [cand]], axis=0)
            trial_m = metrics(val_y, decode(trial_probs, [0.5] * 4))
            if trial_m["qwk"] >= cur_m["qwk"] - 0.005 and trial_m["acc"] > cur_m["acc"]:
                if best_cand is None or trial_m["acc"] > best_m["acc"]:
                    best_cand, best_probs, best_m = cand, trial_probs, trial_m
        if best_cand is None:
            break
        selected.append(best_cand)
        remaining.remove(best_cand)
        cur_m = best_m
        steps.append({"step": step_i, "added": best_cand, "members": list(selected), **cur_m})
        step_i += 1
    final_probs = np.mean([probs_dict[n] for n in selected], axis=0)
    return selected, final_probs, cur_m, steps


def tune_thresholds(probs, y, grid=None, passes=2):
    grid = grid if grid is not None else np.arange(0.30, 0.71, 0.05)
    best = [0.5] * 4
    best_acc = accuracy_score(y, decode(probs, best))
    for _ in range(passes):
        for k in range(4):
            for t in grid:
                trial = best.copy()
                trial[k] = float(t)
                acc = accuracy_score(y, decode(probs, trial))
                if acc > best_acc:
                    best_acc, best = acc, trial
    return best, best_acc


def cv_threshold_gain(probs, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tuned_accs, base_accs = [], []
    for train_idx, test_idx in kf.split(probs):
        thr, _ = tune_thresholds(probs[train_idx], y[train_idx])
        tuned_accs.append(accuracy_score(y[test_idx], decode(probs[test_idx], thr)))
        base_accs.append(accuracy_score(y[test_idx], decode(probs[test_idx], [0.5] * 4)))
    return float(np.mean(tuned_accs) - np.mean(base_accs)), float(np.mean(tuned_accs)), float(np.mean(base_accs))


def save_report(y_true, y_pred, probs, paths, out_prefix, title):
    acc, qwk = accuracy_score(y_true, y_pred), cohen_kappa_score(y_true, y_pred, weights="quadratic")
    macro_f1, mae = f1_score(y_true, y_pred, average="macro"), mean_absolute_error(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=GRADES, digits=4, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.round(4).to_csv(f"{out_prefix}_report.csv")

    prob_cols = {f"prob_thresh{g}gt": probs[:, g] for g in range(probs.shape[1])}
    preds_df = pd.DataFrame({"image_path": paths, "true_grade": y_true, "pred_grade": y_pred, **prob_cols}).round(4)
    preds_path = f"{out_prefix}_predictions.csv"
    preds_df.to_csv(preds_path, index=False)

    cm = confusion_matrix(y_true, y_pred, labels=range(5))
    pct = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None) * 100
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for r in range(5):
        for c in range(5):
            ax.text(c, r, f"{cm[r, c]}\n({pct[r, c]:.1f}%)", ha="center", va="center",
                    color="white" if pct[r, c] > 50 else "black", fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(GRADES)
    ax.set_yticks(range(5)); ax.set_yticklabels(GRADES)
    ax.set_xlabel("Predicted grade"); ax.set_ylabel("Actual grade"); ax.set_title(title)
    plt.tight_layout()
    fig.savefig(f"{out_prefix}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return acc, qwk, macro_f1, mae, preds_path


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(EXTENDED_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log(f"# Part A: ensemble + threshold tuning\n\nDevice: {device}")

    val_p, val_y = list_split("dataset", "val")
    val_y = np.array(val_y)
    log(f"Val: {len(val_p)} images.")

    probs_dict, failed = {}, []
    for spec in MODEL_SPECS:
        ckpt_path = os.path.join(spec["ckpt_dir"], spec["ckpt"])
        for attempt in range(1, 4):
            try:
                if not os.path.exists(ckpt_path):
                    raise FileNotFoundError(ckpt_path)
                model = build_model(spec, device)
                probs = collect_probs(spec, model, val_p, val_y, device, use_amp, tta=False)
                solo = metrics(val_y, decode(probs, [0.5] * 4))
                probs_dict[spec["name"]] = probs
                log(f"- `{spec['name']}` ({spec['kind']}, `{spec['ckpt']}`): val acc {solo['acc']:.4f}, "
                   f"val QWK {solo['qwk']:.4f} (attempt {attempt})")
                del model
                torch.cuda.empty_cache()
                break
            except Exception as e:
                log(f"- `{spec['name']}` attempt {attempt} failed: {type(e).__name__}: {e}")
                if attempt == 3:
                    failed.append(spec["name"])
    if failed:
        log(f"Models unavailable after 3 attempts, excluded from the ensemble: {failed}")
    if not probs_dict:
        log("No models available - aborting Part A.")
        return

    selected, ens_probs, ens_m, steps = greedy_select(probs_dict, val_y)
    log("## Greedy ensemble selection (validation)\n\n" +
       "\n".join(f"- Step {s['step']}: added `{s['added']}` -> members {s['members']} | "
                f"val acc {s['acc']:.4f}, val QWK {s['qwk']:.4f}" for s in steps))
    pd.DataFrame(steps).round(4).to_csv(os.path.join(OUT_DIR, "ensemble_selection_steps.csv"), index=False)
    log(f"**Selected ensemble**: {selected} (val acc {ens_m['acc']:.4f}, val QWK {ens_m['qwk']:.4f}, "
       f"started from the single best model by val accuracy).")

    cv_gain, cv_tuned, cv_base = cv_threshold_gain(ens_probs, val_y)
    adopt_thresholds = cv_gain > 0
    if adopt_thresholds:
        final_thresholds, tuned_full_acc = tune_thresholds(ens_probs, val_y)
        log(f"## Threshold tuning: ADOPTED (5-fold CV gain {cv_gain:+.4f}: tuned {cv_tuned:.4f} vs "
           f"flat-0.5 {cv_base:.4f}). Final thresholds (fit on the full val set): "
           f"{[round(t, 2) for t in final_thresholds]} (full-val accuracy {tuned_full_acc:.4f}).")
    else:
        final_thresholds = [0.5] * 4
        log(f"## Threshold tuning: NOT ADOPTED (5-fold CV gain {cv_gain:+.4f}: tuned {cv_tuned:.4f} vs "
           f"flat-0.5 {cv_base:.4f} - no generalizing improvement). Keeping flat 0.5 thresholds.")

    no_tta_pred = decode(ens_probs, final_thresholds)
    no_tta_m = metrics(val_y, no_tta_pred)

    tta_probs_dict = {}
    for spec in MODEL_SPECS:
        if spec["name"] not in selected:
            continue
        model = build_model(spec, device)
        tta_probs_dict[spec["name"]] = collect_probs(spec, model, val_p, val_y, device, use_amp, tta=True)
        del model
        torch.cuda.empty_cache()
    ens_tta_probs = np.mean([tta_probs_dict[n] for n in selected], axis=0)
    tta_pred = decode(ens_tta_probs, final_thresholds)
    tta_m = metrics(val_y, tta_pred)

    adopt_tta = tta_m["acc"] > no_tta_m["acc"]
    if adopt_tta:
        log(f"## Flip TTA: ADOPTED (val acc {no_tta_m['acc']:.4f} -> {tta_m['acc']:.4f}, "
           f"QWK {no_tta_m['qwk']:.4f} -> {tta_m['qwk']:.4f}).")
        final_val_probs = ens_tta_probs
    else:
        log(f"## Flip TTA: NOT ADOPTED (val acc {no_tta_m['acc']:.4f} vs {tta_m['acc']:.4f} with TTA - no improvement).")
        final_val_probs = ens_probs
    final_val_m = metrics(val_y, decode(final_val_probs, final_thresholds))

    config = {"selected_models": selected, "thresholds": final_thresholds, "use_tta": adopt_tta,
             "val_acc": final_val_m["acc"], "val_qwk": final_val_m["qwk"]}
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)
    log(f"Ensemble config saved to `{CONFIG_PATH}`: {config}")

    # --- Test set: touched exactly once, here ---
    test_p, test_y = list_split("dataset", "test")
    test_y = np.array(test_y)
    test_probs_dict = {}
    for spec in MODEL_SPECS:
        if spec["name"] not in selected:
            continue
        model = build_model(spec, device)
        test_probs_dict[spec["name"]] = collect_probs(spec, model, test_p, test_y, device, use_amp, tta=adopt_tta)
        del model
        torch.cuda.empty_cache()
    ens_test_probs = np.mean([test_probs_dict[n] for n in selected], axis=0)
    test_pred = decode(ens_test_probs, final_thresholds)

    acc, qwk, macro_f1, mae, preds_path = save_report(
        test_y, test_pred, ens_test_probs, test_p, os.path.join(OUT_DIR, "ensemble_test"),
        "Ensemble: test confusion matrix")
    log(f"## Final test evaluation (touched once): accuracy {acc:.4f}, QWK {qwk:.4f}, "
       f"macro F1 {macro_f1:.4f}, MAE {mae:.4f}.")
    print(classification_report(test_y, test_pred, target_names=GRADES, digits=4, zero_division=0))

    sig_results = {}
    for other_name, other_file in [("kl_final", "outputs/results/kl_final_test_predictions.csv"),
                                   ("baseline", "outputs/results/kl_baseline_test_predictions.csv")]:
        out_csv = os.path.join(OUT_DIR, f"significance_ensemble_vs_{other_name}.csv")
        if not os.path.exists(other_file):
            log(f"WARNING: {other_file} not found - skipping bootstrap vs {other_name}.")
            continue
        subprocess.run([sys.executable, "scripts/bootstrap_compare.py", preds_path, other_file,
                       "ensemble", other_name, out_csv], check=True)
        sig_results[other_name] = pd.read_csv(out_csv)

    with open(os.path.join(OUT_DIR, "ensemble_summary.md"), "w") as f:
        f.write("# Ensemble (Part A) summary\n\n")
        f.write("## Selection steps (validation)\n\n")
        f.write(pd.DataFrame(steps).round(4).to_markdown(index=False) + "\n\n")
        f.write(f"Threshold tuning: {'adopted' if adopt_thresholds else 'not adopted'} "
               f"(5-fold CV gain {cv_gain:+.4f}). Final thresholds: {[round(t,2) for t in final_thresholds]}.\n\n")
        f.write(f"Flip TTA: {'adopted' if adopt_tta else 'not adopted'} "
               f"(val acc {no_tta_m['acc']:.4f} -> {tta_m['acc']:.4f}).\n\n")
        f.write(f"Selected models: {selected}\n\n")
        f.write("## Test set (evaluated once)\n\n")
        f.write(f"Accuracy **{acc:.4f}**, QWK **{qwk:.4f}**, Macro F1 **{macro_f1:.4f}**, MAE **{mae:.4f}**.\n\n")
        f.write(f"{'Above' if acc > 0.70 else 'Below'} the 70% accuracy target.\n\n")
        for other_name, df in sig_results.items():
            f.write(f"### vs {other_name}\n\n" + df.round(4).to_markdown(index=False) + "\n\n")
    log(f"Saved outputs/ensemble/ensemble_summary.md. Test accuracy {acc:.4f} "
       f"({'ABOVE' if acc > 0.70 else 'below'} the 70% target).")


if __name__ == "__main__":
    main()
