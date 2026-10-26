"""Part C - XAI extensions on kl_final (reporting only, not model selection): a 4-method
comparison (Grad-CAM, Grad-CAM++, Integrated Gradients, Occlusion), deletion-curve
faithfulness, a model-parameter-randomization sanity check, nearest-neighbor "similar cases,"
and calibration/uncertainty.

    python scripts/xai_extended.py

Outputs under outputs/extended/xai/ only. Decisions logged to
outputs/extended/extended_log.md (shared across Parts A/B/C).

Bug found and fixed (for this script only - kneenext.py is an existing file, not edited):
kneenext.grad_cam_view hooks model.features, which is called TWICE per forward pass (once per
view). Its hooks overwrite a single dict entry, so after the full forward+backward the
captured "activation" ends up being the zoom view's (the last forward call) while the
captured "gradient" ends up being the global view's (the last backward call, since autograd
unwinds in reverse order) - a mismatched, meaningless pair for any config with use_zoom=True
(which includes the winning "asymmetry" configuration, so this affected every committed
KneeNeXt Grad-CAM figure). Verified empirically by comparing against autograd.grad computed
directly on each view's own retained activation tensor. See kneenext_grad_cam_corrected()
below for the fix (capture via retain_grad() instead of an overwritable hook dict).
"""
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus
from captum.attr import IntegratedGradients, Occlusion

from kl_common import (list_split, get_transforms, KLDataset, build_classifier, apply_clahe,
                       run_eval_probs, coral_probs_to_grade, MEAN, STD, IMG_SIZE)
from predict import coral_display_probs
from kneenext import build_kneenext

KL_CKPT_DIR = "outputs/checkpoints"
KNEENEXT_CKPT_DIR = "outputs/kneenext/checkpoints"
EXTENDED_DIR = "outputs/extended"
OUT_DIR = os.path.join(EXTENDED_DIR, "xai")
LOG_PATH = os.path.join(EXTENDED_DIR, "extended_log.md")

PATCH = 16
GRID = IMG_SIZE // PATCH   # 14
N_PATCHES = GRID * GRID    # 196


def log(msg, also_print=True):
    os.makedirs(EXTENDED_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n\n")
    if also_print:
        print(msg)


class CoralSumTarget:
    def __call__(self, model_output):
        return model_output.sum()


def load_kl_final(device):
    m = build_classifier("convnext_tiny", 3, pretrained=False, out_dim=4, attention="cbam").to(device)
    m.load_state_dict(torch.load(os.path.join(KL_CKPT_DIR, "kl_final_best.pt"), map_location=device))
    m.eval()
    return m


def to_tensor(img224, device):
    x = (img224.astype(np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def load_image(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    img = apply_clahe(img)
    return cv2.resize(img, (IMG_SIZE, IMG_SIZE))


def predict_one(model, x):
    with torch.no_grad():
        logits = model(x)
    probs_t = torch.sigmoid(logits)
    thresh = probs_t.squeeze().cpu().numpy()
    grade = int(coral_probs_to_grade(probs_t).item())
    return grade, coral_display_probs(thresh)


def select_images(preds, n_per_grade=3):
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


def kneenext_grad_cam_corrected(model, global_x, zoom_x, global_prior, zoom_prior, feature_module):
    """Fixed dual-view Grad-CAM for the global view - see module docstring for the bug this
    works around. Captures each forward call's activation via retain_grad() (order-independent,
    unlike an overwritable hook dict) instead of kneenext.grad_cam_view's approach."""
    captured = []
    def fwd(_, __, out):
        out.retain_grad()
        captured.append(out)
    h = feature_module.register_forward_hook(fwd)
    model.zero_grad()
    logits = model(global_x, zoom_x, global_prior, zoom_prior)
    logits.sum().backward()
    h.remove()
    act_global = captured[0]   # KneeNeXt.forward always processes the global view first
    weights = act_global.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act_global.detach()).sum(dim=1)).squeeze().cpu().numpy()
    cam = cam / (cam.max() + 1e-8)
    return cam, torch.sigmoid(logits.detach()).squeeze().cpu().numpy()


# --------------------------------------------------------------------------------------------
# 1. Methods comparison
# --------------------------------------------------------------------------------------------
def methods_comparison_kl_final(model, device, chosen_df):
    gradcam = GradCAM(model=model, target_layers=[model.features])
    gradcampp = GradCAMPlusPlus(model=model, target_layers=[model.features])
    def forward_sum(x): return model(x).sum(dim=1)
    ig = IntegratedGradients(forward_sum)
    occ = Occlusion(forward_sum)
    target = [CoralSumTarget()]

    rows = []
    for _, row in chosen_df.iterrows():
        path, stem = row.image_path, os.path.splitext(os.path.basename(row.image_path))[0]
        img = load_image(path)
        x = to_tensor(img, device)

        gc = gradcam(input_tensor=x, targets=target)[0]
        gcpp = gradcampp(input_tensor=x, targets=target)[0]
        ig_raw = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=50).squeeze().detach().cpu().numpy()
        ig_map = np.abs(ig_raw).sum(axis=0); ig_map /= (ig_map.max() + 1e-8)
        occ_raw = occ.attribute(x, strides=(3, 16, 16), sliding_window_shapes=(3, 32, 32),
                                baselines=0).squeeze().detach().cpu().numpy()
        occ_map = np.abs(occ_raw).sum(axis=0); occ_map /= (occ_map.max() + 1e-8)
        maps = {"Grad-CAM": gc, "Grad-CAM++": gcpp, "Integrated Gradients": ig_map, "Occlusion": occ_map}

        fig, axes = plt.subplots(1, 5, figsize=(22, 4.5))
        axes[0].imshow(img); axes[0].set_title("Input"); axes[0].axis("off")
        for i, (name, m) in enumerate(maps.items()):
            axes[i + 1].imshow(img, cmap="gray"); axes[i + 1].imshow(m, cmap="jet", alpha=0.45)
            axes[i + 1].set_title(name); axes[i + 1].axis("off")
        fig.suptitle(f"KL{int(row.true_grade)} true / KL{int(row.pred_grade)} pred - {stem}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"kl_final_methods_{stem}.png"), dpi=110)
        plt.close(fig)

        flat = {k: v.flatten() for k, v in maps.items()}
        names = list(flat.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                rho, _ = spearmanr(flat[names[i]], flat[names[j]])
                rows.append({"image": stem, "method_a": names[i], "method_b": names[j], "spearman_rho": rho})
    corr_df = pd.DataFrame(rows)
    corr_df.to_csv(os.path.join(OUT_DIR, "kl_final_methods_correlation.csv"), index=False)
    summary = corr_df.groupby(["method_a", "method_b"])["spearman_rho"].mean().round(3)
    log("### Methods comparison (kl_final, 15 test images)\n\nSide-by-side figures saved to "
       f"`outputs/extended/xai/kl_final_methods_*.png`. Mean pairwise Spearman rank correlation "
       f"between methods' saliency maps:\n\n{summary.to_string()}")


def methods_comparison_kneenext(device, chosen_df):
    from kneenext import prepare_views
    ckpt = os.path.join(KNEENEXT_CKPT_DIR, "kneenext_best.pt")
    cfg_path = os.path.join(KNEENEXT_CKPT_DIR, "kneenext_best_config.json")
    if not (os.path.exists(ckpt) and os.path.exists(cfg_path)):
        log("KneeNeXt checkpoint not found - methods comparison run on kl_final only.")
        return
    import json
    with open(cfg_path) as f:
        info = json.load(f)
    model = build_kneenext(info["config"], KL_CKPT_DIR, device, init_from=None)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    feature_module = model.features[model.STAGE4_IDX - 1]

    log(f"### KneeNeXt Grad-CAM (global view, bug-corrected - see module docstring)\n\n"
       "Only Grad-CAM is run for KneeNeXt in this comparison (not Grad-CAM++/IG/Occlusion): "
       "those libraries assume a single-tensor-input model, and wrapping KneeNeXt's shared-"
       "backbone dual-forward-pass architecture for each of them was judged not worth the "
       "added complexity for this task - logged as a scope decision, not an oversight.")
    for _, row in chosen_df.iterrows():
        path, stem = row.image_path, os.path.splitext(os.path.basename(row.image_path))[0]
        img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        global_img, zoom_img, g_t, z_t, g_prior, z_prior, _ = prepare_views(img)
        g = g_t.unsqueeze(0).to(device)
        z = z_t.unsqueeze(0).to(device) if model.use_zoom else None
        gp = torch.from_numpy(g_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
        zp = torch.from_numpy(z_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
        cam, _ = kneenext_grad_cam_corrected(model, g, z if z is not None else g, gp, zp, feature_module)
        cam_up = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(global_img); axes[0].set_title("Global view"); axes[0].axis("off")
        axes[1].imshow(global_img, cmap="gray"); axes[1].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM (corrected)"); axes[1].axis("off")
        fig.suptitle(f"KL{int(row.true_grade)} true / KL{int(row.pred_grade)} pred - {stem}")
        plt.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"kneenext_gradcam_corrected_{stem}.png"), dpi=110)
        plt.close(fig)


# --------------------------------------------------------------------------------------------
# 2. Faithfulness: deletion curves
# --------------------------------------------------------------------------------------------
def patch_scores(attr_map):
    return attr_map.reshape(GRID, PATCH, GRID, PATCH).mean(axis=(1, 3)).flatten()


def blur_patches(img, order, k):
    if k == 0:
        return img
    out = img.copy()
    blurred = cv2.GaussianBlur(img, (31, 31), 15)
    for idx in order[:k]:
        r, c = idx // GRID, idx % GRID
        out[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH] = \
            blurred[r * PATCH:(r + 1) * PATCH, c * PATCH:(c + 1) * PATCH]
    return out


def deletion_curve(model, device, img, true_grade, attr_map, rng, steps=11):
    order_imp = np.argsort(-patch_scores(attr_map))
    order_rnd = rng.permutation(N_PATCHES)
    fracs = np.linspace(0, 1, steps)
    p_imp, p_rnd = [], []
    for f in fracs:
        k = int(round(f * N_PATCHES))
        _, disp_imp = predict_one(model, to_tensor(blur_patches(img, order_imp, k), device))
        _, disp_rnd = predict_one(model, to_tensor(blur_patches(img, order_rnd, k), device))
        p_imp.append(disp_imp[true_grade])
        p_rnd.append(disp_rnd[true_grade])
    return fracs, np.array(p_imp), np.array(p_rnd)


def faithfulness(model, device, test_df, n=200):
    rng = np.random.default_rng(42)
    sample = test_df.sample(n=min(n, len(test_df)), random_state=42).reset_index(drop=True)
    gradcam = GradCAM(model=model, target_layers=[model.features])
    def forward_sum(x): return model(x).sum(dim=1)
    ig = IntegratedGradients(forward_sum)
    target = [CoralSumTarget()]

    curves = {"gradcam_importance": [], "gradcam_random": [], "ig_importance": [], "ig_random": []}
    fracs = None
    for _, row in sample.iterrows():
        img = load_image(row.image_path)
        x = to_tensor(img, device)
        true_g = int(row.true_grade)

        gc = gradcam(input_tensor=x, targets=target)[0]
        fracs, p_imp, p_rnd = deletion_curve(model, device, img, true_g, gc, rng)
        curves["gradcam_importance"].append(p_imp); curves["gradcam_random"].append(p_rnd)

        ig_raw = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=20).squeeze().detach().cpu().numpy()
        ig_map = np.abs(ig_raw).sum(axis=0)
        _, p_imp2, p_rnd2 = deletion_curve(model, device, img, true_g, ig_map, rng)
        curves["ig_importance"].append(p_imp2); curves["ig_random"].append(p_rnd2)

    def trapz_auc(y, x):
        y, x = np.asarray(y), np.asarray(x)
        return float(np.sum((y[:-1] + y[1:]) / 2 * np.diff(x)))

    mean_curves = {k: np.mean(v, axis=0) for k, v in curves.items()}
    aucs = {k: trapz_auc(v, fracs) for k, v in mean_curves.items()}

    fig, ax = plt.subplots(figsize=(7, 5))
    for k, v in mean_curves.items():
        ax.plot(fracs, v, label=f"{k} (AUC={aucs[k]:.3f})",
               linestyle="--" if "random" in k else "-")
    ax.set_xlabel("Fraction of image deleted (most-important-first / random)")
    ax.set_ylabel("True-class probability")
    ax.set_title(f"Deletion curves (n={len(sample)} test images)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "deletion_curves.png"), dpi=140)
    plt.close(fig)
    pd.DataFrame({"frac": fracs, **mean_curves}).round(4).to_csv(
        os.path.join(OUT_DIR, "deletion_curves.csv"), index=False)

    log(f"### Faithfulness (deletion curves, n={len(sample)} random test images)\n\n"
       f"AUC (lower = more faithful - true-class probability collapses faster when the "
       f"explanation's claimed-important regions are actually deleted): "
       f"Grad-CAM importance-order **{aucs['gradcam_importance']:.3f}** vs random-order "
       f"{aucs['gradcam_random']:.3f}; Integrated Gradients importance-order "
       f"**{aucs['ig_importance']:.3f}** vs random-order {aucs['ig_random']:.3f}. "
       f"{'Both methods beat random deletion (lower AUC), i.e. both are genuinely faithful, not just plausible-looking.' if aucs['gradcam_importance'] < aucs['gradcam_random'] and aucs['ig_importance'] < aucs['ig_random'] else 'At least one method did not clearly beat random deletion - its explanations may be more plausible-looking than faithful.'}")


# --------------------------------------------------------------------------------------------
# 3. Sanity check: model-parameter randomization
# --------------------------------------------------------------------------------------------
def randomize_module_(module, seed):
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        with torch.no_grad():
            noise = torch.randn(p.shape, generator=g).to(p.device, p.dtype)
            p.copy_(noise * (p.std() if p.numel() > 1 else 1.0))


def sanity_check(model, device, image_path, true_grade):
    img = load_image(image_path)
    x = to_tensor(img, device)
    gradcam = GradCAM(model=model, target_layers=[model.features])
    orig_cam = gradcam(input_tensor=x, targets=[CoralSumTarget()])[0]

    model_copy = copy.deepcopy(model)
    stages = [("classifier+cbam", [model_copy.classifier, model_copy.cbam])] + \
        [(f"+features[{i}]", [model_copy.features[i]]) for i in range(7, -1, -1)]

    maps, corrs = {"original": orig_cam}, {"original": 1.0}
    seed = 0
    for name, modules in stages:
        for m in modules:
            randomize_module_(m, seed); seed += 1
        gc = GradCAM(model=model_copy, target_layers=[model_copy.features])
        cam_step = gc(input_tensor=x, targets=[CoralSumTarget()])[0]
        rho, _ = spearmanr(orig_cam.flatten(), cam_step.flatten())
        maps[name], corrs[name] = cam_step, float(rho)

    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.2))
    for ax, (k, v) in zip(axes, maps.items()):
        ax.imshow(img, cmap="gray"); ax.imshow(v, cmap="jet", alpha=0.5)
        ax.set_title(f"{k}\nrho={corrs[k]:.2f}", fontsize=7); ax.axis("off")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sanity_check_randomization.png"), dpi=130)
    plt.close(fig)
    pd.DataFrame(list(corrs.items()), columns=["stage", "spearman_vs_original"]).to_csv(
        os.path.join(OUT_DIR, "sanity_check_correlations.csv"), index=False)

    final_rho = list(corrs.values())[-1]
    passed = final_rho < 0.3
    log(f"### Sanity check (cascading parameter randomization)\n\n"
       f"Spearman correlation with the original Grad-CAM map, as layers are randomized from "
       f"the output backward: " + ", ".join(f"{k}={v:.2f}" for k, v in corrs.items()) + ".\n\n"
       f"{'PASSED' if passed else 'POSSIBLE CONCERN'}: correlation with the original map "
       f"{'drops to ' + f'{final_rho:.2f}' + ' once the whole backbone is randomized, as expected for a method that genuinely depends on learned weights' if passed else 'stays at ' + f'{final_rho:.2f}' + ' even with the whole backbone randomized - suggests Grad-CAM here may be partly driven by architecture/edge-detection artifacts rather than learned features (the known failure mode this test is designed to catch)'}.")


# --------------------------------------------------------------------------------------------
# 4. Similar cases (nearest-neighbor retrieval)
# --------------------------------------------------------------------------------------------
def extract_features(model, paths, labels, device, batch_size=64):
    _, eval_tf = get_transforms()
    loader = DataLoader(KLDataset(paths, labels, eval_tf, use_clahe=True), batch_size=batch_size,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    feats = []
    model.eval()
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(device)
            f = model.features(imgs)
            f, _ = model.cbam(f)
            f = model.avgpool(f).flatten(1)
            feats.append(f.cpu())
    return torch.cat(feats).numpy()


TRAIN_FEATURE_CACHE = os.path.join(OUT_DIR, "train_features_cache.npz")


def similar_cases(model, device, train_p, train_y, test_p, test_y):
    train_feat = extract_features(model, train_p, train_y, device)
    np.savez(TRAIN_FEATURE_CACHE, features=train_feat, paths=np.array(train_p), labels=np.array(train_y))
    test_feat = extract_features(model, test_p, test_y, device)
    train_norm = train_feat / (np.linalg.norm(train_feat, axis=1, keepdims=True) + 1e-8)
    test_norm = test_feat / (np.linalg.norm(test_feat, axis=1, keepdims=True) + 1e-8)
    sims = test_norm @ train_norm.T
    top3 = np.argsort(-sims, axis=1)[:, :3]
    train_y_arr, test_y_arr = np.array(train_y), np.array(test_y)

    rows, matches = [], 0
    for i in range(len(test_p)):
        neigh_grades = train_y_arr[top3[i]]
        vals, counts = np.unique(neigh_grades, return_counts=True)
        majority = int(vals[np.argmax(counts)])
        match = int(majority == test_y_arr[i])
        matches += match
        rows.append({"test_image": test_p[i], "true_grade": int(test_y_arr[i]),
                    "neighbor1_path": train_p[top3[i, 0]], "neighbor1_grade": int(neigh_grades[0]),
                    "neighbor2_path": train_p[top3[i, 1]], "neighbor2_grade": int(neigh_grades[1]),
                    "neighbor3_path": train_p[top3[i, 2]], "neighbor3_grade": int(neigh_grades[2]),
                    "majority_grade": majority, "match": match})
    acc = matches / len(test_p)
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "similar_cases.csv"), index=False)
    log(f"### Similar cases (3-nearest-neighbor retrieval, cosine similarity on pooled "
       f"CBAM-attended features)\n\nThe 3 nearest training images' majority grade matches the "
       f"test image's true grade **{acc:.4f}** ({matches}/{len(test_p)}) of the time.")
    return acc


# --------------------------------------------------------------------------------------------
# 5. Calibration and uncertainty
# --------------------------------------------------------------------------------------------
def get_display_probs(model, device, paths, labels):
    _, eval_tf = get_transforms()
    loader = DataLoader(KLDataset(paths, labels, eval_tf, use_clahe=True), batch_size=32,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    preds, true, thresh_probs = run_eval_probs(model, loader, None, device, device.type == "cuda", loss_type="coral")
    disp = np.array([coral_display_probs(p) for p in thresh_probs])
    return preds, true, disp


def calibration(model, device, val_p, val_y, test_p, test_y):
    val_pred, val_true, val_disp = get_display_probs(model, device, val_p, val_y)
    test_pred, test_true, test_disp = get_display_probs(model, device, test_p, test_y)

    test_conf = test_disp.max(axis=1)
    test_correct = (test_pred == test_true).astype(int)
    test_entropy = -(test_disp * np.log(test_disp + 1e-12)).sum(axis=1)
    val_entropy = -(val_disp * np.log(val_disp + 1e-12)).sum(axis=1)

    bins = np.linspace(0, 1, 11)
    bin_ids = np.clip(np.digitize(test_conf, bins) - 1, 0, 9)
    ece, bin_accs = 0.0, []
    for b in range(10):
        mask = bin_ids == b
        if mask.sum() == 0:
            bin_accs.append(np.nan); continue
        acc_b, conf_b = test_correct[mask].mean(), test_conf[mask].mean()
        bin_accs.append(acc_b)
        ece += (mask.sum() / len(test_conf)) * abs(acc_b - conf_b)

    fig, ax = plt.subplots(figsize=(6, 6))
    centers = (bins[:-1] + bins[1:]) / 2
    ax.bar(centers, np.nan_to_num(bin_accs), width=0.08, edgecolor="black", color="#6A8CC7", label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence (max predicted probability)"); ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram (test set, ECE={ece:.4f})"); ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "reliability_diagram.png"), dpi=140)
    plt.close(fig)

    thresh = float(np.percentile(val_entropy, 75))
    import json
    with open(os.path.join(OUT_DIR, "uncertainty_threshold.json"), "w") as f:
        json.dump({"entropy_threshold": thresh, "percentile": 75, "chosen_on": "validation"}, f, indent=2)
    confident = test_entropy <= thresh
    acc_confident = float(test_correct[confident].mean()) if confident.sum() else float("nan")
    acc_uncertain = float(test_correct[~confident].mean()) if (~confident).sum() else float("nan")

    pd.DataFrame({"image_path": test_p, "true_grade": test_true, "pred_grade": test_pred,
                 "confidence": test_conf, "entropy": test_entropy, "confident_flag": confident}).round(4).to_csv(
        os.path.join(OUT_DIR, "calibration_test_predictions.csv"), index=False)
    log(f"### Calibration and uncertainty\n\nExpected Calibration Error (test, 10 bins): "
       f"**{ece:.4f}**. Reliability diagram saved.\n\n"
       f"Uncertainty = entropy of the predicted 5-grade distribution. 'Refer to radiologist' "
       f"threshold chosen on validation (75th percentile of val entropy = {thresh:.4f}): on "
       f"test, {int(confident.sum())}/{len(test_p)} ({confident.mean()*100:.1f}%) flagged "
       f"confident (accuracy **{acc_confident:.4f}**), {int((~confident).sum())} flagged "
       f"uncertain (accuracy **{acc_uncertain:.4f}**) - the uncertain subset is "
       f"{'meaningfully less accurate, so the flag is doing useful work' if acc_uncertain < acc_confident - 0.03 else 'not much less accurate, so the flag is only weakly informative'}.")
    return ece, thresh, acc_confident, acc_uncertain


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    log(f"# Part C: XAI extensions\n\nDevice: {device}")

    model = load_kl_final(device)
    test_preds = pd.read_csv("outputs/results/kl_final_test_predictions.csv")
    test_preds["image_path"] = test_preds["image_path"].str.replace("\\", "/", regex=False)
    chosen = select_images(test_preds)

    methods_comparison_kl_final(model, device, chosen)
    methods_comparison_kneenext(device, chosen)
    faithfulness(model, device, test_preds, n=200)

    mid_row = chosen.iloc[len(chosen) // 2]
    sanity_check(model, device, mid_row.image_path, int(mid_row.true_grade))

    train_p, train_y = list_split("dataset", "train")
    test_p, test_y = list_split("dataset", "test")
    similar_cases(model, device, train_p, train_y, test_p, test_y)

    val_p, val_y = list_split("dataset", "val")
    calibration(model, device, val_p, val_y, test_p, test_y)

    log("# Part C finished.")


if __name__ == "__main__":
    main()
