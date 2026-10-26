"""Combined demo: predict.py's behavior (file browser, patient-detail prompts, Grad-CAM,
causal panel), plus:
  - the KL grade from Part A's ensemble if it beat kl_final on validation (checked at
    startup), otherwise kl_final alone - states which was used;
  - Grad-CAM averaged over whichever models were used for the prediction;
  - the 3 most similar training cases (cosine similarity on kl_final's pooled features)
    with their grades;
  - an uncertainty flag ("refer to radiologist") using Part C's validation-chosen threshold;
  - the causal panel extended with this patient's individual obesity effect (Part B's
    CausalForestDML) and the population knee-injury effect, if found.

predict.py and its default model (kl_final) are unchanged.

    python scripts/predict_extended.py
    python scripts/predict_extended.py path/to/xray.png --show [--age N] [--bmi N] [--family-history yes|no]
"""
import os
import json
import pickle
import argparse
import textwrap
import cv2
import numpy as np
import torch
import matplotlib

from kl_common import (KL_TEXT, build_classifier, build_hybrid1, apply_clahe, grad_cam,
                       MEAN, STD, IMG_SIZE, GRADES)
from kneenext import build_kneenext, prepare_views
from preprocess_xray import prepare_xray
from causal_model import load_or_fit_causal_models, out_of_range_messages
from predict import get_patient_inputs, pick_file, coral_display_probs, causal_panel_lines
from xai_extended import kneenext_grad_cam_corrected, TRAIN_FEATURE_CACHE

KL_CKPT_DIR = "outputs/checkpoints"
KNEENEXT_CKPT_DIR = "outputs/kneenext/checkpoints"
ENSEMBLE_CONFIG_PATH = "outputs/ensemble/ensemble_config.json"
CAUSAL_FOREST_PATH = "outputs/extended/causal/causal_forest_model.pkl"
INJURY_ESTIMATES_PATH = "outputs/extended/causal/injury_estimates.csv"
UNCERTAINTY_THRESH_PATH = "outputs/extended/xai/uncertainty_threshold.json"

MODEL_SPECS = {
    "kl_final": {"kind": "classifier", "ckpt_dir": KL_CKPT_DIR, "ckpt": "kl_final_best.pt"},
    "kl_final_seed7": {"kind": "classifier", "ckpt_dir": KL_CKPT_DIR,
                       "ckpt": "kl_baseline_convnext_tiny_seed7_warmup2_clahe_cbam_coral_best.pt"},
    "hybrid1": {"kind": "hybrid1", "ckpt_dir": KL_CKPT_DIR, "ckpt": "kl_hybrid1_best.pt"},
    "kneenext_seed42": {"kind": "kneenext", "config": "asymmetry", "ckpt_dir": KNEENEXT_CKPT_DIR,
                        "ckpt": "kneenext_asymmetry_seed42_best.pt"},
    "kneenext_seed7": {"kind": "kneenext", "config": "asymmetry", "ckpt_dir": KNEENEXT_CKPT_DIR,
                       "ckpt": "kneenext_asymmetry_seed7_best.pt"},
}


def build_model(spec, device):
    if spec["kind"] == "classifier":
        m = build_classifier("convnext_tiny", 3, pretrained=False, out_dim=4, attention="cbam").to(device)
    elif spec["kind"] == "hybrid1":
        m = build_hybrid1(spec["ckpt_dir"], device, init_from=None)
    elif spec["kind"] == "kneenext":
        m = build_kneenext(spec["config"], KL_CKPT_DIR, device, init_from=None)
    m.load_state_dict(torch.load(os.path.join(spec["ckpt_dir"], spec["ckpt"]), map_location=device))
    m.eval()
    return m


def to_tensor(rgb224, device):
    x = (rgb224.astype(np.float32) / 255.0 - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)


def kl_final_val_accuracy(device):
    from kl_common import list_split, get_transforms, KLDataset, run_eval_probs
    from torch.utils.data import DataLoader
    val_p, val_y = list_split("dataset", "val")
    model = build_model(MODEL_SPECS["kl_final"], device)
    _, eval_tf = get_transforms()
    loader = DataLoader(KLDataset(val_p, val_y, eval_tf, use_clahe=True), batch_size=32,
                        shuffle=False, num_workers=0, pin_memory=device.type == "cuda")
    pred, true, _ = run_eval_probs(model, loader, None, device, device.type == "cuda", loss_type="coral")
    del model
    torch.cuda.empty_cache()
    return float((pred == true).mean())


def load_prediction_setup(device):
    """Decides ensemble vs. kl_final-alone by actually comparing validation accuracy (not
    just trusting the ensemble was "selected" - the brief asks to check it beat kl_final)."""
    if not os.path.exists(ENSEMBLE_CONFIG_PATH):
        print("No ensemble config found - using kl_final alone.")
        model_dict = {"kl_final": build_model(MODEL_SPECS["kl_final"], device)}
        return "kl_final", model_dict, ["kl_final"], [0.5] * 4

    with open(ENSEMBLE_CONFIG_PATH) as f:
        cfg = json.load(f)
    kl_final_acc = kl_final_val_accuracy(device)
    beats = cfg["val_acc"] > kl_final_acc
    print(f"Ensemble val accuracy {cfg['val_acc']:.4f} vs kl_final solo val accuracy {kl_final_acc:.4f} "
         f"-> {'using the ENSEMBLE' if beats else 'using kl_final alone'}.")
    if beats:
        members = cfg["selected_models"]
        model_dict = {name: build_model(MODEL_SPECS[name], device) for name in members}
        return "ensemble", model_dict, members, cfg["thresholds"]
    model_dict = {"kl_final": build_model(MODEL_SPECS["kl_final"], device)}
    return "kl_final", model_dict, ["kl_final"], [0.5] * 4


def classify_rgb(img, models, members, thresholds, device):
    """Returns (grade, 5-dim display probs, averaged Grad-CAM (7x7), per-model CAMs dict)."""
    clahe_img = apply_clahe(img)
    all_probs, cams = [], {}
    for name in members:
        spec, model = MODEL_SPECS[name], models[name]
        if spec["kind"] == "kneenext":
            global_img, zoom_img, g_t, z_t, g_prior, z_prior, _ = prepare_views(clahe_img, already_clahe=True)
            g = g_t.unsqueeze(0).to(device)
            z = z_t.unsqueeze(0).to(device) if model.use_zoom else None
            gp = torch.from_numpy(g_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
            zp = torch.from_numpy(z_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
            feature_module = model.features[model.STAGE4_IDX - 1]
            cam, thresh_probs = kneenext_grad_cam_corrected(model, g, z if z is not None else g,
                                                             gp, zp, feature_module)
        else:
            x = to_tensor(cv2.resize(clahe_img, (IMG_SIZE, IMG_SIZE)), device)
            cam, thresh_probs = grad_cam(model, x, model.features)
        all_probs.append(thresh_probs)
        cams[name] = cam
    avg_probs = np.mean(all_probs, axis=0)
    avg_cam = np.mean(list(cams.values()), axis=0)
    grade = int((avg_probs > np.array(thresholds)).sum())
    return grade, coral_display_probs(avg_probs), avg_cam, cams


def nearest_cases(clahe_img224, kl_final_model, device, k=3):
    if not os.path.exists(TRAIN_FEATURE_CACHE):
        return None
    cache = np.load(TRAIN_FEATURE_CACHE, allow_pickle=True)
    train_feat, train_paths, train_labels = cache["features"], cache["paths"], cache["labels"]
    x = to_tensor(clahe_img224, device)
    with torch.no_grad():
        f = kl_final_model.features(x)
        f, _ = kl_final_model.cbam(f)
        f = kl_final_model.avgpool(f).flatten(1).cpu().numpy()[0]
    f_norm = f / (np.linalg.norm(f) + 1e-8)
    train_norm = train_feat / (np.linalg.norm(train_feat, axis=1, keepdims=True) + 1e-8)
    sims = train_norm @ f_norm
    top_idx = np.argsort(-sims)[:k]
    return [(str(train_paths[i]), int(train_labels[i]), float(sims[i])) for i in top_idx]


def uncertainty_info(disp_probs):
    entropy = float(-(disp_probs * np.log(disp_probs + 1e-12)).sum())
    if not os.path.exists(UNCERTAINTY_THRESH_PATH):
        return entropy, None, False
    with open(UNCERTAINTY_THRESH_PATH) as f:
        info = json.load(f)
    thresh = info["entropy_threshold"]
    return entropy, thresh, entropy > thresh


def extended_causal_lines(age, bmi, famkr, causal_models, causal_forest, injury_df):
    lines = causal_panel_lines(age, bmi, famkr, causal_models)
    if age is not None and bmi is not None and not out_of_range_messages(causal_models, age, bmi) and causal_forest is not None:
        famkr_used = famkr if famkr is not None else causal_models["most_common_famkr"]
        X = np.array([[age, famkr_used]])
        effect = float(causal_forest.effect(X)[0])
        lo, hi = causal_forest.effect_interval(X, alpha=0.05)
        lines.append(f"  This patient's individual obesity effect (causal forest, Part B): {effect:+.2f} "
                    f"KL grade points (95% CI [{float(np.ravel(lo)[0]):+.2f}, {float(np.ravel(hi)[0]):+.2f}]).")
    if injury_df is not None:
        lin = float(injury_df.loc[injury_df.Method == "linear_regression", "ATE"].iloc[0])
        psm = float(injury_df.loc[injury_df.Method == "propensity_score_matching", "ATE"].iloc[0])
        lines.append(f"  Population finding (Part B): prior knee injury is associated with "
                    f"{min(lin,psm):+.2f} to {max(lin,psm):+.2f} KL grade points, adjusted for "
                    f"age/BMI/family history.")
    return lines


def report(image_path, img, grade, probs, avg_cam, neighbors, entropy, unc_thresh, is_uncertain,
          model_label, members, out_dir, plt, age=None, bmi=None, famkr=None, causal_coefs=None,
          causal_forest=None, injury_df=None):
    print(f"\nImage: {image_path}")
    print(f"Model used: {model_label} ({', '.join(members)})")
    print(f"Predicted KL grade: {grade} - {KL_TEXT[grade]} ({probs[grade] * 100:.1f}% confidence)")
    for g in range(5):
        print(f"  KL{g}: {probs[g] * 100:5.1f}%")
    flag_text = "REFER TO RADIOLOGIST (uncertain)" if is_uncertain else "confident"
    print(f"Uncertainty: entropy {entropy:.3f}" + (f" (threshold {unc_thresh:.3f})" if unc_thresh else "")
         + f" -> {flag_text}")
    if neighbors:
        print("Most similar training cases:")
        for p, g, s in neighbors:
            print(f"  KL{g} - {os.path.basename(p)} (cosine similarity {s:.3f})")

    causal_lines = extended_causal_lines(age, bmi, famkr, causal_coefs, causal_forest, injury_df)
    print()
    for line in causal_lines:
        print(f"  {line}")

    n_thumb = len(neighbors) if neighbors else 0
    n_panels = 3 + n_thumb + 1   # input, Grad-CAM, prob, [neighbor thumbs], causal
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    idx = 0
    axes[idx].imshow(img)
    axes[idx].set_title(f"Input\n({model_label})"); axes[idx].axis("off"); idx += 1

    cam_up = cv2.resize(avg_cam, (IMG_SIZE, IMG_SIZE))
    axes[idx].imshow(cv2.resize(img, (IMG_SIZE, IMG_SIZE)), cmap="gray")
    axes[idx].imshow(cam_up, cmap="jet", alpha=0.45)
    axes[idx].set_title(f"Grad-CAM (avg over {len(members)} model(s))"); axes[idx].axis("off"); idx += 1

    for p, g, s in (neighbors or []):
        n_img = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) if os.path.exists(p) else None
        if n_img is not None:
            axes[idx].imshow(n_img)
        axes[idx].set_title(f"Similar: KL{g}\n(sim {s:.2f})", fontsize=9); axes[idx].axis("off"); idx += 1

    prob_ax, causal_ax = axes[idx], axes[idx + 1]
    colors = ["#888888"] * 5
    colors[grade] = "#D85A30"
    prob_ax.bar([f"KL{g}" for g in range(5)], probs * 100, color=colors)
    prob_ax.set_ylabel("Probability (%)"); prob_ax.set_ylim(0, 100)
    prob_ax.set_title(f"Predicted: KL{grade} ({KL_TEXT[grade]})\n{flag_text}")

    causal_ax.axis("off")
    causal_wrapped = "\n".join(textwrap.fill(line, 34, subsequent_indent="  ") for line in causal_lines)
    causal_ax.text(0.0, 1.0, causal_wrapped, va="top", ha="left", fontsize=7.2, family="monospace",
                   transform=causal_ax.transAxes)
    causal_ax.set_title("Causal risk panel (extended)", fontsize=10)

    fig.canvas.manager.set_window_title(f"Extended prediction - {os.path.basename(image_path)}")
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_extended_prediction.png")
    fig.savefig(out_path, dpi=110)
    print(f"Saved result image to {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?")
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--bmi", type=float, default=None)
    ap.add_argument("--family-history", choices=["yes", "no"], default=None)
    ap.add_argument("--out-dir", default="outputs/predictions")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--auto-crop", action="store_true")
    args = ap.parse_args()

    interactive = args.image is None
    if not interactive and not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on {device}...")
    model_label, models, members, thresholds = load_prediction_setup(device)
    kl_final_for_neighbors = models.get("kl_final") or models.get("kl_final_seed7") or build_model(MODEL_SPECS["kl_final"], device)
    crop_seg_model = None
    if args.auto_crop:
        from kl_common import load_seg_model
        crop_seg_model = load_seg_model("unetpp", KL_CKPT_DIR, device)

    causal_coefs = load_or_fit_causal_models()
    causal_forest = None
    if os.path.exists(CAUSAL_FOREST_PATH):
        with open(CAUSAL_FOREST_PATH, "rb") as f:
            causal_forest = pickle.load(f)
    injury_df = None
    if os.path.exists(INJURY_ESTIMATES_PATH):
        import pandas as pd
        injury_df = pd.read_csv(INJURY_ESTIMATES_PATH)

    def run_one(img_rgb, tag, age, bmi, famkr):
        grade, probs, avg_cam, _ = classify_rgb(img_rgb, models, members, thresholds, device)
        clahe224 = cv2.resize(apply_clahe(img_rgb), (IMG_SIZE, IMG_SIZE))
        neighbors = nearest_cases(clahe224, kl_final_for_neighbors, device)
        entropy, unc_thresh, is_uncertain = uncertainty_info(probs)
        report(tag, img_rgb, grade, probs, avg_cam, neighbors, entropy, unc_thresh, is_uncertain,
              model_label, members, args.out_dir, plt, age=age, bmi=bmi, famkr=famkr,
              causal_coefs=causal_coefs, causal_forest=causal_forest, injury_df=injury_df)

    def handle(path):
        age, bmi, famkr = get_patient_inputs(args)
        if args.auto_crop:
            for label, crop in prepare_xray(path, crop_seg_model, device):
                tag = f"{os.path.splitext(os.path.basename(path))[0]}_{label}"
                run_one(crop, tag, age, bmi, famkr)
                if args.show or interactive:
                    plt.show()
                plt.close("all")
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            run_one(img, path, age, bmi, famkr)
            if args.show or interactive:
                plt.show()
            plt.close("all")

    if not interactive:
        handle(args.image)
        return

    print("Choose an X-ray in the file browser. Close the result window to pick another; press Cancel to quit.")
    while True:
        path = pick_file()
        if not path:
            print("No file selected. Exiting.")
            break
        try:
            handle(path)
        except ValueError as e:
            print(e)
            continue


if __name__ == "__main__":
    main()
