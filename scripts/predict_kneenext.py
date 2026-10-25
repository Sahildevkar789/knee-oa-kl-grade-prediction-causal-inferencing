"""Predicts the KL grade of knee X-rays using KneeNeXt (the experimental knee-specific
ConvNeXt variant - see scripts/kneenext.py) instead of kl_final. Same behavior as
predict.py: file-browser mode, direct-path mode, patient-detail prompts, Grad-CAM, and the
causal risk panel - plus KneeNeXt's zoom (joint-line crop) view, since that view is central
to its architecture. predict.py itself and its default model (kl_final) are untouched; this
is a separate, additive script.

Run without an image path to pick files with a file browser (repeats until you press Cancel):
    python scripts/predict_kneenext.py
Or give a path directly:
    python scripts/predict_kneenext.py path/to/xray.png --show
"""
import os
import json
import argparse
import textwrap
import cv2
import torch
import matplotlib

from kl_common import KL_TEXT, load_seg_model
from kneenext import build_kneenext, prepare_views, grad_cam_view, IMG_SIZE
from preprocess_xray import prepare_xray
from causal_model import load_or_fit_causal_models
from predict import get_patient_inputs, causal_panel_lines, pick_file, coral_display_probs

KNEENEXT_CKPT_DIR = "outputs/kneenext/checkpoints"


def load_model(ckpt_dir, device):
    with open(os.path.join(ckpt_dir, "kneenext_best_config.json")) as f:
        info = json.load(f)
    model = build_kneenext(info["config"], "outputs/checkpoints", device, init_from=None)
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "kneenext_best.pt"), map_location=device))
    model.eval()
    return model, info


def classify_rgb(img, model, device):
    """Classifies an already-loaded RGB image. Returns (grade, 5-length display probs,
    global-view Grad-CAM, global view image, zoom view image, joint-detector fallback flag)."""
    global_img, zoom_img, g_t, z_t, g_prior, z_prior, fallback = prepare_views(img)
    g = g_t.unsqueeze(0).to(device)
    z = z_t.unsqueeze(0).to(device) if model.use_zoom else None
    gp = torch.from_numpy(g_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
    zp = torch.from_numpy(z_prior).unsqueeze(0).to(device) if model.use_joint_cbam else None
    feature_module = model.features[model.STAGE4_IDX - 1]
    cam, thresh_probs = grad_cam_view(model, g, z if z is not None else g, gp, zp, feature_module)
    grade = int((thresh_probs > 0.5).sum())
    probs = coral_display_probs(thresh_probs)
    return grade, probs, cam, global_img, zoom_img, fallback


def predict_image(image_path, model, device):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {image_path}. Use a PNG or JPG file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return classify_rgb(img, model, device)


def predict_image_auto_crop(image_path, crop_seg_model, model, device):
    """Auto-crops (bilateral-aware) then classifies each knee. Returns a list of
    (label, grade, probs, cam, global_img, zoom_img, fallback)."""
    crops = prepare_xray(image_path, crop_seg_model, device)
    return [(label, *classify_rgb(crop, model, device)) for label, crop in crops]


def report(image_path, grade, probs, cam, global_img, zoom_img, fallback, out_dir, plt,
          age=None, bmi=None, famkr=None, causal_coefs=None):
    """Prints the prediction and builds + saves the result figure: global view, zoom
    (joint-line crop) view, Grad-CAM overlay, probability bar chart, causal risk panel."""
    print(f"\nImage: {image_path}")
    if fallback:
        print("  (joint-line detector fell back to the vertical center for the zoom crop)")
    print(f"Predicted KL grade: {grade} - {KL_TEXT[grade]} ({probs[grade] * 100:.1f}% confidence)")
    for g in range(5):
        print(f"  KL{g}: {probs[g] * 100:5.1f}%")

    causal_lines = causal_panel_lines(age, bmi, famkr, causal_coefs)
    print()
    for line in causal_lines:
        print(f"  {line}")

    fig, axes = plt.subplots(1, 5, figsize=(27.5, 5))
    axes[0].imshow(global_img); axes[0].set_title("Global view"); axes[0].axis("off")
    axes[1].imshow(zoom_img); axes[1].set_title("Zoom view (joint-line crop)"); axes[1].axis("off")

    cam_up = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    axes[2].imshow(global_img, cmap="gray"); axes[2].imshow(cam_up, cmap="jet", alpha=0.45)
    axes[2].set_title("Grad-CAM (global view)"); axes[2].axis("off")

    prob_ax, causal_ax = axes[3], axes[4]
    causal_ax.axis("off")
    causal_wrapped = "\n".join(textwrap.fill(line, 38, subsequent_indent="  ") for line in causal_lines)
    causal_ax.text(0.0, 1.0, causal_wrapped, va="top", ha="left", fontsize=7.8, family="monospace",
                   transform=causal_ax.transAxes)
    causal_ax.set_title("Causal risk panel", fontsize=10)

    fig.canvas.manager.set_window_title(f"KneeNeXt prediction - {os.path.basename(image_path)}")
    colors = ["#888888"] * 5
    colors[grade] = "#D85A30"
    prob_ax.bar([f"KL{g}" for g in range(5)], probs * 100, color=colors)
    prob_ax.set_ylabel("Probability (%)"); prob_ax.set_ylim(0, 100)
    prob_ax.set_title(f"Predicted: KL{grade} ({KL_TEXT[grade]})")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_kneenext_prediction.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved result image to {out_path}")
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="Path to a knee X-ray. Leave out to choose with a file browser.")
    ap.add_argument("--age", type=float, default=None, help="Patient age - for the causal risk panel. Skips the terminal prompt.")
    ap.add_argument("--bmi", type=float, default=None, help="Patient BMI - for the causal risk panel. Skips the terminal prompt.")
    ap.add_argument("--family-history", choices=["yes", "no"], default=None,
                    help="Family history of knee replacement - for the causal risk panel only. Skips the terminal prompt.")
    ap.add_argument("--ckpt-dir", default=KNEENEXT_CKPT_DIR)
    ap.add_argument("--base-ckpt-dir", default="outputs/checkpoints", help="Where kl_final_best.pt and the segmentation checkpoints live")
    ap.add_argument("--out-dir", default="outputs/predictions")
    ap.add_argument("--show", action="store_true", help="Open the result window (always on in file-browser mode)")
    ap.add_argument("--auto-crop", action="store_true",
                    help="Auto-crop the input to a 224x224 joint crop before classifying")
    args = ap.parse_args()

    interactive = args.image is None
    if not interactive and not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading KneeNeXt on {device}...")
    model, info = load_model(args.ckpt_dir, device)
    print(f"KneeNeXt config: {info['config']} (seed {info['seed']}, val QWK {info['val_qwk']:.4f})")
    crop_seg_model = load_seg_model("unetpp", args.base_ckpt_dir, device) if args.auto_crop else None
    causal_coefs = load_or_fit_causal_models()

    def handle(path):
        age, bmi, famkr = get_patient_inputs(args)
        if args.auto_crop:
            for label, grade, probs, cam, g_img, z_img, fb in predict_image_auto_crop(path, crop_seg_model, model, device):
                tag = f"{os.path.splitext(os.path.basename(path))[0]}_{label}"
                report(tag, grade, probs, cam, g_img, z_img, fb, args.out_dir, plt,
                      age=age, bmi=bmi, famkr=famkr, causal_coefs=causal_coefs)
                if args.show or interactive:
                    plt.show()
                plt.close("all")
        else:
            grade, probs, cam, g_img, z_img, fb = predict_image(path, model, device)
            report(path, grade, probs, cam, g_img, z_img, fb, args.out_dir, plt,
                  age=age, bmi=bmi, famkr=famkr, causal_coefs=causal_coefs)
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
