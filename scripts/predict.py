"""Predict the KL grade of knee X-rays.

Run without an image path to pick files with a file browser (repeats until you press Cancel):
    python scripts/predict.py
    python scripts/predict.py --run unetpp

Or give a path directly:
    python scripts/predict.py path/to/xray.png --show

Pass --age/--bmi to additionally use Hybrid 2's tabular branch - though per Phase 3's results
(val QWK 0.8170 vs kl_final's 0.8416), Hybrid 2 never actually wins, so kl_final is used
either way and a note explains why (see HYBRID2_BEAT_FINAL below).
"""
import os
import argparse
import textwrap
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

from kl_common import (build_classifier, build_hybrid2, load_seg_model, apply_clahe, coral_probs_to_grade,
                       grad_cam, MEAN, STD, IMG_SIZE, SEG_SIZE, KL_TEXT)
from preprocess_xray import prepare_xray
from measure_jsw import measure_jsw_array, assign_compartments
from causal_model import load_or_fit_causal_regression, expected_kl

RUNS = ["final", "baseline", "unet", "unetpp"]

# Phase 3's result: Hybrid 2's 2-seed mean val QWK (0.8170) never beat kl_final's (0.8416).
# --age/--bmi are still accepted (the brief asks for this switch logic to exist), but since
# the condition is false, kl_final is used regardless - this constant documents why, rather
# than silently ignoring the flags.
HYBRID2_BEAT_FINAL = False
HYBRID2_TABULAR_COLS = ["V00AGE", "P01BMI"]   # sex omitted - no sex column exists in the dataset (see Phase 3 log)
HYBRID2_TAB_MEAN = np.array([61.0, 28.58], dtype=np.float32)   # training-set stats, from Phase 3's log
HYBRID2_TAB_STD = np.array([9.24, 4.88], dtype=np.float32)

CAUSAL_CAVEAT = ("Population-level estimate from OAI data under the causal model's assumptions; "
                 "not a diagnosis or individual prediction.")


def to_tensor(rgb, size, device):
    x = (cv2.resize(rgb, (size, size)) / 255.0 - np.array(MEAN)) / np.array(STD)
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)


def coral_display_probs(thresh_probs):
    """Converts CORAL's 4 'P(grade>k)' outputs into a 5-length per-grade distribution for display
    (p0=1-P(>0), pk=P(>k-1)-P(>k), p4=P(>3)), clipped and renormalized in case thresholds aren't
    perfectly monotonic."""
    p = np.concatenate([[1.0], thresh_probs]) - np.concatenate([thresh_probs, [0.0]])
    p = np.clip(p, 0, None)
    return p / p.sum()


def infer_side(path):
    """Best-effort knee side from a dataset-style filename (...L.png / ...R.png); None if unclear."""
    stem = os.path.splitext(os.path.basename(path))[0]
    if stem.endswith("L"):
        return "L"
    if stem.endswith("R"):
        return "R"
    return None


def prompt_optional(prompt_text, parse=float):
    """Asks in the terminal, returns None on a blank/unparseable answer (the skip path) or on
    EOF (so this also behaves when stdin isn't interactive, e.g. automated testing)."""
    try:
        raw = input(f"{prompt_text} (press Enter to skip): ").strip()
    except EOFError:
        raw = ""
    if not raw:
        return None
    try:
        return parse(raw)
    except ValueError:
        print(f"  Couldn't parse '{raw}', skipping.")
        return None


def get_patient_inputs(args):
    """Age/BMI/family-history for the causal risk panel only - never fed to the classifier
    (the predicted KL grade always comes from the image alone). Values given via --age/--bmi/
    --family-history skip their prompt; everything else is asked for, Enter to skip."""
    age = args.age if args.age is not None else prompt_optional("Patient age")
    bmi = args.bmi if args.bmi is not None else prompt_optional("Patient BMI")
    if args.family_history is not None:
        famkr = {"yes": 1, "no": 0}[args.family_history]
    else:
        famkr = prompt_optional("Family history of knee replacement? (yes/no)", parse=str)
        if famkr is not None:
            famkr = {"y": 1, "yes": 1, "1": 1, "n": 0, "no": 0, "0": 0}.get(famkr.strip().lower())
            if famkr is None:
                print("  Couldn't parse that as yes/no, skipping.")
    return age, bmi, famkr


def causal_panel_lines(age, bmi, famkr, coefs):
    """Lines for both the terminal printout and the result figure's causal risk panel. Built
    once so the two stay identical (Phase 5's population finding, plus a patient-specific
    estimate from the fitted regression-adjustment model if age and BMI were given)."""
    lines = [
        "Causal risk panel (Phase 5 finding - population-level, not derived from this image):",
        "Obesity (BMI>=30) -> KL grade, adjusting for age + family history of knee replacement:",
        "  Population ATE: +0.48 to +0.58 KL grade points (linear regression / propensity score matching).",
    ]
    if age is not None and bmi is not None:
        obese = bmi >= 30
        lines.append(f"  This patient: age {age:.0f}, BMI {bmi:.1f} -> "
                     f"{'obese' if obese else 'not obese'} (BMI>=30 threshold).")
        if famkr is None:
            famkr_used = coefs["most_common_famkr"]
            lines.append(f"  Family history skipped - using the population's most common value "
                         f"({'yes' if famkr_used else 'no'}).")
        else:
            famkr_used = famkr
        kl_obese = expected_kl(coefs, age, famkr_used, 1)
        kl_not = expected_kl(coefs, age, famkr_used, 0)
        lines.append(f"  Fitted model at this age/family history: expected KL grade "
                     f"{kl_not:.2f} if not obese vs {kl_obese:.2f} if obese "
                     f"(+{kl_obese - kl_not:.2f}).")
    else:
        lines.append("  Age/BMI not given - showing only the population-level finding above.")
    lines.append(f"  Caveat: {CAUSAL_CAVEAT}")
    return lines


def load_models(run, ckpt_dir, device, use_hybrid2=False):
    """Loads the segmentation model (only needed for the unet/unetpp mask channel) and the classifier once."""
    if use_hybrid2:
        seg_model = None
        classifier = build_hybrid2(ckpt_dir, device, tabular_dim=len(HYBRID2_TABULAR_COLS), init_from=None)
        classifier.load_state_dict(torch.load(os.path.join(ckpt_dir, "kl_hybrid2_best.pt"), map_location=device))
    elif run == "final":
        seg_model = None
        classifier = build_classifier("convnext_tiny", 3, pretrained=False, out_dim=4, attention="cbam").to(device)
        classifier.load_state_dict(torch.load(os.path.join(ckpt_dir, "kl_final_best.pt"), map_location=device))
    else:
        seg_model = load_seg_model("unet" if run == "unet" else "unetpp", ckpt_dir, device) if run != "baseline" else None
        classifier = build_classifier("efficientnet_b0", 3 if run == "baseline" else 4, pretrained=False).to(device)
        classifier.load_state_dict(torch.load(os.path.join(ckpt_dir, f"kl_{run}_best.pt"), map_location=device))
    classifier.eval()
    return seg_model, classifier


def classify_rgb(img, seg_model, classifier, run, device, tab=None):
    """Classifies an already-loaded RGB image. Returns (predicted grade, 5-length display probabilities,
    cam_or_None). cam is only computed for the final/hybrid2 (CORAL, single/paired-input) models."""
    uses_coral = run == "final" or tab is not None
    if uses_coral:
        img = apply_clahe(img)
    x = to_tensor(img, IMG_SIZE, device)

    if tab is not None:
        tab_t = torch.tensor(tab, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = classifier(x, tab_t)
        cam = None   # Grad-CAM not wired up for the two-input fusion model's forward signature
    elif run in ("unet", "unetpp"):
        with torch.no_grad():
            prob_map = torch.sigmoid(seg_model(F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)))
            x_masked = torch.cat([x, F.interpolate(prob_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)], dim=1)
            logits = classifier(x_masked)
        cam = None
    elif run == "final":
        cam, _ = grad_cam(classifier, x, classifier.features)
        with torch.no_grad():
            logits = classifier(x)
    else:
        with torch.no_grad():
            logits = classifier(x)
        cam = None

    if uses_coral:
        probs_t = torch.sigmoid(logits.detach())
        thresh_probs = probs_t.squeeze().cpu().numpy()
        grade = int(coral_probs_to_grade(probs_t).item())
        probs = coral_display_probs(thresh_probs)
    else:
        probs = torch.softmax(logits.detach(), dim=1).squeeze().cpu().numpy()
        grade = int(probs.argmax())
    return grade, probs, cam


def predict_image(image_path, seg_model, classifier, run, device, tab=None):
    """Loads an image from disk and returns it (RGB), the predicted grade, display probabilities, and CAM."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {image_path}. Use a PNG or JPG file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grade, probs, cam = classify_rgb(img, seg_model, classifier, run, device, tab=tab)
    return img, grade, probs, cam


def predict_image_auto_crop(image_path, crop_seg_model, seg_model, classifier, run, device, tab=None):
    """Auto-crops (bilateral-aware) then classifies each knee. Returns a list of (label, crop_rgb, grade, probs, cam)."""
    crops = prepare_xray(image_path, crop_seg_model, device)
    return [(label, crop, *classify_rgb(crop, seg_model, classifier, run, device, tab=tab)) for label, crop in crops]


def report(image_path, img, grade, probs, out_dir, plt, cam=None, side=None,
          age=None, bmi=None, famkr=None, causal_coefs=None):
    """Prints the prediction and builds + saves the result figure. With cam given (final/hybrid2
    runs), shows input/Grad-CAM/JSW/probabilities (4 panels); otherwise input/probabilities (2).
    Either way, one more panel is appended: the causal risk panel (Phase 5's population finding,
    plus a patient-specific estimate if age/bmi were given). age/bmi/famkr are for this panel
    only - the predicted grade above always comes from the image alone."""
    print(f"\nImage: {image_path}")
    print(f"Predicted KL grade: {grade} - {KL_TEXT[grade]} ({probs[grade] * 100:.1f}% confidence)")
    for g in range(5):
        print(f"  KL{g}: {probs[g] * 100:5.1f}%")

    jsw = None
    if cam is not None:
        jsw = measure_jsw_array(img)
        comp = assign_compartments(side, jsw) if side else {
            "medial_min_jsw": None, "lateral_min_jsw": None, "compartment_label": "unknown"}
        if comp["medial_min_jsw"] is not None:
            print(f"  JSW (normalized by joint width) - medial min: {comp['medial_min_jsw']:.3f}, "
                 f"lateral min: {comp['lateral_min_jsw']:.3f}")
        else:
            print(f"  JSW (side unknown - labeled left/right of image) - left min: {jsw['left_min_jsw']:.3f}, "
                 f"right min: {jsw['right_min_jsw']:.3f}")

    causal_lines = causal_panel_lines(age, bmi, famkr, causal_coefs)
    print()
    for line in causal_lines:
        print(f"  {line}")

    n_panels = 3 if cam is None else 5
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5 * n_panels, 5))
    axes[0].imshow(img); axes[0].set_title("Input X-ray"); axes[0].axis("off")

    if cam is None:
        prob_ax = axes[1]
        causal_ax = axes[2]
    else:
        cam_up = cv2.resize(cam, (img.shape[1], img.shape[0]))
        axes[1].imshow(img, cmap="gray"); axes[1].imshow(cam_up, cmap="jet", alpha=0.45)
        axes[1].set_title("Grad-CAM"); axes[1].axis("off")

        axes[2].imshow(img)
        axes[2].axhline(jsw["joint_row"], color="yellow", linestyle="--", linewidth=1, alpha=0.7)
        for c in jsw["left_cols"][::3]:
            axes[2].plot(c, jsw["joint_row"], ".", color="lime", markersize=2)
        for c in jsw["right_cols"][::3]:
            axes[2].plot(c, jsw["joint_row"], ".", color="orange", markersize=2)
        if comp["medial_min_jsw"] is not None:
            jsw_label = f"medial min={comp['medial_min_jsw']:.3f}  lateral min={comp['lateral_min_jsw']:.3f}"
        else:
            jsw_label = f"left min={jsw['left_min_jsw']:.3f}  right min={jsw['right_min_jsw']:.3f} (side unknown)"
        axes[2].set_title(f"JSW\n{jsw_label}", fontsize=10); axes[2].axis("off")
        prob_ax = axes[3]
        causal_ax = axes[4]

    causal_ax.axis("off")
    causal_wrapped = "\n".join(textwrap.fill(line, 38, subsequent_indent="  ") for line in causal_lines)
    causal_ax.text(0.0, 1.0, causal_wrapped, va="top", ha="left", fontsize=7.8, family="monospace",
                   transform=causal_ax.transAxes)
    causal_ax.set_title("Causal risk panel", fontsize=10)

    fig.canvas.manager.set_window_title(f"KL grade prediction - {os.path.basename(image_path)}")
    colors = ["#888888"] * 5
    colors[grade] = "#D85A30"
    prob_ax.bar([f"KL{g}" for g in range(5)], probs * 100, color=colors)
    prob_ax.set_ylabel("Probability (%)"); prob_ax.set_ylim(0, 100)
    prob_ax.set_title(f"Predicted: KL{grade} ({KL_TEXT[grade]})")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(image_path))[0] + "_prediction.png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved result image to {out_path}")
    return fig


def pick_file():
    """Opens a file browser and returns the chosen image path, or None if cancelled."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)   # bring the dialog to the front
    path = filedialog.askopenfilename(
        title="Select a knee X-ray",
        filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
    )
    root.destroy()
    return path or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", nargs="?", help="Path to a knee X-ray. Leave out to choose with a file browser.")
    ap.add_argument("--run", choices=RUNS, default="final",
                    help="Which trained classifier to use (final = the locked ConvNeXt-tiny+CLAHE+CBAM+CORAL model)")
    ap.add_argument("--age", type=float, default=None,
                    help="Patient age - for the causal risk panel; with --bmi, also considers Hybrid 2 (see module docstring). Skips the terminal prompt.")
    ap.add_argument("--bmi", type=float, default=None,
                    help="Patient BMI - for the causal risk panel; with --age, also considers Hybrid 2. Skips the terminal prompt.")
    ap.add_argument("--family-history", choices=["yes", "no"], default=None,
                    help="Family history of knee replacement - for the causal risk panel only, never fed to the classifier. Skips the terminal prompt.")
    ap.add_argument("--sex", choices=["M", "F"], default=None, help="Accepted for forward-compatibility; not used by any current model (no sex data available)")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    ap.add_argument("--out-dir", default="outputs/predictions")
    ap.add_argument("--show", action="store_true", help="Open the result window (always on in file-browser mode)")
    ap.add_argument("--auto-crop", action="store_true",
                     help="Auto-crop the input to a 224x224 joint crop before classifying "
                          "(handles full-leg X-rays, bilateral images, arbitrary sizes)")
    args = ap.parse_args()

    interactive = args.image is None
    if not interactive and not args.show:
        matplotlib.use("Agg")   # save to file only, no pop-up window
    import matplotlib.pyplot as plt

    use_hybrid2 = args.age is not None and args.bmi is not None and HYBRID2_BEAT_FINAL
    if args.age is not None and args.bmi is not None and not HYBRID2_BEAT_FINAL:
        print("Note: --age/--bmi given, but Hybrid 2 didn't beat kl_final on validation "
             f"(0.8170 vs 0.8416) - using kl_final (--run {args.run}) regardless.")
    tab = None
    if use_hybrid2:
        tab = (np.array([args.age, args.bmi], dtype=np.float32) - HYBRID2_TAB_MEAN) / HYBRID2_TAB_STD

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models ({'hybrid2' if use_hybrid2 else args.run}) on {device}...")
    seg_model, classifier = load_models(args.run, args.ckpt_dir, device, use_hybrid2=use_hybrid2)
    crop_seg_model = load_seg_model("unetpp", args.ckpt_dir, device) if args.auto_crop else None
    causal_coefs = load_or_fit_causal_regression()

    def handle(path):
        # Age/BMI/family history are for the causal risk panel only - asked here (or taken from
        # --age/--bmi/--family-history, which skip the prompt) per image, never fed to the
        # classifier above: the predicted grade always comes from the image alone.
        age, bmi, famkr = get_patient_inputs(args)
        if args.auto_crop:
            for label, crop, grade, probs, cam in predict_image_auto_crop(path, crop_seg_model, seg_model, classifier, args.run, device, tab=tab):
                tag = f"{os.path.splitext(os.path.basename(path))[0]}_{label}"
                side = label if label in ("left", "right") else infer_side(path)
                report(tag, crop, grade, probs, args.out_dir, plt, cam=cam, side=side,
                      age=age, bmi=bmi, famkr=famkr, causal_coefs=causal_coefs)
                if args.show or interactive:
                    plt.show()
                plt.close("all")
        else:
            img, grade, probs, cam = predict_image(path, seg_model, classifier, args.run, device, tab=tab)
            report(path, img, grade, probs, args.out_dir, plt, cam=cam, side=infer_side(path),
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
