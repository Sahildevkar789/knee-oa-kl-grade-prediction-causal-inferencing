"""Predict the KL grade of knee X-rays.

Run without an image path to pick files with a file browser (repeats until you press Cancel):
    python scripts/predict.py
    python scripts/predict.py --run unetpp

Or give a path directly:
    python scripts/predict.py path/to/xray.png --show
"""
import os
import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

from kl_common import (build_classifier, load_seg_model, MEAN, STD, IMG_SIZE, SEG_SIZE, KL_TEXT)
from preprocess_xray import prepare_xray


def to_tensor(rgb, size, device):
    x = (cv2.resize(rgb, (size, size)) / 255.0 - np.array(MEAN)) / np.array(STD)
    return torch.tensor(x.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(device)


def load_models(run, ckpt_dir, device):
    """Loads the segmentation model (only needed for the unet/unetpp mask channel) and the classifier once."""
    seg_model = load_seg_model("unet" if run == "unet" else "unetpp", ckpt_dir, device) if run != "baseline" else None
    classifier = build_classifier(3 if run == "baseline" else 4, pretrained=False).to(device)
    classifier.load_state_dict(torch.load(os.path.join(ckpt_dir, f"kl_{run}_best.pt"), map_location=device))
    classifier.eval()
    return seg_model, classifier


def classify_rgb(img, seg_model, classifier, run, device):
    """Classifies an already-loaded RGB image. Returns the 5 class probabilities."""
    with torch.no_grad():
        x = to_tensor(img, IMG_SIZE, device)
        if run != "baseline":
            prob_map = torch.sigmoid(seg_model(F.interpolate(x, size=(SEG_SIZE, SEG_SIZE), mode="bilinear", align_corners=False)))
            x = torch.cat([x, F.interpolate(prob_map, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)], dim=1)
        probs = torch.softmax(classifier(x), dim=1).squeeze().cpu().numpy()
    return probs


def predict_image(image_path, seg_model, classifier, run, device):
    """Loads an image from disk and returns it (RGB) plus the 5 class probabilities."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {image_path}. Use a PNG or JPG file.")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    probs = classify_rgb(img, seg_model, classifier, run, device)
    return img, probs


def predict_image_auto_crop(image_path, crop_seg_model, seg_model, classifier, run, device):
    """Auto-crops (bilateral-aware) then classifies each knee. Returns a list of (label, crop_rgb, probs)."""
    crops = prepare_xray(image_path, crop_seg_model, device)
    return [(label, crop, classify_rgb(crop, seg_model, classifier, run, device)) for label, crop in crops]


def report(image_path, img, probs, out_dir, plt):
    """Prints the prediction and builds + saves the result figure."""
    grade = int(probs.argmax())
    print(f"\nImage: {image_path}")
    print(f"Predicted KL grade: {grade} - {KL_TEXT[grade]} ({probs[grade] * 100:.1f}% confidence)")
    for g in range(5):
        print(f"  KL{g}: {probs[g] * 100:5.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.canvas.manager.set_window_title(f"KL grade prediction - {os.path.basename(image_path)}")
    axes[0].imshow(img); axes[0].set_title("Input X-ray"); axes[0].axis("off")
    colors = ["#888888"] * 5
    colors[grade] = "#D85A30"
    axes[1].bar([f"KL{g}" for g in range(5)], probs * 100, color=colors)
    axes[1].set_ylabel("Probability (%)"); axes[1].set_ylim(0, 100)
    axes[1].set_title(f"Predicted: KL{grade} ({KL_TEXT[grade]})")
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
    ap.add_argument("--run", choices=["baseline", "unet", "unetpp"], default="baseline",
                    help="Which trained classifier to use")
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models ({args.run}) on {device}...")
    seg_model, classifier = load_models(args.run, args.ckpt_dir, device)
    crop_seg_model = load_seg_model("unetpp", args.ckpt_dir, device) if args.auto_crop else None

    def handle(path):
        if args.auto_crop:
            for label, crop, probs in predict_image_auto_crop(path, crop_seg_model, seg_model, classifier, args.run, device):
                tag = f"{os.path.splitext(os.path.basename(path))[0]}_{label}"
                report(tag, crop, probs, args.out_dir, plt)
                if args.show or interactive:
                    plt.show()
                plt.close("all")
        else:
            img, probs = predict_image(path, seg_model, classifier, args.run, device)
            report(path, img, probs, args.out_dir, plt)
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
