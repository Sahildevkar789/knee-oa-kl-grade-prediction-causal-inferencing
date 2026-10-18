"""Converts an arbitrary knee X-ray into 224x224 crop(s) matching the training images.

Handles PNG/JPG/BMP/TIFF/DICOM, trims empty borders, splits bilateral (two-knee)
images, and locates the joint automatically with the U-Net++ segmentation model
(it was trained on full-leg X-rays, so it works well on uncropped inputs). Falls
back to an intensity heuristic, then to a manual click, if the mask is unusable.

CLI:
    python scripts/preprocess_xray.py input_path --out-dir outputs/preprocessed
"""
import os
import argparse
import cv2
import numpy as np
import torch

from kl_common import load_seg_model, SEG_SIZE, IMG_SIZE, MEAN, STD

CROP_SIZE = IMG_SIZE
SCALE_FACTOR = 1.15   # crop side = knee_width_at_joint * SCALE_FACTOR; calibrated against dataset/test crops


def load_any(path):
    """Loads PNG/JPG/BMP/TIFF/DICOM as a grayscale uint8 2D array."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        import pydicom
        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        arr = arr * slope + intercept

        wc, ww = ds.get("WindowCenter"), ds.get("WindowWidth")
        if wc is not None and ww is not None:
            wc = float(wc[0] if hasattr(wc, "__iter__") else wc)
            ww = float(ww[0] if hasattr(ww, "__iter__") else ww)
            arr = np.clip(arr, wc - ww / 2, wc + ww / 2)

        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        gray = (arr * 255).astype(np.uint8)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            gray = 255 - gray
        return gray

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read {path}")
    if img.ndim == 3:
        code = cv2.COLOR_BGRA2GRAY if img.shape[2] == 4 else cv2.COLOR_BGR2GRAY
        img = cv2.cvtColor(img, code)
    return img


def trim_borders(gray, bg_thresh=8, min_content_frac=0.02):
    """Removes large near-uniform white/black margins. Returns (trimmed, (x0,y0,x1,y1)) in original coords."""
    h, w = gray.shape
    corners = [gray[0, 0], gray[0, -1], gray[-1, 0], gray[-1, -1]]
    bg_is_white = np.mean(corners) > 127
    content_mask = (gray < 255 - bg_thresh) if bg_is_white else (gray > bg_thresh)

    cols = np.where(content_mask.any(axis=0))[0]
    rows = np.where(content_mask.any(axis=1))[0]
    if len(cols) == 0 or len(rows) == 0:
        return gray, (0, 0, w, h)

    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    if (x1 - x0) < w * min_content_frac or (y1 - y0) < h * min_content_frac:
        return gray, (0, 0, w, h)   # trimming would remove almost everything - something's off, skip it
    return gray[y0:y1, x0:x1], (x0, y0, x1, y1)


def split_bilateral(gray, aspect_thresh=1.35):
    """Splits into left/right halves if the image is much wider than tall (two knees side by side)."""
    h, w = gray.shape
    if w / h < aspect_thresh:
        return [("single", gray, (0, 0))]
    mid = w // 2
    return [("left", gray[:, :mid], (0, 0)), ("right", gray[:, mid:], (mid, 0))]


def run_seg_mask(gray, seg_model, device):
    """Runs the segmentation model on a grayscale image, returns a (SEG_SIZE,SEG_SIZE) probability map."""
    img3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    x = cv2.resize(img3, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
    x = (x - np.array(MEAN)) / np.array(STD)   # same ImageNet normalization the model was trained with
    x = torch.tensor(x.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        prob = torch.sigmoid(seg_model(x)).squeeze().cpu().numpy()
    return prob


def find_joint_from_mask(mask, mask_thresh=0.5, bone_thresh=0.1):
    """Finds the joint row/center/width from a bone-probability mask (femur above, tibia below, gap between).

    Locates the contiguous bone-containing region first, then searches for the dip
    only inside it - otherwise argmin latches onto the empty rows above/below the
    leg in frame instead of the actual joint gap.
    """
    binm = mask > mask_thresh
    h, w = binm.shape
    row_coverage = binm.sum(axis=1).astype(np.float32) / w
    smoothed = np.convolve(row_coverage, np.ones(5) / 5, mode="same")

    bone_rows = np.where(smoothed > bone_thresh)[0]
    if len(bone_rows) == 0:
        return None
    bone_top, bone_bottom = int(bone_rows[0]), int(bone_rows[-1])
    if bone_bottom - bone_top < h * 0.1:
        return None   # too small a bone region to be meaningful

    margin = max(5, int((bone_bottom - bone_top) * 0.15))
    search_lo, search_hi = bone_top + margin, bone_bottom - margin
    if search_hi <= search_lo:
        return None
    joint_row = int(np.argmin(smoothed[search_lo:search_hi])) + search_lo

    win = 15
    above = smoothed[max(bone_top, joint_row - win):joint_row]
    below = smoothed[joint_row + 1:min(bone_bottom + 1, joint_row + 1 + win)]
    if len(above) == 0 or len(below) == 0:
        return None
    joint_val = smoothed[joint_row]
    if above.max() < joint_val + 0.08 or below.max() < joint_val + 0.08:
        return None   # no clear two-bone separation - not confident this is a real joint gap

    band_half = 10
    r0, r1 = max(0, joint_row - band_half), min(h, joint_row + band_half)
    xs = np.where(binm[r0:r1, :].any(axis=0))[0]
    if len(xs) == 0:
        return None
    cx = int((xs[0] + xs[-1]) / 2)
    knee_width = int(xs[-1] - xs[0])
    if knee_width < w * 0.1:
        return None
    return joint_row, cx, knee_width


def intensity_width_at_row(gray, row, band_half=10):
    """Bone width at a given row via Otsu thresholding on raw pixels - independent of the
    segmentation mask, which is known to be unreliable on OAI's tight joint crops (the
    model was trained on CGMH's full-leg images, not this scale - see CLAUDE.md)."""
    h, w = gray.shape
    r0, r1 = max(0, row - band_half), min(h, row + band_half)
    band = gray[r0:r1, :]
    if band.size == 0:
        return 0
    thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    xs = np.where((band > thresh).any(axis=0))[0]
    return int(xs[-1] - xs[0]) if len(xs) else 0


def intensity_fallback(gray):
    """Darkest horizontal band in the central region, used when the mask approach is inconclusive.

    Returns None if the image has no meaningful horizontal structure at all (e.g. a mostly
    blank/featureless image, such as a screenshot with excess white margin) - argmin would
    otherwise just pick an arbitrary row with no real joint there.
    """
    h, w = gray.shape
    lo, hi = int(h * 0.25), int(h * 0.75)
    row_mean = gray[lo:hi, :].astype(np.float32).mean(axis=1)
    if row_mean.std() < 3.0:   # essentially flat - no identifiable joint dip
        return None
    joint_row = int(np.argmin(row_mean)) + lo
    cx = w // 2
    knee_width = int(w * 0.7)
    return joint_row, cx, knee_width


def click_fallback(gray, window_title):
    """Opens a window for the user to click the joint center. Returns (row, cx, default_width)."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    clicked = {}
    fig, ax = plt.subplots()
    fig.canvas.manager.set_window_title(window_title)
    ax.imshow(gray, cmap="gray")
    ax.set_title("Click the center of the knee joint (the dark gap between femur and tibia)")

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            clicked["x"], clicked["y"] = event.xdata, event.ydata
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    if "x" not in clicked:
        raise RuntimeError("No click received - cannot locate the joint.")
    cx, cy = int(clicked["x"]), int(clicked["y"])
    knee_width = int(min(gray.shape) * 0.6)
    return cy, cx, knee_width


def looks_already_cropped(gray, joint_row, cx, knee_width, trim_retained_frac):
    """Roughly square, joint near vertical center, bones filling most of the width, and
    barely any padding was trimmed off (real dataset crops have none at all - a full-leg
    image can coincidentally end up near-square/bone-filled after trimming its letterbox
    bars, so that alone isn't enough to tell the two apart)."""
    h, w = gray.shape
    aspect_ok = 0.8 <= (w / h) <= 1.25
    joint_centered = 0.2 <= (joint_row / h) <= 0.8
    bones_fill_width = knee_width >= w * 0.6
    barely_trimmed = trim_retained_frac >= 0.92
    return aspect_ok and joint_centered and bones_fill_width and barely_trimmed


def crop_square(gray, cy, cx, side):
    """Crops a `side`x`side` square centered on (cy, cx), padding with black if it runs off the edge."""
    h, w = gray.shape
    side = max(int(side), 16)
    half = side // 2
    x0, x1 = cx - half, cx + half
    y0, y1 = cy - half, cy + half
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x1 - w), max(0, y1 - h)
    if pad_l or pad_t or pad_r or pad_b:
        gray = cv2.copyMakeBorder(gray, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        x0, x1, y0, y1 = x0 + pad_l, x1 + pad_l, y0 + pad_t, y1 + pad_t
    crop = gray[y0:y1, x0:x1]
    box = (x0 - pad_l, y0 - pad_t, x1 - pad_l, y1 - pad_t)   # back in the un-padded coordinate space
    return crop, box


def _process_half(half_img, label, offset, seg_model, device, trim_retained_frac=1.0):
    """Runs the full joint-finding + crop pipeline on one knee. Returns a result dict."""
    hx0, hy0 = offset
    warning = None

    mask = run_seg_mask(half_img, seg_model, device)
    found = find_joint_from_mask(mask)
    method = "mask"
    if found is None:
        found = intensity_fallback(half_img)
        method = "intensity"
    if found is None:
        found = click_fallback(half_img, f"Locate the joint - {label}")
        method = "click"

    mh, mw = mask.shape
    sy, sx = half_img.shape[0] / mh, half_img.shape[1] / mw
    joint_row_m, cx_m, knee_width_m = found
    if method == "mask":
        joint_row = int(joint_row_m * sy)
        cx = int(cx_m * sx)
        knee_width = int(knee_width_m * sx)
    else:
        joint_row, cx, knee_width = joint_row_m, cx_m, knee_width_m

    width_for_check = intensity_width_at_row(half_img, joint_row)
    already_cropped = looks_already_cropped(half_img, joint_row, cx, width_for_check, trim_retained_frac)
    if already_cropped:
        crop = half_img
        box = (0, 0, half_img.shape[1], half_img.shape[0])
    else:
        side = knee_width * SCALE_FACTOR
        crop, box = crop_square(half_img, joint_row, cx, side)

    if crop.shape[0] < CROP_SIZE or crop.shape[1] < CROP_SIZE:
        warning = (f"Source crop ({crop.shape[1]}x{crop.shape[0]}) is smaller than "
                   f"{CROP_SIZE}x{CROP_SIZE} - upscaling, some detail will be lost.")

    resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                          interpolation=cv2.INTER_AREA if crop.shape[0] > CROP_SIZE else cv2.INTER_CUBIC)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)

    box_in_original = (box[0] + hx0, box[1] + hy0, box[2] + hx0, box[3] + hy0)
    return {"label": label, "image": rgb, "box": box_in_original,
            "already_cropped": already_cropped, "method": method, "warning": warning}


def prepare_xray(image_path, seg_model=None, device=None, ckpt_dir="outputs/checkpoints", return_debug=False):
    """Converts a knee X-ray into one or more 224x224 crops matching the training data.

    Returns a list of (label, image) tuples, where `image` is a 224x224x3 uint8 RGB array.
    Pass `return_debug=True` to instead get the full per-crop info (box, method, warning).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if seg_model is None:
        seg_model = load_seg_model("unetpp", ckpt_dir, device)

    gray = load_any(image_path)
    trimmed, (tx0, ty0, tx1, ty1) = trim_borders(gray)
    trim_retained_frac = ((tx1 - tx0) * (ty1 - ty0)) / (gray.shape[0] * gray.shape[1])
    halves = split_bilateral(trimmed)

    results = []
    for label, half_img, (hx0, hy0) in halves:
        r = _process_half(half_img, label, (tx0 + hx0, ty0 + hy0), seg_model, device, trim_retained_frac)
        results.append(r)

    if return_debug:
        return results
    return [(r["label"], r["image"]) for r in results]


def draw_debug(gray, results, title):
    """Builds a figure of the original image with each crop box drawn on it."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(gray, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    colors = ["#D85A30", "#3080D8"]
    for i, r in enumerate(results):
        x0, y0, x1, y1 = r["box"]
        c = colors[i % len(colors)]
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=c, linewidth=2))
        tag = f"{r['label']} ({r['method']}{', already cropped' if r['already_cropped'] else ''})"
        ax.text(x0, max(0, y0 - 8), tag, color=c, fontsize=9, weight="bold")
    plt.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="Path to a knee X-ray (PNG/JPG/BMP/TIFF/DICOM)")
    ap.add_argument("--out-dir", default="outputs/preprocessed")
    ap.add_argument("--ckpt-dir", default="outputs/checkpoints")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_model = load_seg_model("unetpp", args.ckpt_dir, device)

    results = prepare_xray(args.image, seg_model, device, args.ckpt_dir, return_debug=True)
    gray = load_any(args.image)

    stem = os.path.splitext(os.path.basename(args.image))[0]
    for r in results:
        out_path = os.path.join(args.out_dir, f"{stem}_{r['label']}.png")
        cv2.imwrite(out_path, cv2.cvtColor(r["image"], cv2.COLOR_RGB2BGR))
        print(f"{r['label']}: method={r['method']} already_cropped={r['already_cropped']} -> {out_path}")
        if r["warning"]:
            print(f"  WARNING: {r['warning']}")

    import matplotlib
    matplotlib.use("Agg")
    fig = draw_debug(gray, results, f"Crop regions - {os.path.basename(args.image)}")
    debug_path = os.path.join(args.out_dir, f"{stem}_debug.png")
    fig.savefig(debug_path, dpi=120)
    print(f"Debug image saved to {debug_path}")


if __name__ == "__main__":
    main()
