"""Measures joint space width (JSW) from knee X-rays using an intensity-based method on the
CLAHE-enhanced image. For each image: locate the joint line (darkest horizontal band near the
vertical center), then for each column scan up/down from that line to find the femur's lower
edge and the tibia's upper edge - the gap between them is JSW. The central ~20% of the joint
width is skipped (tibial spines/intercondylar eminence, unreliable here), and the rest is split
into two compartments reported as min/mean JSW, normalized by joint width.

CLI:
    python scripts/measure_jsw.py --out data/jsw_features.csv
"""
import os
import argparse
import cv2
import numpy as np
import pandas as pd

from kl_common import apply_clahe

CENTRAL_SKIP_FRAC = 0.20   # fraction of joint width skipped around the center (tibial spines)
SCAN_FRAC = 0.18           # per-column up/down scan window, as a fraction of image height
LOCAL_SEARCH_FRAC = 0.10   # per-column local joint-line search radius around the global row,
                           # as a fraction of image height - the joint line is rarely perfectly
                           # horizontal (limb tilt, X-ray angle), so anchoring every column's
                           # scan to one single global row made most columns spuriously read
                           # zero gap (see Phase 2 log) once checked across the full dataset


def find_joint_row(gray):
    """Darkest horizontal band in the central region of the image."""
    h, w = gray.shape
    lo, hi = int(h * 0.25), int(h * 0.75)
    row_mean = gray[lo:hi, :].astype(np.float32).mean(axis=1)
    return int(np.argmin(row_mean)) + lo


def joint_extent(gray, joint_row, threshold, band_half=10):
    """Horizontal bone extent (left_x, right_x) in a band around the joint row, via Otsu threshold."""
    h, w = gray.shape
    r0, r1 = max(0, joint_row - band_half), min(h, joint_row + band_half)
    band = gray[r0:r1, :]
    cols = np.where((band > threshold).any(axis=0))[0]
    if len(cols) == 0:
        return 0, w - 1
    return int(cols[0]), int(cols[-1])


def column_jsw(gray, col, joint_row, threshold, max_scan, search_radius):
    """Finds this column's own local joint line (darkest point within search_radius of the
    global joint_row, allowing for normal joint-line tilt) then scans up/down from there for
    the femur's lower edge and the tibia's upper edge. Returns None if no edge was found within
    the scan window in either direction - a measurement failure (bone curving away near a
    compartment's outer edge), not a genuine zero-width reading; must not be conflated with a
    real narrow/bone-on-bone gap (femur_edge and tibia_edge both found, close together)."""
    h = gray.shape[0]
    r0, r1 = max(0, joint_row - search_radius), min(h, joint_row + search_radius)
    local_row = int(np.argmin(gray[r0:r1, col].astype(np.float32))) + r0

    femur_edge = None
    for dy in range(max_scan + 1):
        r = local_row - dy
        if r < 0:
            break
        if gray[r, col] > threshold:
            femur_edge = r
            break
    tibia_edge = None
    for dy in range(max_scan + 1):
        r = local_row + dy
        if r >= h:
            break
        if gray[r, col] > threshold:
            tibia_edge = r
            break
    if femur_edge is None or tibia_edge is None:
        return None
    return max(0.0, tibia_edge - femur_edge)


def measure_jsw(image_path):
    """Loads an image from disk and measures JSW on it - see measure_jsw_array for the core logic."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read {image_path}")
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return measure_jsw_array(rgb)


def measure_jsw_array(rgb):
    """Returns a dict of raw (pixel) and normalized min/mean JSW for the left/right halves of
    the joint (medial/lateral assignment is done by the caller, which knows the knee side).
    rgb: an already-loaded RGB uint8 array (e.g. from predict.py's in-memory auto-crop output,
    which has no file path to re-read from)."""
    clahe_rgb = apply_clahe(rgb)
    gray = cv2.cvtColor(clahe_rgb, cv2.COLOR_RGB2GRAY)

    h, w = gray.shape
    joint_row = find_joint_row(gray)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    left_x, right_x = joint_extent(gray, joint_row, threshold)
    joint_width = max(right_x - left_x, 1)
    max_scan = max(int(h * SCAN_FRAC), 5)
    search_radius = max(int(h * LOCAL_SEARCH_FRAC), 3)

    # Skip the central ~20% (tibial spines/intercondylar notch) and a matching margin at each
    # compartment's outer edge (femoral condyle curving away from the vertical scan direction) -
    # both regions make the single-row-anchored vertical scan anatomically unreliable, not just
    # the center; see Phase 2 log for the diagnostic that found this.
    skip_half = CENTRAL_SKIP_FRAC * joint_width / 2
    outer_margin = CENTRAL_SKIP_FRAC * joint_width
    center_x = (left_x + right_x) / 2
    left_cols = [c for c in range(left_x, right_x + 1) if left_x + outer_margin < c < center_x - skip_half]
    right_cols = [c for c in range(left_x, right_x + 1) if center_x + skip_half < c < right_x - outer_margin]

    def summarize(cols):
        raw = [column_jsw(gray, c, joint_row, threshold, max_scan, search_radius) for c in cols]
        arr = np.array([v if v is not None else np.nan for v in raw])
        valid = arr[~np.isnan(arr)]
        # Record JSW=0 only if every column failed to measure at all (true fallback case);
        # otherwise min/mean use only columns where an actual femur/tibia edge pair was found,
        # so curvature-driven measurement failures at a compartment's outer edge don't get
        # conflated with genuine bone-on-bone narrowing (see Phase 2 log for why this matters).
        if len(valid) == 0:
            return 0.0, 0.0, 0, len(cols), arr
        return float(valid.min()), float(valid.mean()), len(valid), len(cols), arr

    left_min, left_mean, left_n_valid, left_n, left_arr = summarize(left_cols)
    right_min, right_mean, right_n_valid, right_n, right_arr = summarize(right_cols)

    return {
        "joint_row": joint_row, "left_x": left_x, "right_x": right_x, "joint_width_px": joint_width,
        "left_min_jsw": left_min / joint_width, "left_mean_jsw": left_mean / joint_width,
        "right_min_jsw": right_min / joint_width, "right_mean_jsw": right_mean / joint_width,
        "left_valid_frac": left_n_valid / left_n if left_n else 0.0,
        "right_valid_frac": right_n_valid / right_n if right_n else 0.0,
        "left_cols": left_cols, "right_cols": right_cols,
        "left_jsw_px": left_arr, "right_jsw_px": right_arr,
        "gray": gray,
    }


def assign_compartments(side_code, measured):
    """Maps left/right-of-image JSW to medial/lateral using the derived AP-view convention:
    for a RIGHT knee, medial is the image-RIGHT half (closer to the body midline when the
    patient faces the camera, non-mirrored); for a LEFT knee, medial is the image-LEFT half.
    This convention is the thing being checked in the Spearman-correlation validation step
    (see final_build_log.md) - if it's backwards, swap() flips it after the fact."""
    if side_code == "R":
        medial, lateral = "right", "left"
    elif side_code == "L":
        medial, lateral = "left", "right"
    else:
        return {"medial_min_jsw": None, "medial_mean_jsw": None,
                "lateral_min_jsw": None, "lateral_mean_jsw": None, "compartment_label": "unknown"}
    return {
        "medial_min_jsw": measured[f"{medial}_min_jsw"], "medial_mean_jsw": measured[f"{medial}_mean_jsw"],
        "lateral_min_jsw": measured[f"{lateral}_min_jsw"], "lateral_mean_jsw": measured[f"{lateral}_mean_jsw"],
        "compartment_label": "medial/lateral (side-based)",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/image_manifest.csv")
    ap.add_argument("--out", default="data/jsw_features.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"ID": str})
    rows = []
    n_errors = 0
    for i, r in manifest.iterrows():
        try:
            m = measure_jsw(r["IMAGE_PATH"])
        except Exception as e:
            n_errors += 1
            continue
        comp = assign_compartments(r["SIDE_CODE"], m)
        rows.append({
            "ID": r["ID"], "SIDE_CODE": r["SIDE_CODE"], "SPLIT": r["SPLIT"],
            "IMAGE_NAME": r["IMAGE_NAME"], "KL_GRADE": r["KL_GRADE"],
            "joint_width_px": m["joint_width_px"],
            "left_min_jsw": m["left_min_jsw"], "left_mean_jsw": m["left_mean_jsw"],
            "right_min_jsw": m["right_min_jsw"], "right_mean_jsw": m["right_mean_jsw"],
            **comp,
        })
        if (i + 1) % 1000 == 0:
            print(f"{i + 1}/{len(manifest)} processed ({n_errors} errors so far)")

    df = pd.DataFrame(rows).round(4)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Done. {len(df)} rows written to {args.out} ({n_errors} image read errors).")


if __name__ == "__main__":
    main()
