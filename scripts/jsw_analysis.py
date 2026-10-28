"""High-precision Joint Space Width (JSW) and gap measurement module for knee X-rays.
Extends scripts/measure_jsw.py to retain full 2D coordinates for femoral condyles
and tibial plateau, calculate quantitative millimeter metrics (min/mean JSW, OARSI JSN Grade,
compartment ratio), and render medical-grade annotated X-ray visualizations with
highlighted gap ribbons and calipers.
"""
import base64
import cv2
import numpy as np

STANDARD_TIBIAL_PLATEAU_WIDTH_MM = 75.0  # Average adult knee AP tibial plateau width in mm
CENTRAL_SKIP_FRAC = 0.20
SCAN_FRAC = 0.18
LOCAL_SEARCH_FRAC = 0.10


def apply_fast_clahe(img_rgb):
    """CLAHE on RGB image without torch/albumentations dependencies."""
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB), enhanced


def find_joint_row(gray):
    """Darkest horizontal band in the central region of the image."""
    h, w = gray.shape
    lo, hi = int(h * 0.25), int(h * 0.75)
    row_mean = gray[lo:hi, :].astype(np.float32).mean(axis=1)
    return int(np.argmin(row_mean)) + lo


def joint_extent(gray, joint_row, threshold, band_half=10):
    """Horizontal bone extent (left_x, right_x) in a band around the joint row."""
    h, w = gray.shape
    r0, r1 = max(0, joint_row - band_half), min(h, joint_row + band_half)
    band = gray[r0:r1, :]
    cols = np.where((band > threshold).any(axis=0))[0]
    if len(cols) == 0:
        return 0, w - 1
    return int(cols[0]), int(cols[-1])


def column_jsw_detailed(gray, col, joint_row, threshold, max_scan, search_radius):
    """
    Finds this column's local joint line and scans up for femur edge and down for tibia edge.
    Returns (gap, femur_edge_y, tibia_edge_y), or (None, None, None) if measurement failed.
    """
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
        return None, None, None

    gap = max(0.0, float(tibia_edge - femur_edge))
    return gap, femur_edge, tibia_edge


def estimate_knee_jsw(rgb_img, side_code="R", custom_mm_scale=None):
    """
    Measures joint space width (JSW) across medial and lateral compartments from an RGB knee image.
    
    Parameters:
      rgb_img: uint8 numpy array (H, W, 3) in RGB
      side_code: 'R' (Right knee) or 'L' (Left knee)
      custom_mm_scale: optional float mm_per_pixel override

    Returns:
      dict with quantitative metrics, compartment curves, base64 annotated image, and calipers.
    """
    h, w, _ = rgb_img.shape
    clahe_rgb, gray = apply_fast_clahe(rgb_img)

    joint_row = find_joint_row(gray)
    threshold, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    left_x, right_x = joint_extent(gray, joint_row, threshold)
    joint_width = max(right_x - left_x, 10)
    max_scan = max(int(h * SCAN_FRAC), 5)
    search_radius = max(int(h * LOCAL_SEARCH_FRAC), 3)

    mm_per_px = float(custom_mm_scale) if custom_mm_scale else (STANDARD_TIBIAL_PLATEAU_WIDTH_MM / joint_width)

    # Compartment columns
    skip_half = CENTRAL_SKIP_FRAC * joint_width / 2
    outer_margin = CENTRAL_SKIP_FRAC * joint_width
    center_x = (left_x + right_x) / 2
    left_cols = [c for c in range(left_x, right_x + 1) if left_x + outer_margin < c < center_x - skip_half]
    right_cols = [c for c in range(left_x, right_x + 1) if center_x + skip_half < c < right_x - outer_margin]

    def process_compartment(cols):
        if not cols:
            return None
        valid_cols = []
        femur_pts = []
        tibia_pts = []
        gaps = []

        for c in cols:
            gap, fy, ty = column_jsw_detailed(gray, c, joint_row, threshold, max_scan, search_radius)
            if gap is not None:
                valid_cols.append(c)
                femur_pts.append(fy)
                tibia_pts.append(ty)
                gaps.append(gap)

        if len(valid_cols) == 0:
            return {
                "cols": cols,
                "femur_y": [joint_row] * len(cols),
                "tibia_y": [joint_row] * len(cols),
                "gaps": [0.0] * len(cols),
                "min_gap_px": 0.0,
                "mean_gap_px": 0.0,
                "min_col": cols[len(cols) // 2],
                "min_femur_y": joint_row,
                "min_tibia_y": joint_row,
                "valid_count": 0,
            }

        # Apply moving average filter to soften single-pixel threshold stepping
        k = min(3, len(valid_cols))
        if k >= 2:
            kernel = np.ones(k) / k
            fy_smooth = np.convolve(np.array(femur_pts, dtype=np.float32), kernel, mode="same")
            ty_smooth = np.convolve(np.array(tibia_pts, dtype=np.float32), kernel, mode="same")
            smooth_gaps = np.maximum(0.0, ty_smooth - fy_smooth)
        else:
            fy_smooth = np.array(femur_pts, dtype=np.float32)
            ty_smooth = np.array(tibia_pts, dtype=np.float32)
            smooth_gaps = np.array(gaps, dtype=np.float32)

        # Robust minimum calculation excluding extreme peripheral columns (tibial border margins)
        if len(smooth_gaps) >= 8:
            margin_trim = max(2, int(len(smooth_gaps) * 0.12))
            core_slice = slice(margin_trim, len(smooth_gaps) - margin_trim)
            min_sub_idx = int(np.argmin(smooth_gaps[core_slice]))
            min_idx = margin_trim + min_sub_idx
        else:
            min_idx = int(np.argmin(smooth_gaps))

        min_gap_val = float(smooth_gaps[min_idx])

        return {
            "cols": valid_cols,
            "femur_y": [round(float(y), 1) for y in fy_smooth],
            "tibia_y": [round(float(y), 1) for y in ty_smooth],
            "gaps": [round(float(g), 1) for g in smooth_gaps],
            "min_gap_px": round(min_gap_val, 1),
            "mean_gap_px": round(float(np.mean(smooth_gaps)), 1),
            "min_col": valid_cols[min_idx],
            "min_femur_y": round(float(fy_smooth[min_idx]), 1),
            "min_tibia_y": round(float(ty_smooth[min_idx]), 1),
            "valid_count": len(valid_cols),
        }

    left_curve = process_compartment(left_cols)
    right_curve = process_compartment(right_cols)

    # Side mapping (standard AP Knee View):
    # R knee: Patient's Right is image-Left (Lateral), Patient's Left is image-Right (Medial)
    # L knee: Patient's Left is image-Right (Lateral), Patient's Right is image-Left (Medial)
    side = side_code.upper() if side_code in ("L", "LEFT") else "R"
    if side == "R":
        medial_curve = right_curve
        lateral_curve = left_curve
        medial_pos_label = "Image Right (Patient Medial)"
        lateral_pos_label = "Image Left (Patient Lateral)"
    else:
        medial_curve = left_curve
        lateral_curve = right_curve
        medial_pos_label = "Image Left (Patient Medial)"
        lateral_pos_label = "Image Right (Patient Lateral)"

    def comp_metrics(curve, name, pos_tag):
        if not curve or curve["valid_count"] == 0:
            return {
                "name": name, "position": pos_tag,
                "min_mm": 0.0, "mean_mm": 0.0, "min_px": 0.0, "mean_px": 0.0,
                "caliper": None, "valid_count": 0,
            }
        min_mm = round(curve["min_gap_px"] * mm_per_px, 2)
        mean_mm = round(curve["mean_gap_px"] * mm_per_px, 2)
        return {
            "name": name,
            "position": pos_tag,
            "min_mm": min_mm,
            "mean_mm": mean_mm,
            "min_px": curve["min_gap_px"],
            "mean_px": curve["mean_gap_px"],
            "valid_count": curve["valid_count"],
            "caliper": {
                "x": curve["min_col"],
                "femur_y": curve["min_femur_y"],
                "tibia_y": curve["min_tibia_y"],
                "gap_px": curve["min_gap_px"],
                "gap_mm": min_mm,
            }
        }

    medial_stats = comp_metrics(medial_curve, "Medial Compartment", medial_pos_label)
    lateral_stats = comp_metrics(lateral_curve, "Lateral Compartment", lateral_pos_label)

    # OARSI Joint Space Narrowing (JSN) grading based on the narrowest compartment
    min_overall_mm = min(medial_stats["min_mm"], lateral_stats["min_mm"])
    if min_overall_mm >= 4.0:
        jsn_grade = 0
        jsn_label = "Grade 0 — Normal joint space (> 4.0 mm)"
        jsn_severity = "normal"
        jsn_desc = "Preserved joint space width in both compartments."
    elif min_overall_mm >= 3.0:
        jsn_grade = 1
        jsn_label = "Grade 1 — Mild narrowing (3.0 – 4.0 mm)"
        jsn_severity = "mild"
        jsn_desc = "Early joint space narrowing; minimal cartilage loss."
    elif min_overall_mm >= 1.5:
        jsn_grade = 2
        jsn_label = "Grade 2 — Moderate narrowing (1.5 – 3.0 mm)"
        jsn_severity = "moderate"
        jsn_desc = "Definite joint space narrowing; moderate articular cartilage loss."
    else:
        jsn_grade = 3
        jsn_label = "Grade 3 — Severe narrowing / bone-on-bone (< 1.5 mm)"
        jsn_severity = "severe"
        jsn_desc = "Marked joint space narrowing with near-complete or complete cartilage loss."

    # Medial/Lateral Ratio and Coronal Alignment Assessment
    if lateral_stats["mean_mm"] > 0.1:
        ml_ratio = round(medial_stats["mean_mm"] / lateral_stats["mean_mm"], 2)
    else:
        ml_ratio = 1.0

    if ml_ratio < 0.70:
        alignment_note = "Medial-predominant narrowing (varus deformity / bow-leg pattern)"
    elif ml_ratio > 1.35:
        alignment_note = "Lateral-predominant narrowing (valgus deformity / knock-knee pattern)"
    else:
        alignment_note = "Bicompartmentally balanced joint space width"

    # 6. Generate annotated visualization image
    annotated = rgb_img.copy()
    overlay_gap = np.zeros_like(rgb_img)

    # Draw translucent filled ribbons for the joint space
    for curve in (left_curve, right_curve):
        if not curve or not curve["cols"] or curve["valid_count"] == 0:
            continue
        cols = curve["cols"]
        fy = curve["femur_y"]
        ty = curve["tibia_y"]

        poly = []
        for c, y in zip(cols, fy):
            poly.append([int(c), int(y)])
        for c, y in reversed(list(zip(cols, ty))):
            poly.append([int(c), int(y)])

        poly_np = np.array([poly], dtype=np.int32)
        cur_min_mm = curve["min_gap_px"] * mm_per_px
        if cur_min_mm < 1.5:
            ribbon_color = (239, 68, 68)     # Coral Red (Severe)
        elif cur_min_mm < 3.0:
            ribbon_color = (245, 158, 11)    # Amber (Moderate)
        else:
            ribbon_color = (6, 182, 212)     # Cyan/Teal (Normal/Mild)

        cv2.fillPoly(overlay_gap, poly_np, ribbon_color)

    annotated = cv2.addWeighted(annotated, 0.72, overlay_gap, 0.55, 0)

    # Draw contour lines & caliper markers
    items = [
        (left_curve, "LAT" if side == "R" else "MED"),
        (right_curve, "MED" if side == "R" else "LAT")
    ]
    for curve, badge_label in items:
        if not curve or not curve["cols"] or curve["valid_count"] == 0:
            continue
        cols = curve["cols"]
        fy = curve["femur_y"]
        ty = curve["tibia_y"]

        # Femur condyle line (sky blue)
        pts_f = np.array([[int(c), int(y)] for c, y in zip(cols, fy)], dtype=np.int32)
        cv2.polylines(annotated, [pts_f], isClosed=False, color=(56, 189, 248), thickness=2, lineType=cv2.LINE_AA)

        # Tibial plateau line (emerald green)
        pts_t = np.array([[int(c), int(y)] for c, y in zip(cols, ty)], dtype=np.int32)
        cv2.polylines(annotated, [pts_t], isClosed=False, color=(52, 211, 153), thickness=2, lineType=cv2.LINE_AA)

        # Minimum Caliper
        cx = curve["min_col"]
        cy1 = int(curve["min_femur_y"])
        cy2 = int(curve["min_tibia_y"])
        min_mm = curve["min_gap_px"] * mm_per_px

        # Vertical caliper line
        cv2.line(annotated, (cx, cy1), (cx, cy2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        # End ticks
        cv2.line(annotated, (cx - 4, cy1), (cx + 4, cy1), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(annotated, (cx - 4, cy2), (cx + 4, cy2), (255, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy1), 2, (56, 189, 248), -1, lineType=cv2.LINE_AA)
        cv2.circle(annotated, (cx, cy2), 2, (52, 211, 153), -1, lineType=cv2.LINE_AA)

        # Caliper badge label
        txt = f"{badge_label}: {min_mm:.1f}mm"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        bx = max(4, min(w - tw - 8, cx - tw // 2))
        by = max(th + 4, cy1 - 7)
        # Badge background
        cv2.rectangle(annotated, (bx - 2, by - th - 3), (bx + tw + 2, by + 3), (15, 23, 42), -1)
        cv2.rectangle(annotated, (bx - 2, by - th - 3), (bx + tw + 2, by + 3), (56, 189, 248), 1)
        cv2.putText(annotated, txt, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Encode annotated image to base64 PNG
    ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
    annotated_base64 = base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None

    return {
        "side": side,
        "side_label": "Right Knee" if side == "R" else "Left Knee",
        "joint_width_px": joint_width,
        "mm_per_px": round(mm_per_px, 4),
        "joint_row": joint_row,
        "medial": medial_stats,
        "lateral": lateral_stats,
        "ml_ratio": ml_ratio,
        "alignment_note": alignment_note,
        "jsn_classification": {
            "grade": jsn_grade,
            "label": jsn_label,
            "severity": jsn_severity,
            "description": jsn_desc,
            "min_overall_mm": min_overall_mm,
        },
        "annotated_png_base64": annotated_base64,
        "raw_curves": {
            "left": left_curve,
            "right": right_curve,
        }
    }


if __name__ == "__main__":
    for i in range(5):
        img = cv2.imread(f"ui/static/samples/sample_kl{i}.png")
        res = estimate_knee_jsw(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), side_code="R")
        print(f"KL{i} ({res['side_label']}): Medial min={res['medial']['min_mm']}mm (mean {res['medial']['mean_mm']}mm) | Lateral min={res['lateral']['min_mm']}mm (mean {res['lateral']['mean_mm']}mm) | {res['jsn_classification']['label']}")
