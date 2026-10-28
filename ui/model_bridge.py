"""Single entry point the UI calls: analyze(image_path, age=None, bmi=None,
family_history=None). Imports and calls the project's existing code (the same functions
predict_extended.py uses) - no model logic is copied or reimplemented here. The final
ensemble (or kl_final alone, whichever predict_extended.py's own validation check selects)
is loaded once, lazily, on first call.

Usage:
    from ui.model_bridge import analyze
    result = analyze("path/to/xray.png", age=65, bmi=32, family_history="yes")
"""
import os
import sys
import base64
import pickle

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(ROOT, "scripts")
for _p in (ROOT, SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from kl_common import apply_clahe, IMG_SIZE
from predict_extended import (load_prediction_setup, classify_rgb, nearest_cases, uncertainty_info,
                              to_tensor, MODEL_SPECS, build_model)
from causal_model import load_or_fit_causal_models, out_of_range_messages, contribution_age, \
    contribution_bmi, contribution_famkr, expected_kl
from jsw_analysis import estimate_knee_jsw

CAUSAL_FOREST_PATH = os.path.join(ROOT, "outputs", "extended", "causal", "causal_forest_model.pkl")
INJURY_ESTIMATES_PATH = os.path.join(ROOT, "outputs", "extended", "causal", "injury_estimates.csv")
DISCLAIMER = "Research prototype — not for clinical use."

_state = {}


def _ensure_loaded():
    """Loads everything once: the final prediction setup (ensemble or kl_final - whichever
    predict_extended.py's own validation-accuracy check selects), the causal models, the
    causal forest, and the injury association estimates."""
    if _state:
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_label, models, members, thresholds = load_prediction_setup(device)
    kl_final_for_neighbors = models.get("kl_final") or models.get("kl_final_seed7") \
        or build_model(MODEL_SPECS["kl_final"], device)

    causal_models = load_or_fit_causal_models()
    causal_forest = None
    if os.path.exists(CAUSAL_FOREST_PATH):
        try:
            with open(CAUSAL_FOREST_PATH, "rb") as f:
                causal_forest = pickle.load(f)
        except Exception:
            causal_forest = None
    injury_df = None
    if os.path.exists(INJURY_ESTIMATES_PATH):
        import pandas as pd
        injury_df = pd.read_csv(INJURY_ESTIMATES_PATH)

    _state.update(device=device, model_label=model_label, models=models, members=members,
                  thresholds=thresholds, kl_final_for_neighbors=kl_final_for_neighbors,
                  causal_models=causal_models, causal_forest=causal_forest, injury_df=injury_df)


def _png_base64(rgb_uint8):
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else None


def _overlay_cam(cam_7x7, base_img224):
    import matplotlib.cm as cm
    cam_up = cv2.resize(cam_7x7, (IMG_SIZE, IMG_SIZE))
    cam_up = np.clip(cam_up, 0, 1)
    heatmap = (cm.jet(cam_up)[:, :, :3] * 255).astype(np.uint8)
    return cv2.addWeighted(base_img224.astype(np.uint8), 0.6, heatmap, 0.4, 0)


def _integrated_gradients_image(model, img_rgb, device):
    """Integrated Gradients on kl_final (or its seed-7 sibling, whichever is loaded), same
    target convention as xai_extended.py (sum of the 4 CORAL threshold logits)."""
    from captum.attr import IntegratedGradients
    clahe_img = apply_clahe(img_rgb)
    img224 = cv2.resize(clahe_img, (IMG_SIZE, IMG_SIZE))
    x = to_tensor(img224, device)

    def forward_sum(xx):
        return model(xx).sum(dim=1)

    ig = IntegratedGradients(forward_sum)
    attr = ig.attribute(x, baselines=torch.zeros_like(x), n_steps=50).squeeze().detach().cpu().numpy()
    ig_map = np.abs(attr).sum(axis=0)
    ig_map = ig_map / (ig_map.max() + 1e-8)
    base = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return _overlay_cam(ig_map, base)


def _build_causal_panel(age, bmi, family_history):
    causal_models = _state["causal_models"]
    panel = {"caveat": "Population-level estimates under this DAG's assumptions, not individual predictions."}

    famkr = {"yes": 1, "no": 0, True: 1, False: 0, 1: 1, 0: 0}.get(family_history, None)

    if age is None or bmi is None:
        panel["available"] = False
        panel["message"] = "Age and BMI not given - provide both for a per-factor causal breakdown."
        return panel

    refusal = out_of_range_messages(causal_models, age, bmi)
    if refusal:
        # Same refusal behavior as causal_query.py: out-of-range age/BMI is refused, not computed.
        panel["available"] = False
        panel["refused"] = True
        panel["messages"] = refusal
        return panel

    famkr_used = famkr if famkr is not None else causal_models["most_common_famkr"]
    panel["available"] = True
    panel["age"] = age
    panel["bmi"] = bmi
    panel["obese"] = bool(bmi >= 30)
    panel["family_history_used"] = bool(famkr_used)
    panel["family_history_was_given"] = famkr is not None
    clipped, was_clipped, raw = expected_kl(causal_models, age, bmi, famkr_used)
    panel["expected_kl"] = {"value": clipped, "clipped": bool(was_clipped), "raw": raw}

    age_c = contribution_age(causal_models, age)
    panel["age_contribution"] = age_c
    panel["bmi_contribution"] = contribution_bmi(causal_models, bmi)
    panel["family_history_contribution"] = contribution_famkr(causal_models, famkr_used)

    if _state["causal_forest"] is not None:
        X = np.array([[age, famkr_used]])
        effect = float(_state["causal_forest"].effect(X)[0])
        lo, hi = _state["causal_forest"].effect_interval(X, alpha=0.05)
        panel["individual_obesity_effect"] = {"value": effect, "ci_low": float(np.ravel(lo)[0]),
                                              "ci_high": float(np.ravel(hi)[0])}

    rng = causal_models["data_ranges"]
    whatif_bmis = sorted({round(b, 1) for b in (bmi - 10, bmi - 5, bmi, bmi + 5, bmi + 10)
                          if rng["bmi_min"] <= b <= rng["bmi_max"]})
    panel["bmi_whatif"] = []
    for b in whatif_bmis:
        c, wc, r = expected_kl(causal_models, age, b, famkr_used)
        panel["bmi_whatif"].append({"bmi": b, "expected_kl": c})

    if _state["injury_df"] is not None:
        df = _state["injury_df"]
        lin = float(df.loc[df.Method == "linear_regression", "ATE"].iloc[0])
        psm = float(df.loc[df.Method == "propensity_score_matching", "ATE"].iloc[0])
        panel["knee_injury"] = {
            "linear_regression_ate": lin, "propensity_score_matching_ate": psm,
            "label": "association, not a demonstrated cause (self-reported history, timing relative to KL grade uncertain)",
        }

    return panel


def analyze(image_path, age=None, bmi=None, family_history=None, side="R"):
    """Runs the full prediction pipeline on one image and returns a plain (JSON-serializable)
    dict. age/bmi: float or None. family_history: "yes"/"no"/True/False/None. Never raises on
    a missing/invalid causal profile - that just narrows what the causal panel can report;
    out-of-range age/BMI is refused (not computed) the same way causal_query.py refuses it."""
    _ensure_loaded()
    s = _state
    device = s["device"]

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    grade, probs, avg_cam, _ = classify_rgb(img, s["models"], s["members"], s["thresholds"], device)
    base224 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gradcam_png = _png_base64(_overlay_cam(avg_cam, base224))

    ig_png = None
    try:
        ig_png = _png_base64(_integrated_gradients_image(s["kl_final_for_neighbors"], img, device))
    except Exception as e:
        ig_png = None  # Integrated Gradients is an optional extra visualization, not required for a grade.

    clahe224 = cv2.resize(apply_clahe(img), (IMG_SIZE, IMG_SIZE))
    neighbors = nearest_cases(clahe224, s["kl_final_for_neighbors"], device) or []
    entropy, unc_thresh, is_uncertain = uncertainty_info(probs)

    # High-precision Joint Space Width (JSW) and gap measurement
    jsw_data = None
    try:
        jsw_data = estimate_knee_jsw(img, side_code=side or "R")
    except Exception as e:
        jsw_data = None

    return {
        "grade": int(grade),
        "grade_label": ["No osteoarthritis", "Doubtful", "Minimal", "Moderate", "Severe"][int(grade)],
        "probabilities": {f"KL{g}": float(probs[g]) for g in range(5)},
        "confidence": float(probs[grade]),
        "model_used": s["model_label"],
        "model_members": list(s["members"]),
        "uncertainty": {
            "entropy": entropy, "threshold": unc_thresh, "flag": bool(is_uncertain),
            "message": "REFER TO RADIOLOGIST (uncertain)" if is_uncertain else "confident",
        },
        "gradcam_png_base64": gradcam_png,
        "integrated_gradients_png_base64": ig_png,
        "similar_cases": [{"path": p, "grade": int(g), "similarity": float(sim)} for p, g, sim in neighbors],
        "causal_panel": _build_causal_panel(age, bmi, family_history),
        "jsw": jsw_data,
        "disclaimer": DISCLAIMER,
    }
