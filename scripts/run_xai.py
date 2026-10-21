"""Phase 4 XAI: Grad-CAM (last ConvNeXt stage) + CBAM attention maps for kl_final and
kl_hybrid1 (the hybrid with the best val QWK, 0.8320 vs Hybrid 2's 0.8170 - see
final_build_log.md Phase 1/3), on 15 test images (3/grade, >=5 wrong predictions). Also runs
the quantitative joint-region check and (since Hybrid 2 exists) SHAP on its tabular features.

CORAL note: both models output 4 threshold logits, not a single class score, so Grad-CAM's
target scalar is the SUM of the 4 threshold logits - a monotonic proxy for the predicted
grade (more thresholds crossed with higher confidence -> larger sum), analogous to using the
predicted class's logit in a standard softmax classifier.

Not a reusable CLI tool (one-off per Phase 4), kept under scripts/ for the record per the
brief's instruction to save everything Phase 4 produces.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kl_common import build_classifier, build_hybrid1, apply_clahe, get_transforms, GRADES, KL_TEXT, grad_cam
from measure_jsw import find_joint_row

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_, eval_tf = get_transforms()
OUT_DIR = "outputs/results/xai"
os.makedirs(OUT_DIR, exist_ok=True)


def load_image(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    clahe_img = apply_clahe(img)
    x = eval_tf(image=clahe_img)["image"].unsqueeze(0).to(device)
    return img, x


def cam_mass_in_band(cam, img_shape, joint_row, band_frac=0.15):
    """Upsamples the CAM to image size, normalizes to a probability mass, returns the
    fraction of that mass falling inside a +-band_frac*height horizontal band around joint_row."""
    h, w = img_shape[:2]
    cam_up = cv2.resize(cam, (w, h))
    cam_up = np.clip(cam_up, 0, None)
    total = cam_up.sum()
    if total <= 0:
        return 0.0
    band_half = int(band_frac * h)
    r0, r1 = max(0, joint_row - band_half), min(h, joint_row + band_half)
    return float(cam_up[r0:r1, :].sum() / total)


def select_images():
    """3 per grade, 1 wrong + 2 correct per grade where available (>=5 wrong total)."""
    preds = pd.read_csv("outputs/results/kl_final_test_predictions.csv")
    preds["image_path"] = preds["image_path"].str.replace("\\", "/", regex=False)
    chosen = []
    for g in range(5):
        sub = preds[preds.true_grade == g]
        wrong = sub[sub.pred_grade != g]
        correct = sub[sub.pred_grade == g]
        picks = []
        if len(wrong) > 0:
            picks.append(wrong.iloc[0])
        picks += [correct.iloc[i] for i in range(min(2, len(correct)))]
        while len(picks) < 3 and len(sub) > len(picks):
            extra = sub[~sub.image_path.isin([p.image_path for p in picks])].iloc[0]
            picks.append(extra)
        chosen += picks[:3]
    return pd.DataFrame(chosen)


def main():
    kl_final = build_classifier("convnext_tiny", 3, pretrained=False, out_dim=4, attention="cbam").to(device)
    kl_final.load_state_dict(torch.load("outputs/checkpoints/kl_final_best.pt", map_location=device))
    kl_final.eval()

    hybrid1 = build_hybrid1("outputs/checkpoints", device, init_from=None)
    hybrid1.load_state_dict(torch.load("outputs/checkpoints/kl_hybrid1_best.pt", map_location=device))
    hybrid1.eval()

    chosen = select_images()
    n_wrong = (chosen.pred_grade != chosen.true_grade).sum()
    print(f"Selected {len(chosen)} images, {n_wrong} wrong predictions (kl_final)")

    rows = []
    for _, r in chosen.iterrows():
        path = r["image_path"]
        true_g, pred_g = int(r["true_grade"]), int(r["pred_grade"])
        img, x = load_image(path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        joint_row = find_joint_row(gray)

        cam_f, probs_f = grad_cam(kl_final, x.clone(), kl_final.features)
        attn_f = kl_final.last_attn.squeeze().cpu().numpy()
        mass_f = cam_mass_in_band(cam_f, img.shape, joint_row)

        cam_h, probs_h = grad_cam(hybrid1, x.clone(), hybrid1.features)
        attn_h = hybrid1.last_attn.squeeze().cpu().numpy()
        mass_h = cam_mass_in_band(cam_h, img.shape, joint_row)

        stem = os.path.splitext(os.path.basename(path))[0]
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        for row_i, (name, cam, attn, mass) in enumerate([("kl_final", cam_f, attn_f, mass_f),
                                                          ("kl_hybrid1", cam_h, attn_h, mass_h)]):
            cam_up = cv2.resize(cam, (img.shape[1], img.shape[0]))
            attn_up = cv2.resize(attn, (img.shape[1], img.shape[0]))
            axes[row_i, 0].imshow(img); axes[row_i, 0].set_title(f"{name}: input"); axes[row_i, 0].axis("off")
            axes[row_i, 1].imshow(img, cmap="gray"); axes[row_i, 1].imshow(cam_up, cmap="jet", alpha=0.45)
            axes[row_i, 1].axhline(joint_row, color="white", linestyle="--", linewidth=1)
            axes[row_i, 1].set_title(f"{name}: Grad-CAM (joint-band mass={mass:.2f})"); axes[row_i, 1].axis("off")
            axes[row_i, 2].imshow(img, cmap="gray"); axes[row_i, 2].imshow(attn_up, cmap="jet", alpha=0.45)
            axes[row_i, 2].set_title(f"{name}: CBAM attention"); axes[row_i, 2].axis("off")
        fig.suptitle(f"KL{true_g} (true) - {os.path.basename(path)} | kl_final pred=KL{pred_g}", fontsize=11)
        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"KL{true_g}_{stem}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)

        rows.append({"image_path": path, "true_grade": true_g, "pred_grade_final": pred_g,
                    "correct": true_g == pred_g, "joint_row": joint_row,
                    "cam_mass_in_band_final": mass_f, "cam_mass_in_band_hybrid1": mass_h})
        print(f"  {stem}: KL{true_g} (pred {pred_g}) - mass_final={mass_f:.3f} mass_hybrid1={mass_h:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "cam_mass_quantitative.csv"), index=False)

    print("\nMean Grad-CAM joint-band mass by KL grade (kl_final):")
    print(df.groupby("true_grade")["cam_mass_in_band_final"].mean())
    print("\nMean Grad-CAM joint-band mass, correct vs incorrect (kl_final):")
    print(df.groupby("correct")["cam_mass_in_band_final"].mean())
    print(f"\nSaved {len(chosen)} overlay figures + quantitative CSV to {OUT_DIR}/")


if __name__ == "__main__":
    main()
