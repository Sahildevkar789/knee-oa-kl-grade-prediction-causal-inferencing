"""Phase 4 XAI (Hybrid 2 exists -> SHAP required): SHAP values for Hybrid 2's tabular
features (age, BMI) on a sample of the test set. Since Hybrid 2 takes two inputs (image +
tabular), each explained instance holds its OWN image fixed and perturbs only the tabular
features against a background of training-set tabular rows - a standard per-instance
multi-modal SHAP setup (explains "how would this patient's predicted grade change if their
age/BMI were different, given their actual X-ray"). Target explained: the sum of the 4 CORAL threshold sigmoid outputs (a continuous proxy for
the predicted grade, 0-4) - NOT the discretized predicted grade, which is piecewise-constant
and gives near-zero SHAP values unless a perturbation happens to cross a 0.5 threshold
(found via a 3-instance test run before committing to the full 60 - see final_build_log.md).

Not a reusable CLI tool (one-off per Phase 4), kept under scripts/ for the record.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
import cv2
import numpy as np
import pandas as pd
import torch
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kl_common import build_hybrid2, apply_clahe, get_transforms
from train_hybrid2 import load_tabular_split, TABULAR_COLS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EXPLAIN = 60     # test instances to explain (kept modest - each needs its own KernelExplainer call)
N_BACKGROUND = 20  # background tabular rows per explained instance
_, eval_tf = get_transforms()


def load_image_tensor(path):
    img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    clahe_img = apply_clahe(img)
    return eval_tf(image=clahe_img)["image"].unsqueeze(0).to(device)


def main():
    model = build_hybrid2("outputs/checkpoints", device, tabular_dim=len(TABULAR_COLS), init_from=None)
    model.load_state_dict(torch.load("outputs/checkpoints/kl_hybrid2_best.pt", map_location=device))
    model.eval()

    df = pd.read_csv("data/clinical_features.csv", dtype={"ID": str})
    train_p, train_y, train_tab, bmi_median, _ = load_tabular_split(df, "train")
    test_p, test_y, test_tab, _, _ = load_tabular_split(df, "test", impute_median=bmi_median)
    tab_mean, tab_std = train_tab.mean(axis=0), train_tab.std(axis=0)
    tab_std[tab_std == 0] = 1.0
    train_tab_std = (train_tab - tab_mean) / tab_std
    test_tab_std = (test_tab - tab_mean) / tab_std

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(train_tab_std), size=N_BACKGROUND, replace=False)
    background = train_tab_std[bg_idx]
    explain_idx = rng.choice(len(test_tab_std), size=min(N_EXPLAIN, len(test_tab_std)), replace=False)

    all_shap = []
    for n, i in enumerate(explain_idx):
        x_img = load_image_tensor(test_p[i])

        def predict_fn(tab_batch):
            tab_t = torch.tensor(tab_batch, dtype=torch.float32).to(device)
            img_rep = x_img.repeat(len(tab_batch), 1, 1, 1)
            with torch.no_grad():
                logits = model(img_rep, tab_t)
                probs = torch.sigmoid(logits.float())
                continuous_grade = probs.sum(dim=1)   # continuous 0-4 proxy, not the discretized grade
            return continuous_grade.cpu().numpy()

        explainer = shap.KernelExplainer(predict_fn, background, silent=True)
        sv = explainer.shap_values(test_tab_std[i:i + 1], nsamples=100, silent=True)
        all_shap.append(np.array(sv).reshape(-1))
        if (n + 1) % 10 == 0:
            print(f"{n + 1}/{len(explain_idx)} instances explained")

    shap_values = np.array(all_shap)   # (N_EXPLAIN, 2)
    explained_tab = test_tab_std[explain_idx]

    os.makedirs("outputs/results/xai", exist_ok=True)
    pd.DataFrame({"image_path": [test_p[i] for i in explain_idx],
                 **{f"shap_{c}": shap_values[:, j] for j, c in enumerate(TABULAR_COLS)},
                 **{c: explained_tab[:, j] for j, c in enumerate(TABULAR_COLS)}}
                ).round(4).to_csv("outputs/results/xai/shap_hybrid2_values.csv", index=False)

    mean_abs = np.abs(shap_values).mean(axis=0)
    print("\nMean |SHAP| per feature:")
    for c, v in zip(TABULAR_COLS, mean_abs):
        print(f"  {c}: {v:.4f}")
    pd.DataFrame({"feature": TABULAR_COLS, "mean_abs_shap": mean_abs}).round(4).to_csv(
        "outputs/results/xai/shap_hybrid2_mean_abs.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(TABULAR_COLS, mean_abs, color="#D85A30")
    ax.set_xlabel("Mean |SHAP value| (effect on predicted grade)")
    ax.set_title("Hybrid 2 - tabular feature importance")
    plt.tight_layout()
    plt.savefig("outputs/results/xai/shap_hybrid2_bar.png", dpi=130)
    plt.close(fig)

    plt.figure()
    shap.summary_plot(shap_values, explained_tab, feature_names=TABULAR_COLS, show=False)
    plt.title("Hybrid 2 - SHAP summary (tabular features)")
    plt.tight_layout()
    plt.savefig("outputs/results/xai/shap_hybrid2_summary.png", dpi=130)
    plt.close()
    print("\nSaved SHAP outputs to outputs/results/xai/shap_hybrid2_*")


if __name__ == "__main__":
    main()
