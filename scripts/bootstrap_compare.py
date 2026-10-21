"""One-off: paired bootstrap (2000 resamples) of QWK/accuracy between two prediction CSVs.
Usage: python scripts/_bootstrap.py <model_a_predictions.csv> <model_b_predictions.csv> <label_a> <label_b> <out_csv>
Reports metric_a - metric_b. Not part of the permanent CLI."""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, accuracy_score

path_a, path_b, label_a, label_b, out_csv = sys.argv[1:6]
a = pd.read_csv(path_a)
b = pd.read_csv(path_b)
a["image_path"] = a["image_path"].str.replace("\\", "/", regex=False)
b["image_path"] = b["image_path"].str.replace("\\", "/", regex=False)
a, b = a.set_index("image_path"), b.set_index("image_path")
assert set(a.index) == set(b.index), "Test sets differ between the two runs"
b = b.loc[a.index]
assert (a["true_grade"].values == b["true_grade"].values).all(), "Ground truth mismatch"

y_true = a["true_grade"].values
pred_a, pred_b = a["pred_grade"].values, b["pred_grade"].values
n = len(y_true)

rng = np.random.default_rng(42)
qwk_diffs, acc_diffs = [], []
for _ in range(2000):
    idx = rng.integers(0, n, n)
    yt, pa, pb = y_true[idx], pred_a[idx], pred_b[idx]
    qwk_diffs.append(cohen_kappa_score(yt, pa, weights="quadratic") - cohen_kappa_score(yt, pb, weights="quadratic"))
    acc_diffs.append(accuracy_score(yt, pa) - accuracy_score(yt, pb))

qwk_diffs, acc_diffs = np.array(qwk_diffs), np.array(acc_diffs)


def summarize(diffs, label):
    mean_diff = diffs.mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    significant = lo > 0 or hi < 0
    print(f"{label}: mean diff = {mean_diff:.4f}, 95% CI = [{lo:.4f}, {hi:.4f}], significant = {significant}")
    return {"Metric": label, "MeanDiff": mean_diff, "CI_low": lo, "CI_high": hi,
            "Significant_95pct": significant, "Resamples": 2000}


rows = [summarize(qwk_diffs, f"QWK ({label_a} - {label_b})"),
       summarize(acc_diffs, f"Accuracy ({label_a} - {label_b})")]
pd.DataFrame(rows).round(4).to_csv(out_csv, index=False)
print(f"Saved {out_csv}")
