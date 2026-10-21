"""Fits (or loads a cached) linear regression for Phase 5's causal model, for predict.py's
per-patient causal risk panel. Model: KL_grade ~ Age + FamilyHistoryKneeReplacement + Obese,
fit on the exact same primary one-knee-per-patient data (seed 42) as run_causal.py's
backdoor.linear_regression estimate, so this Obese coefficient matches Phase 5's reported ATE
(+0.485) by construction - this is a refit of that same model to recover the full coefficient
vector (DoWhy's estimator exposes only the treatment ATE, not the fitted intercept/covariate
coefficients needed to predict an expected KL grade for a specific age/family-history).

Not a reusable CLI tool - imported by predict.py, which calls load_or_fit_causal_regression().
"""
import os
import json
import numpy as np
import pandas as pd

COEF_PATH = "outputs/results/causal/kl_regression_coefs.json"


def _refit():
    df = pd.read_csv("data/clinical_features.csv", dtype={"ID": str})
    df["famkr"] = df["P01FAMKR"].str.extract(r"^(\d)").astype(float)
    df = df.dropna(subset=["famkr", "P01BMI", "V00AGE"]).reset_index(drop=True)
    df["obese"] = (df["P01BMI"] >= 30).astype(int)

    # Same one-knee-per-patient sampling as run_causal.py's one_knee_per_patient(seed=42).
    rng = np.random.default_rng(42)
    primary = df.groupby("ID", group_keys=False).apply(
        lambda g: g.sample(1, random_state=rng.integers(0, 2**31))).reset_index(drop=True)

    X = np.column_stack([np.ones(len(primary)), primary["V00AGE"].values,
                         primary["famkr"].values, primary["obese"].values])
    y = primary["KL_GRADE"].values.astype(np.float64)
    coefs, *_ = np.linalg.lstsq(X, y, rcond=None)
    intercept, age_coef, famkr_coef, obese_coef = (float(c) for c in coefs)
    most_common_famkr = int(primary["famkr"].mode().iloc[0])

    result = {
        "intercept": intercept, "age_coef": age_coef, "famkr_coef": famkr_coef,
        "obese_coef": obese_coef, "most_common_famkr": most_common_famkr, "n": len(primary),
        "source": "refit by causal_model.py - same spec and same one-knee-per-patient "
                 "(seed 42) sample as run_causal.py's backdoor.linear_regression estimate",
    }
    os.makedirs(os.path.dirname(COEF_PATH), exist_ok=True)
    with open(COEF_PATH, "w") as f:
        json.dump(result, f, indent=2)
    return result


def load_or_fit_causal_regression():
    if os.path.exists(COEF_PATH):
        with open(COEF_PATH) as f:
            return json.load(f)
    return _refit()


def expected_kl(coefs, age, famkr, obese):
    """Expected KL grade (0-4, unclipped linear prediction) for given age, family history
    (0/1) and obesity status (0/1), from the fitted regression-adjustment model."""
    return (coefs["intercept"] + coefs["age_coef"] * age
           + coefs["famkr_coef"] * famkr + coefs["obese_coef"] * obese)


if __name__ == "__main__":
    c = load_or_fit_causal_regression()
    print(json.dumps(c, indent=2))
    print(f"Sanity check vs causal_estimates.csv's linear_regression ATE (+0.485): "
         f"obese_coef = {c['obese_coef']:.4f}")
