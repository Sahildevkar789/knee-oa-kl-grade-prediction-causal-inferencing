"""Phase 5: causal effect of obesity (BMI>=30) on KL grade, via DoWhy.

Data limitation (same as Phase 3, logged again here): no sex/gender column exists anywhere
in unified_oai_dataset.csv. Adjustment set is age + family history of knee replacement
(P01FAMKR) only - sex omitted, not fabricated.

Hard rules followed: WOMAC, JSN (P01SV*JS*) and osteophyte (P01SV*OST) columns are never
used as adjustment covariates (they're outcomes/consequences of OA, not causes). Primary
analysis uses one knee per patient (randomly chosen, seed 42) to avoid within-patient
correlation; a sensitivity analysis repeats with both knees.

Not a reusable CLI tool (one-off per Phase 5), kept under scripts/ for the record.
"""
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dowhy import CausalModel

OUT_DIR = "outputs/results/causal"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv("data/clinical_features.csv", dtype={"ID": str})
print(f"Raw: {len(df)} rows, {df.ID.nunique()} unique patients")

# Parse family history (categorical string with a ".: Missing..." sentinel) to 0/1/NaN
df["famkr"] = df["P01FAMKR"].str.extract(r"^(\d)").astype(float)
n_fam_missing = df["famkr"].isna().sum()
n_bmi_missing = df["P01BMI"].isna().sum()
print(f"Dropping {n_fam_missing} rows with missing family history, {n_bmi_missing} with "
     f"missing BMI (small counts - dropping rather than imputing for this analysis, logged).")
df = df.dropna(subset=["famkr", "P01BMI", "V00AGE"]).reset_index(drop=True)
df["obese"] = (df["P01BMI"] >= 30).astype(int)
df["kl_ge2"] = (df["KL_GRADE"] >= 2).astype(int)
print(f"After dropping: {len(df)} rows, {df.ID.nunique()} patients")


def one_knee_per_patient(data, seed=42):
    rng = np.random.default_rng(seed)
    picks = data.groupby("ID", group_keys=False).apply(
        lambda g: g.sample(1, random_state=rng.integers(0, 2**31)))
    return picks.reset_index(drop=True)


def draw_dag():
    g = nx.DiGraph()
    g.add_edges_from([("Age", "Obese"), ("Age", "KL_grade"), ("FamilyHistoryKneeReplacement", "KL_grade"),
                      ("Obese", "KL_grade")])
    pos = {"Age": (0, 1), "FamilyHistoryKneeReplacement": (2, 1), "Obese": (0, 0), "KL_grade": (2, 0)}
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(g, pos, ax=ax, with_labels=True, node_color="#D8E4F0", node_size=3200,
           font_size=8, arrowsize=20, edge_color="#555")
    ax.set_title("Causal DAG: obesity -> KL grade\n(age, family history as confounders)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dag.png"), dpi=130)
    plt.close(fig)
    # Node names here must match the dataframe columns DoWhy actually sees (outcome is always
    # renamed to "y" in run_analysis, regardless of which outcome is being estimated) - the
    # PNG above uses the readable "KL_grade" label purely for the diagram, not for DoWhy.
    return "digraph { Age -> Obese; Age -> y; FamilyHistoryKneeReplacement -> y; Obese -> y; }"


dag_str = draw_dag()
print(f"Saved DAG to {OUT_DIR}/dag.png")


def run_analysis(data, outcome_col, label):
    """Returns (model, identified_estimand, estimate_psm, estimate_linreg)."""
    d = data.rename(columns={"KL_GRADE": "KL_grade", "obese": "Obese", "famkr": "FamilyHistoryKneeReplacement",
                             "V00AGE": "Age"})[["Age", "Obese", "FamilyHistoryKneeReplacement", "KL_grade", "kl_ge2"]]
    target_col = "KL_grade" if outcome_col == "KL_GRADE" else "kl_ge2"
    d = d.rename(columns={target_col: "y"})

    model = CausalModel(data=d, treatment="Obese", outcome="y", graph=dag_str,
                        common_causes=["Age", "FamilyHistoryKneeReplacement"])
    identified = model.identify_effect(proceed_when_unidentifiable=True)

    est_psm = model.estimate_effect(identified, method_name="backdoor.propensity_score_matching")
    est_lin = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    print(f"\n[{label}] Propensity score matching: ATE = {est_psm.value:.4f}")
    print(f"[{label}] Linear regression adjustment: ATE = {est_lin.value:.4f}")
    return model, identified, est_psm, est_lin


results = []

# --- Primary analysis: one knee per patient, all splits ---
primary = one_knee_per_patient(df, seed=42)
print(f"\n=== PRIMARY (one knee/patient, n={len(primary)}) ===")
model_kl, ident_kl, psm_kl, lin_kl = run_analysis(primary, "KL_GRADE", "primary/KL_grade")
results.append({"Analysis": "primary_one_knee", "Outcome": "KL_grade (0-4)", "N": len(primary),
                "Method": "propensity_score_matching", "ATE": psm_kl.value})
results.append({"Analysis": "primary_one_knee", "Outcome": "KL_grade (0-4)", "N": len(primary),
                "Method": "linear_regression", "ATE": lin_kl.value})

model_bin, ident_bin, psm_bin, lin_bin = run_analysis(primary, "kl_ge2", "primary/KL>=2")
results.append({"Analysis": "primary_one_knee", "Outcome": "KL>=2 (binary OA)", "N": len(primary),
                "Method": "propensity_score_matching", "ATE": psm_bin.value})
results.append({"Analysis": "primary_one_knee", "Outcome": "KL>=2 (binary OA)", "N": len(primary),
                "Method": "linear_regression", "ATE": lin_bin.value})

# --- Refuters (on the primary KL_grade linear regression estimate) ---
print("\n=== Refuters (primary KL_grade, linear regression estimate) ===")
refute_rows = []
for method, kwargs in [("placebo_treatment_refuter", {"placebo_type": "permute"}),
                       ("random_common_cause", {}),
                       ("data_subset_refuter", {"subset_fraction": 0.8})]:
    ref = model_kl.refute_estimate(ident_kl, lin_kl, method_name=method, **kwargs)
    print(f"{method}: {ref}")
    rr = getattr(ref, "refutation_result", None)
    p_value = rr.get("p_value") if isinstance(rr, dict) else None
    refute_rows.append({"Refuter": method, "New_effect": ref.new_effect,
                        "Original_effect": ref.estimated_effect, "p_value": p_value})
pd.DataFrame(refute_rows).round(4).to_csv(os.path.join(OUT_DIR, "refutation_results.csv"), index=False)

# --- Sensitivity: both knees (no one-knee-per-patient restriction) ---
print(f"\n=== SENSITIVITY (both knees, n={len(df)}) ===")
model_both, ident_both, psm_both, lin_both = run_analysis(df, "KL_GRADE", "sensitivity/both-knees")
results.append({"Analysis": "sensitivity_both_knees", "Outcome": "KL_grade (0-4)", "N": len(df),
                "Method": "propensity_score_matching", "ATE": psm_both.value})
results.append({"Analysis": "sensitivity_both_knees", "Outcome": "KL_grade (0-4)", "N": len(df),
                "Method": "linear_regression", "ATE": lin_both.value})

results_df = pd.DataFrame(results).round(4)
results_df.to_csv(os.path.join(OUT_DIR, "causal_estimates.csv"), index=False)
print(f"\nSaved {OUT_DIR}/causal_estimates.csv and refutation_results.csv")
print(results_df.to_string(index=False))

# JSW mediation analysis: SKIPPED - Phase 2 found JSW not validated (rho=-0.179, threshold -0.3)
with open(os.path.join(OUT_DIR, "mediation_analysis_SKIPPED.txt"), "w") as f:
    f.write("Mediation analysis (BMI -> medial min JSW -> KL grade) was conditional on JSW "
           "being validated in Phase 2. Phase 2 found medial_min_jsw vs medial JSN "
           "rho=-0.179, below the rho<=-0.3 threshold -> NOT validated. Skipped per the brief.\n")
print(f"Mediation analysis skipped (JSW not validated) - see {OUT_DIR}/mediation_analysis_SKIPPED.txt")
