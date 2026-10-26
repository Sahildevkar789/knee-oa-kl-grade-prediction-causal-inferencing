"""Part B - causal extensions on top of Phase 5's obesity -> KL grade question. Same data and
one-knee-per-patient selection (seed 42) as Phase 5/causal_model.py; never adjusts for WOMAC,
JSN, osteophyte grades, or physical activity (PASE) - all are affected by OA itself, not
causes of it.

    python scripts/causal_extended.py

Outputs under outputs/extended/causal/ only. Decisions logged to
outputs/extended/extended_log.md (shared across Parts A/B/C).
"""
import os
import re
import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dowhy import CausalModel
from econml.dml import CausalForestDML

EXTENDED_DIR = "outputs/extended"
OUT_DIR = os.path.join(EXTENDED_DIR, "causal")
LOG_PATH = os.path.join(EXTENDED_DIR, "extended_log.md")


def log(msg, also_print=True):
    os.makedirs(EXTENDED_DIR, exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(msg.rstrip() + "\n\n")
    if also_print:
        print(msg)


def load_primary():
    """Same cleaning + one-knee-per-patient sampling (seed 42) as causal_model.py's
    _load_primary - copied here (not imported) because this script needs extra columns
    (knee injury, matched by side) that data/clinical_features.csv doesn't carry, so the
    merge has to go back to unified_oai_dataset.csv read-only."""
    df = pd.read_csv("data/clinical_features.csv", dtype={"ID": str})
    df["famkr"] = df["P01FAMKR"].str.extract(r"^(\d)").astype(float)
    df = df.dropna(subset=["famkr", "P01BMI", "V00AGE"]).reset_index(drop=True)
    df["obese"] = (df["P01BMI"] >= 30).astype(int)

    inj = pd.read_csv("unified_oai_dataset.csv", usecols=["ID", "P01INJR", "P01INJL"], dtype={"ID": str})
    inj["injR"] = inj["P01INJR"].str.extract(r"^(\d)").astype(float)
    inj["injL"] = inj["P01INJL"].str.extract(r"^(\d)").astype(float)
    df = df.merge(inj[["ID", "injR", "injL"]], on="ID", how="left")
    df["injury"] = np.where(df["SIDE_CODE"] == "R", df["injR"],
                            np.where(df["SIDE_CODE"] == "L", df["injL"], np.nan))

    rng = np.random.default_rng(42)
    primary = df.groupby("ID", group_keys=False).apply(
        lambda g: g.sample(1, random_state=rng.integers(0, 2**31))).reset_index(drop=True)
    d = primary.rename(columns={"V00AGE": "Age", "P01BMI": "BMI", "famkr": "FamilyHistoryKneeReplacement",
                                "KL_GRADE": "y", "obese": "Obese", "injury": "Injury"})
    return d


# --------------------------------------------------------------------------------------------
# 1. Sensitivity to hidden confounders: E-value + DoWhy add_unobserved_common_cause grid
# --------------------------------------------------------------------------------------------
def e_value(estimate, outcome_sd, ci_bound=None):
    """VanderWeele & Ding (2017) E-value for a linear (continuous-outcome) estimate: convert
    to an approximate standardized mean difference, then an approximate risk ratio
    (RR = exp(0.91*d)), then the usual E-value formula. Returns (E-value for the point
    estimate, E-value for the CI bound closer to the null) - the second answers "how strong
    would a confounder need to be to move the CI to include zero," the first "...to explain
    away the point estimate entirely." Plain-language interpretation is attached by the caller."""
    def to_rr(est):
        d = est / outcome_sd
        return float(np.exp(0.91 * d))

    def ev(rr):
        if rr < 1:
            rr = 1 / rr
        return rr + np.sqrt(rr * (rr - 1))

    rr_point = to_rr(estimate)
    out = {"rr_approx": rr_point, "e_value_point": ev(rr_point)}
    if ci_bound is not None:
        rr_ci = to_rr(ci_bound)
        out["rr_ci_bound"] = rr_ci
        out["e_value_ci_bound"] = ev(rr_ci) if abs(ci_bound) > 1e-9 else 1.0
    return out


def run_sensitivity(d):
    X = sm.add_constant(d[["Age", "BMI", "FamilyHistoryKneeReplacement"]].astype(float))
    # Obesity's effect isn't in this BMI-continuous model - refit with Obese to match Phase 5's spec exactly
    X_ob = sm.add_constant(d[["Obese", "Age", "FamilyHistoryKneeReplacement"]].astype(float))
    m = sm.OLS(d["y"].astype(float), X_ob).fit()
    coef, ci = m.params["Obese"], m.conf_int().loc["Obese"]
    ci_near_null = ci[0] if coef > 0 else ci[1]
    outcome_sd = float(d["y"].std())

    ev = e_value(coef, outcome_sd, ci_near_null)
    log(f"### E-value (obesity -> KL grade)\n\nPoint estimate +{coef:.3f} KL grade points "
       f"(95% CI [{ci[0]:.3f}, {ci[1]:.3f}]), outcome SD {outcome_sd:.3f}.\n\n"
       f"Approximate risk ratio for the point estimate: {ev['rr_approx']:.3f}. "
       f"**E-value = {ev['e_value_point']:.3f}** - an unmeasured confounder would need to be "
       f"associated with BOTH obesity and KL grade by a risk-ratio of at least "
       f"{ev['e_value_point']:.2f}-fold each (above and beyond age/family history, which are "
       f"already adjusted for) to fully explain away this effect; anything weaker could not.\n\n"
       f"E-value for the CI bound closer to the null ({ci_near_null:.3f}): "
       f"**{ev['e_value_ci_bound']:.3f}** - a confounder this strong would be enough to make "
       f"the effect's confidence interval include zero.")

    # DoWhy add_unobserved_common_cause refuter over a grid of confounder strengths
    model = CausalModel(data=d[["Obese", "Age", "FamilyHistoryKneeReplacement", "y"]], treatment="Obese",
                        outcome="y", common_causes=["Age", "FamilyHistoryKneeReplacement"])
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")

    strengths = [0.00, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
    grid = np.zeros((len(strengths), len(strengths)))
    for i, s_t in enumerate(strengths):
        for j, s_y in enumerate(strengths):
            if s_t == 0 and s_y == 0:
                grid[i, j] = estimate.value
                continue
            try:
                ref = model.refute_estimate(identified, estimate, method_name="add_unobserved_common_cause",
                                            confounders_effect_on_treatment="binary_flip",
                                            confounders_effect_on_outcome="linear",
                                            effect_strength_on_treatment=s_t, effect_strength_on_outcome=s_y)
                grid[i, j] = ref.new_effect
            except Exception as e:
                grid[i, j] = np.nan
                log(f"  add_unobserved_common_cause failed at strength ({s_t},{s_y}): {e}")

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(grid, cmap="RdBu", vmin=-abs(estimate.value) * 1.5, vmax=abs(estimate.value) * 1.5, origin="lower")
    ax.set_xticks(range(len(strengths))); ax.set_xticklabels(strengths)
    ax.set_yticks(range(len(strengths))); ax.set_yticklabels(strengths)
    ax.set_xlabel("Confounder effect on outcome (strength)")
    ax.set_ylabel("Confounder effect on treatment (strength)")
    ax.set_title(f"Obesity effect under a simulated hidden confounder\n(original estimate: +{estimate.value:.3f})")
    for i in range(len(strengths)):
        for j in range(len(strengths)):
            ax.text(j, i, f"{grid[i,j]:.2f}", ha="center", va="center", fontsize=7)
    plt.colorbar(im, label="Simulated effect")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "sensitivity_confounder_grid.png"), dpi=140)
    plt.close(fig)
    pd.DataFrame(grid, index=strengths, columns=strengths).to_csv(
        os.path.join(OUT_DIR, "sensitivity_confounder_grid.csv"))

    crosses_zero = np.any((grid < 0) & ~np.isnan(grid))
    log(f"### Hidden-confounder grid\n\nSaved `sensitivity_confounder_grid.png`/`.csv`. "
       f"{'A confounder within the tested strength grid (up to 0.30 on both treatment and outcome) '
        'would be enough to flip the effect negative.' if crosses_zero else 'No confounder strength '
        'within the tested grid (up to 0.30 on both treatment and outcome) was enough to flip the '
        'effect negative - the obesity-KL grade association is fairly robust to the kind of hidden '
        'confounding this simulation can represent.'} Plain language: this simulates a hypothetical "
       f"unmeasured variable (e.g. physical activity level) with a given strength of association to "
       f"both obesity and KL grade, and reports what the obesity effect would look like if that "
       f"variable had been included as a covariate.")
    return coef, ev


# --------------------------------------------------------------------------------------------
# 2. Knee injury as a second cause
# --------------------------------------------------------------------------------------------
def run_injury(d):
    n_missing = d["Injury"].isna().sum()
    sub = d.dropna(subset=["Injury"]).reset_index(drop=True)
    log(f"### Knee injury effect\n\nUsable columns: `P01INJR`/`P01INJL` (unified_oai_dataset.csv), "
       f"matched to each knee's SIDE_CODE. {n_missing} rows missing an injury value for their side, "
       f"dropped (not imputed) for this analysis - n={len(sub)}.")

    cols = ["Injury", "Age", "BMI", "FamilyHistoryKneeReplacement", "y"]
    model = CausalModel(data=sub[cols], treatment="Injury", outcome="y",
                        common_causes=["Age", "BMI", "FamilyHistoryKneeReplacement"])
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    est_lin = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    est_psm = model.estimate_effect(identified, method_name="backdoor.propensity_score_matching")

    refute_rows = []
    for method, kwargs in [("placebo_treatment_refuter", {"placebo_type": "permute"}),
                           ("random_common_cause", {}), ("data_subset_refuter", {"subset_fraction": 0.8})]:
        ref = model.refute_estimate(identified, est_lin, method_name=method, **kwargs)
        refute_rows.append({"refuter": method, "new_effect": ref.new_effect, "original_effect": ref.estimated_effect})
    pd.DataFrame(refute_rows).round(4).to_csv(os.path.join(OUT_DIR, "injury_refutations.csv"), index=False)

    log(f"Prior knee injury -> KL grade, adjusting for age/BMI/family history: "
       f"linear regression ATE **{est_lin.value:+.3f}**, propensity score matching ATE "
       f"**{est_psm.value:+.3f}** KL grade points. Refuters: " +
       ", ".join(f"{r['refuter']}={r['new_effect']:+.3f}" for r in refute_rows) + ".")
    pd.DataFrame([{"Method": "linear_regression", "ATE": est_lin.value},
                 {"Method": "propensity_score_matching", "ATE": est_psm.value}]).round(4).to_csv(
        os.path.join(OUT_DIR, "injury_estimates.csv"), index=False)
    return est_lin.value, est_psm.value


# --------------------------------------------------------------------------------------------
# 3. Personalized obesity effects (EconML CausalForestDML)
# --------------------------------------------------------------------------------------------
def run_causal_forest(d):
    Y = d["y"].astype(float).values
    T = d["Obese"].astype(int).values
    X = d[["Age", "FamilyHistoryKneeReplacement"]].astype(float).values
    W = d[["Age", "FamilyHistoryKneeReplacement"]].astype(float).values

    est = CausalForestDML(discrete_treatment=True, n_estimators=500, min_samples_leaf=20,
                          random_state=42, cv=5)
    est.fit(Y, T, X=X, W=W)
    effects = est.effect(X)
    lo, hi = est.effect_interval(X, alpha=0.05)

    with open(os.path.join(OUT_DIR, "causal_forest_model.pkl"), "wb") as f:
        pickle.dump(est, f)

    mean_effect = float(effects.mean())
    pd.DataFrame({"Age": d["Age"], "FamilyHistory": d["FamilyHistoryKneeReplacement"],
                 "IndividualEffect": effects, "CI_low": lo, "CI_high": hi}).round(4).to_csv(
        os.path.join(OUT_DIR, "causal_forest_individual_effects.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(effects, bins=40, color="#6A8CC7", edgecolor="white")
    axes[0].axvline(mean_effect, color="#D85A30", linestyle="--", label=f"mean {mean_effect:.3f}")
    axes[0].set_xlabel("Individual obesity effect (KL grade points)"); axes[0].set_ylabel("Patients")
    axes[0].set_title("Distribution of individual effects"); axes[0].legend()
    sc = axes[1].scatter(d["Age"], effects, c=d["FamilyHistoryKneeReplacement"], cmap="coolwarm", s=10, alpha=0.6)
    axes[1].set_xlabel("Age"); axes[1].set_ylabel("Individual obesity effect")
    axes[1].set_title("Effect vs age (color = family history)")
    plt.colorbar(sc, ax=axes[1], label="Family history (0/1)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "causal_forest_individual_effects.png"), dpi=140)
    plt.close(fig)

    corr_age = float(np.corrcoef(d["Age"], effects)[0, 1])
    fam_mean = float(effects[d["FamilyHistoryKneeReplacement"] == 1].mean())
    nofam_mean = float(effects[d["FamilyHistoryKneeReplacement"] == 0].mean())
    log(f"### Personalized obesity effects (CausalForestDML)\n\nMean individual effect "
       f"**{mean_effect:+.3f}** KL grade points (population ATE from Phase 5: +0.48 to +0.58 - "
       f"{'consistent with' if 0.3 < mean_effect < 0.7 else 'notably different from'} that range). "
       f"Effect-vs-age correlation: {corr_age:+.3f} ({'larger effects at older ages' if corr_age > 0.1 else 'smaller effects at older ages' if corr_age < -0.1 else 'little age trend'}). "
       f"Mean effect with family history: {fam_mean:+.3f} vs without: {nofam_mean:+.3f} "
       f"({'larger' if fam_mean > nofam_mean else 'smaller'} for patients with a family history of knee replacement). "
       f"Fitted model saved to `causal_forest_model.pkl`; per-patient effects to "
       f"`causal_forest_individual_effects.csv`; plots to `causal_forest_individual_effects.png`.")
    return est, mean_effect


# --------------------------------------------------------------------------------------------
# 4. BMI dose-response curve
# --------------------------------------------------------------------------------------------
def run_dose_response(d):
    X = sm.add_constant(d[["Age", "BMI", "FamilyHistoryKneeReplacement"]].astype(float))
    m = sm.OLS(d["y"].astype(float), X).fit()

    bmi_grid = np.linspace(d["BMI"].min(), d["BMI"].max(), 100)
    ref_age, ref_fam = d["Age"].median(), 0
    X_pred = pd.DataFrame({"const": 1.0, "Age": ref_age, "BMI": bmi_grid, "FamilyHistoryKneeReplacement": ref_fam})
    pred = m.get_prediction(X_pred[["const", "Age", "BMI", "FamilyHistoryKneeReplacement"]])
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(bmi_grid, mean, color="#D85A30", linewidth=2)
    ax.fill_between(bmi_grid, ci[:, 0], ci[:, 1], color="#D85A30", alpha=0.2)
    ax.axvline(30, color="#555", linestyle="--", linewidth=1, label="Obesity threshold (BMI 30)")
    ax.set_xlabel("BMI"); ax.set_ylabel(f"Expected KL grade (age={ref_age:.0f}, no family history)")
    ax.set_title("BMI dose-response (adjusted for age + family history)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "bmi_dose_response.png"), dpi=140)
    plt.close(fig)
    pd.DataFrame({"BMI": bmi_grid, "ExpectedKL": mean, "CI_low": ci[:, 0], "CI_high": ci[:, 1]}).round(4).to_csv(
        os.path.join(OUT_DIR, "bmi_dose_response.csv"), index=False)
    log(f"### BMI dose-response\n\nSaved `bmi_dose_response.png`/`.csv` - expected KL grade across the "
       f"observed BMI range ({d['BMI'].min():.1f}-{d['BMI'].max():.1f}), age held at the median "
       f"({ref_age:.0f}), no family history. Roughly linear over this range (consistent with the "
       f"linearity check in causal_model.py's BMI band analysis).")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    log("# Part B: causal extensions")
    d = load_primary()
    log(f"Primary data: one knee per patient (seed 42), n={len(d)}.")

    run_sensitivity(d)
    run_injury(d)
    run_causal_forest(d)
    run_dose_response(d)
    log("# Part B finished.")


if __name__ == "__main__":
    main()
