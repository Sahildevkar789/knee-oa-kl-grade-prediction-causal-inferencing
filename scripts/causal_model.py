"""Fits (or loads a cached) set of causal models behind the "what-if" tool (causal_query.py)
and predict.py's causal risk panel. Same data and DAG spirit as Phase 5's run_causal.py: one
knee per patient (seed 42), age + family history + BMI only (no sex - unavailable, see
CLAUDE.md), never adjusting for WOMAC/JSN/osteophyte grades (they're consequences of OA, not
causes). Point estimates and refuters go through DoWhy; confidence intervals come from an
equivalent statsmodels OLS fit of the same specification (DoWhy's linear_regression estimator
is itself an OLS wrapper with no CI accessor in this dowhy version, so this is a direct
re-fit of the same model, not a different method - the two point estimates are cross-checked
and logged to catch any mismatch).

DAG (extended from Phase 5 to make BMI, not just the Obese threshold, a first-class node, so
age's effect can be split into direct and BMI-mediated parts):
    Age -> BMI; Age -> y; FamilyHistoryKneeReplacement -> y; BMI -> y
and, for the age-group obesity analysis specifically (matches Phase 5 exactly):
    Age -> Obese; Age -> y; FamilyHistoryKneeReplacement -> y; Obese -> y

Not a CLI tool - imported by causal_query.py and predict.py.
"""
import os
import json
import numpy as np
import pandas as pd
import statsmodels.api as sm
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dowhy import CausalModel

OUT_DIR = "outputs/results/causal"
MODELS_PATH = os.path.join(OUT_DIR, "causal_query_models.json")
RESULTS_CSV = os.path.join(OUT_DIR, "causal_query_results.csv")
REFUTE_CSV = os.path.join(OUT_DIR, "causal_query_refutations.csv")
DAG_PATH = os.path.join(OUT_DIR, "causal_query_dag.png")

REFERENCE = {"age": 50.0, "bmi": 25.0, "famkr": 0}
AGE_BANDS = ["<55", "55-65", "65+"]
BMI_BAND_LABELS = ["under25", "25to30", "30to35", "35plus"]
BMI_BAND_DISPLAY = {"under25": "<25", "25to30": "25-30", "30to35": "30-35", "35plus": "35+"}

CAUSAL_CAVEAT = ("These are population-level causal estimates from OAI data under this DAG's "
                 "assumptions, not individual predictions.")


def age_group_label(age):
    if age < 55:
        return "<55"
    if age < 65:
        return "55-65"
    return "65+"


def _load_primary():
    """Same cleaning + one-knee-per-patient sampling (seed 42) as run_causal.py, columns
    renamed once to the names used throughout this module and in the DAG strings below."""
    df = pd.read_csv("data/clinical_features.csv", dtype={"ID": str})
    df["famkr"] = df["P01FAMKR"].str.extract(r"^(\d)").astype(float)
    df = df.dropna(subset=["famkr", "P01BMI", "V00AGE"]).reset_index(drop=True)
    df["obese"] = (df["P01BMI"] >= 30).astype(int)

    rng = np.random.default_rng(42)
    primary = df.groupby("ID", group_keys=False).apply(
        lambda g: g.sample(1, random_state=rng.integers(0, 2**31))).reset_index(drop=True)

    d = primary.rename(columns={"V00AGE": "Age", "P01BMI": "BMI", "famkr": "FamilyHistoryKneeReplacement",
                                "KL_GRADE": "y", "obese": "Obese"})
    return d[["Age", "BMI", "FamilyHistoryKneeReplacement", "y", "Obese"]].astype(
        {"Age": float, "BMI": float, "FamilyHistoryKneeReplacement": float, "y": float, "Obese": float})


def _ols(x_cols, d, y_col="y"):
    """sm.OLS of d[y_col] against an intercept + the given columns (plain additive, no interactions)."""
    X = sm.add_constant(d[x_cols].astype(float))
    return sm.OLS(d[y_col].astype(float), X).fit()


def _coef_block(fit, name, scale=1.0):
    ci = fit.conf_int().loc[name]
    return {"coef": float(fit.params[name] * scale), "se": float(fit.bse[name] * scale),
           "ci_low": float(ci[0] * scale), "ci_high": float(ci[1] * scale)}


def _dowhy_ate(d, treatment, common_causes, label, refute_rows, run_refuters=True):
    """Fits the same spec via DoWhy (for the point-estimate cross-check and, where requested,
    the placebo/random-common-cause refuters), returns the ATE.

    Deliberately NOT passing a graph here: DoWhy's identify_effect(), when given both a graph
    and common_causes, derives the backdoor set purely from graph structure and silently
    ignores common_causes (verified empirically - with a graph, requesting an adjustment
    variable that isn't a graph-parent of the treatment, e.g. the mediator BMI for Age's
    "direct effect", or FamilyHistory for Age's "total effect", gets dropped, and e.g. the
    age_total/age_direct ATEs below came out byte-identical, both effectively just y~Age with
    no adjustment at all). With no graph and only common_causes, DoWhy instead treats
    common_causes as the backdoor set directly - confirmed to reproduce the equivalent
    statsmodels OLS coefficient exactly. The DAG is still drawn (_draw_dag) for the record and
    the write-up; it just isn't passed to CausalModel here."""
    model = CausalModel(data=d, treatment=treatment, outcome="y", common_causes=common_causes)
    identified = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(identified, method_name="backdoor.linear_regression")
    if run_refuters:
        for method, kwargs in [("placebo_treatment_refuter", {"placebo_type": "permute"}),
                               ("random_common_cause", {})]:
            try:
                ref = model.refute_estimate(identified, estimate, method_name=method, **kwargs)
                rr = getattr(ref, "refutation_result", None)
                p_value = rr.get("p_value") if isinstance(rr, dict) else None
                refute_rows.append({"effect": label, "refuter": method,
                                    "original_effect": float(ref.estimated_effect), "new_effect": float(ref.new_effect),
                                    "p_value": p_value})
            except Exception as e:
                refute_rows.append({"effect": label, "refuter": method, "original_effect": float(estimate.value),
                                    "new_effect": None, "p_value": None, "error": str(e)})
    return float(estimate.value)


def _draw_dag():
    g = nx.DiGraph()
    g.add_edges_from([("Age", "BMI"), ("Age", "KL_grade"), ("FamilyHistoryKneeReplacement", "KL_grade"),
                      ("BMI", "KL_grade")])
    pos = {"Age": (0, 1), "FamilyHistoryKneeReplacement": (2, 1), "BMI": (0, 0), "KL_grade": (2, 0)}
    fig, ax = plt.subplots(figsize=(6, 5))
    nx.draw(g, pos, ax=ax, with_labels=True, node_color="#D8E4F0", node_size=3400,
           font_size=8, arrowsize=20, edge_color="#555")
    ax.set_title("Extended causal DAG: Age -> BMI -> KL grade\n(+ direct Age effect, family history)")
    plt.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.savefig(DAG_PATH, dpi=130)
    plt.close(fig)


def _refit():
    d = _load_primary()
    n = len(d)
    refute_rows = []

    # --- Combined model: y ~ Age + BMI + FamilyHistoryKneeReplacement (this IS the age direct
    # effect, holding BMI fixed, and the BMI/family-history effects adjusted for everything else) ---
    m_combined = _ols(["Age", "BMI", "FamilyHistoryKneeReplacement"], d)
    combined_model = {
        "intercept": float(m_combined.params["const"]),
        "age_coef": float(m_combined.params["Age"]), "bmi_coef": float(m_combined.params["BMI"]),
        "famkr_coef": float(m_combined.params["FamilyHistoryKneeReplacement"]),
    }

    # --- (a) Age: total (unconditional on BMI), direct (conditional on BMI), indirect (via BMI) ---
    m_total = _ols(["Age", "FamilyHistoryKneeReplacement"], d)
    m_mediator = _ols(["Age"], d, y_col="BMI")
    a_coef, a_se = float(m_mediator.params["Age"]), float(m_mediator.bse["Age"])
    b_coef, b_se = combined_model["bmi_coef"], float(m_combined.bse["BMI"])
    indirect_per_year = a_coef * b_coef
    se_indirect_per_year = float(np.sqrt(a_coef**2 * b_se**2 + b_coef**2 * a_se**2))  # Sobel SE

    total_per_10y = _coef_block(m_total, "Age", scale=10)
    direct_per_10y = _coef_block(m_combined, "Age", scale=10)
    indirect_per_10y = {"coef": indirect_per_year * 10, "se": se_indirect_per_year * 10,
                        "ci_low": (indirect_per_year - 1.96 * se_indirect_per_year) * 10,
                        "ci_high": (indirect_per_year + 1.96 * se_indirect_per_year) * 10}

    total_per_10y["dowhy_ate_per_10y"] = _dowhy_ate(
        d[["Age", "FamilyHistoryKneeReplacement", "y"]], "Age", ["FamilyHistoryKneeReplacement"],
        "age_total", refute_rows) * 10
    direct_per_10y["dowhy_ate_per_10y"] = _dowhy_ate(
        d[["Age", "BMI", "FamilyHistoryKneeReplacement", "y"]], "Age", ["BMI", "FamilyHistoryKneeReplacement"],
        "age_direct", refute_rows) * 10

    age_effect = {
        "total_per_10y": total_per_10y, "direct_per_10y": direct_per_10y, "indirect_per_10y": indirect_per_10y,
        "consistency_check_per_10y": {"direct_plus_indirect": direct_per_10y["coef"] + indirect_per_10y["coef"],
                                      "total": total_per_10y["coef"]},
    }

    # --- (b) BMI: per-unit effect (adjusting for age + family history) + linearity check ---
    bmi_per_unit = _coef_block(m_combined, "BMI")
    bmi_per_unit["dowhy_ate"] = _dowhy_ate(
        d[["Age", "BMI", "FamilyHistoryKneeReplacement", "y"]], "BMI", ["Age", "FamilyHistoryKneeReplacement"],
        "bmi", refute_rows)

    bins = [-np.inf, 25, 30, 35, np.inf]
    d = d.assign(bmi_band=pd.cut(d["BMI"], bins=bins, labels=BMI_BAND_LABELS))
    dum = pd.get_dummies(d["bmi_band"])[BMI_BAND_LABELS].astype(float)
    X_band = pd.concat([dum.drop(columns=["under25"]), d[["Age", "FamilyHistoryKneeReplacement"]]], axis=1)
    m_band = sm.OLS(d["y"].astype(float), sm.add_constant(X_band)).fit()
    band_mean_bmi = d.groupby("bmi_band", observed=True)["BMI"].mean()
    ref_mean = band_mean_bmi["under25"]

    band_coef, linear_pred = {}, {}
    for lbl in BMI_BAND_LABELS:
        disp = BMI_BAND_DISPLAY[lbl]
        if lbl == "under25":
            band_coef[disp] = {"coef": 0.0, "ci_low": 0.0, "ci_high": 0.0}
        else:
            ci = m_band.conf_int().loc[lbl]
            band_coef[disp] = {"coef": float(m_band.params[lbl]), "ci_low": float(ci[0]), "ci_high": float(ci[1])}
        linear_pred[disp] = float(bmi_per_unit["coef"] * (band_mean_bmi[lbl] - ref_mean))
    roughly_linear = all(
        band_coef[disp]["ci_low"] <= linear_pred[disp] <= band_coef[disp]["ci_high"]
        for disp in list(BMI_BAND_DISPLAY.values())[1:])

    bmi_effect = {
        "per_unit": bmi_per_unit,
        "linearity": {"band_coef": band_coef, "linear_pred": linear_pred, "roughly_linear": bool(roughly_linear),
                     "note": ("The linear BMI effect's predicted step at each band's mean BMI falls inside that "
                             "band's own (dummy-coded) confidence interval - the linear fit is a reasonable "
                             "approximation over this range." if roughly_linear else
                             "At least one band's dummy-coded effect falls outside what the linear BMI fit "
                             "predicts at that band - the relationship is probably not perfectly linear, treat "
                             "the per-unit BMI effect as an average slope over the observed range, not exact "
                             "at every BMI value.")},
    }

    # --- (c) Family history (adjusting for age + BMI) ---
    family_history_effect = _coef_block(m_combined, "FamilyHistoryKneeReplacement")
    family_history_effect["dowhy_ate"] = _dowhy_ate(
        d[["Age", "BMI", "FamilyHistoryKneeReplacement", "y"]], "FamilyHistoryKneeReplacement", ["Age", "BMI"],
        "family_history", refute_rows)

    # --- (d) Obesity effect by age group (matches Phase 5's spec, run per age stratum) ---
    obesity_by_age_group = {}
    for band in AGE_BANDS:
        sub = d[d["Age"].apply(age_group_label) == band].reset_index(drop=True)
        m_sub = _ols(["Obese", "Age", "FamilyHistoryKneeReplacement"], sub)
        block = _coef_block(m_sub, "Obese")
        block["n"] = int(len(sub))
        block["dowhy_ate"] = _dowhy_ate(
            sub[["Obese", "Age", "FamilyHistoryKneeReplacement", "y"]], "Obese",
            ["Age", "FamilyHistoryKneeReplacement"], f"obesity_{band}", refute_rows)
        obesity_by_age_group[band] = block

    _draw_dag()

    result = {
        "reference": REFERENCE, "n": n,
        "data_ranges": {"age_min": float(d["Age"].min()), "age_max": float(d["Age"].max()),
                        "bmi_min": float(d["BMI"].min()), "bmi_max": float(d["BMI"].max())},
        "most_common_famkr": int(d["FamilyHistoryKneeReplacement"].mode().iloc[0]),
        "combined_model": combined_model, "age_effect": age_effect, "bmi_effect": bmi_effect,
        "family_history_effect": family_history_effect, "obesity_by_age_group": obesity_by_age_group,
        "refutations": refute_rows,
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(MODELS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame(refute_rows).round(4).to_csv(REFUTE_CSV, index=False)

    rows = [
        {"Effect": "Age - total (per 10y)", **{k: total_per_10y[k] for k in ("coef", "ci_low", "ci_high", "dowhy_ate_per_10y")}},
        {"Effect": "Age - direct, controlling for BMI (per 10y)", **{k: direct_per_10y[k] for k in ("coef", "ci_low", "ci_high", "dowhy_ate_per_10y")}},
        {"Effect": "Age - indirect via BMI (per 10y)", "coef": indirect_per_10y["coef"], "ci_low": indirect_per_10y["ci_low"], "ci_high": indirect_per_10y["ci_high"]},
        {"Effect": "BMI (per 1 unit, adj. age+family history)", "coef": bmi_per_unit["coef"], "ci_low": bmi_per_unit["ci_low"], "ci_high": bmi_per_unit["ci_high"], "dowhy_ate_per_10y": bmi_per_unit["dowhy_ate"]},
        {"Effect": "Family history (adj. age+BMI)", "coef": family_history_effect["coef"], "ci_low": family_history_effect["ci_low"], "ci_high": family_history_effect["ci_high"], "dowhy_ate_per_10y": family_history_effect["dowhy_ate"]},
    ]
    for band in AGE_BANDS:
        b = obesity_by_age_group[band]
        rows.append({"Effect": f"Obesity, age {band} (adj. age+family history)", "coef": b["coef"],
                    "ci_low": b["ci_low"], "ci_high": b["ci_high"], "dowhy_ate_per_10y": b["dowhy_ate"], "n": b["n"]})
    pd.DataFrame(rows).round(4).to_csv(RESULTS_CSV, index=False)

    return result


def load_or_fit_causal_models(force_refit=False):
    if not force_refit and os.path.exists(MODELS_PATH):
        with open(MODELS_PATH) as f:
            return json.load(f)
    return _refit()


def warn_if_out_of_range(models, age=None, bmi=None):
    """List of warning strings for an age/bmi outside the range actually seen in the fitting
    data - the model should not be trusted to extrapolate silently beyond this range."""
    warnings = []
    r = models["data_ranges"]
    if age is not None and not (r["age_min"] <= age <= r["age_max"]):
        warnings.append(f"Age {age:.0f} is outside the data's observed range "
                        f"({r['age_min']:.0f}-{r['age_max']:.0f}) - this is extrapolation, treat with caution.")
    if bmi is not None and not (r["bmi_min"] <= bmi <= r["bmi_max"]):
        warnings.append(f"BMI {bmi:.1f} is outside the data's observed range "
                        f"({r['bmi_min']:.1f}-{r['bmi_max']:.1f}) - this is extrapolation, treat with caution.")
    return warnings


def expected_kl(models, age, bmi, famkr):
    """Expected KL grade (0-4, unclipped linear prediction) for a given profile, from the
    combined model (y ~ Age + BMI + FamilyHistoryKneeReplacement)."""
    m = models["combined_model"]
    return m["intercept"] + m["age_coef"] * age + m["bmi_coef"] * bmi + m["famkr_coef"] * famkr


def _scale_ci(block, delta):
    """Scales a {coef,ci_low,ci_high} block (stated per unit of its variable) by delta - valid
    because every effect here is a linear-model coefficient, so CIs scale linearly too. A
    negative delta flips which scaled bound is numerically smaller, so re-sort rather than
    assume scaled ci_low stays <= scaled ci_high."""
    lo, hi = block["ci_low"] * delta, block["ci_high"] * delta
    return {"value": block["coef"] * delta, "ci_low": min(lo, hi), "ci_high": max(lo, hi)}


def contribution_age(models, age):
    """This patient's age vs. the reference age (50), decomposed into total/direct/indirect
    (via BMI) contributions to expected KL grade, each per-10-years coefficients scaled to the
    actual age gap."""
    delta_decades = (age - models["reference"]["age"]) / 10.0
    e = models["age_effect"]
    return {"total": _scale_ci(e["total_per_10y"], delta_decades),
           "direct": _scale_ci(e["direct_per_10y"], delta_decades),
           "indirect": _scale_ci(e["indirect_per_10y"], delta_decades)}


def contribution_bmi(models, bmi):
    delta = bmi - models["reference"]["bmi"]
    return _scale_ci(models["bmi_effect"]["per_unit"], delta)


def contribution_famkr(models, famkr):
    delta = famkr - models["reference"]["famkr"]
    return _scale_ci(models["family_history_effect"], delta)


if __name__ == "__main__":
    m = load_or_fit_causal_models(force_refit=True)
    print(json.dumps(m, indent=2))
