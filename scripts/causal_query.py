"""Interactive causal "what-if" tool for the OAI knee data: given a patient's age, BMI, and
family history of knee replacement, shows the expected KL grade and a per-factor breakdown of
how each one contributes relative to a reference profile (age 50, BMI 25, no family history),
plus how the obesity effect looks for this patient's age group, and a what-if "change BMI to?"
follow-up. Every underlying effect is fit once (cached, see causal_model.py) via the same data
and DAG spirit as Phase 5 (one knee per patient, seed 42; never adjusting for WOMAC/JSN/
osteophyte grades - they're consequences of OA, not causes).

An age or BMI outside what OAI actually enrolled (45-79 / 16.9-48.7) is refused, not computed -
the model is a straight line fit to that range, and extrapolating it (e.g. to a child) can give
a nonsensical expected KL grade (below 0 or above 4).

Interactive:
    python scripts/causal_query.py
Non-interactive (skips the matching prompt):
    python scripts/causal_query.py --age 65 --bmi 33 --family-history yes
"""
import argparse

from causal_model import (load_or_fit_causal_models, age_group_label, out_of_range_messages,
                          fmt_expected_kl, contribution_age, contribution_bmi, contribution_famkr,
                          fmt_contribution, CAUSAL_CAVEAT)


def prompt_optional(prompt_text, parse=float):
    try:
        raw = input(f"{prompt_text}: ").strip()
    except EOFError:
        raw = ""
    if not raw:
        return None
    try:
        return parse(raw)
    except ValueError:
        print(f"  Couldn't parse '{raw}', try again or leave blank.")
        return prompt_optional(prompt_text, parse)


def parse_yn(raw):
    v = {"y": 1, "yes": 1, "1": 1, "n": 0, "no": 0, "0": 0}.get(raw.strip().lower())
    if v is None:
        raise ValueError(raw)
    return v


def get_profile(args):
    """Age/BMI/family history for this query - --age/--bmi/--family-history skip their prompt.
    Returns None if age or BMI is left blank (profile incomplete - nothing to compute)."""
    age = args.age if args.age is not None else prompt_optional("Age")
    if age is None:
        return None
    bmi = args.bmi if args.bmi is not None else prompt_optional("BMI")
    if bmi is None:
        return None
    if args.family_history is not None:
        famkr = {"yes": 1, "no": 0}[args.family_history]
    else:
        famkr = prompt_optional("Family history of knee replacement (y/n)", parse=parse_yn)
    return age, bmi, famkr


def fmt_age_line(age, age_c):
    if age_c["total"]["is_reference"]:
        return f"Age ({age:.0f} vs 50): same as reference (no contribution)"
    return (f"Age ({age:.0f} vs 50): total {fmt_contribution(age_c['total'])} "
           f"= direct {fmt_contribution(age_c['direct'])} + via BMI {fmt_contribution(age_c['indirect'])}")


def print_profile_report(models, age, bmi, famkr):
    """Prints the full per-factor report, or - if age/BMI falls outside the OAI study
    population - a refusal message instead. Returns (age, bmi, famkr_used) on success, None on
    refusal (nothing was computed)."""
    refusal = out_of_range_messages(models, age, bmi)
    if refusal:
        for m in refusal:
            print(f"  {m}")
        return None

    ref = models["reference"]
    famkr_used = famkr if famkr is not None else models["most_common_famkr"]
    if famkr is None:
        print(f"  Family history skipped - using the population's most common value "
             f"({'yes' if famkr_used else 'no'}).")

    obese = bmi >= 30
    print(f"\nProfile: age {age:.0f}, BMI {bmi:.1f} ({'obese' if obese else 'not obese'}, "
         f"threshold 30), family history {'yes' if famkr_used else 'no'}")
    print(f"Expected KL grade for this profile: {fmt_expected_kl(models, age, bmi, famkr_used)} "
         f"(reference profile age 50/BMI 25/no family history: "
         f"{fmt_expected_kl(models, ref['age'], ref['bmi'], ref['famkr'])})")

    print("\nPer-factor contribution to expected KL grade, relative to the reference profile:")
    age_c = contribution_age(models, age)
    print(f"  {fmt_age_line(age, age_c)}")
    bmi_c = contribution_bmi(models, bmi)
    print(f"  BMI ({bmi:.1f} vs 25): {fmt_contribution(bmi_c)}")
    famkr_c = contribution_famkr(models, famkr_used)
    print(f"  Family history ({'yes' if famkr_used else 'no'} vs no): {fmt_contribution(famkr_c)}")

    band = age_group_label(age)
    ob = models["obesity_by_age_group"][band]
    print(f"\nObesity effect for this patient's age group ({band}, n={ob['n']}): "
         f"+{ob['coef']:.2f} KL grade points (95% CI {ob['ci_low']:.2f} to {ob['ci_high']:.2f})")

    print(f"\nCaveat: {CAUSAL_CAVEAT}")
    return age, bmi, famkr_used


def whatif_loop(models, age, famkr_used):
    """'Change BMI to?' follow-up - loops until a blank answer, then returns to the outer loop.
    A new BMI outside the OAI range is refused the same way as the main profile."""
    while True:
        new_bmi = prompt_optional("Change BMI to? (blank to stop)")
        if new_bmi is None:
            return
        refusal = out_of_range_messages(models, bmi=new_bmi)
        if refusal:
            for m in refusal:
                print(f"  {m}")
            continue
        print(f"  At BMI {new_bmi:.1f} (age {age:.0f}, family history "
             f"{'yes' if famkr_used else 'no'}): expected KL grade "
             f"{fmt_expected_kl(models, age, new_bmi, famkr_used)}")
        print(f"  Caveat: {CAUSAL_CAVEAT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--age", type=float, default=None)
    ap.add_argument("--bmi", type=float, default=None)
    ap.add_argument("--family-history", choices=["yes", "no"], default=None)
    ap.add_argument("--refit", action="store_true", help="Refit all models from data instead of using the cache")
    args = ap.parse_args()

    print("Fitting/loading causal models..." if args.refit else "Loading causal models (cached)...")
    models = load_or_fit_causal_models(force_refit=args.refit)
    print(f"Ready ({models['n']} patients, one knee each, seed 42). "
         f"Data ranges: age {models['data_ranges']['age_min']:.0f}-{models['data_ranges']['age_max']:.0f}, "
         f"BMI {models['data_ranges']['bmi_min']:.1f}-{models['data_ranges']['bmi_max']:.1f}.\n")

    non_interactive = args.age is not None and args.bmi is not None
    while True:
        print("Enter a patient profile (blank age or BMI to quit):")
        profile = get_profile(args)
        if profile is None:
            print("Exiting.")
            return
        age, bmi, famkr = profile
        result = print_profile_report(models, age, bmi, famkr)
        if result is None:
            print()
            if non_interactive:
                return
            continue   # refused - ask for a fresh profile instead of starting the what-if loop
        age, bmi, famkr_used = result
        whatif_loop(models, age, famkr_used)
        print()
        if non_interactive:
            return   # --age/--bmi given: answer once, don't loop asking for a new profile


if __name__ == "__main__":
    main()
