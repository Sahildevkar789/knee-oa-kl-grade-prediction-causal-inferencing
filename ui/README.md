# Knee OA Assessment UI (research prototype)

A small Flask UI over this project's final model, via `ui/model_bridge.py` - no model logic
is duplicated here; everything goes through the same code `scripts/predict_extended.py` uses.

**Research prototype — not for clinical use.**

## Install and run

From the project root, with the main project venv active (the UI needs everything already
installed there - torch, econml, captum, pytorch-grad-cam, etc. - plus Flask):

```
venv\Scripts\pip install -r ui\requirements_ui.txt
venv\Scripts\python ui\app.py
```

Then open `http://127.0.0.1:5050/` in a browser. Upload a knee X-ray (PNG/JPG), optionally
fill in age/BMI/family history (used only by the causal risk panel, never by the classifier),
and click "Run assessment."

## What it shows

- The predicted KL grade and per-grade probabilities, from the final model (the validated
  ensemble if it beat `kl_final` on validation, otherwise `kl_final` alone - decided once at
  startup, the same way `predict_extended.py` decides it; which one was used is stated).
- An uncertainty flag ("confident" / "refer to radiologist"), using the entropy threshold
  chosen on validation data in Part C of the extended work.
- Integrated Gradients (on `kl_final`) and a bug-fixed Grad-CAM, generated fresh for each
  request.
- The 3 most similar training cases (cosine similarity on pooled features) with their grades.
- A causal risk panel: population findings, this patient's individual obesity effect (from
  the causal forest, if age/BMI/family history were given and in range), a small BMI
  what-if table, and the knee injury association. Age/BMI outside the OAI study population
  (45-79 / 16.9-48.7) is refused, not silently extrapolated - same rule as
  `scripts/causal_query.py`.

## Files

- `model_bridge.py` - the one function the UI (or anything else) needs: `analyze(image_path,
  age=None, bmi=None, family_history=None)`.
- `app.py` - the Flask app; thin routing/upload-handling only.
- `templates/`, `static/` - the page itself.
- `test_samples/` - test evidence (see `../ui/integration_log.md` for why there are no
  browser screenshots: no browser-automation tool was available in the session that built
  this).
- `integration_log.md` - the full decision trail, including what was and wasn't taken from
  the teammate's fork.
