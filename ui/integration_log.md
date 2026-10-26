# UI integration log

## Fork comparison (teammate's fork vs. our `main`)

Remote added: `https://github.com/sanchitgharat07-coder/knee-oa-kl-grade-prediction-causal-inferencing`
(as `teammate_fork`), fetched. Compared via `git diff master teammate_fork/master --stat`.

**Framework**: Flask (confirmed from `webapp/app.py`'s imports - `from flask import Flask,
render_template, request, jsonify` - plus a `templates/`/`static/` folder structure; not
Streamlit or Gradio).

**Files the fork adds** (none of these exist on our `main`): `webapp/app.py` (627 lines),
`webapp/templates/index.html`, `webapp/static/css/app.css`, `webapp/static/js/app.js`,
`webapp/README.md`, `webapp/requirements-webapp.txt`.

**Files the fork changes that exist on our `main`**: **`scripts/kl_common.py`** - it rewrites
`grad_cam()` with more defensive error handling (explicit shape checks, `retain_grad()`
instead of a forward/backward hook pair, clearer exceptions) but the same inputs/outputs and
the same underlying math - not a bug fix of anything that was broken (that function is only
buggy in the KneeNeXt dual-view case, which `kl_common.grad_cam` was never used for). **Per
the brief, this change was NOT brought in** - `scripts/kl_common.py` is one of our existing
files and this task's rule is add-only; the fork's version is logged here for the record but
ignored otherwise. No other existing file was touched by the fork (the rest of the diff is
entirely files that exist on our side but not the fork's - `final_output/`,
`scripts/make_final_figures.py`, CLAUDE.md's newer sections - because the fork predates all
of Part 1's work, not because the fork deleted anything).

**Does the fork bring in checkpoints, dataset images, or large files?** No - everything it
adds is source code (`.py`/`.html`/`.css`/`.js`/`.md`/`.txt`). Its `app.py` does reference
checkpoints at runtime (`CHECKPOINT_DIR / "kl_final_best.pt"` etc.) but does not add the
checkpoint files themselves to the repo. Nothing needed appending to `.gitignore` for this
reason.

**What was taken vs. rewritten**: only the UI *files* (templates/static/the Flask app
skeleton) were taken as a starting point, into `ui/`. The *logic* inside `app.py` was
substantially rewritten (see below) - the brief asks for the UI to call
`ui/model_bridge.analyze()` instead of loading checkpoints itself, which the fork's original
`app.py` did directly.

## What changed when adapting the fork's `webapp/` into `ui/`

- `ui/app.py`: the hardcoded `Path.home() / "Desktop" / "knee-oa-kl-grade-prediction-causal-
  inferencing"` project-path assumption (the fork author's own machine's folder name, not
  ours) is gone - replaced with a path relative to this file, so it works on any machine.
  Direct `from scripts.predict import load_models, classify_rgb, ...` is gone - every
  prediction now goes through `ui/model_bridge.analyze()` instead. The `kl_final`/`Hybrid 2`
  model dropdown is gone - `model_bridge` always serves this project's actual final
  configuration (the validated ensemble, or `kl_final` alone if the ensemble didn't beat it -
  decided once, the same way `predict_extended.py` decides it), not a user-selectable
  under-tested alternative. Age/BMI/family history are no longer Hybrid-2-only fields - they
  now feed the causal panel on every request (never the classifier), matching
  `predict_extended.py`'s own rule.
- `ui/templates/index.html` / `ui/static/js/app.js`: kept the overall clinical-workspace
  layout and design language, but replaced the model-selection form with age/BMI/family-
  history fields, and added three result sections the fork's placeholder UI didn't have:
  uncertainty flag, 3 similar training cases, and the extended causal panel (individual
  obesity effect, BMI what-if, knee injury association). The "Patient history" and "Clinical
  report" tabs from the fork (pure UI placeholders with no backend behind them even in the
  fork) were dropped to keep this focused on what `model_bridge.analyze()` actually returns,
  rather than carrying forward empty placeholder views.
- `ui/static/css/app.css`: copied from the fork as-is, plus one small appended block for
  styles the new sections need (`.warn-text`/`.ok-text` for the uncertainty flag,
  `.grade-probabilities`, list styling) that the fork's original CSS didn't have.
- A real bug in the fork's own `app.js` was found and not carried forward: a stray top-level
  statement, `window.latestModelInputImage = data.model_input_image || null;`, references an
  undefined `data` variable outside any function - this would throw a `ReferenceError` on
  page load in the original fork. Not present in `ui/static/js/app.js`.

## Screenshots

**No screenshots could be captured** - this session had no browser-automation tool available
(checked; none was loaded, and the generic web-fetch tool explicitly refuses `localhost`
URLs, which a local Flask server always is). Per the brief's own allowance ("Save screenshots
**or outputs**"), `ui/test_samples/` instead holds: the raw HTML the server actually returned
for `/` (`index_page_rendered.html`, HTTP 200, confirming the page renders), and the full,
real JSON response from a real end-to-end HTTP POST to `/api/analyze` (`api_analyze_response.json`).
Nothing was faked or mocked up as a "screenshot." Because there are no real screenshots,
`final_output/figures/ui/` was not created on this branch - copying placeholder images into a
figures folder would misrepresent them as real UI captures, which this project avoids
throughout.

## Test: analyze() vs. predict_extended.py

`ui/test_samples/test_analyze_vs_predict_extended.py` calls `model_bridge.analyze()` on one
test image per KL grade (same images, same `--age 65 --bmi 32 --family-history yes` inputs
used in `predict_extended.py`'s own demo-check run). Results (`test_analyze_results.txt`):

| Image | True grade | `analyze()` grade | `analyze()` confidence |
|---|---|---|---|
| `9003175L.png` | 0 | 0 | 0.942 |
| `9001400L.png` | 1 | 0 | 0.854 |
| `9003316R.png` | 2 | 0 | 0.491 (flagged uncertain) |
| `9011053L.png` | 3 | 3 | 0.998 |
| `9012867R.png` | 4 | 4 | 0.998 |

These exactly match `predict_extended.py`'s own earlier run on the same five images with the
same inputs (grade, confidence, and the uncertain flag on the KL2 image all identical) -
confirmed by direct comparison against that session's recorded output, not re-derived or
approximated.

## End-to-end HTTP test

Flask server started (`python ui/app.py`, port 5050). `GET /api/health` returned
`{"ok": true, "device_available": "cuda"}`. `GET /` returned HTTP 200 with the rendered page
(6,306 bytes). A real multipart `POST /api/analyze` (image `dataset/test/2/9003316R.png`,
age=65, bmi=32, family_history=yes) returned grade 0, confidence 0.4905, model "ensemble",
uncertainty "REFER TO RADIOLOGIST (uncertain)", 3 similar cases, and both a Grad-CAM and an
Integrated Gradients image (base64 PNG) - matching `analyze()`'s direct-call result exactly.
Server stopped after the test.

## New UI packages

`Flask>=3.0,<4.0` and `Werkzeug>=3.0,<4.0` were installed (not previously in the project venv)
and added to `ui/requirements_ui.txt`. `torch.cuda.is_available()` confirmed `True` both
before and after.
