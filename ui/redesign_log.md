# UI redesign log (`ui-redesign` branch)

Branched from `ui-integration` (not `master`) because PR #1 ("UI integration: connect
teammate's fork UI to the final model") was still open/unmerged at the time this work
started - verified via `gh pr list`. `ui-integration` was pulled to latest `origin` first
(already up to date, nothing new). This matches the brief's own fallback instruction.

## Scope discipline

Only `ui/templates/`, `ui/static/` and `ui/app.py` were touched, exactly as the rules
required. `ui/model_bridge.py` was **not edited** - confirmed via `git diff` (no output) at
the end of the task. `scripts/`, `outputs/`, `final_output/`, every checkpoint, and
`ui/README.md`/`ui/integration_log.md`/`ui/requirements_ui.txt`/CLAUDE.md/`.gitignore` are
all unchanged - confirmed the same way. The only pre-existing untracked files
(`outputs/kneenext/checkpoints/*.pt`, left untracked on purpose by the prior KneeNeXt task)
are still untracked and were not touched.

`ui/app.py` changes are additive and don't touch how a prediction is made: one helper
(`_attach_case_thumbnails`, pure file I/O - reads the image path `model_bridge.analyze()`
already returned and resizes/encodes it, nothing model-related) and two new routes:

- `POST /api/whatif` - the redesign's live BMI what-if slider. Calls
  `causal_model.load_or_fit_causal_models()` / `expected_kl()` / `out_of_range_messages()` /
  `contribution_bmi()` directly - the exact same functions `model_bridge.analyze()` already
  calls for its own causal panel, not a reimplementation. Out-of-range age/BMI is refused
  the same way `causal_query.py` refuses it (tested: age 11 → refused with the same message
  text the CLI tool would print).
- `GET /api/causal_insights` - assembles the redesigned Causal insights page's numbers by
  reading `outputs/results/causal/causal_estimates.csv`, `causal_query_results.csv`,
  `refutation_results.csv`, `outputs/extended/causal/injury_estimates.csv` /
  `injury_refutations.csv`, and `final_output/tables/T7_causal.csv` directly - the same
  "never invent a number, read it from the existing results file" discipline
  `scripts/make_final_figures.py` uses. No number in this route is computed on the fly;
  it's reformatting rows that already exist on disk into JSON the frontend can render.

Neither new route calls the classifier, so predictions are unaffected - `/api/analyze`
still calls `model_bridge.analyze()` exactly as before; the only addition there is the
thumbnail attachment, which runs *after* the prediction is already computed and only adds
a cosmetic field to the response.

## What the old UI looked like vs. what changed

The previous `ui-integration` UI (`ui/templates/index.html` + `ui/static/css/app.css` +
`ui/static/js/app.js`) was a flat, light, static clinical-form layout: a static "K+" text
logo, Unicode-glyph icons (`&#9635;`, `&#9121;`, `&mdash;`, etc. - inconsistent rendering
across fonts/platforms, the "fix all broken icons" complaint), a Google Fonts `@import` for
DM Sans/Manrope (**a hard offline blocker** - the viva room may have no internet, so this
had to go), plain prediction text with no severity visualization, raw method names
(`propensity_score_matching`) straight from the CSVs, and developer wording in the subtitle
("ensemble if it beat kl_final on validation" and `predict_extended.py` by name).

Every one of those became a concrete change:

- **Logo**: inline SVG knee/bone glyph, not a text box.
- **Icons**: every icon anywhere in the page (nav, cards, upload, uncertainty banner,
  disclaimer, modal close, download) is now hand-drawn inline SVG (`stroke="currentColor"`,
  so it recolors automatically with the theme) - nothing depends on a font glyph existing.
- **Fonts**: the Google Fonts `@import` is gone. `--font` is a system-font stack
  (`-apple-system, "Segoe UI", Roboto, Helvetica, Arial, system-ui, sans-serif`) - every
  platform has one of these already installed, so the page never makes a network request
  for type. Confirmed by `grep -rniE "https?://|cdn\.|fonts\.|unpkg|jsdelivr"` across
  `ui/templates` and `ui/static/{css,js}` - the only hits are SVG `xmlns` namespace strings
  (not network references, just XML boilerplate).
- **Developer wording**: the prediction card's subtitle and the result's "model line" are
  now built from a small `MEMBER_LABELS` map in `app.js` ("ConvNeXt (kl_final)", "KneeNeXt")
  and read "Final model: ensemble of ConvNeXt (kl_final) and KneeNeXt." - no filename, no
  "beat kl_final on validation" phrasing, anywhere in the rendered page.
  `propensity_score_matching` / `linear_regression` are relabeled "Propensity score
  matching" / "Regression adjustment" via a `METHOD_LABELS` map before they ever reach the
  page.
- **Disclaimer**: kept, but shown once per view as a single `.disclaimer` card with an icon,
  not repeated in three different places the way the old layout did.

## Theme and background animation

Dark ("reading room") is the default (`<html data-theme="dark">`); a sidebar toggle flips
`data-theme` to `"light"` and remembers the choice in `localStorage` (a per-viewer UI
convenience only - no prediction data is ever stored client-side). All colors are CSS
variables on `:root` / `:root[data-theme="light"]`, so no component hardcodes a color.

The background (`.bg-scene`, `position:fixed`, `z-index:0`, `pointer-events:none` so it
never intercepts clicks) is an inline SVG line drawing of a femur and tibia (two closed
`<path>` contours with a gap between them suggesting the joint line) plus a blurred
diagonal "scanning light" bar that sweeps across on a slow 11s loop. Both bone paths use
`pathLength="1"` so the draw-in animation (`stroke-dasharray:1` → `stroke-dashoffset` 1→0→1)
is exact regardless of the path's real geometric length, instead of guessing a dash value
that might not match. Opacity is capped at 16% (10% in light mode) specifically so it never
competes with real content in front of it. `@media (prefers-reduced-motion: reduce)` freezes
the knee-drawing animation at its resting (fully drawn) state and hides the scan sweep -
checked by reading the rule back from the stylesheet, since this sandbox has no way to
toggle the OS-level reduced-motion flag to see it live.

## Interactive features, and how each is actually wired

- **Upload**: the dropzone is a `<label>` wrapping a hidden file input (native
  keyboard/screen-reader accessible) plus `dragenter`/`dragover`/`dragleave`/`drop`
  listeners that toggle `.drag-over` (glowing border, defined as a box-shadow using
  `--accent-soft`) and, on drop, feed the dropped file through the same `acceptFile()` path
  as a normal click-to-browse selection or a demo-sample click - one code path for all three
  entry points.
- **Analysis animation**: `runProgressAnimation()` renders the 6-step checklist from the
  brief and ticks one item "active" → "done" every ~480ms, while the *actual* request is a
  single `fetch("/api/analyze")` running concurrently - exactly as the brief specifies
  ("these are visual stages only; the real call is still one request"). If the real request
  finishes before the checklist does, the checklist is force-completed immediately in
  `progress.finish()`; it never blocks on itself. A scanning line (`.lightbox.scanning`) runs
  over the X-ray preview for the same duration.
- **Grade result**: an animated badge (CSS pop-in), a confidence ring (an SVG `<circle>` with
  `stroke-dasharray` set to its own circumference and the offset animated from "empty" to the
  confidence fraction via a CSS transition), a 5-segment colour-coded severity track
  (`--kl0`...`--kl4`, green→red) with a circular marker whose `left%` is computed from the
  predicted grade, animated probability bars (same technique: set width 0, then set the real
  width on the next animation frame so the CSS transition actually plays), and one of five
  fixed plain-English KL descriptions (standard Kellgren-Lawrence grade descriptions, phrased
  consistently with the grade labels `model_bridge.py` already returns - not a new clinical
  claim). An uncertain prediction shows an amber banner with a CSS `box-shadow` pulse
  (disabled under reduced motion); a confident one shows a plain green check line instead.
- **Explainability viewer**: Integrated Gradients and Grad-CAM tabs stack the heatmap PNG
  `model_bridge.analyze()` already returns *on top of* the plain X-ray preview the browser
  already has locally (the just-uploaded file's own object URL) and controls the heatmap
  layer's CSS `opacity` with a range slider. This is an honest implementation of "fade the
  heatmap in and out": the heatmap images from the backend are pre-blended 60/40 PNGs (see
  `model_bridge._overlay_cam`, unchanged), so stacking a variable-opacity copy of that PNG
  over the plain X-ray still produces the intended fade - more of the plain X-ray shows
  through as opacity drops, more of the heatmap's colour shows through as it rises - without
  asking `model_bridge.py` to return raw, unblended heatmap arrays it doesn't currently
  expose. Clicking either layer opens the zoom modal.
- **Similar cases**: `app.py`'s new `_attach_case_thumbnails()` reads each case's path
  (already returned by `nearest_cases()`, relative to the project root, e.g.
  `dataset\train\0\...png`) and attaches a 160×160 base64 thumbnail, with a path-containment
  check (`os.path.realpath(...).startswith(DATASET_DIR)`) so a card can never be made to
  serve a file outside `dataset/`. Clicking a card opens the zoom modal.
- **Causal panel**: individual obesity effect and the knee-injury association are rendered
  straight from `analyze()`'s existing `causal_panel` fields (unchanged shape). The BMI
  what-if is a genuinely live `<input type="range" min="16.9" max="48.7">` that posts to the
  new `/api/whatif` route on a 180ms debounce as it's dragged, displaying the real expected
  KL grade for that exact BMI (not just the 5 discrete points `model_bridge.analyze()`
  precomputes) - using the age and family-history the patient-details form had at submit
  time. Dragging it below the studied BMI range immediately shows the same refusal message
  `causal_query.py` would print, instead of extrapolating.
- **Causal insights page**: method cards (readable labels, point estimates), a confidence-
  interval bar chart - but only for numbers that actually have a CI in the project's results
  (the three age-group-stratified obesity estimates in `causal_query_results.csv`; the
  primary pooled obesity ATE from `causal_estimates.csv` has no stored CI, so it's shown as a
  plain point estimate rather than inventing an interval for it), the E-value (point and
  CI-bound), the refutation-check summary, the knee-injury association, and the two existing
  report figures (`final_output/figures/bmi_dose_response.png` /
  `obesity_effect.png`, copied once into `ui/static/images/` as the brief allows) with real
  alt text and captions.
- **Demo mode**: `ui/static/samples/sample_kl{0..4}.png` are the same five images already
  used in this project's own `predict_extended.py` demo check and the previous `ui-integration`
  test (`9003175L.png`, `9001400L.png`, `9003316R.png`, `9011053L.png`, `9012867R.png`) -
  reused rather than picked fresh, so there's one consistent set of "known" images across the
  whole project's test history. A click fetches the static file, wraps it in a `File`, and
  feeds it through the exact same `acceptFile()` path as a real upload - no special-cased
  demo logic in the backend at all.
- **Download report**: `window.print()` on a dedicated button; a `@media print` block hides
  the sidebar, background animation, nav, upload controls and footer, and lets the result
  cards (grade, probabilities, heatmap, causal panel, disclaimer) print/PDF plainly on a white
  background.

## Testing

1. **Exact-match test** (`ui/test_samples/redesign/test_redesign_match.py` +
   `test_http_analyze.py`): ran the same 5 test images (one per KL grade, same
   `age=65/bmi=32/family_history=yes` inputs used throughout this project's prior UI tests)
   both directly through `model_bridge.analyze()` and through a live HTTP
   `POST /api/analyze` against the running redesigned server. Result
   (`http_vs_direct_report.txt`): **grade and confidence match exactly** for all 5 images
   (`ALL_MATCH: True`), and the ensemble-vs-kl_final selection log line
   (`Ensemble val accuracy 0.7107 vs kl_final solo val accuracy 0.6755 -> using the ENSEMBLE`)
   is identical to previous sessions' runs - confirming the redesign changed nothing about
   how a prediction is produced. Thumbnails, Grad-CAM, Integrated Gradients and the causal
   panel were all present on every request.
2. **Edge cases** (`test_skip_fields.py`, and a direct `/api/whatif` call): all three patient
   fields skipped → `causal_panel.available == False` with the expected message, not a
   crash; age 11 (out of the 45-79 studied range) via `/api/whatif` → refused with the exact
   `causal_model.out_of_range_messages()` text, not silently computed.
3. **Route smoke test**: `GET /`, `GET /api/health`, `GET /api/causal_insights`,
   `POST /api/whatif` all return 200 with the expected JSON shape (one real bug was caught
   and fixed here: `injury_refutations.csv`'s columns are lowercase
   (`refuter`/`new_effect`/`original_effect`), unlike `refutation_results.csv`'s
   capitalized ones - the route 500'd until that was corrected).
4. **Offline check**: no browser/network-disabling tool was available in this session to
   literally turn off the network adapter, so this was verified the other reliable way - a
   full-text search confirmed zero external hostnames anywhere in `ui/templates/` or
   `ui/static/{css,js}` (the SVG `xmlns` strings are namespace declarations, not network
   requests), the Google Fonts `@import` that was the one real offline blocker in the
   previous version is gone, and every image/script/style reference in the page resolves to
   `ui/static/...` served by the same local Flask process. A page with zero external
   references cannot be blocked by having no internet.
5. **Screenshots**: Playwright was checked (`import playwright` in the project venv) and is
   **not installed** - per the brief's own "if it's available" condition, no browser
   automation was used and no screenshot was faked. `ui/test_samples/redesign/` instead holds
   the same kind of real evidence the previous `ui-integration` task used when it hit the
   same limitation: the raw rendered `GET /` HTML (`index_page_rendered.html`), the real JSON
   test reports above, and the Flask server log (`flask_run.log`).
6. **Scope check**: `git status` and `git diff --stat` (against `scripts/`, `outputs/`,
   `final_output/`, `ui/model_bridge.py`, `ui/README.md`, `ui/integration_log.md`,
   `ui/requirements_ui.txt`, CLAUDE.md, `.gitignore`) both confirm nothing outside
   `ui/templates/`, `ui/static/` and `ui/app.py` changed; the only untracked files are the
   pre-existing KneeNeXt ablation checkpoints (left alone, as before) and this task's own new
   static assets/test evidence.

## New assets added

- `ui/static/samples/sample_kl{0-4}.png` - copies of 5 existing `dataset/test/` images
  (not new data).
- `ui/static/images/bmi_dose_response.png`, `obesity_effect.png` - copies of existing
  `final_output/figures/` PNGs (not regenerated, not new numbers).

No new Python packages were needed for the redesign itself (Flask was already added by the
previous `ui-integration` task); `requests` turned out to be unavailable for this task's own
HTTP test script, so that script uses the standard-library `urllib.request` with a hand-built
multipart body instead of adding a new dependency.
