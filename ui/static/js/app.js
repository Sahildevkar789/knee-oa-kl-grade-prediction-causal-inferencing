const $ = (id) => document.getElementById(id);
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ========================= Theme ========================= */
function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const toggle = $("theme-toggle");
  const moonIcon = toggle?.querySelector(".icon-moon");
  const sunIcon = toggle?.querySelector(".icon-sun");
  const label = $("theme-toggle-label");
  if (theme === "light") {
    moonIcon?.classList.add("hidden");
    sunIcon?.classList.remove("hidden");
    if (label) label.textContent = "Light mode";
    toggle?.setAttribute("aria-pressed", "true");
  } else {
    sunIcon?.classList.add("hidden");
    moonIcon?.classList.remove("hidden");
    if (label) label.textContent = "Dark mode";
    toggle?.setAttribute("aria-pressed", "false");
  }
}
(function initTheme() {
  let saved = "dark";
  try { saved = localStorage.getItem("koa_theme") || "dark"; } catch (e) { /* ignore */ }
  applyTheme(saved);
})();
$("theme-toggle")?.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  applyTheme(next);
  try { localStorage.setItem("koa_theme", next); } catch (e) { /* ignore */ }
});

/* ========================= View switching (Reading Room vs Causal Insights) ========================= */
const viewNames = ["analysis", "causal"];
function openView(name) {
  viewNames.forEach((view) => {
    const element = $("view-" + view);
    if (element) element.classList.toggle("hidden", view !== name);
  });
  document.querySelectorAll(".sidebar nav .nav").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: prefersReducedMotion ? "auto" : "smooth" });
  if (name === "causal" && !window.__causalLoaded) {
    window.__causalLoaded = true;
    loadCausalInsights();
  }
}
document.querySelectorAll(".sidebar nav .nav").forEach((button) => {
  button.addEventListener("click", () => openView(button.dataset.view));
});

/* ========================= Reading Room Internal Tabs ========================= */
const tabNames = ["jsw", "xai", "causal", "similar"];
function openReadingRoomTab(name) {
  tabNames.forEach((tab) => {
    const pane = $("tab-pane-" + tab);
    if (pane) pane.classList.toggle("hidden", tab !== name);
  });
  document.querySelectorAll(".reading-room-tabs .rr-tab").forEach((btn) => {
    const isActive = btn.dataset.tab === name;
    btn.classList.toggle("active", isActive);
    btn.setAttribute("aria-selected", isActive ? "true" : "false");
  });
}
document.querySelectorAll(".reading-room-tabs .rr-tab").forEach((btn) => {
  btn.addEventListener("click", () => openReadingRoomTab(btn.dataset.tab));
});

/* ========================= Toast Notifications ========================= */
function toast(message) {
  const element = $("toast");
  if (!element) return;
  element.textContent = message;
  element.style.display = "block";
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => { element.style.display = "none"; }, 3400);
}

/* ========================= Zoom Modal ========================= */
function openZoom(src, alt) {
  const modal = $("zoom-modal");
  const img = $("modal-img");
  if (!modal || !img) return;
  img.src = src;
  img.alt = alt || "Enlarged view";
  modal.classList.remove("hidden");
}
function closeZoom() {
  $("zoom-modal")?.classList.add("hidden");
  if ($("modal-img")) $("modal-img").src = "";
}
$("modal-close")?.addEventListener("click", closeZoom);
$("zoom-modal")?.addEventListener("click", (e) => { if (e.target.id === "zoom-modal") closeZoom(); });

/* ========================= Laterality Picker ========================= */
function setLaterality(side) {
  const s = side.toUpperCase() === "L" ? "L" : "R";
  const hiddenInput = $("side");
  if (hiddenInput) hiddenInput.value = s;
  document.querySelectorAll(".lat-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.side === s);
  });
}
document.querySelectorAll(".lat-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    setLaterality(btn.dataset.side);
    // If an assessment has already been run and image is present, live update JSW
    if (window.__lastResult && selectedFile) {
      recalculateJsw(btn.dataset.side);
    }
  });
});

/* ========================= X-ray Selection ========================= */
let selectedImageUrl = null;
let selectedFile = null;

function acceptFile(file) {
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    toast("Please choose a PNG or JPEG radiograph.");
    return;
  }
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedFile = file;
  selectedImageUrl = URL.createObjectURL(file);
  if ($("xray-preview")) $("xray-preview").src = selectedImageUrl;
  if ($("image-name")) $("image-name").textContent = file.name;
  $("image-view")?.classList.remove("hidden");
  $("empty-image")?.classList.add("hidden");
  resetResults();
  toast("Radiograph loaded. Ready to run assessment.");
}

$("xray-input")?.addEventListener("change", (event) => {
  acceptFile(event.target.files?.[0]);
});

const dropzone = $("dropzone");
["dragenter", "dragover"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add("drag-over"); });
});
["dragleave", "drop"].forEach((evt) => {
  dropzone?.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove("drag-over"); });
});
dropzone?.addEventListener("drop", (e) => {
  const file = e.dataTransfer?.files?.[0];
  if (file) acceptFile(file);
});

$("remove-image")?.addEventListener("click", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = null;
  selectedFile = null;
  if ($("xray-input")) $("xray-input").value = "";
  $("xray-preview")?.removeAttribute("src");
  $("image-view")?.classList.add("hidden");
  $("empty-image")?.classList.remove("hidden");
  resetResults();
});

$("xray-preview")?.addEventListener("click", () => {
  if (selectedImageUrl) openZoom(selectedImageUrl, "Selected knee X-ray, enlarged");
});
$("inspect-full-btn")?.addEventListener("click", () => {
  if (selectedImageUrl) openZoom(selectedImageUrl, "Selected knee X-ray, enlarged");
});

/* Demo samples with complete clinical profiles */
const DEMO_SAMPLES = [
  { grade: 0, file: "sample_kl0.png", label: "KL0 Normal", side: "R", age: 58, bmi: 24.2, fh: "no", pain: 0, injury: "no" },
  { grade: 1, file: "sample_kl1.png", label: "KL1 Doubtful", side: "L", age: 63, bmi: 27.5, fh: "no", pain: 2, injury: "no" },
  { grade: 2, file: "sample_kl2.png", label: "KL2 Minimal", side: "R", age: 67, bmi: 29.1, fh: "yes", pain: 5, injury: "no" },
  { grade: 3, file: "sample_kl3.png", label: "KL3 Moderate", side: "R", age: 71, bmi: 33.4, fh: "yes", pain: 7, injury: "yes" },
  { grade: 4, file: "sample_kl4.png", label: "KL4 Severe", side: "R", age: 76, bmi: 36.8, fh: "yes", pain: 9, injury: "yes" },
];

(function buildDemoRow() {
  const row = $("demo-row");
  if (!row) return;
  DEMO_SAMPLES.forEach((sample) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "demo-chip";
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 3l14 9-14 9V3z"/></svg> ${sample.label} (${sample.side})`;
    btn.addEventListener("click", async () => {
      try {
        const resp = await fetch(`/static/samples/${sample.file}`);
        const blob = await resp.blob();
        const file = new File([blob], sample.file, { type: "image/png" });
        setLaterality(sample.side);
        if ($("age")) $("age").value = sample.age;
        if ($("bmi")) $("bmi").value = sample.bmi;
        if ($("family_history")) $("family_history").value = sample.fh;
        if ($("pain_vas")) $("pain_vas").value = sample.pain;
        if ($("knee_injury")) $("knee_injury").value = sample.injury;
        acceptFile(file);
      } catch (e) {
        toast("Could not load sample radiograph.");
      }
    });
    row.appendChild(btn);
  });
})();

function resetResults() {
  $("open-report-btn")?.setAttribute("disabled", "true");
  $("download-report-btn")?.setAttribute("disabled", "true");
  $("progress-list")?.classList.add("hidden");
  if ($("progress-list")) $("progress-list").innerHTML = "";
  window.__lastResult = null;
}

/* ========================= Clinical Definitions ========================= */
const GRADE_LABELS = ["No Osteoarthritis", "Doubtful OA", "Minimal OA", "Moderate OA", "Severe OA"];
const GRADE_DESCRIPTIONS = [
  "Normal joint space width. No radiographic evidence of osteophytes or sclerosis.",
  "Doubtful joint space narrowing; minute osteophytic spicule of questionable significance.",
  "Definite osteophytes present with possible joint space narrowing; subchondral integrity maintained.",
  "Multiple moderate osteophytes, definite joint space narrowing, some sclerosis, possible bone contour deformity.",
  "Severe joint space narrowing / bone-on-bone contact, marked subchondral sclerosis, and gross contour destruction.",
];
const GRADE_COLOR_VARS = ["--kl0", "--kl1", "--kl2", "--kl3", "--kl4"];
const MEMBER_LABELS = {
  kl_final: "ConvNeXt (kl_final)",
  kl_final_seed7: "ConvNeXt (kl_final)",
  hybrid1: "Transformer hybrid",
  kneenext_seed42: "KneeNeXt",
  kneenext_seed7: "KneeNeXt",
};
function memberLabel(key) { return MEMBER_LABELS[key] || key; }

/* ========================= Progress Animation ========================= */
const PROGRESS_STEPS = [
  "Preprocessing & CLAHE contrast",
  "Segmenting joint boundaries",
  "Measuring Medial & Lateral JSW",
  "Running ConvNeXt feature extractor",
  "Running KneeNeXt ordinal model",
  "Ensembling & calculating saliency",
];
function runProgressAnimation() {
  const list = $("progress-list");
  if (!list) return { finish: () => {} };
  list.classList.remove("hidden");
  list.innerHTML = PROGRESS_STEPS.map((label, i) =>
    `<li data-i="${i}"><span class="dot"></span><span>${label}</span></li>`).join("");
  const items = Array.from(list.children);
  let i = 0;
  function tick() {
    if (i > 0) items[i - 1].classList.replace("active", "done");
    if (i < items.length - 1) {
      items[i].classList.add("active");
      i += 1;
      timer = setTimeout(tick, prefersReducedMotion ? 0 : 420);
    } else {
      items[i].classList.add("active");
    }
  }
  let timer = setTimeout(tick, 0);
  $("image-view")?.classList.add("scanning");
  return {
    finish(success) {
      clearTimeout(timer);
      items.forEach((el) => { el.classList.remove("active"); el.classList.add(success ? "done" : "active"); });
      $("image-view")?.classList.remove("scanning");
    },
  };
}

/* ========================= Rendering: KL Prediction ========================= */
function renderPrediction(data) {
  const panel = $("prediction-panel");
  if (!panel) return;
  panel.replaceChildren();
  panel.classList.remove("prediction-empty");

  const top = document.createElement("div");
  top.className = "result-top";

  const badge = document.createElement("div");
  badge.className = "grade-badge";
  badge.style.borderColor = `var(${GRADE_COLOR_VARS[data.grade]})`;
  badge.innerHTML = `<span class="num">${data.grade}</span><span class="lbl">KL GRADE</span>`;

  const ring = document.createElement("div");
  ring.className = "conf-ring-wrap";
  const circumference = 2 * Math.PI * 36;
  ring.innerHTML = `
    <svg viewBox="0 0 80 80">
      <circle class="ring-bg" cx="40" cy="40" r="36"/>
      <circle class="ring-fg" cx="40" cy="40" r="36" style="stroke-dasharray:${circumference};stroke-dashoffset:${circumference};stroke:var(${GRADE_COLOR_VARS[data.grade]})"/>
    </svg>
    <div class="ring-label">${Math.round(data.confidence * 100)}%</div>`;

  const text = document.createElement("div");
  text.className = "result-text";
  const memberNames = (data.model_members || []).map(memberLabel);
  const modelLine = data.model_used === "ensemble"
    ? `Ensemble of ${memberNames.join(" and ")}`
    : `${memberNames[0] || "ConvNeXt"}`;
  text.innerHTML = `
    <b>Grade ${data.grade} — ${GRADE_LABELS[data.grade]}</b>
    <p class="grade-desc">${GRADE_DESCRIPTIONS[data.grade]}</p>
    <p class="model-line">${modelLine}</p>`;

  top.append(badge, ring, text);
  panel.appendChild(top);

  requestAnimationFrame(() => {
    const offset = circumference * (1 - data.confidence);
    const fg = ring.querySelector(".ring-fg");
    if (fg) fg.style.strokeDashoffset = String(offset);
  });

  const scale = document.createElement("div");
  scale.className = "severity-scale";
  scale.innerHTML = `
    <div class="severity-track"><span></span><span></span><span></span><span></span><span></span></div>
    <div class="severity-labels"><span>KL0</span><span>KL1</span><span>KL2</span><span>KL3</span><span>KL4</span></div>
    <div class="severity-marker" id="severity-marker"></div>`;
  panel.appendChild(scale);
  requestAnimationFrame(() => {
    const marker = scale.querySelector("#severity-marker");
    if (marker) marker.style.left = `${(data.grade + 0.5) * 20}%`;
  });

  const bars = document.createElement("div");
  bars.className = "prob-bars";
  Object.entries(data.probabilities).forEach(([label, value], idx) => {
    const pct = Math.max(0, Math.min(100, value * 100));
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML = `<span class="k-label">${label}</span><div class="track"><div class="fill" style="background:var(${GRADE_COLOR_VARS[idx]})"></div></div><span class="pct">${pct.toFixed(1)}%</span>`;
    bars.appendChild(row);
    requestAnimationFrame(() => {
      const fill = row.querySelector(".fill");
      if (fill) fill.style.width = `${pct}%`;
    });
  });
  panel.appendChild(bars);

  const banner = document.createElement("div");
  const isUncertain = !!data.uncertainty?.flag;
  banner.className = `uncertainty-banner ${isUncertain ? "warn" : "ok"}`;
  banner.innerHTML = isUncertain
    ? `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2 1 21h22L12 2z"/><path d="M12 9v5M12 17v.01"/></svg><span>High uncertainty / borderline grade — specialist radiologist review advised.</span>`
    : `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6 9 17l-5-5"/></svg><span>High model confidence across ordinal threshold margins.</span>`;
  panel.appendChild(banner);

  const tag = $("model-used-tag");
  if (tag) { tag.textContent = `KL ${data.grade} CLASSIFIED`; tag.classList.remove("muted"); }
  const hint = $("model-line-hint");
  if (hint) hint.textContent = modelLine;

  $("open-report-btn")?.removeAttribute("disabled");
  $("download-report-btn")?.removeAttribute("disabled");
}

/* ========================= Rendering: Joint Space Width & Gap Studio ========================= */
let currentDicomFilters = { brightness: 0, contrast: 0, invert: false };

function updateDicomFilters() {
  const b = 1 + currentDicomFilters.brightness / 100;
  const c = 1 + currentDicomFilters.contrast / 100;
  const inv = currentDicomFilters.invert ? "invert(100%)" : "invert(0%)";
  const filterStr = `brightness(${b}) contrast(${c}) ${inv}`;

  const imgAnnotated = $("jsw-annotated-img");
  const imgOriginal = $("jsw-original-img");
  const splitLeft = $("jsw-split-left");
  const splitRight = $("jsw-split-right");

  [imgAnnotated, imgOriginal, splitLeft, splitRight].forEach((img) => {
    if (img) img.style.filter = filterStr;
  });
}

$("dicom-brightness")?.addEventListener("input", (e) => {
  currentDicomFilters.brightness = Number(e.target.value);
  updateDicomFilters();
});
$("dicom-contrast")?.addEventListener("input", (e) => {
  currentDicomFilters.contrast = Number(e.target.value);
  updateDicomFilters();
});
$("dicom-invert")?.addEventListener("click", () => {
  currentDicomFilters.invert = !currentDicomFilters.invert;
  $("dicom-invert")?.classList.toggle("active", currentDicomFilters.invert);
  updateDicomFilters();
});
$("dicom-reset")?.addEventListener("click", () => {
  currentDicomFilters = { brightness: 0, contrast: 0, invert: false };
  if ($("dicom-brightness")) $("dicom-brightness").value = "0";
  if ($("dicom-contrast")) $("dicom-contrast").value = "0";
  $("dicom-invert")?.classList.remove("active");
  updateDicomFilters();
});

// View mode switcher (Highlighted Gap / Original / Side-by-side)
document.querySelectorAll(".jsw-mode-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".jsw-mode-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const mode = btn.dataset.mode;
    const ann = $("jsw-annotated-img");
    const orig = $("jsw-original-img");
    const split = $("jsw-split-wrap");
    if (!ann || !orig || !split) return;

    if (mode === "highlighted") {
      ann.classList.remove("hidden");
      orig.classList.add("hidden");
      split.classList.add("hidden");
    } else if (mode === "original") {
      ann.classList.add("hidden");
      orig.classList.remove("hidden");
      split.classList.add("hidden");
    } else if (mode === "split") {
      ann.classList.add("hidden");
      orig.classList.add("hidden");
      split.classList.remove("hidden");
    }
  });
});

function renderJswStudio(jsw, originalSrc) {
  const emptyState = $("jsw-empty-state");
  const displayArea = $("jsw-display-area");
  const metricsBody = $("jsw-metrics-body");
  if (!jsw || !metricsBody) return;

  if (emptyState) emptyState.classList.add("hidden");
  if (displayArea) displayArea.classList.remove("hidden");

  // Populate images
  const annSrc = jsw.annotated_png_base64 ? `data:image/png;base64,${jsw.annotated_png_base64}` : originalSrc;
  const annImg = $("jsw-annotated-img");
  const origImg = $("jsw-original-img");
  const splitL = $("jsw-split-left");
  const splitR = $("jsw-split-right");

  if (annImg) { annImg.src = annSrc; annImg.onclick = () => openZoom(annSrc, "Annotated Joint Space Width with Gap Ribbon"); }
  if (origImg) { origImg.src = originalSrc; origImg.onclick = () => openZoom(originalSrc, "Original Radiograph"); }
  if (splitL) splitL.src = originalSrc;
  if (splitR) splitR.src = annSrc;

  // Download annotated image button handler
  const dlBtn = $("jsw-download-img-btn");
  if (dlBtn) {
    dlBtn.onclick = () => {
      const a = document.createElement("a");
      a.href = annSrc;
      a.download = `knee_jsw_${jsw.side}_annotated.png`;
      a.click();
      toast("Downloaded annotated JSW radiograph.");
    };
  }

  // Header laterality text
  const latHint = $("jsw-laterality-hint");
  if (latHint) latHint.textContent = `Laterality: ${jsw.side_label} (${jsw.side === "R" ? "Image-Right = Medial" : "Image-Left = Medial"})`;

  // Render Morphometrics Dashboard
  metricsBody.replaceChildren();

  // 1. OARSI JSN Grade Banner
  const jsn = jsw.jsn_classification;
  const banner = document.createElement("div");
  banner.className = `jsn-grade-banner ${jsn.severity}`;
  banner.innerHTML = `
    <div class="jsn-banner-top">
      <b>${jsn.label}</b>
      <span class="jsn-pill ${jsn.severity}">${jsn.severity} Narrowing</span>
    </div>
    <p class="jsn-desc">${jsn.description}</p>
    <div class="jsn-meter">
      <div class="jsn-meter-track"><span></span><span></span><span></span><span></span></div>
      <div class="jsn-meter-labels"><span>Normal (&gt;4mm)</span><span>Mild (3-4mm)</span><span>Mod (1.5-3mm)</span><span>Severe (&lt;1.5mm)</span></div>
    </div>`;
  metricsBody.appendChild(banner);

  // 2. Medial & Lateral Compartment Cards Grid
  const compGrid = document.createElement("div");
  compGrid.className = "compartment-grid";

  const med = jsw.medial;
  const lat = jsw.lateral;

  const medCard = document.createElement("div");
  medCard.className = `comp-card ${med.min_mm < 3.0 ? "active-narrow" : ""}`;
  medCard.innerHTML = `
    <div class="comp-card-head">
      <span class="comp-title">${med.name}</span>
      <span class="comp-pos">${med.position}</span>
    </div>
    <div class="comp-main-stat">
      <span class="stat-val">${med.min_mm.toFixed(2)}</span>
      <span class="stat-unit">mm min</span>
    </div>
    <div class="comp-sub-stat"><span>Mean Compartment JSW:</span><b>${med.mean_mm.toFixed(2)} mm</b></div>
    <div class="comp-sub-stat"><span>Raw Min Pixel Gap:</span><b>${med.min_px.toFixed(1)} px</b></div>
    <div class="comp-sub-stat"><span>Status:</span><b style="color:${med.min_mm < 1.5 ? "var(--kl4)" : med.min_mm < 3.0 ? "var(--kl2)" : "var(--kl0)"}">${med.min_mm < 1.5 ? "Severe Narrowing" : med.min_mm < 3.0 ? "Moderate Narrowing" : med.min_mm < 4.0 ? "Mild Narrowing" : "Normal Joint Space"}</b></div>`;

  const latCard = document.createElement("div");
  latCard.className = `comp-card ${lat.min_mm < 3.0 ? "active-narrow" : ""}`;
  latCard.innerHTML = `
    <div class="comp-card-head">
      <span class="comp-title">${lat.name}</span>
      <span class="comp-pos">${lat.position}</span>
    </div>
    <div class="comp-main-stat">
      <span class="stat-val">${lat.min_mm.toFixed(2)}</span>
      <span class="stat-unit">mm min</span>
    </div>
    <div class="comp-sub-stat"><span>Mean Compartment JSW:</span><b>${lat.mean_mm.toFixed(2)} mm</b></div>
    <div class="comp-sub-stat"><span>Raw Min Pixel Gap:</span><b>${lat.min_px.toFixed(1)} px</b></div>
    <div class="comp-sub-stat"><span>Status:</span><b style="color:${lat.min_mm < 1.5 ? "var(--kl4)" : lat.min_mm < 3.0 ? "var(--kl2)" : "var(--kl0)"}">${lat.min_mm < 1.5 ? "Severe Narrowing" : lat.min_mm < 3.0 ? "Moderate Narrowing" : lat.min_mm < 4.0 ? "Mild Narrowing" : "Normal Joint Space"}</b></div>`;

  compGrid.append(medCard, latCard);
  metricsBody.appendChild(compGrid);

  // 3. Alignment Assessment Card
  const alignCard = document.createElement("div");
  alignCard.className = "alignment-block";
  alignCard.innerHTML = `
    <div class="alignment-head">
      <b>Coronal Alignment Assessment</b>
      <span class="alignment-ratio">M:L Ratio = ${jsw.ml_ratio.toFixed(2)}</span>
    </div>
    <p class="alignment-note">${jsw.alignment_note}</p>`;
  metricsBody.appendChild(alignCard);

  // 4. Calibration scale info
  const calNote = document.createElement("div");
  calNote.className = "calibration-note";
  calNote.innerHTML = `
    <span>Plateau Width: <b>${jsw.joint_width_px} px</b></span>
    <span>Calibration Scale: <b>${jsw.mm_per_px.toFixed(4)} mm/px</b></span>
    <span>Anatomical Reference: <b>75 mm</b></span>`;
  metricsBody.appendChild(calNote);
}

// Quick Side Switcher button (Right Knee ↔ Left Knee)
async function recalculateJsw(newSide) {
  if (!selectedFile) return;
  const btn = $("quick-toggle-side-btn");
  if (btn) btn.disabled = true;
  toast(`Re-aligning for ${newSide === "R" ? "Right" : "Left"} Knee…`);

  const fd = new FormData();
  fd.append("xray", selectedFile);
  fd.append("side", newSide);

  try {
    const resp = await fetch("/api/measure_jsw", { method: "POST", body: fd });
    const data = await resp.json();
    if (data.ok && data.jsw) {
      if (window.__lastResult) window.__lastResult.jsw = data.jsw;
      setLaterality(newSide);
      renderJswStudio(data.jsw, selectedImageUrl);
      toast(`Updated for ${data.jsw.side_label}.`);
    } else {
      toast("Could not update joint space measurement.");
    }
  } catch (err) {
    toast("Re-measurement error.");
  } finally {
    if (btn) btn.disabled = false;
  }
}
$("quick-toggle-side-btn")?.addEventListener("click", () => {
  const currentSide = $("side")?.value || "R";
  const newSide = currentSide === "R" ? "L" : "R";
  recalculateJsw(newSide);
});

/* ========================= Rendering: Explainability Heatmaps ========================= */
function renderXai(data) {
  const el = $("xai-panel");
  if (!el || !selectedImageUrl) return;
  el.replaceChildren();

  const methods = [];
  if (data.integrated_gradients_png_base64) {
    methods.push({ key: "ig", label: "Integrated Gradients (Main Attributions)", src: `data:image/png;base64,${data.integrated_gradients_png_base64}` });
  }
  if (data.gradcam_png_base64) {
    methods.push({ key: "cam", label: "Grad-CAM Activation Map", src: `data:image/png;base64,${data.gradcam_png_base64}` });
  }
  if (methods.length === 0) {
    el.innerHTML = `<div class="empty-state"><h3>No explanation heatmaps generated</h3><p>Gradient heatmaps could not be created for this image format.</p></div>`;
    return;
  }

  const tabs = document.createElement("div");
  tabs.className = "xai-tabs";
  methods.forEach((m, i) => {
    const tab = document.createElement("button");
    tab.type = "button";
    tab.className = `xai-tab ${i === 0 ? "active" : ""}`;
    tab.textContent = m.label;
    tab.dataset.key = m.key;
    tabs.appendChild(tab);
  });
  el.appendChild(tabs);

  const viewer = document.createElement("div");
  viewer.className = "xai-viewer";
  const baseImg = document.createElement("img");
  baseImg.className = "layer base";
  baseImg.src = selectedImageUrl;
  baseImg.alt = "Original Knee Radiograph";
  const overlayImg = document.createElement("img");
  overlayImg.className = "layer overlay";
  overlayImg.alt = "Saliency Activation Overlay";
  overlayImg.src = methods[0].src;
  overlayImg.style.opacity = "0.85";
  viewer.append(baseImg, overlayImg);
  el.appendChild(viewer);

  [baseImg, overlayImg].forEach((img) => img.addEventListener("click", () => openZoom(overlayImg.src, overlayImg.alt)));

  const opacityRow = document.createElement("div");
  opacityRow.className = "xai-opacity-row";
  opacityRow.innerHTML = `<label for="xai-opacity">Heatmap Opacity</label><input type="range" id="xai-opacity" min="0" max="100" value="85" aria-label="Heatmap overlay opacity">`;
  el.appendChild(opacityRow);
  opacityRow.querySelector("input").addEventListener("input", (e) => {
    overlayImg.style.opacity = String(Number(e.target.value) / 100);
  });

  tabs.querySelectorAll(".xai-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.querySelectorAll(".xai-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      const m = methods.find((mm) => mm.key === tab.dataset.key);
      if (m) overlayImg.src = m.src;
    });
  });

  const caption = document.createElement("p");
  caption.className = "xai-caption";
  caption.textContent = "Integrated Gradients computes path integrals over input pixel gradients; Grad-CAM highlights deep convolutional feature maps concentrated on condylar margins.";
  el.appendChild(caption);
}

/* ========================= Rendering: Similar Reference Cases ========================= */
function renderSimilarCases(data) {
  const el = $("similar-cases-panel");
  if (!el) return;
  el.replaceChildren();
  if (!data.similar_cases || data.similar_cases.length === 0) {
    el.innerHTML = `<div class="empty-state"><h3>No Reference Cases Located</h3><p>Training latent embeddings database was not found.</p></div>`;
    return;
  }
  const grid = document.createElement("div");
  grid.className = "case-grid";
  data.similar_cases.forEach((c) => {
    const card = document.createElement("div");
    card.className = "case-card";
    const thumbSrc = c.thumbnail_png_base64 ? `data:image/png;base64,${c.thumbnail_png_base64}` : null;
    card.innerHTML = `
      ${thumbSrc ? `<img src="${thumbSrc}" alt="Reference Case KL${c.grade}">` : `<div class="empty-image" style="height:100px"><span>No preview</span></div>`}
      <div class="meta">
        <b>Grade ${c.grade} — ${GRADE_LABELS[c.grade]}</b>
        <span>${(c.similarity * 100).toFixed(1)}% Latent Similarity</span>
      </div>`;
    if (thumbSrc) card.addEventListener("click", () => openZoom(thumbSrc, `Reference case Grade ${c.grade}`));
    grid.appendChild(card);
  });
  el.appendChild(grid);
}

/* ========================= Rendering: Causal Panel ========================= */
function renderCausalPanel(panel, baseAge, baseFamilyHistory) {
  const el = $("causal-panel-result");
  if (!el) return;
  el.replaceChildren();

  if (panel.refused) {
    const div = document.createElement("div");
    div.className = "small-warning";
    div.textContent = panel.messages.join(" ");
    el.appendChild(div);
    return;
  }
  if (!panel.available) {
    el.innerHTML = `<div class="empty-state"><h3>Add Age and BMI for Causal Modeling</h3><p>${panel.message || "Provide patient age and BMI for counterfactual risk breakdown."}</p></div>`;
    return;
  }

  const summary = document.createElement("div");
  summary.className = "causal-block";
  const lines = [];
  lines.push(`<div class="stat-line"><span>Patient Anthropometric Profile</span><b>${panel.obese ? "Obese" : "Non-obese"} (BMI ${panel.bmi})</b></div>`);
  lines.push(`<div class="stat-line"><span>Expected Radiographic KL Severity</span><b>${panel.expected_kl.value.toFixed(2)} pts ${panel.expected_kl.clipped ? "(clipped)" : ""}</b></div>`);
  if (panel.individual_obesity_effect) {
    const e = panel.individual_obesity_effect;
    lines.push(`<div class="stat-line"><span>Individual Obesity Excess Risk</span><b>${e.value >= 0 ? "+" : ""}${e.value.toFixed(2)} KL points (95% CI: ${e.ci_low.toFixed(2)} to ${e.ci_high.toFixed(2)})</b></div>`);
  }
  summary.innerHTML = `<h4>Individual Risk Decomposition</h4>${lines.join("")}`;
  el.appendChild(summary);

  if (panel.knee_injury) {
    const injury = document.createElement("div");
    injury.className = "causal-block";
    injury.innerHTML = `<h4>Knee Injury Association</h4>
      <div class="stat-line"><span>Regression Adjustment ATE</span><b>+${panel.knee_injury.linear_regression_ate.toFixed(2)} KL pts</b></div>
      <div class="stat-line"><span>Propensity Score Matching ATE</span><b>+${panel.knee_injury.propensity_score_matching_ate.toFixed(2)} KL pts</b></div>
      <p class="hint" style="margin-top:6px">Observational association from patient history; does not prove individual knee trauma etiology.</p>`;
    el.appendChild(injury);
  }

  const whatif = document.createElement("div");
  whatif.className = "causal-block";
  whatif.innerHTML = `<h4>Lifestyle Intervention & BMI What-If Simulator</h4>
    <p class="hint">Adjust target BMI to evaluate anticipated counterfactual change in expected OA severity:</p>
    <div class="slider-row">
      <input type="range" id="bmi-whatif-slider" min="16.9" max="48.7" step="0.1" value="${panel.bmi}" aria-label="Target BMI slider">
      <span class="slider-readout" id="bmi-whatif-readout">${panel.bmi}</span>
    </div>
    <div class="whatif-result" id="bmi-whatif-result">Expected KL grade at target BMI ${panel.bmi}: ${panel.expected_kl.value.toFixed(2)}</div>`;
  el.appendChild(whatif);

  const slider = whatif.querySelector("#bmi-whatif-slider");
  const readout = whatif.querySelector("#bmi-whatif-readout");
  const resultBox = whatif.querySelector("#bmi-whatif-result");
  let debounceTimer = null;
  slider.addEventListener("input", () => {
    const bmiVal = Number(slider.value);
    readout.textContent = bmiVal.toFixed(1);
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      try {
        const resp = await fetch("/api/whatif", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ age: baseAge, bmi: bmiVal, family_history: baseFamilyHistory }),
        });
        const d = await resp.json();
        if (d.ok) {
          resultBox.textContent = `Expected KL score at target BMI ${bmiVal.toFixed(1)}: ${d.expected_kl.value.toFixed(2)}${d.expected_kl.clipped ? " (clipped)" : ""}`;
        } else if (d.refused) {
          resultBox.textContent = d.messages.join(" ");
        }
      } catch (e) { /* keep last good value */ }
    }, 180);
  });

  const caveat = document.createElement("div");
  caveat.className = "small-warning";
  caveat.textContent = panel.caveat;
  el.appendChild(caveat);
}

/* ========================= Run Assessment Handler ========================= */
$("analyze-btn")?.addEventListener("click", async () => {
  if (!selectedFile) { toast("Select a knee X-ray radiograph first."); return; }

  const button = $("analyze-btn");
  button.disabled = true;
  if ($("analysis-status")) $("analysis-status").textContent = "Running neural network and JSW measurement…";
  const progress = runProgressAnimation();

  const formData = new FormData();
  formData.append("xray", selectedFile);
  const age = $("age")?.value; if (age) formData.append("age", age);
  const bmi = $("bmi")?.value; if (bmi) formData.append("bmi", bmi);
  const fh = $("family_history")?.value; if (fh) formData.append("family_history", fh);
  const side = $("side")?.value || "R"; formData.append("side", side);
  const pain = $("pain_vas")?.value; if (pain) formData.append("pain_vas", pain);
  const injury = $("knee_injury")?.value; if (injury) formData.append("knee_injury", injury);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData, cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.message || `Assessment failed (HTTP ${response.status}).`);
    }
    progress.finish(true);
    renderPrediction(data);
    if (data.jsw) {
      renderJswStudio(data.jsw, selectedImageUrl);
      openReadingRoomTab("jsw"); // Auto-focus the Joint Space Studio tab
    }
    renderXai(data);
    renderSimilarCases(data);
    renderCausalPanel(data.causal_panel, age ? Number(age) : null, fh || null);
    window.__lastResult = data;
    if ($("analysis-status")) $("analysis-status").textContent = `Assessment complete. ${data.disclaimer}`;
    toast("Assessment and Joint Space Width measurement completed.");
  } catch (error) {
    progress.finish(false);
    if ($("analysis-status")) $("analysis-status").textContent = error.message || "Assessment failed.";
    toast(error.message || "Assessment failed.");
  } finally {
    button.disabled = false;
  }
});

/* ========================= Clinical Diagnostic Report Modal ========================= */
function populateAndOpenReport() {
  const data = window.__lastResult;
  if (!data) { toast("Run an assessment first."); return; }

  const body = $("report-body-content");
  if (!body) return;

  const age = $("age")?.value || "Not provided";
  const bmi = $("bmi")?.value || "Not provided";
  const side = $("side")?.value === "L" ? "Left Knee (OS)" : "Right Knee (OD)";
  const pain = $("pain_vas")?.value ? `${$("pain_vas").value} / 10` : "Not reported";
  const injury = $("knee_injury")?.value === "yes" ? "Yes (prior trauma/meniscal history)" : "None reported";

  const jsw = data.jsw || {};
  const med = jsw.medial || {};
  const lat = jsw.lateral || {};
  const jsn = jsw.jsn_classification || {};

  body.innerHTML = `
    <!-- Patient Profile -->
    <div class="report-patient-summary">
      <div class="rep-stat-item">Study Type<b>Standing AP Knee Radiograph</b></div>
      <div class="rep-stat-item">Laterality<b>${side}</b></div>
      <div class="rep-stat-item">Patient Age<b>${age} yrs</b></div>
      <div class="rep-stat-item">Patient BMI<b>${bmi} kg/m²</b></div>
      <div class="rep-stat-item">Joint Pain VAS<b>${pain}</b></div>
      <div class="rep-stat-item">Prior Knee Trauma<b>${injury}</b></div>
    </div>

    <!-- Section 1: Radiographic KL Grade -->
    <div class="report-section">
      <h4>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M13 2 3 14h7l-1 8 11-14h-7z"/></svg>
        Primary Finding: Radiographic Kellgren-Lawrence Grade
      </h4>
      <table class="report-table">
        <tr><th style="width:30%">Overall KL Grade</th><td><b>Grade ${data.grade} — ${GRADE_LABELS[data.grade]}</b></td></tr>
        <tr><th>Diagnostic Confidence</th><td><b>${(data.confidence * 100).toFixed(1)}%</b> (${data.model_used})</td></tr>
        <tr><th>Radiologist Triage</th><td><b>${data.uncertainty?.flag ? "BORDERLINE / REFER TO SPECIALIST" : "CONFIDENT PREDICTION"}</b></td></tr>
        <tr><th>Radiographic Features</th><td>${GRADE_DESCRIPTIONS[data.grade]}</td></tr>
      </table>
    </div>

    <!-- Section 2: Joint Space Morphometrics -->
    <div class="report-section">
      <h4>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M4 6h16M4 12h16M4 18h16M8 6v12M16 6v12"/></svg>
        Quantitative Joint Space Width (JSW) Morphometrics
      </h4>
      <table class="report-table">
        <thead>
          <tr>
            <th>Compartment</th>
            <th>Minimum JSW</th>
            <th>Mean JSW</th>
            <th>Pixel Gap</th>
            <th>Narrowing Status</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><b>${med.name || "Medial Compartment"}</b></td>
            <td><b>${med.min_mm ? med.min_mm.toFixed(2) : "0.00"} mm</b></td>
            <td>${med.mean_mm ? med.mean_mm.toFixed(2) : "0.00"} mm</td>
            <td>${med.min_px ? med.min_px.toFixed(1) : "0"} px</td>
            <td>${med.min_mm < 1.5 ? "Severe Narrowing" : med.min_mm < 3.0 ? "Moderate Narrowing" : med.min_mm < 4.0 ? "Mild Narrowing" : "Preserved"}</td>
          </tr>
          <tr>
            <td><b>${lat.name || "Lateral Compartment"}</b></td>
            <td><b>${lat.min_mm ? lat.min_mm.toFixed(2) : "0.00"} mm</b></td>
            <td>${lat.mean_mm ? lat.mean_mm.toFixed(2) : "0.00"} mm</td>
            <td>${lat.min_px ? lat.min_px.toFixed(1) : "0"} px</td>
            <td>${lat.min_mm < 1.5 ? "Severe Narrowing" : lat.min_mm < 3.0 ? "Moderate Narrowing" : lat.min_mm < 4.0 ? "Mild Narrowing" : "Preserved"}</td>
          </tr>
        </tbody>
      </table>
      <div style="background:var(--panel-2);padding:10px 12px;border-radius:8px;border:1px solid var(--border);font-size:11px">
        <div><b>OARSI Joint Space Narrowing (JSN):</b> ${jsn.label || "N/A"}</div>
        <div style="margin-top:4px"><b>Coronal Alignment:</b> ${jsw.alignment_note || "N/A"} (M:L Ratio = ${jsw.ml_ratio || "1.00"})</div>
        <div style="margin-top:4px;color:var(--text-faint)"><b>Tibial Plateau Reference:</b> Width = ${jsw.joint_width_px || "0"} px, Scaling = ${jsw.mm_per_px || "0.42"} mm/px</div>
      </div>
    </div>

    <!-- Section 3: Causal Risk Breakdown -->
    ${data.causal_panel?.available ? `
    <div class="report-section">
      <h4>
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><path d="M4 19V5M4 19h16"/></svg>
        Causal Risk & Lifestyle Prognosis
      </h4>
      <table class="report-table">
        <tr><th style="width:30%">Current Expected KL</th><td><b>${data.causal_panel.expected_kl.value.toFixed(2)} pts</b></td></tr>
        ${data.causal_panel.individual_obesity_effect ? `<tr><th>Obesity Attributable Risk</th><td>+${data.causal_panel.individual_obesity_effect.value.toFixed(2)} KL points</td></tr>` : ""}
        <tr><th>Causal Recommendation</th><td>Target weight reduction to non-obese threshold (BMI &lt; 25–28) is modeled to reduce progression trajectory.</td></tr>
      </table>
    </div>` : ""}

    <!-- Signoff and Disclaimer -->
    <div class="report-signoff">
      <div>
        <b>Study Generated:</b> ${new Date().toLocaleString()}<br>
        <b>Software Version:</b> Knee OA Reading Room v2.4 (ConvNeXt + KneeNeXt + JSW Studio)
      </div>
      <div style="text-align:right;max-width:320px">
        <b>${data.disclaimer}</b><br>
        Clinical decision support prototype. Final diagnosis must be confirmed by qualified medical practitioner.
      </div>
    </div>`;

  $("report-modal")?.classList.remove("hidden");
}

$("open-report-btn")?.addEventListener("click", populateAndOpenReport);
$("download-report-btn")?.addEventListener("click", () => {
  populateAndOpenReport();
  setTimeout(() => window.print(), 300);
});
$("report-print-btn")?.addEventListener("click", () => window.print());

$("report-close-btn")?.addEventListener("click", () => {
  $("report-modal")?.classList.add("hidden");
});
$("report-modal")?.addEventListener("click", (e) => {
  if (e.target.id === "report-modal") $("report-modal")?.classList.add("hidden");
});

// JSON Export handler
$("report-export-json-btn")?.addEventListener("click", () => {
  if (!window.__lastResult) return;
  const jsonBlob = new Blob([JSON.stringify(window.__lastResult, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(jsonBlob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `knee_oa_assessment_${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
  toast("Exported clinical assessment JSON.");
});

/* ========================= Causal Insights Page ========================= */
const METHOD_LABELS = {
  propensity_score_matching: "Propensity score matching",
  linear_regression: "Regression adjustment",
  random_common_cause: "Random common cause",
  placebo_treatment_refuter: "Placebo treatment",
  data_subset_refuter: "Data subset",
};
function readable(name) { return METHOD_LABELS[name] || name; }

async function loadCausalInsights() {
  const methodsEl = $("causal-methods");
  const ciEl = $("ci-bars");
  const robustEl = $("causal-robustness");
  if (methodsEl) methodsEl.textContent = "Loading population causal estimates…";
  try {
    const resp = await fetch("/api/causal_insights", { cache: "no-store" });
    const data = await resp.json();
    if (!resp.ok || !data.ok) throw new Error("Could not load causal insights.");

    if (methodsEl) {
      methodsEl.replaceChildren();
      data.methods.forEach((m) => {
        const card = document.createElement("div");
        card.className = "method-card";
        card.innerHTML = `<b>+${m.ate.toFixed(3)}</b><span>${m.label}</span><small>KL grade points${m.n ? `, n = ${m.n}` : ""}</small>`;
        methodsEl.appendChild(card);
      });
    }

    if (ciEl) {
      ciEl.replaceChildren();
      ciEl.style.display = "block";
      const wrap = document.createElement("div");
      data.age_group_ci.forEach((row) => {
        const lo = row.ci_low, hi = row.ci_high, est = row.estimate;
        const scaleMax = 0.8;
        const pct = (v) => Math.max(0, Math.min(100, (v / scaleMax) * 100));
        const line = document.createElement("div");
        line.className = "ci-bar-row";
        line.innerHTML = `
          <span class="name">${row.label.replace(" (adj. age+family history)", "")}</span>
          <div class="ci-bar-track">
            <div class="ci-bar-fill" style="left:${pct(lo)}%;width:${pct(hi) - pct(lo)}%"></div>
            <div class="ci-bar-point" style="left:${pct(est)}%"></div>
          </div>`;
        wrap.appendChild(line);
      });
      ciEl.appendChild(wrap);
      const caption = document.createElement("p");
      caption.className = "hint";
      caption.style.marginTop = "6px";
      caption.textContent = "Obesity effect on KL grade by age group, with 95% confidence intervals.";
      ciEl.appendChild(caption);
    }

    if (robustEl) {
      robustEl.replaceChildren();
      if (data.evalue.point) {
        const card = document.createElement("div");
        card.className = "method-card";
        card.innerHTML = `<b>${data.evalue.point.toFixed(2)}</b><span>E-value</span><small>An unmeasured confounder would need at least this strong an association with both obesity and KL grade to explain the effect away (CI-bound E-value: ${data.evalue.ci_bound?.toFixed(2) ?? "—"}).</small>`;
        robustEl.appendChild(card);
      }
      const refCard = document.createElement("div");
      refCard.className = "method-card";
      refCard.innerHTML = `<b>Stable</b><span>Refutation checks</span><small>${data.refuters.map((r) => `${readable(r.name)}: ${r.new_effect.toFixed(3)}`).join(" · ")}</small>`;
      robustEl.appendChild(refCard);

      if (data.injury.linear_regression !== undefined) {
        const injuryCard = document.createElement("div");
        injuryCard.className = "method-card";
        injuryCard.innerHTML = `<b>+${data.injury.propensity_score_matching.toFixed(2)}</b><span>Knee injury (association)</span><small>Regression adjustment: +${data.injury.linear_regression.toFixed(2)} KL pts. Self-reported history; not a demonstrated cause.</small>`;
        robustEl.appendChild(injuryCard);
      }
    }
  } catch (error) {
    if (methodsEl) methodsEl.textContent = error.message || "Could not load causal insights.";
  }
}

document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeZoom();
    $("report-modal")?.classList.add("hidden");
  }
});

window.addEventListener("beforeunload", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
});
