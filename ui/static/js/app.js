const $ = (id) => document.getElementById(id);

const viewNames = ["analysis", "causal"];

function openView(name) {
  viewNames.forEach((view) => {
    const element = $("view-" + view);
    if (element) element.classList.toggle("hidden", view !== name);
  });
  document.querySelectorAll(".nav").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === name);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
}

document.querySelectorAll(".nav").forEach((button) => {
  button.addEventListener("click", () => openView(button.dataset.view));
});

function toast(message) {
  const element = $("toast");
  if (!element) return;
  element.textContent = message;
  element.style.display = "block";
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => { element.style.display = "none"; }, 3200);
}

let selectedImageUrl = null;

// --- X-ray selection and local preview ---
$("xray-input")?.addEventListener("change", (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  if (!["image/png", "image/jpeg"].includes(file.type)) {
    toast("Choose a PNG or JPEG image.");
    event.target.value = "";
    return;
  }
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = URL.createObjectURL(file);
  if ($("xray-preview")) $("xray-preview").src = selectedImageUrl;
  if ($("image-name")) $("image-name").textContent = file.name;
  $("image-view")?.classList.remove("hidden");
  $("empty-image")?.classList.add("hidden");
  toast("Image preview opened locally. No prediction has been run yet.");
});

$("remove-image")?.addEventListener("click", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
  selectedImageUrl = null;
  if ($("xray-input")) $("xray-input").value = "";
  $("xray-preview")?.removeAttribute("src");
  $("image-view")?.classList.add("hidden");
  $("empty-image")?.classList.remove("hidden");
});

const GRADE_LABELS = ["KL 0 — None", "KL 1 — Doubtful", "KL 2 — Mild", "KL 3 — Moderate", "KL 4 — Severe"];

function renderPrediction(data) {
  const panel = $("prediction-panel");
  if (!panel) return;
  panel.replaceChildren();

  const gradeIcon = document.createElement("div");
  gradeIcon.className = "grade-icon";
  gradeIcon.textContent = `KL ${data.grade}`;

  const heading = document.createElement("b");
  heading.textContent = GRADE_LABELS[data.grade] || `Grade ${data.grade}`;

  const confLine = document.createElement("p");
  confLine.textContent = `Confidence: ${(data.confidence * 100).toFixed(1)}%  |  Model: ${data.model_used} (${data.model_members.join(" + ")})`;

  const uncertainty = document.createElement("p");
  uncertainty.className = data.uncertainty.flag ? "warn-text" : "ok-text";
  uncertainty.textContent = `Uncertainty: ${data.uncertainty.message}`;

  const probContainer = document.createElement("div");
  probContainer.className = "grade-probabilities";
  Object.entries(data.probabilities).forEach(([label, value]) => {
    const pct = Math.max(0, Math.min(100, value * 100));
    const row = document.createElement("div");
    row.style.margin = "8px 0";
    const line = document.createElement("div");
    line.style.display = "flex"; line.style.justifyContent = "space-between"; line.style.gap = "12px";
    const l = document.createElement("span"); l.textContent = label;
    const n = document.createElement("b"); n.textContent = `${pct.toFixed(2)}%`;
    line.append(l, n);
    const track = document.createElement("div");
    track.style.cssText = "height:6px;background:#e5e7eb;border-radius:5px;overflow:hidden";
    const fill = document.createElement("div");
    fill.style.cssText = `height:100%;width:${pct}%;background:#3976c5`;
    track.appendChild(fill);
    row.append(line, track);
    probContainer.appendChild(row);
  });

  panel.append(gradeIcon, heading, confLine, uncertainty, probContainer);

  const tag = $("model-used-tag");
  if (tag) { tag.textContent = "RESEARCH OUTPUT"; tag.classList.remove("muted"); }
}

function imgFigure(title, src) {
  const figure = document.createElement("figure");
  figure.style.margin = "0"; figure.style.minWidth = "0";
  const heading = document.createElement("b");
  heading.textContent = title; heading.style.display = "block"; heading.style.marginBottom = "8px";
  const image = document.createElement("img");
  image.src = src; image.style.width = "100%"; image.style.borderRadius = "8px"; image.style.background = "#f1f5f9";
  figure.append(heading, image);
  return figure;
}

function renderXai(data) {
  const el = $("xai-panel");
  if (!el) return;
  el.replaceChildren();
  const grid = document.createElement("div");
  grid.style.display = "grid"; grid.style.gridTemplateColumns = "repeat(auto-fit, minmax(200px, 1fr))"; grid.style.gap = "14px";
  if (data.gradcam_png_base64) grid.appendChild(imgFigure("Grad-CAM (bug-fixed)", `data:image/png;base64,${data.gradcam_png_base64}`));
  if (data.integrated_gradients_png_base64) grid.appendChild(imgFigure("Integrated Gradients (kl_final)", `data:image/png;base64,${data.integrated_gradients_png_base64}`));
  el.appendChild(grid);
  const caption = document.createElement("p");
  caption.className = "hint";
  caption.textContent = "Exploratory visualizations of regions influencing the model output - not proof of clinical reasoning.";
  el.appendChild(caption);
}

function renderSimilarCases(data) {
  const el = $("similar-cases-panel");
  if (!el) return;
  el.replaceChildren();
  if (!data.similar_cases || data.similar_cases.length === 0) {
    el.innerHTML = "<p class=\"hint\">No similar cases found.</p>";
    return;
  }
  const list = document.createElement("ul");
  data.similar_cases.forEach((c) => {
    const li = document.createElement("li");
    li.textContent = `KL${c.grade} — ${c.path.split(/[\\/]/).pop()} (cosine similarity ${c.similarity.toFixed(3)})`;
    list.appendChild(li);
  });
  el.appendChild(list);
}

function renderCausalPanel(panel) {
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
    const p = document.createElement("p");
    p.className = "hint";
    p.textContent = panel.message || "Causal panel not available.";
    el.appendChild(p);
    return;
  }

  const lines = [];
  lines.push(`This patient: age ${panel.age}, BMI ${panel.bmi} → ${panel.obese ? "obese" : "not obese"} (threshold 30).`);
  lines.push(`Expected KL grade: ${panel.expected_kl.value.toFixed(2)}${panel.expected_kl.clipped ? " (clipped to the valid 0-4 range)" : ""}.`);
  if (panel.individual_obesity_effect) {
    const e = panel.individual_obesity_effect;
    lines.push(`Individual obesity effect (causal forest): ${e.value >= 0 ? "+" : ""}${e.value.toFixed(2)} KL grade points (95% CI [${e.ci_low.toFixed(2)}, ${e.ci_high.toFixed(2)}]).`);
  }
  if (panel.knee_injury) {
    lines.push(`Knee injury (${panel.knee_injury.label}): +${panel.knee_injury.linear_regression_ate.toFixed(2)} to +${panel.knee_injury.propensity_score_matching_ate.toFixed(2)} KL grade points.`);
  }

  const ul = document.createElement("ul");
  lines.forEach((t) => { const li = document.createElement("li"); li.textContent = t; ul.appendChild(li); });
  el.appendChild(ul);

  if (panel.bmi_whatif && panel.bmi_whatif.length) {
    const h = document.createElement("b"); h.textContent = "BMI what-if (expected KL grade):";
    const ul2 = document.createElement("ul");
    panel.bmi_whatif.forEach((w) => {
      const li = document.createElement("li");
      li.textContent = `BMI ${w.bmi}: ${w.expected_kl.toFixed(2)}`;
      ul2.appendChild(li);
    });
    el.append(h, ul2);
  }

  const caveat = document.createElement("div");
  caveat.className = "small-warning";
  caveat.textContent = panel.caveat;
  el.appendChild(caveat);
}

// --- Run assessment ---
$("analyze-btn")?.addEventListener("click", async () => {
  const file = $("xray-input")?.files?.[0];
  if (!file) { toast("Select an X-ray first."); return; }

  const button = $("analyze-btn");
  button.disabled = true;
  if ($("analysis-status")) $("analysis-status").textContent = "Running the model…";

  const formData = new FormData();
  formData.append("xray", file);
  const age = $("age")?.value; if (age) formData.append("age", age);
  const bmi = $("bmi")?.value; if (bmi) formData.append("bmi", bmi);
  const fh = $("family_history")?.value; if (fh) formData.append("family_history", fh);

  try {
    const response = await fetch("/api/analyze", { method: "POST", body: formData, cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) {
      throw new Error(data.message || `Assessment failed (HTTP ${response.status}).`);
    }
    renderPrediction(data);
    renderXai(data);
    renderSimilarCases(data);
    renderCausalPanel(data.causal_panel);
    if ($("analysis-status")) $("analysis-status").textContent = `Done. ${data.disclaimer}`;
    toast("Assessment completed.");
  } catch (error) {
    if ($("analysis-status")) $("analysis-status").textContent = error.message || "Assessment failed.";
    toast(error.message || "Assessment failed.");
  } finally {
    button.disabled = false;
  }
});

// --- Historical causal insights tab ---
async function loadCausalData() {
  const container = $("causal-results");
  if (!container) return;
  container.textContent = "Loading…";
  try {
    const response = await fetch("/api/causal", { cache: "no-store" });
    const data = await response.json().catch(() => ({}));
    if (!response.ok || !data.ok) throw new Error(data.error || "Causal API request failed.");
    container.replaceChildren();
    (data.causal_estimates || [])
      .filter((r) => r.Analysis === "primary_one_knee" && r.Outcome === "KL_grade (0-4)")
      .forEach((row) => {
        const card = document.createElement("div");
        card.className = "effect-card";
        const label = document.createElement("span"); label.textContent = row.Method;
        const strong = document.createElement("strong"); strong.textContent = `+${Number(row.ATE).toFixed(4)}`;
        const small = document.createElement("small"); small.textContent = `Estimated effect on KL grade; n = ${row.N}`;
        card.append(label, strong, small);
        container.appendChild(card);
      });
  } catch (error) {
    container.textContent = error.message || "Could not load causal results.";
  }
}
if ($("causal-results")) loadCausalData();

window.addEventListener("beforeunload", () => {
  if (selectedImageUrl) URL.revokeObjectURL(selectedImageUrl);
});
