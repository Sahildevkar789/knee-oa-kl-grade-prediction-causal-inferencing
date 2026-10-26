"""Flask UI, adapted from the teammate's fork (webapp/app.py) to call this project's final
model through ui/model_bridge.py instead of loading checkpoints directly. No model logic
lives here - every prediction, explanation, similar-case lookup and causal estimate comes
from model_bridge.analyze().

Changes from the fork's original app.py (logged in detail in ui/integration_log.md):
  - the hardcoded Path.home()/"Desktop"/<fork's own folder name> project-path assumption is
    gone; this file locates the project root relative to itself, so it works on any machine.
  - no direct `from scripts.predict import ...` - everything model-related goes through
    model_bridge.analyze() instead.
  - the kl_final/Hybrid 2 model dropdown is gone - model_bridge always serves this project's
    final configuration (the validated ensemble, or kl_final alone if the ensemble didn't
    beat it - decided once, at startup, exactly as predict_extended.py decides it).
  - age/BMI/family history are no longer Hybrid-2-only fields - they feed the causal panel
    for every request now (never the classifier itself, same rule as predict_extended.py).

    python ui/app.py
"""
import os
import sys
import tempfile
import traceback

from flask import Flask, render_template, request, jsonify

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ui.model_bridge import analyze, DISCLAIMER

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_IMAGE_BYTES = 12 * 1024 * 1024
CAUSAL_DIR = os.path.join(ROOT, "outputs", "results", "causal")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


def allowed_file(filename):
    return bool(filename) and "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_optional_float(raw):
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


@app.get("/")
def index():
    return render_template("index.html", disclaimer=DISCLAIMER)


@app.get("/api/health")
def health():
    import torch
    return jsonify({"ok": True, "service": "Knee OA KL-grade research prototype",
                    "device_available": "cuda" if torch.cuda.is_available() else "cpu"})


@app.get("/api/causal")
def causal_results():
    """Serves the project's existing causal-analysis output files, unmodified and read-only."""
    import csv
    import json as json_mod

    def read_csv(name):
        path = os.path.join(CAUSAL_DIR, name)
        if not os.path.isfile(path):
            return None
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            return list(csv.DictReader(fh))

    estimates = read_csv("causal_estimates.csv")
    query_results = read_csv("causal_query_results.csv")
    models_path = os.path.join(CAUSAL_DIR, "causal_query_models.json")
    models = None
    if os.path.isfile(models_path):
        with open(models_path, "r", encoding="utf-8") as fh:
            models = json_mod.load(fh)

    if estimates is None and query_results is None and models is None:
        return jsonify({"ok": False, "error": "No causal output files were found."}), 404

    return jsonify({"ok": True, "causal_estimates": estimates or [], "query_results": query_results or [],
                    "models": models or {}})


@app.post("/api/patient")
def save_patient():
    """Validates patient details for this prototype; nothing is persisted."""
    data = request.get_json(silent=True) or {}
    required = ["patient_id", "age", "side"]
    missing = [k for k in required if not str(data.get(k, "")).strip()]
    if missing:
        return jsonify({"ok": False, "error": "Missing required fields: " + ", ".join(missing)}), 400
    try:
        age = float(data["age"])
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "Age must be a valid number."}), 400
    if not 18 <= age <= 120:
        return jsonify({"ok": False, "error": "Age must be between 18 and 120."}), 400
    side = str(data["side"]).strip().upper()
    if side not in {"L", "R", "LEFT", "RIGHT"}:
        return jsonify({"ok": False, "error": "Side must be left or right."}), 400
    return jsonify({"ok": True, "message": "Patient details validated. No data was stored.",
                    "patient": {"patient_id": str(data["patient_id"]).strip(), "age": age, "side": side}})


@app.post("/api/analyze")
def run_analyze():
    if "xray" not in request.files or not request.files["xray"].filename:
        return jsonify({"ok": False, "message": "Select a knee X-ray first."}), 400
    file = request.files["xray"]
    if not allowed_file(file.filename):
        return jsonify({"ok": False, "message": "Only PNG, JPG and JPEG images are supported."}), 400

    age = parse_optional_float(request.form.get("age"))
    bmi = parse_optional_float(request.form.get("bmi"))
    family_history = request.form.get("family_history") or None
    if family_history not in ("yes", "no"):
        family_history = None

    tmp_path = None
    try:
        suffix = "." + file.filename.rsplit(".", 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        result = analyze(tmp_path, age=age, bmi=bmi, family_history=family_history)
        result["ok"] = True
        result["disclaimer"] = DISCLAIMER
        return jsonify(result)
    except Exception:
        app.logger.error("Inference failed:\n%s", traceback.format_exc())
        return jsonify({"ok": False, "message": "Inference failed. Check the Flask terminal for details."}), 500
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"ok": False, "message": "Image exceeds the 12 MB upload limit."}), 413


if __name__ == "__main__":
    print("=" * 55)
    print("Knee OA KL-grade research webapp (ui/)")
    print(f"Project root: {ROOT}")
    print(f"Disclaimer: {DISCLAIMER}")
    print("=" * 55)
    app.run(host="127.0.0.1", port=5050, debug=False)
