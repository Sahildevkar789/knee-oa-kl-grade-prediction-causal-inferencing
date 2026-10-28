import sys
import json

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from ui.model_bridge import analyze

IMAGES = [
    ("dataset/test/0/9003175L.png", 0),
    ("dataset/test/1/9001400L.png", 1),
    ("dataset/test/2/9003316R.png", 2),
    ("dataset/test/3/9011053L.png", 3),
    ("dataset/test/4/9012867R.png", 4),
]

results = []
for path, true_g in IMAGES:
    r = analyze(path, age=65, bmi=32, family_history="yes")
    row = {
        "path": path, "true_grade": true_g, "pred_grade": r["grade"],
        "confidence": r["confidence"], "probabilities": r["probabilities"],
        "model_used": r["model_used"], "uncertain": r["uncertainty"]["flag"],
    }
    results.append(row)
    print(f"{path}: true={true_g} pred={r['grade']} conf={r['confidence']:.4f} model={r['model_used']}")

with open("ui/test_samples/redesign/direct_bridge_results.json", "w") as f:
    json.dump(results, f, indent=2)
