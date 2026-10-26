import sys
sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from ui.model_bridge import analyze

images = [
    ("dataset/test/0/9003175L.png", 0),
    ("dataset/test/1/9001400L.png", 1),
    ("dataset/test/2/9003316R.png", 2),
    ("dataset/test/3/9011053L.png", 3),
    ("dataset/test/4/9012867R.png", 4),
]
for path, true_g in images:
    r = analyze(path, age=65, bmi=32, family_history="yes")
    print(f"{path}: true={true_g} pred={r['grade']} conf={r['confidence']:.3f} "
         f"model={r['model_used']} uncertain={r['uncertainty']['flag']} "
         f"n_neighbors={len(r['similar_cases'])} causal_available={r['causal_panel']['available']} "
         f"ig_present={r['integrated_gradients_png_base64'] is not None} "
         f"cam_present={r['gradcam_png_base64'] is not None}")
