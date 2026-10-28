import json
import uuid
import urllib.request

IMAGES = [
    ("dataset/test/0/9003175L.png", 0),
    ("dataset/test/1/9001400L.png", 1),
    ("dataset/test/2/9003316R.png", 2),
    ("dataset/test/3/9011053L.png", 3),
    ("dataset/test/4/9012867R.png", 4),
]


def post_multipart(url, path, fields):
    boundary = uuid.uuid4().hex
    with open(path, "rb") as fh:
        file_bytes = fh.read()
    parts = []
    for key, value in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n{value}\r\n".encode())
    parts.append(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"xray\"; filename=\"{path.split('/')[-1]}\"\r\n"
        f"Content-Type: image/png\r\n\r\n".encode() + file_bytes + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(url, data=body, method="POST",
                                  headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


with open("ui/test_samples/redesign/direct_bridge_results.json") as f:
    direct = {row["path"]: row for row in json.load(f)}

all_match = True
report = []
for path, true_g in IMAGES:
    r = post_multipart("http://127.0.0.1:5050/api/analyze", path,
                        {"age": "65", "bmi": "32", "family_history": "yes"})
    d = direct[path]
    grade_match = r["grade"] == d["pred_grade"]
    conf_match = abs(r["confidence"] - d["confidence"]) < 1e-6
    thumbs_ok = all(c.get("thumbnail_png_base64") for c in r.get("similar_cases", []))
    ok = grade_match and conf_match
    all_match = all_match and ok
    line = (f"{path}: http_grade={r['grade']} direct_grade={d['pred_grade']} "
            f"http_conf={r['confidence']:.4f} direct_conf={d['confidence']:.4f} "
            f"match={ok} thumbs_ok={thumbs_ok} ig_present={r['integrated_gradients_png_base64'] is not None} "
            f"cam_present={r['gradcam_png_base64'] is not None} causal_available={r['causal_panel']['available']}")
    print(line)
    report.append(line)

print("ALL_MATCH:", all_match)
with open("ui/test_samples/redesign/http_vs_direct_report.txt", "w") as f:
    f.write("\n".join(report) + f"\n\nALL_MATCH: {all_match}\n")
