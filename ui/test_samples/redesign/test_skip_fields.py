import json
import uuid
import urllib.request

boundary = uuid.uuid4().hex
with open("dataset/test/2/9003316R.png", "rb") as fh:
    file_bytes = fh.read()
body = (
    f"--{boundary}\r\nContent-Disposition: form-data; name=\"xray\"; filename=\"x.png\"\r\n"
    f"Content-Type: image/png\r\n\r\n".encode() + file_bytes + b"\r\n"
    + f"--{boundary}--\r\n".encode()
)
req = urllib.request.Request("http://127.0.0.1:5050/api/analyze", data=body, method="POST",
                              headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
with urllib.request.urlopen(req, timeout=60) as resp:
    r = json.loads(resp.read().decode())
print("grade:", r["grade"], "causal_panel:", r["causal_panel"])
