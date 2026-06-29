import requests
import os

MODELS = {
    "bunny.obj": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/stanford-bunny.obj",
    "armadillo.obj": "https://raw.githubusercontent.com/alecjacobson/common-3d-test-models/master/data/armadillo.obj",
}

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

for filename, url in MODELS.items():
    out_path = os.path.join(OUT_DIR, filename)
    if os.path.exists(out_path):
        print(f"already exists: {filename}")
        continue
    print(f"downloading {filename} ...")
    r = requests.get(url)
    with open(out_path, "wb") as f:
        f.write(r.content)
    print(f"saved → {out_path}")