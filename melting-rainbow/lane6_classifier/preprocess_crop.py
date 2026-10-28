"""Make the corrected model input for an image model: a small crop of the liquid in front of the white
paper, divided by the paper beside the vial (removes exposure / white-balance differences).

   python lane6_classifier/preprocess_crop.py <labels.csv> <image_root> <out_dir>
labels.csv needs columns: image, total_ml  (twin_dataset_v2/labels.csv or video5_eval/video5_eval.csv).
Writes <out_dir>/<same relative path>.png (32x32, paper = white) and <out_dir>/features.csv
(normalized color T_r/g/b of the band) -- feed the crop AND recipe_id to the model.
"""
import sys, os, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.measure import band_rows, VIAL_X, PAPER, measure

labels, root, out = sys.argv[1], sys.argv[2], sys.argv[3]
d = pd.read_csv(labels); rows = []
for r in d.itertuples():
    img = cv2.imread(os.path.join(root, r.image))
    if img is None:
        print("missing", r.image); continue
    r0, r1 = band_rows(r.total_ml)
    p = PAPER; paper = np.median(img[p[0]:p[1], p[2]:p[3]].reshape(-1, 3), axis=0)
    crop = img[r0:r1, VIAL_X[0] - 5:VIAL_X[1] + 5].astype(np.float32) / np.maximum(paper, 1)
    crop = cv2.resize(np.clip(crop * 255, 0, 255).astype(np.uint8), (32, 32), interpolation=cv2.INTER_AREA)
    dst = os.path.join(out, os.path.splitext(r.image)[0] + ".png"); os.makedirs(os.path.dirname(dst), exist_ok=True)
    cv2.imwrite(dst, crop)
    T = measure(img, r.total_ml)["T"]
    rows.append(dict(image=r.image, crop=os.path.relpath(dst, out), T_r=T[0], T_g=T[1], T_b=T[2]))
pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "features.csv"), index=False)
print("wrote", len(rows), "crops to", out)
