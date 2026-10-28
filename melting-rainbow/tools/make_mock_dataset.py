"""Build a small MOCK dataset with the same layout and columns as the real
csi-vial-fill-dataset (per its README), so the starter scripts can be tested
before the real data is downloaded. The colors here are invented: carryover
is injected on purpose (first vial = 70% current mix + 30% previous mix).

Usage: python tools/make_mock_dataset.py mock_data
"""
import sys, os
from datetime import datetime, timedelta
import numpy as np, pandas as pd, cv2

out = sys.argv[1] if len(sys.argv) > 1 else "mock_data"
rng = np.random.default_rng(0)

# Recipes 5-14 exactly as in the dataset README (mL per tank)
RECIPES = {
    5: (10, 10, 10), 6: (30, 10, None), 7: (20, None, 50), 8: (20, 20, 20),
    9: (10, 35, 35), 10: (45, 15, 15), 11: (10, 10, 50), 12: (10, 50, 10),
    13: (None, 50, 20), 14: (60, None, 10),
}
# Invented dye colors (BGR) for the mock only
DYE = {"t100": np.array([235, 235, 235.]),  # clear
       "t200": np.array([30, 140, 245.]),   # orange
       "t400": np.array([200, 90, 20.])}    # blue

def order(t100, t200, t400):
    o = [n for n, v in (("T200", t200), ("T400", t400), ("T100", t100)) if v]
    return ">".join(o)

def mix_color(rid):
    t100, t200, t400 = RECIPES[rid]
    vols = {"t100": t100 or 0, "t200": t200 or 0, "t400": t400 or 0}
    tot = sum(vols.values())
    return sum(DYE[k] * v / tot for k, v in vols.items())

def draw_frame(color, fill_frac):
    img = np.full((480, 640, 3), 60, np.uint8)          # dark background
    cv2.rectangle(img, (280, 120), (360, 400), (180, 180, 180), 2)  # vial outline
    top = int(400 - 260 * fill_frac)
    if fill_frac > 0:
        c = np.clip(color + rng.normal(0, 4, 3), 0, 255)
        cv2.rectangle(img, (282, top), (358, 398), c.tolist(), -1)
        cv2.rectangle(img, (300, top), (306, 398), (255, 255, 255), -1)  # glare stripe
    return img

for cam in ("vial_fill", "tank_level", "cap_seal"):
    os.makedirs(f"{out}/{cam}/images", exist_ok=True)
os.makedirs(f"{out}/vial_fill/labels", exist_ok=True)

t0 = datetime(2026, 9, 30, 10, 0, 0)
fps, step = 30, 2.0            # stills ~every 2 s (README)
fill_len, gap = 20.0, 30.0     # seconds (mock)
rows, frame_rows, idx = [], [], {c: [] for c in ("vial_fill", "tank_level", "cap_seal")}

vial_spans = []
t = 5.0
prev_rid = None
for vial in range(1, 31):
    rid = 5 + (vial - 1) // 3
    t100, t200, t400 = RECIPES[rid]
    color = mix_color(rid)
    if (vial - 1) % 3 == 0 and prev_rid is not None:       # first vial of a new recipe
        color = 0.7 * color + 0.3 * mix_color(prev_rid)    # injected carryover (H1)
    total = sum(v or 0 for v in (t100, t200, t400))
    vial_spans.append((t, t + fill_len, vial, rid, color, total / 80))
    rows.append(dict(vial=vial, recipe_id=rid,
                     type="Triple" if all((t100, t200, t400)) else "Pair",
                     t100_ml=t100, t200_ml=t200, t400_ml=t400, total_ml=total,
                     dispense_order=order(t100, t200, t400), color_purpose="",
                     start_time=f"{int(t//60)}:{t%60:04.1f}", end_time=f"{int((t+fill_len)//60)}:{(t+fill_len)%60:04.1f}",
                     start_s=t, end_s=t + fill_len, duration_s=fill_len,
                     start_frame=int(t * fps), end_frame=int((t + fill_len) * fps)))
    prev_rid = rid
    t += fill_len + gap

video_len = t
pd.DataFrame(rows).to_csv(f"{out}/vial_fill/labels/video5_vial_recipes.csv", index=False)

# Stills. Simulate a small clock offset on tank_level (+0.4 s) and dropped cap_seal frames.
s = 0.0
while s < video_len:
    cur = next((sp for sp in vial_spans if sp[0] <= s <= sp[1]), None)
    last_done = [sp for sp in vial_spans if sp[1] < s]
    for cam, offset in (("vial_fill", 0.0), ("tank_level", 0.4), ("cap_seal", 0.15)):
        if cam == "cap_seal" and rng.random() < 0.05:
            continue  # dropped frame
        ts = t0 + timedelta(seconds=s + offset)
        fn = f"{cam}_{ts:%Y%m%d_%H%M%S}_{ts.microsecond:06d}.jpg"
        if cam == "vial_fill":
            if cur:
                frac = (s - cur[0]) / fill_len * cur[5]
                img = draw_frame(cur[4], frac)
            else:
                img = draw_frame(np.zeros(3), 0)
        else:
            img = np.full((480, 640, 3), 90, np.uint8)
        cv2.imwrite(f"{out}/{cam}/images/{fn}", img)
        idx[cam].append(dict(filename=fn, timestamp=ts.isoformat(), video_id=5))
        if cam == "vial_fill":
            r = rows[cur[2] - 1] if cur else None
            frame_rows.append(dict(filename=fn, timestamp=ts.isoformat(), video_s=round(s, 3),
                                   in_fill=int(cur is not None), vial=cur[2] if cur else "",
                                   recipe_id=cur[3] if cur else "",
                                   **{k: (r[k] if r else "") for k in
                                      ("type", "t100_ml", "t200_ml", "t400_ml", "total_ml",
                                       "dispense_order", "color_purpose")}))
    s += step + rng.normal(0, 0.05)

for cam in idx:
    pd.DataFrame(idx[cam]).to_csv(f"{out}/{cam}/image_index.csv", index=False)
pd.DataFrame(frame_rows).to_csv(f"{out}/vial_fill/labels/video5_frame_labels.csv", index=False)
print(f"mock dataset written to {out}: {len(frame_rows)} vial_fill frames")
