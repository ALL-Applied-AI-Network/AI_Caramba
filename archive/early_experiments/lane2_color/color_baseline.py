"""Lane 2: measure the liquid color of every labeled vial (video 5).

Two ways to get frames (same output):
  --video  (recommended) reads vial_fill_video_5.mp4 and grabs frames just before each
           fill ends (end_frame - offset), when the vial is full and still under the nozzle.
  default  uses the ~2 s stills listed in video5_frame_labels.csv (last N in-fill stills).

Crops a fixed box over the liquid, takes the median color (resists glare/labels), converts
to Lab, and compares each vial to its recipe's LAST vial (the clean reference, per the
dataset page) as dE.

Usage:
  python lane2_color/color_baseline.py <dataset_root> --video --box 300 330 28 50 --preview
Tip: open crop_preview.jpg; the green box must sit inside the liquid.
"""
import argparse
import numpy as np, pandas as pd, cv2

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--box", nargs=4, type=int, required=True, metavar=("X", "Y", "W", "H"))
ap.add_argument("--video", action="store_true", help="read frames from vial_fill_video_5.mp4")
ap.add_argument("--offset", type=int, default=15, help="video mode: frames before end_frame")
ap.add_argument("--last", type=int, default=2, help="stills mode: final in-fill stills per vial")
ap.add_argument("--out", default="vial_colors.csv")
ap.add_argument("--preview", action="store_true")
a = ap.parse_args()
x, y, w, h = a.box

def lab_of(bgr):
    px = np.uint8([[np.clip(bgr, 0, 255)]])
    L, A, B = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return L * 100 / 255, A - 128, B - 128            # OpenCV 8-bit Lab -> CIE Lab ranges

def row_for(r, vial, bgr):
    L, A, B = lab_of(bgr)
    return dict(vial=vial, recipe_id=int(r.recipe_id), pos_in_batch=(vial - 1) % 3 + 1,
                t100_ml=r.t100_ml, t200_ml=r.t200_ml, t400_ml=r.t400_ml,
                dispense_order=r.dispense_order, B=bgr[0], G=bgr[1], R=bgr[2], L=L, a=A, b=B)

rows = []
if a.video:
    rec = pd.read_csv(f"{a.root}/vial_fill/labels/video5_vial_recipes.csv")
    cap = cv2.VideoCapture(f"{a.root}/vial_fill/videos/vial_fill_video_5.mp4")
    for _, r in rec.iterrows():
        px = []
        for off in (a.offset + 6, a.offset, a.offset - 6):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.end_frame) - off)
            ok, img = cap.read()
            if not ok:
                continue
            px.append(img[y:y + h, x:x + w].reshape(-1, 3))
            if a.preview and int(r.vial) == 7 and off == a.offset:
                p = img.copy(); cv2.rectangle(p, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("crop_preview.jpg", p)
        rows.append(row_for(r, int(r.vial), np.median(np.vstack(px), axis=0)))
else:
    lab = pd.read_csv(f"{a.root}/vial_fill/labels/video5_frame_labels.csv")
    fills = lab[lab["in_fill"] == 1].copy()
    fills["vial"] = fills["vial"].astype(int)
    for vial, d in fills.groupby("vial"):
        d = d.sort_values("video_s").tail(a.last)
        crops = []
        for fn in d["filename"]:
            img = cv2.imread(f"{a.root}/vial_fill/images/{fn}")
            if img is None:
                print(f"missing image {fn}"); continue
            crops.append(img[y:y + h, x:x + w].reshape(-1, 3))
            if a.preview and vial == fills["vial"].min():
                p = img.copy(); cv2.rectangle(p, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imwrite("crop_preview.jpg", p)
        rows.append(row_for(d.iloc[-1], vial, np.median(np.vstack(crops), axis=0)))

df = pd.DataFrame(rows)
ref = df[df.pos_in_batch == 3].set_index("recipe_id")[["L", "a", "b"]]
df["dE_to_ref"] = df.apply(lambda r: float(np.linalg.norm(
    r[["L", "a", "b"]].to_numpy(float) - ref.loc[r.recipe_id].to_numpy(float))), axis=1)
df.to_csv(a.out, index=False)

print(df.pivot(index="recipe_id", columns="pos_in_batch", values="dE_to_ref").round(1)
        .rename(columns={1: "vial1_dE", 2: "vial2_dE", 3: "vial3_dE(ref)"}))
clean = df[(df.pos_in_batch == 2)].dE_to_ref
first = df[(df.pos_in_batch == 1) & (df.recipe_id > df.recipe_id.min())].dE_to_ref
print(f"\nclean vial-2 dE: max {clean.max():.1f} | first-vial dE: min {first.min():.1f}")
if first.min() > clean.max():
    print(f"separable -> a threshold anywhere in ({clean.max():.1f}, {first.min():.1f}) splits them")
else:
    print("Overlap: some vial 2s score as high as first vials. On the real data this is real carryover\n"
          "lasting two vials (R6, R14), not noise. Check by eye, then score with --truth in lane 5.")
