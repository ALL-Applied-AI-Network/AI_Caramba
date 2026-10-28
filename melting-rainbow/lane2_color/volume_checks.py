"""Lane 2: volume checks that work with the cameras we have (video 5).

The fill camera can't see the liquid line for most recipes: the black vial holder
hides it between roughly 45 and 75 mL. So volume is checked two other ways:

  1. FILL TIME (labels): each recipe's fill time is very repeatable. A fill that runs
     much longer or shorter than that recipe normally takes = possible volume problem
     (low tank, clog, wrong amount). Prints each recipe's normal time and spread.
  2. TANK LEVEL (tank_level camera): the liquid line in T200 and T400 drops as fluid
     is used. Fits pixels-of-drop vs. labeled mL dispensed, so we get a real
     measured volume, accurate to roughly +/-20 mL. Good for catching a wrong or
     empty tank, not a 10% error on one vial.

Usage: python lane2_color/volume_checks.py <inner dataset folder>
Outputs: fill_time_bands.csv, tank_levels.csv
"""
import sys
import cv2, numpy as np, pandas as pd

root = sys.argv[1]
rec = pd.read_csv(f"{root}/vial_fill/labels/video5_vial_recipes.csv").fillna({"t100_ml": 0, "t200_ml": 0, "t400_ml": 0})

# ---- 1. fill time per recipe -------------------------------------------------
bands = rec.groupby("recipe_id").agg(total_ml=("total_ml", "first"), mean_s=("duration_s", "mean"),
                                     sd_s=("duration_s", "std"), min_s=("duration_s", "min"), max_s=("duration_s", "max")).round(2)
bands["flag_below_s"] = (bands.mean_s - 3).round(1)   # +/-3 s band: ~2x the worst spread we saw (R13)
bands["flag_above_s"] = (bands.mean_s + 3).round(1)
bands.to_csv("fill_time_bands.csv")
print("FILL TIME per recipe (s). Flag a fill outside the band:\n", bands.to_string())

# ---- 2. tank level vs dispensed volume -----------------------------------------
cap = cv2.VideoCapture(f"{root}/tank_level/videos/tank_level_video_5.mp4")
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cols = {"T200": (300, 325, "t200_ml"), "T400": (520, 545, "t400_ml")}   # x-range of a clear strip of each tank
y0, y1 = 325, 375                                                    # rows where the liquid line sits in video 5
rows = []
for fi in range(0, n, 150):                                          # every 5 s
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ok, f = cap.read()
    if not ok:
        continue
    g = cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(float), (5, 5), 0)
    r = {"t_s": fi / 30}
    for k, (x0, x1, _) in cols.items():
        prof = g[y0:y1, x0:x1].mean(1)
        grad = np.abs(np.diff(prof)); i = int(np.argmax(grad))
        lo = max(0, i - 2); w = grad[lo:i + 3]; idx = np.arange(lo, lo + len(w))
        r[k + "_line_px"] = y0 + (w * idx).sum() / w.sum() + 0.5      # sub-pixel edge
    rows.append(r)
lv = pd.DataFrame(rows)
print("\nTANK LEVEL vs labeled volume used:")
for k, (_, _, col) in cols.items():
    used = np.array([rec.loc[rec.end_s <= t, col].sum() for t in lv.t_s])
    A = np.c_[np.ones(len(used)), used]
    b, *_ = np.linalg.lstsq(A, lv[k + "_line_px"], rcond=None)
    ml_per_px = 1 / b[1]
    lv[k + "_used_ml_labels"] = used
    lv[k + "_used_ml_measured"] = (lv[k + "_line_px"] - b[0]) * ml_per_px
    err = lv[k + "_used_ml_measured"] - used
    print(f"  {k}: {ml_per_px:.0f} mL per pixel, {used[-1]:.0f} mL used, measurement error sd {err.std():.0f} mL, "
          f"corr {np.corrcoef(used, lv[k + '_line_px'])[0, 1]:.3f}")
lv.round(2).to_csv("tank_levels.csv", index=False)
print("  T100 (clear) has no visible edge with this method; T300 is out of service.")
