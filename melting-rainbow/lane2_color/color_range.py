"""Lane 2: lightest to darkest recipe, and where the camera stops telling colors apart.

Reads vial_colors.csv (from color_baseline.py) and ranks the clean last vial of each
recipe by lightness (Lab L, 0 = black, 100 = white). Then prints the color difference
(dE) between every pair of clean recipes: pairs under ~5 look the same to the camera,
so carryover between them can't be seen.

Known issue: on 30 mL fills (recipe 5) the default box (y 300) sits on the liquid's top
edge and reads too light. Re-measure 30 mL vials with --box 300 340 28 25.

Usage: python lane2_color/color_range.py vial_colors.csv
"""
import sys
import numpy as np, pandas as pd

d = pd.read_csv(sys.argv[1])
ref = d[d.pos_in_batch == 3].copy()
v = ref[["t100_ml", "t200_ml", "t400_ml"]].fillna(0)
ref["water_pct"] = (100 * v.t100_ml / v.sum(axis=1)).round()
ref["chroma"] = np.hypot(ref.a, ref.b).round(1)
ref = ref.sort_values("L", ascending=False)
print("Lightest to darkest (clean vials):")
print(ref[["recipe_id", "t100_ml", "t200_ml", "t400_ml", "water_pct", "L", "a", "b", "chroma"]].round(1).to_string(index=False))

ids = ref.recipe_id.tolist()
lab = ref.set_index("recipe_id")[["L", "a", "b"]]
pairs = []
for i, p in enumerate(ids):
    for q in ids[i + 1:]:
        pairs.append((p, q, float(np.linalg.norm(lab.loc[p] - lab.loc[q]))))
close = sorted([x for x in pairs if x[2] < 5], key=lambda x: x[2])
print("\nRecipe pairs the camera can barely tell apart (dE < 5):")
for p, q, e in close:
    print(f"  R{p} vs R{q}: dE {e:.1f}")
