"""Lane 4: write one twin file per recipe color, plus the render commands for each fill level.

What we found in the twin (csi_digital_twin_v1, fill station):
  - Liquid material: /FillStation/_materials/USD_blue_liquid_approximation (only its color changes here).
  - The built-in animation is recipe 7: blue fills (frames 21-368), pause, clear added (586-661).
    Liquid height -> volume (70 mL at the end):
      10 mL = frame 82, 20 = 148, 30 = 217, 40 = 290, 50 = 362, 60 = 619, 70 = 657.
    75 and 80 mL are NOT in the animation (would need the liquid mesh scaled).
  - The black vial holder is modeled too, so the twin hides the liquid line the same way the real camera does.

Colors: STARTING GUESSES. The twin's own blue (0.006, 0.20, 0.30) is treated as recipe 7
(50 blue / 70 total); orange is a guess. Mixes are done like dyes (multiply transmittance).
Calibrate: render a recipe, crop it with lane2_color/color_baseline.py's box, compare with the
real clean vial, adjust ORANGE / BLUE below, repeat.

Run from the TWIN ROOT folder (the one with csi_line.usda):
  python <repo>/lane4_twin/make_variants.py
Writes variants/R<id>_<name>.usda and render_commands.txt (one Isaac Sim render per line).
"""
import os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
STATION = "stations/fill_station/fill_station.usda"
MATERIAL = "/FillStation/_materials/USD_blue_liquid_approximation"
# The pouring stream keeps the twin's teal color and is visible during the clear pour (frames 586-661),
# so it would show at the 60/70 mL frames. Real vials are measured after filling (no stream), so hide it.
STREAM = "/FillStation/FillStation_LocalRoot/Fill_dispensing_stream"
FRAME_FOR_ML = {30: 217, 40: 290, 60: 619, 70: 657}      # nearest available for 75/80 mL is 70 (frame 657)

BLUE_R7 = np.array([0.006, 0.20, 0.30])   # twin's built-in liquid = recipe 7 (blue fraction 50/70)
ORANGE = np.array([0.35, 0.12, 0.02])     # guess for an orange-heavy mix (treated as fraction 50/70)
od_blue = -np.log(np.clip(BLUE_R7, 1e-3, 1)) / (50 / 70)
od_orng = -np.log(np.clip(ORANGE, 1e-3, 1)) / (50 / 70)

# Recipes from the dataset README: (T100 clear, T200 orange, T400 blue) in mL
RECIPES = {5: (10, 10, 10), 6: (30, 10, 0), 7: (20, 0, 50), 8: (20, 20, 20), 9: (10, 35, 35),
           10: (45, 15, 15), 11: (10, 10, 50), 12: (10, 50, 10), 13: (0, 50, 20), 14: (60, 0, 10)}

def color(t100, t200, t400):
    tot = t100 + t200 + t400
    od = (t200 / tot) * od_orng + (t400 / tot) * od_blue
    return np.round(np.exp(-od), 4)

def frame_for(total):
    return FRAME_FOR_ML[min(FRAME_FOR_ML, key=lambda m: abs(m - total))]

os.makedirs("variants", exist_ok=True)
lines = []
for rid, (a, b, c) in RECIPES.items():
    rgb = color(a, b, c)
    out = f"variants/R{rid}.usda"
    subprocess.run([sys.executable, os.path.join(HERE, "recolor_liquid.py"), "color", STATION,
                    "--material", MATERIAL, "--rgb", *map(str, rgb), "--out", out, "--hide", STREAM], check=True)
    total = a + b + c
    f = frame_for(total)
    lines.append(f"python tools/render_example.py --camera FillCam --stage {out} --frame {f} --out renders/R{rid}_{total}mL_f{f}.png")
    print(f"R{rid}: {total} mL -> frame {f}, color {tuple(float(x) for x in rgb)}")
open("render_commands.txt", "w").write("\n".join(lines) + "\n")
print("\nwrote render_commands.txt  (on Brev/Linux use <isaac-sim>/python.sh instead of python)")
