"""Lane 4: list materials in the fill station, or write a recolored COPY for one recipe/defect.

Does NOT edit the original twin files. It writes a small override layer that
sublayers the station file and changes only the liquid material's diffuseColor,
so you can render one variant per color without touching the shared twin.

Runs with plain Python + `pip install usd-core` (no GPU needed), or with Isaac Sim's python.sh.

  # 1) find the liquid material(s)
  python lane4_twin/recolor_liquid.py list <twin>/stations/fill_station/fill_station.usda

  # 2) make a variant (RGB 0-1), then render it with the twin's render_example.py --stage <variant>
  python lane4_twin/recolor_liquid.py color <twin>/stations/fill_station/fill_station.usda \
      --material /Path/To/LiquidMaterial --rgb 0.2 0.6 0.4 --out variants/r7_carryover.usda
"""
import argparse, os
from pxr import Usd, UsdShade, Sdf, Gf

ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["list", "color"])
ap.add_argument("stage")
ap.add_argument("--material"); ap.add_argument("--rgb", nargs=3, type=float)
ap.add_argument("--out")
ap.add_argument("--hide", action="append", default=[],
                help="prim path to hide in every frame (e.g. the dispensing stream); repeatable")
a = ap.parse_args()

stage = Usd.Stage.Open(a.stage)
if a.cmd == "list":
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Material):
            colors = []
            for child in Usd.PrimRange(prim):
                sh = UsdShade.Shader(child)
                if sh and sh.GetIdAttr().Get() == "UsdPreviewSurface":
                    inp = sh.GetInput("diffuseColor")
                    colors.append(inp.Get() if inp else None)
            print(prim.GetPath(), colors)
    raise SystemExit

assert a.material and a.rgb and a.out, "color needs --material, --rgb, --out"
os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
layer = Sdf.Layer.CreateNew(a.out)
layer.subLayerPaths.append(os.path.relpath(os.path.abspath(a.stage), os.path.dirname(os.path.abspath(a.out))).replace("\\", "/"))
# Copy timing/units from the source so frame numbers mean the same thing (twin = 30 fps, frames 1-686).
src = Sdf.Layer.FindOrOpen(a.stage)
for key in ("timeCodesPerSecond", "framesPerSecond", "startTimeCode", "endTimeCode", "upAxis", "metersPerUnit", "defaultPrim"):
    if src.pseudoRoot.GetInfo(key) is not None and src.pseudoRoot.HasInfo(key):
        layer.pseudoRoot.SetInfo(key, src.pseudoRoot.GetInfo(key))
var = Usd.Stage.Open(layer)
var.SetEditTarget(layer)
n = 0
for child in Usd.PrimRange(var.GetPrimAtPath(a.material)):
    sh = UsdShade.Shader(child)
    if sh and sh.GetIdAttr().Get() == "UsdPreviewSurface":
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*a.rgb))
        n += 1
for path in a.hide:
    # A plain (non-animated) value in this stronger layer overrides the station's animated visibility.
    p = var.OverridePrim(path)
    p.CreateAttribute("visibility", Sdf.ValueTypeNames.Token).Set("invisible")
    print(f"hidden in all frames: {path}")
layer.Save()
print(f"wrote {a.out}: set diffuseColor on {n} shader(s) under {a.material}")
if n == 0:
    print("no UsdPreviewSurface found under that path - run `list` and pick the liquid material")
