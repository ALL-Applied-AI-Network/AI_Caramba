"""In-memory scene fixes so the twin matches the real fill station (organizers' files are never changed).

add_paper_sheet: the real line has a white paper sheet hung behind the vial that reaches down to image row
~320 at the vial (checked on all 30 video 5 vials). The twin's backdrop stops higher, so in the twin the
liquid below the holder is seen against the dark rear trough. This adds a thin white sheet on the session
layer at the twin's splash-guard plane (x = 5.28 m, the panel behind the vial), from z = 0.997 m (the height
that projects to row 320 through the organizers' FillCam calibration, cameras/FillCam.json) up to the top of
the guard (z = 1.16 m), across the guard's width (y = -2.34 .. -1.63 m).
"""
from pxr import Gf, Sdf, UsdGeom, UsdShade

SHEET = "/FillStation/MR_paper_sheet"   # under the identity root (LocalRoot is in inches and rotated)
SHEET_MAT = "/FillStation/_materials/MR_paper_white"


def add_paper_sheet(stage, albedo=0.85, x=5.286, y=(-2.34, -1.63), z=(0.997, 1.16), thick=0.002):
    """Call with the session layer as edit target. Returns the sheet prim."""
    mat = UsdShade.Material.Define(stage, SHEET_MAT)
    sh = UsdShade.Shader.Define(stage, SHEET_MAT + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(albedo, albedo, albedo))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    cube = UsdGeom.Cube.Define(stage, SHEET)
    cube.CreateSizeAttr(1.0)
    # /FillStation has an identity transform, so these are the station file's world coordinates (meters)
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(x, (y[0] + y[1]) / 2, (z[0] + z[1]) / 2))
    xf.AddScaleOp().Set(Gf.Vec3f(thick, y[1] - y[0], z[1] - z[0]))
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(mat)
    return cube.GetPrim()
