"""Exposure calibration: render the 10 clean recipes at several camera exposures (in memory only).

Why: twin vials come out much lighter than the real camera (L* 26-66 vs 5-26). Cruz approved changing the
CAMERA EXPOSURE only (liquid opacity stays at the organizers' 0.6).
How it is used (analysis done afterwards, see calibrate_report.py):
  - the exposure is CHOSEN using only R6 (orange only) and R14 (blue only), as agreed;
  - the other 8 recipes are held out and only used to report how well that choice works.
Round 1 (Oct 22) showed: the USD `exposure` attribute is IGNORED by Isaac Sim, but `exposure:time` WORKS --
and once exposure is authored, RTX switches to physical camera exposure (f-stop 1, ISO 100), which at
time = 1 s is massively over-exposed (crops near white). So round 2 sweeps exposure:time = 2**EV over a
wide range, for all 10 clean recipes.
Round 4 (only with Cruz's OK): --opacity tries other liquid opacities in the ladder, because at the organizers'
0.6 a full 70 mL vial cannot get darker than ~13 per channel, which hides carryover between dark recipes.
Organizers' files are not changed: edits go to the in-memory session layer.

Run from the twin folder:  C:\\isaac\\env\\Scripts\\python.exe calibrate_exposure.py
"""
import argparse, csv, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_colors import RUN_ORDER, color, frame_for, total_ml

p = argparse.ArgumentParser()
p.add_argument("--stage", default="stations/fill_station/fill_station.usda")
p.add_argument("--out_dir", default="calibration")
p.add_argument("--ev", type=float, nargs="+", default=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12])
p.add_argument("--subframes", type=int, default=32)
p.add_argument("--ladder", action="store_true",
               help="round 3: at one exposure (--ev, one value), render a ladder of liquid colors at the 40 mL "
                    "and 70 mL fill levels to measure how a liquid color turns into a camera color")
p.add_argument("--opacity", type=float, nargs="+", default=[0.6],
               help="round 4 (needs Cruz's OK): liquid opacity values to try in the ladder; 0.6 = organizers' value")
args = p.parse_args()

jobs = [("exposure_time", ev, rid) for ev in args.ev for rid in RUN_ORDER]
LADDER = []
if args.ladder:
    ev = args.ev[0]
    grays = [0.003, 0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]
    tints = [(0.6, 0.05, 0.01), (0.3, 0.05, 0.01), (0.01, 0.2, 0.4), (0.01, 0.5, 0.6), (0.2, 0.4, 0.1), (0.5, 0.3, 0.05)]
    for op in args.opacity:
        for rid in (6, 14):                   # 40 mL (R6 level) and 70 mL (R14 level)
            for v in grays:
                LADDER.append((f"ladder_op{op}", ev, rid, (v, v, v), op))
            for t in tints:
                LADDER.append((f"ladder_op{op}", ev, rid, t, op))
    jobs = [(m, e, r) for (m, e, r, _, _) in LADDER]
out = os.path.abspath(args.out_dir); os.makedirs(out, exist_ok=True)

from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "open_usd": args.stage, "renderer": "PathTracing",
                         "width": 640, "height": 480})
log = open(os.path.join(out, "calibration_log.txt"), "a", buffering=1)
def say(m):
    print(m, flush=True); log.write(m + "\n")
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdShade, Gf, Sdf

    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())
    tl = omni.timeline.get_timeline_interface(); fps = stage.GetTimeCodesPerSecond() or 30.0
    diffuse = UsdShade.Shader(stage.GetPrimAtPath(
        "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF")).GetInput("diffuseColor")
    stream = stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot/Fill_dispensing_stream")
    UsdGeom.Imageable(stream).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    cam_path = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
    cam = stage.GetPrimAtPath(cam_path)
    a_exp = cam.GetAttribute("exposure")
    a_time = cam.GetAttribute("exposure:time")
    opacity_in = UsdShade.Shader(stage.GetPrimAtPath(
        "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF")).GetInput("opacity")
    say(f"organizers' liquid opacity = {opacity_in.Get()}")
    say("=" * 60)
    say(f"Started {time.ctime()}  original camera exposure={a_exp.Get()}  exposure:time={a_time.Get()}  "
        f"jobs={len(jobs)}")

    rp = rep.create.render_product(cam_path, (640, 480))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    for _ in range(10):
        sim_app.update()
    clip = UsdGeom.Camera(cam).GetClippingRangeAttr(); nf = clip.Get()
    clip.Set(Gf.Vec2f(nf[0] * 0.5, nf[1])); sim_app.update(); clip.Set(nf); sim_app.update()
    tl.set_current_time(1.0 / fps)
    for _ in range(30):
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

    with open(os.path.join(out, "calibration.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "ev", "recipe_id", "frame", "liquid_r", "liquid_g", "liquid_b",
                    "crop_box", "crop_R", "crop_G", "crop_B", "image", "seconds"])
        for k, (method, ev, rid) in enumerate(jobs, 1):
            t0 = time.time()
            a_time.Set(float(2.0 ** ev))
            rgb = LADDER[k - 1][3] if args.ladder else color(rid)
            if args.ladder:
                opacity_in.Set(float(LADDER[k - 1][4]))   # session layer, in memory only
            diffuse.Set(Gf.Vec3f(*rgb))
            fr = frame_for(rid)
            tl.set_current_time(fr / fps); sim_app.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            img = np.asarray(ann.get_data())[..., :3].copy()
            name = (f"{method}_{k:03d}_ml{total_ml(rid)}.png" if args.ladder else f"{method}_ev{ev:+.1f}_R{rid}.png")
            cv2.imwrite(os.path.join(out, name), img[..., ::-1])
            x, y, bw, bh = (300, 340, 28, 25) if total_ml(rid) <= 40 else (300, 300, 28, 30)
            med = np.median(img[y:y + bh, x:x + bw].reshape(-1, 3), axis=0)
            w.writerow([method, ev, rid, fr, *rgb, f"{x} {y} {bw} {bh}", *[round(float(v), 1) for v in med],
                        name, round(time.time() - t0, 1)]); f.flush()
            say(f"[{k}/{len(jobs)}] {method} EV {ev:+.1f} R{rid}: crop RGB {tuple(round(float(v)) for v in med)}")
    say(f"Finished {time.ctime()}")
except BaseException:
    import traceback
    say("FAILED:\n" + traceback.format_exc()); log.close(); sim_app.close(); raise
log.close()
sim_app.close()
