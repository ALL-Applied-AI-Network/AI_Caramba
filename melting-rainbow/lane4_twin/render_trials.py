"""Render labeled twin vials for training (approach A) -- built on the teammate's render_vials.py.

KEPT from the teammate's script (it runs on Cruz's laptop):
  - opening the station through SimulationApp(open_usd=...), headless path tracing, 640x480 FillCam
  - camera clipping-range re-sync and the path-tracing warm-up
  - video 5 run order (R5 -> R14), one folder per recipe, a CSV index
CHANGED so the images are usable training data:
  - each vial gets ITS OWN liquid color (twin_colors.py: recipe mL + dye mixing), set in memory only
  - after each recipe change, vials with 10/20/40/60% leftover of the PREVIOUS recipe in run order
  - each recipe is shown at its own fill level (30 / 40 / 60 / 70 mL frames)
  - the pouring stream is hidden (real vials are judged after filling)
  - one real render per image (no copies), and the CSV has labels: carryover %, good / off-color, cause
  - label rule: off-color only when the color change is VISIBLE (dE vs the clean vial >= --tolerance);
    leftover that does not change the color is labeled good ("leftover present but not visible")
  - a log with timing, and a check that warns if two different colors produce the same image

The organizers' files are NOT changed: color and stream edits go to the in-memory session layer,
which is never saved.

Run from the twin folder (the one with csi_line.usda), with Isaac Sim's Python:
  C:\\isaac\\env\\Scripts\\python.exe render_trials.py --dry-run     # writes the plan only, no Isaac Sim
  C:\\isaac\\env\\Scripts\\python.exe render_trials.py --limit 3     # quick test: first 3 images
  C:\\isaac\\env\\Scripts\\python.exe render_trials.py               # full set (46 images; --extra-pairs adds 324)
"""
import argparse, csv, hashlib, itertools, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_colors import RECIPES, RUN_ORDER, color, frame_for, total_ml

p = argparse.ArgumentParser(description="Render labeled twin vials (recipe colors + carryover)")
p.add_argument("--stage", default="stations/fill_station/fill_station.usda")
p.add_argument("--out_dir", default="twin_trials")
p.add_argument("--carry", type=float, nargs="+", default=[0.1, 0.2, 0.4, 0.6],
               help="leftover fractions of the previous recipe to render after each recipe change")
p.add_argument("--tolerance", type=float, default=5.0,
               help="label rule (agreed Oct 22): off-color when the vial's color differs from the clean vial "
                    "of the same recipe by at least this many dE (CIE Lab). Value set after calibration.")
p.add_argument("--ev", type=float, default=0.0,
               help="camera exposure:time = 2**ev seconds (0 = leave the camera untouched); from calibration on R6+R14")
p.add_argument("--extra-pairs", action="store_true",
               help="also render first vials for every other recipe pair (not just video 5 order)")
p.add_argument("--limit", type=int, default=0, help="render only the first N images (0 = all)")
p.add_argument("--paper", action="store_true",
               help="add the real line's white paper sheet behind the vial (in memory; twin_scene.py)")
p.add_argument("--subframes", type=int, default=32, help="path-tracing quality per capture")
p.add_argument("--dry-run", action="store_true", help="write the plan CSV only; do not start Isaac Sim")
args = p.parse_args()

# ---------------------------------------------------------------- plan (pure Python, no Isaac Sim)
plan = []
# In the twin, vial 1, 2 and 3 of a recipe look identical unless their liquid differs, so each image is a
# UNIQUE (recipe, previous recipe, leftover fraction) combination -- no duplicate images.
def add(rid, prev, carry, kind):
    rgb = color(rid, prev, carry)
    plan.append(dict(
        image="", kind=kind, recipe_id=rid, prev_recipe_id=prev if prev is not None else "",
        carryover=carry, total_ml=total_ml(rid), frame=frame_for(rid),
        liquid_r=rgb[0], liquid_g=rgb[1], liquid_b=rgb[2],
        label="good" if carry == 0 else "pending",      # decided after rendering, by visible dE
        cause="none"))

for i, rid in enumerate(RUN_ORDER):
    prev = RUN_ORDER[i - 1] if i > 0 else None   # what ran before R5 in video 5 is unknown -> clean only
    add(rid, prev, 0.0, "clean")
    if prev:
        for c in args.carry:
            add(rid, prev, c, "video5_order")

if args.extra_pairs:
    consecutive = set(zip(RUN_ORDER, RUN_ORDER[1:]))
    for prev, rid in itertools.permutations(RUN_ORDER, 2):
        if (prev, rid) not in consecutive:
            for c in args.carry:
                add(rid, prev, c, "extra_pair")

for n, r in enumerate(plan, 1):
    tag = f"after_R{r['prev_recipe_id']}_carry{int(round(r['carryover'] * 100)):02d}" if r["carryover"] > 0 else "clean"
    r["image"] = f"recipe_{r['recipe_id']}/{n:03d}_R{r['recipe_id']}_{tag}_f{r['frame']}.png"
if args.limit:
    plan = plan[:args.limit]

out = os.path.abspath(args.out_dir)
os.makedirs(out, exist_ok=True)
fields = list(plan[0].keys()) + ["crop_r", "crop_g", "crop_b", "crop_L", "crop_a", "crop_b_lab",
                                 "dE_to_clean", "md5", "seconds", "status"]
plan_csv = os.path.join(out, "plan.csv")
with open(plan_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(plan[0].keys())); w.writeheader(); w.writerows(plan)
n_carry = sum(r["carryover"] > 0 for r in plan)
print(f"Plan: {len(plan)} images ({len(plan) - n_carry} clean, {n_carry} with leftover; "
      f"off-color/good decided after rendering by dE >= {args.tolerance}) -> {plan_csv}")
if args.dry_run:
    sys.exit(0)

# ---------------------------------------------------------------- render (teammate's Isaac Sim setup)
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "open_usd": args.stage, "renderer": "PathTracing",
                         "width": 640, "height": 480})
log_path = os.path.join(out, "render_log.txt")
log = open(log_path, "a", buffering=1)
def say(msg):
    print(msg, flush=True); log.write(msg + "\n")

try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import numpy as np, cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdShade, Gf

    say("=" * 60)
    say(f"Started {time.ctime()}  stage={os.path.abspath(args.stage)}  images={len(plan)}  "
        f"tolerance_dE={args.tolerance}  exposure_EV={args.ev}  subframes={args.subframes}")
    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())          # in-memory only; organizers' files untouched
    timeline = omni.timeline.get_timeline_interface()
    fps = stage.GetTimeCodesPerSecond() or 30.0

    shader = UsdShade.Shader(stage.GetPrimAtPath(
        "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF"))
    if not shader:
        raise RuntimeError("liquid shader not found - is this the organizers' fill_station.usda?")
    diffuse = shader.GetInput("diffuseColor")
    stream = stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot/Fill_dispensing_stream")
    if stream.IsValid():
        UsdGeom.Imageable(stream).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
        say("Pouring stream hidden (session layer)")

    if args.paper:
        from twin_scene import add_paper_sheet
        add_paper_sheet(stage)
        print("White paper sheet added behind the vial (session layer)", flush=True)
    cam_path = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
    rp = rep.create.render_product(cam_path, (640, 480))
    ann = rep.AnnotatorRegistry.get_annotator("rgb"); ann.attach([rp])
    rep.orchestrator.set_capture_on_play(False)
    for _ in range(10):
        sim_app.update()

    # teammate's RTX camera re-sync
    cam = stage.GetPrimAtPath(cam_path)
    if cam.IsValid():
        clip = UsdGeom.Camera(cam).GetClippingRangeAttr(); nf = clip.Get()
        if nf:
            clip.Set(Gf.Vec2f(nf[0] * 0.5, nf[1])); sim_app.update(); clip.Set(nf); sim_app.update()

    # teammate's path-tracing warm-up
    timeline.set_current_time(1.0 / fps)
    for _ in range(30):
        rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)

    if args.ev != 0.0:
        # Isaac Sim ignores USD 'exposure' but honours 'exposure:time' (physical exposure; calibration Oct 22)
        cam.GetAttribute("exposure:time").Set(float(2.0 ** args.ev))   # session layer, in memory only
        sim_app.update()
        say(f"Camera exposure:time set to 2^{args.ev:+.2f} = {2.0 ** args.ev:.5f} s (session layer)")

    def lab(rgb):
        px = np.uint8([[np.clip(rgb, 0, 255)]])
        L, A, B = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        return np.array([L * 100 / 255, A - 128, B - 128])          # same conversion as Lane 2
    clean_lab = {}

    results_csv = os.path.join(out, "labels.csv")
    last = None
    with open(results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for k, r in enumerate(plan, 1):
            t0 = time.time()
            diffuse.Set(Gf.Vec3f(r["liquid_r"], r["liquid_g"], r["liquid_b"]))
            timeline.set_current_time(r["frame"] / fps)
            sim_app.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            data = np.asarray(ann.get_data())
            if data.ndim != 3 or data.shape[0] != 480 or data.shape[1] != 640:
                r.update(status=f"FAILED (buffer {data.shape})", seconds=round(time.time() - t0, 1))
                say(f"[{k}/{len(plan)}] FAILED {r['image']}  buffer {data.shape}")
                w.writerow(r); continue
            img = data[..., :3].copy()
            path = os.path.join(out, r["image"])
            os.makedirs(os.path.dirname(path), exist_ok=True)
            cv2.imwrite(path, img[..., ::-1])
            # Lane 2 box; 30-40 mL fills sit lower (in the twin the 40 mL surface is at row ~300)
            x, y, bw, bh = (300, 340, 28, 25) if r["total_ml"] <= 40 else (300, 300, 28, 30)
            crop = np.median(img[y:y + bh, x:x + bw].reshape(-1, 3), axis=0)   # median, like Lane 2
            L = lab(crop)
            if r["carryover"] == 0:
                clean_lab[r["recipe_id"]] = L
                de = 0.0
            else:
                de = float(np.linalg.norm(L - clean_lab[r["recipe_id"]]))
                visible = de >= args.tolerance
                r["label"] = "off-color" if visible else "good"
                r["cause"] = (f"carryover from R{r['prev_recipe_id']}" if visible
                              else "leftover present but not visible")
            md5 = hashlib.md5(img.tobytes()).hexdigest()
            rgb = (r["liquid_r"], r["liquid_g"], r["liquid_b"])
            status = "OK"
            if last and last[0] == md5 and last[1] != rgb:
                status = "WARNING: identical to previous image despite a different color"
            last = (md5, rgb)
            r.update(crop_r=round(float(crop[0]), 1), crop_g=round(float(crop[1]), 1),
                     crop_b=round(float(crop[2]), 1), crop_L=round(L[0], 1), crop_a=round(L[1], 1),
                     crop_b_lab=round(L[2], 1), dE_to_clean=round(de, 1), md5=md5, seconds=round(time.time() - t0, 1), status=status)
            w.writerow(r); f.flush()
            say(f"[{k}/{len(plan)}] {status}  {r['image']}  {r['label']}  dE {r['dE_to_clean']}  crop RGB "
                f"({r['crop_r']}, {r['crop_g']}, {r['crop_b']})  {r['seconds']} s")
    say(f"Finished {time.ctime()}  -> {results_csv}")
except BaseException:
    import traceback
    say("FAILED:\n" + traceback.format_exc())
    log.close()
    sim_app.close()
    raise
log.close()
sim_app.close()
