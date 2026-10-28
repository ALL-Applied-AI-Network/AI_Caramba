"""Render a large, varied, labeled twin dataset for training (dataset v2, ~2000 images).

Same setup as render_trials.py (teammate's Isaac Sim script + our calibrated colors, exposure 1/256 s,
pouring stream hidden, organizers' files untouched), but built for training volume WITHOUT duplicates:

  RENDERS (each a different liquid):
    - 10 clean reference vials (one per recipe)
    - --renders leftover vials: a random "previous -> current" recipe pair (all 90 ordered pairs, not just
      video 5's order) with a random leftover fraction between 5% and 100% of the previous recipe
  LABEL (agreed rule): off-color when the color change vs the clean vial of the same recipe is VISIBLE,
    dE >= --tolerance (default 5; two clean REAL vials of one recipe differ by 1.6-4.9). Measured on the raw
    render, before any augmentation.
  AUGMENTED COPIES (what the model trains on; raw PNGs are kept too):
    brightness +/- 1/3 stop, white balance +/- 7% per channel, shift up to 2 px, sensor noise, JPEG quality
    75-95 -- like a real camera. (White-balance jitter added Oct 22: the real camera's colors are not fully
    consistent between recipes -- e.g. real R12 is redder than any dye mix with R14's blue share allows --
    which points to automatic exposure/white balance in the real camera.)
    Leftover renders get 1 augmented copy each; each clean vial gets --clean-copies copies so that clean
    examples are a fair share of the set (these copies differ only by augmentation; marked in the CSV).

Known limit (README): at the organizers' liquid opacity 0.6 the twin can't show leftover between some dark
recipes (e.g. R11->R12), so those images are labeled "good" because they LOOK unchanged in the twin.

Run from the twin folder:
  C:\\isaac\\env\\Scripts\\python.exe render_dataset.py --renders 1400 --clean-copies 60
  (about 35-40 min; labels.csv is written as it goes, so partial results are usable)
"""
import argparse, csv, hashlib, itertools, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from twin_colors import RUN_ORDER, CAMERA_EV, color, frame_for, total_ml

p = argparse.ArgumentParser()
p.add_argument("--stage", default="stations/fill_station/fill_station.usda")
p.add_argument("--out_dir", default="twin_dataset_v2")
p.add_argument("--renders", type=int, default=1400, help="number of leftover renders (each unique)")
p.add_argument("--clean-copies", type=int, default=60, help="augmented copies per clean recipe")
p.add_argument("--tolerance", type=float, default=5.0)
p.add_argument("--ev", type=float, default=CAMERA_EV)
p.add_argument("--seed", type=int, default=2026)
p.add_argument("--paper", action="store_true",
               help="add the real line's white paper sheet behind the vial (in memory; twin_scene.py)")
p.add_argument("--subframes", type=int, default=32)
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()
rng = np.random.default_rng(args.seed)

# ---------------------------------------------------------------- plan
pairs = list(itertools.permutations(RUN_ORDER, 2))            # 90 ordered (previous, current) pairs
consecutive = set(zip(RUN_ORDER, RUN_ORDER[1:]))
plan = [dict(kind="clean", recipe_id=rid, prev_recipe_id="", carryover=0.0) for rid in RUN_ORDER]
# No two renders may have the same liquid at the same fill level (e.g. R5 and R8 share a 1:1:1 mix, so
# "R8 with R5 leftover" would be pixel-identical to clean R8) -- such draws are skipped and redrawn.
seen = {(color(r, None, 0.0), frame_for(r)) for r in RUN_ORDER}
while len(plan) < len(RUN_ORDER) + args.renders:
    prev, rid = pairs[rng.integers(len(pairs))]
    carry = round(float(rng.uniform(0.05, 1.0)), 3)
    key = (color(rid, prev, carry), frame_for(rid))
    if key in seen:
        continue
    seen.add(key)
    plan.append(dict(kind="video5_order" if (prev, rid) in consecutive else "other_pair",
                     recipe_id=rid, prev_recipe_id=prev, carryover=carry))
for n, r in enumerate(plan, 1):
    rgb = color(r["recipe_id"], r["prev_recipe_id"] or None, r["carryover"])
    r.update(render_id=n, total_ml=total_ml(r["recipe_id"]), frame=frame_for(r["recipe_id"]),
             liquid_r=rgb[0], liquid_g=rgb[1], liquid_b=rgb[2])
    tag = "clean" if r["kind"] == "clean" else f"after_R{r['prev_recipe_id']}_carry{int(round(r['carryover']*100)):03d}"
    r["raw_image"] = f"raw/recipe_{r['recipe_id']}/{n:04d}_R{r['recipe_id']}_{tag}.png"
out = os.path.abspath(args.out_dir); os.makedirs(out, exist_ok=True)
n_aug = args.renders + len(RUN_ORDER) * args.clean_copies
print(f"Plan: {len(plan)} renders ({len(RUN_ORDER)} clean + {args.renders} leftover) -> {n_aug} training images")
if args.dry_run:
    with open(os.path.join(out, "plan.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(plan[0].keys())); w.writeheader(); w.writerows(plan)
    sys.exit(0)

# ---------------------------------------------------------------- render
from isaacsim import SimulationApp
sim_app = SimulationApp({"headless": True, "open_usd": args.stage, "renderer": "PathTracing",
                         "width": 640, "height": 480})
log = open(os.path.join(out, "render_log.txt"), "a", buffering=1)
def say(m):
    print(m, flush=True); log.write(m + "\n")
try:
    from isaacsim.core.utils.extensions import enable_extension
    enable_extension("omni.replicator.core")
    import cv2
    import omni.usd, omni.timeline
    import omni.replicator.core as rep
    from pxr import UsdGeom, UsdShade, Gf

    stage = omni.usd.get_context().get_stage()
    stage.SetEditTarget(stage.GetSessionLayer())                     # in memory only
    tl = omni.timeline.get_timeline_interface(); fps = stage.GetTimeCodesPerSecond() or 30.0
    diffuse = UsdShade.Shader(stage.GetPrimAtPath(
        "/FillStation/_materials/USD_blue_liquid_approximation/Principled_BSDF")).GetInput("diffuseColor")
    UsdGeom.Imageable(stage.GetPrimAtPath("/FillStation/FillStation_LocalRoot/Fill_dispensing_stream")
                      ).GetVisibilityAttr().Set(UsdGeom.Tokens.invisible)
    if args.paper:
        from twin_scene import add_paper_sheet
        add_paper_sheet(stage)
        print("White paper sheet added behind the vial (session layer)", flush=True)
    cam_path = "/FillStation/FillStation_LocalRoot/FillCam/FillCam"
    cam = stage.GetPrimAtPath(cam_path)
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
    cam.GetAttribute("exposure:time").Set(float(2.0 ** args.ev)); sim_app.update()
    say("=" * 60)
    say(f"Started {time.ctime()}  renders={len(plan)}  training images={n_aug}  exposure:time=2^{args.ev}  "
        f"tolerance_dE={args.tolerance}  seed={args.seed}")

    def lab(rgb):
        px = np.uint8([[np.clip(rgb, 0, 255)]])
        L, A, B = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
        return np.array([L * 100 / 255, A - 128, B - 128])

    def augment(img, k):
        """img RGB uint8 -> augmented RGB uint8 + parameters (seeded per image)."""
        g = np.random.default_rng(args.seed * 100003 + k)
        stops = float(g.uniform(-1 / 3, 1 / 3)); dx, dy = (int(v) for v in g.integers(-2, 3, size=2))
        sigma = float(g.uniform(1.0, 3.0)); q = int(g.integers(75, 96))
        wb = g.uniform(0.93, 1.07, size=3)          # white-balance jitter: real camera colors drift per scene
        a = np.clip(img.astype(np.float32) * (2.0 ** stops) * wb, 0, 255)
        a = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (640, 480), borderMode=cv2.BORDER_REPLICATE)
        a = np.clip(a + g.normal(0, sigma, a.shape), 0, 255).astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", a[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        return enc, dict(aug_brightness_stops=round(stops, 3), aug_wb_r=round(float(wb[0]), 3),
                         aug_wb_g=round(float(wb[1]), 3), aug_wb_b=round(float(wb[2]), 3), aug_shift_x=dx, aug_shift_y=dy,
                         aug_noise_sigma=round(sigma, 2), aug_jpeg_quality=q)

    fields = ["image", "raw_image", "render_id", "copy", "kind", "recipe_id", "prev_recipe_id", "carryover",
              "total_ml", "frame", "liquid_r", "liquid_g", "liquid_b", "crop_r", "crop_g", "crop_b",
              "dE_to_clean", "label", "cause", "aug_brightness_stops", "aug_wb_r", "aug_wb_g", "aug_wb_b", "aug_shift_x", "aug_shift_y",
              "aug_noise_sigma", "aug_jpeg_quality", "raw_md5"]
    clean_lab, k_img, t_start = {}, 0, time.time()
    with open(os.path.join(out, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, r in enumerate(plan, 1):
            diffuse.Set(Gf.Vec3f(r["liquid_r"], r["liquid_g"], r["liquid_b"]))
            tl.set_current_time(r["frame"] / fps); sim_app.update()
            for _ in range(2):
                rep.orchestrator.step(rt_subframes=args.subframes, delta_time=0.0)
            img = np.asarray(ann.get_data())[..., :3].copy()
            raw_path = os.path.join(out, r["raw_image"]); os.makedirs(os.path.dirname(raw_path), exist_ok=True)
            cv2.imwrite(raw_path, img[..., ::-1])
            x, y, bw, bh = (300, 340, 28, 25) if r["total_ml"] <= 40 else (300, 300, 28, 30)
            crop = np.median(img[y:y + bh, x:x + bw].reshape(-1, 3), axis=0); L = lab(crop)
            if r["kind"] == "clean":
                clean_lab[r["recipe_id"]] = L; de = 0.0; label, cause = "good", "none"
            else:
                de = float(np.linalg.norm(L - clean_lab[r["recipe_id"]]))
                vis = de >= args.tolerance
                label = "off-color" if vis else "good"
                cause = f"carryover from R{r['prev_recipe_id']}" if vis else "leftover present but not visible"
            base = dict(r, crop_r=round(float(crop[0]), 1), crop_g=round(float(crop[1]), 1),
                        crop_b=round(float(crop[2]), 1), dE_to_clean=round(de, 1), label=label, cause=cause,
                        raw_md5=hashlib.md5(img.tobytes()).hexdigest())
            base.pop("raw_image", None); base["raw_image"] = r["raw_image"]
            for c in range(args.clean_copies if r["kind"] == "clean" else 1):
                k_img += 1
                enc, aug = augment(img, k_img)
                name = f"images/{label}/recipe_{r['recipe_id']}/{k_img:05d}_R{r['recipe_id']}_" + \
                       ("clean" if r["kind"] == "clean" else f"after_R{r['prev_recipe_id']}") + f"_c{c}.jpg"
                path = os.path.join(out, name); os.makedirs(os.path.dirname(path), exist_ok=True)
                enc.tofile(path)
                w.writerow({**{k: base.get(k, "") for k in fields}, "image": name, "copy": c, **aug})
            f.flush()
            if i % 25 == 0 or i <= 10:
                el = time.time() - t_start
                say(f"[{i}/{len(plan)}] R{r['recipe_id']} {r['kind']} carry {r['carryover']:.2f} -> {label} "
                    f"(dE {de:.1f})  {el/60:.1f} min elapsed, ~{el/i*(len(plan)-i)/60:.0f} min left")
    say(f"Finished {time.ctime()}  training images={k_img}")
except BaseException:
    import traceback
    say("FAILED:\n" + traceback.format_exc()); log.close(); sim_app.close(); raise
log.close()
sim_app.close()
