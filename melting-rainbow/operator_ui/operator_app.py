"""Melting Rainbow - operator station (browser UI). Rule (Approach A) + AI second opinion (pair model).

Start it, and a page opens in the browser (http://localhost:8765). The operator picks the camera (or a video file),
presses Start, and watches each fill. When a batch of vials is done, every vial gets:
    PASS     - the rule and the AI both say it matches the batch's clean reference vial -> leave it in the good lane
    REJECT   - both say it's off-color (with the likely cause and leftover) -> operator pulls it to scrap
    REVIEW   - they disagree -> the operator looks and decides Pass / Reject
and the operator releases or holds the batch. Every decision is written to operator_log.csv.

Why an operator in the loop (organizers, Oct 26): our system can't talk to the line's sensors/actuators yet. The line's
existing Teledyne sorter checks color against a static, non-updatable color list with no link to the recipe, so it sends
almost every vial to the "good" location. The page therefore (1) tells the operator which vials to pull, (2) lets the
operator record which lane the existing sorter actually used ("Sorter: good / scrap") - that is the measured baseline -
and (3) lets the operator enter the batch's recipe, so the check is tied to the recipe. PLC integration comes after.

  python operator_ui/operator_app.py                                  # then use the page
  python operator_ui/operator_app.py --ai-model pair_model.pt         # with the AI second opinion
  python operator_ui/operator_app.py --video <...>/vial_fill_video_5.mp4 --autostart   # demo on a recording

Only needs what the demo needs (python, opencv-python, numpy). The AI part also needs torch (+ timm for the
EfficientNet version); without them the page runs on the rule alone and says so.

Where the decisions come from (nothing tuned on video 5):
  rule ..... demo/live_inspect.py + lane6_classifier/approach_a.py: compare with the batch's LAST vial, off if dE >= 7
             (cutoff from videos 1-4 noise); cause with sure/unsure (threshold set on the twin).
  AI ....... lane6_classifier/pair_model/pair_model.py: vial crop + reference crop -> P(off), trained on the twin only.
             Off if P >= 0.5. Oct 26 small-CNN model on video 5 (scored once): 0 misses, 2 false alarms.
  batches .. every N vials (default 3, like our CSI plan), or the operator presses "New batch" at a recipe change.
"""
import argparse, csv, io, json, os, re, sys, threading, time, webbrowser
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from demo.live_inspect import Inspector, holder_present, steady_frames, auto_ml, liquid_score   # noqa: E402
from lane2_color.detect_fills_level import LevelFillDetector, level_label, level_pct, final_level   # noqa: E402
from flywheel.examples import save_example   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

LIVE_URL_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://", "udp://")
VIDEO_DEVICE_RE = re.compile(r"^/dev/video(\d+)$")


def is_hardware_camera(source):
    """True for a numeric camera or a Linux V4L2 device/by-id path."""
    value = str(source).strip()
    return value.isdigit() or value.startswith(("/dev/video", "/dev/v4l/"))


def is_network_camera(source):
    return str(source).strip().lower().startswith(LIVE_URL_PREFIXES)


def resolve_camera_index(source):
    """Resolve a numeric or stable /dev/v4l/by-id source to /dev/videoN."""
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    if not os.path.exists(value):
        raise RuntimeError(
            f"{value!r} does not exist - reconnect the camera or check /dev/v4l/by-id"
        )
    real = os.path.realpath(value)
    match = VIDEO_DEVICE_RE.fullmatch(real)
    if match is None:
        raise RuntimeError(f"{value!r} resolved to {real!r}, not a /dev/videoN device")
    return int(match.group(1))


def open_linewatch_camera(source):
    """Open a local camera with the proven Linewatch Vial Fill settings."""
    index = resolve_camera_index(source)
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(
            f"could not open {source!r} (resolved to /dev/video{index}); "
            "check that another application is not using it"
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Match Linewatch: let exposure and white balance settle, then freeze WB
    # so recipe colours remain comparable while exposure stays automatic.
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
    cap.set(cv2.CAP_PROP_AUTO_WB, 1)
    got_frame = False
    for _ in range(30):
        ok, frame = cap.read()
        got_frame = got_frame or (ok and frame is not None)
    if not got_frame:
        cap.release()
        raise RuntimeError(
            f"{source!r} (/dev/video{index}) opened but returned no frames; "
            "check USB bandwidth and camera connections"
        )

    wb = cap.get(cv2.CAP_PROP_WB_TEMPERATURE)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)
    if wb > 0:
        cap.set(cv2.CAP_PROP_WB_TEMPERATURE, wb)

    settings = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "auto_exposure": float(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE)),
        "auto_wb": float(cap.get(cv2.CAP_PROP_AUTO_WB)),
        "wb_temperature": float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
    }
    return cap, index, settings

# ----------------------------------------------------------------------------------------------- camera alignment
# Our measuring windows are fixed pixel spots, set for the organizers' vial-fill camera (640x480). A different camera
# (e.g. our own webcam at CSI) sees the same rig from a slightly different spot. The operator clicks the two BOTTOM
# corners of the black holder's front face; we map them onto where they are in the organizers' picture (measured on
# videos 1-5: left (259,258), right (371,270), +/- 2 px) with a shift + uniform scale + small rotation, and warp every
# frame. That works when our camera looks at the rig from about the same direction (no strong perspective change).
REF_PTS = np.array([[259.0, 258.0], [371.0, 270.0]])
REF_FRAME = os.path.join(HERE, "reference_frame.jpg")      # organizers' video 5 frame, for the overlay check


def similarity_from_2pts(src, dst):
    """2x3 matrix (scale + rotation + shift) taking the 2 src points onto the 2 dst points."""
    (x1, y1), (x2, y2) = src
    (u1, v1), (u2, v2) = dst
    sv, dv = complex(x2 - x1, y2 - y1), complex(u2 - u1, v2 - v1)
    if abs(sv) < 5:
        raise ValueError("the two points are too close together")
    a = dv / sv                                            # scale * e^(i*angle)
    t = complex(u1, v1) - a * complex(x1, y1)
    return np.array([[a.real, -a.imag, t.real], [a.imag, a.real, t.imag]], dtype=np.float64)


# ----------------------------------------------------------------------------------------------- AI (optional)
class AI:
    def __init__(self, path):
        self.ok, self.msg, self.net = False, "AI off (no model file given)", None
        if not path:
            return
        try:
            import torch
            sys.path.insert(0, os.path.join(ROOT, "lane6_classifier", "pair_model"))
            import pair_model as pm
            self.torch, self.pm = torch, pm
            self.net, self.arch, self.size = pm.load_model(path)
            self.ok, self.msg = True, f"AI on: {os.path.basename(path)} ({self.arch})"
        except Exception as e:                                   # torch missing, bad file, ...
            self.msg = f"AI off: could not load {path} ({type(e).__name__}: {e})"

    def p_off(self, vial_img, vial_ml, ref_img, ref_ml):
        a = self.torch.tensor(self.pm.crop(vial_img, vial_ml, self.size))[None]
        b = self.torch.tensor(self.pm.crop(ref_img, ref_ml, self.size))[None]
        with self.torch.no_grad():
            logit, w = self.net(a, b)
        return float(self.torch.sigmoid(logit)[0]), float(w[0])


# ----------------------------------------------------------------------------------------------- inspection engine
class OperatorInspector(Inspector):
    """Inspector with operator batches: every N vials, or 'New batch' pressed. Keeps one frame per vial for the AI
    and the thumbnails."""

    def __init__(self, batch_size, ai, tol):
        super().__init__(None, tol, 30.0, batch_size)
        self.ai = ai
        self.batch_no = 1
        self.in_batch = 0
        self.force_break = False
        self.thumbs = {}          # vial index -> jpeg bytes
        self.batch_of = {}        # vial index -> batch name
        self.decisions = {}       # batch name -> "RELEASED"/"HELD", vial index -> "PASS"/"REJECT" overrides
        self.batch_recipe = {}    # batch name -> recipe the operator entered (e.g. "R7"), optional
        self.sorter = {}          # vial index -> lane the existing Teledyne sorter used ("good"/"scrap"), baseline
        self.views = {}           # vial index -> full camera frame (jpeg) for the zoom view
        self.plan = []            # run plan, one row per vial (what the line is told to make), in order
        self.plan_offset = 0      # operator correction if a planned vial was skipped / an extra vial was filled

    def planned(self, k):
        """Plan row for detected vial index k (None past the end of the plan)."""
        i = k + self.plan_offset
        return self.plan[i] if 0 <= i < len(self.plan) else None

    def add_vial(self, frames, start_f, leave_f, fill_line=None):
        # name the vial's batch before the parent class files it
        name = f"B{self.batch_no}"
        k = len(self.vials)
        mid = frames[len(frames) // 2]
        ml = float(np.median([auto_ml(img) for img in frames]))
        Ts, L, P = [], [], []
        from common.measure import measure, lab_of_T
        from demo.live_inspect import srgb_lin
        for img in frames:
            m = measure(img, ml); Ts.append(m["T"]); L.append(m["liquid_rgb"]); P.append(m["paper_rgb"])
        T, Lm, Pm = np.median(Ts, 0), np.median(L, 0), np.median(P, 0)
        plan_row = self.planned(k)
        v = dict(n=k + 1, recipe=name, ml=ml, start_frame=start_f, leave_frame=leave_f, T=T, planned=plan_row,
                 OD=-np.log(np.clip(srgb_lin(Lm) / srgb_lin(Pm), 1e-4, None)), lab=lab_of_T(np.clip(T, 0, 1)),
                 frame=mid, t=time.strftime("%H:%M:%S"), fill_line=fill_line,
                 video_s=None)
        self.vials.append(v)
        if not self.batches or self.vials[self.batches[-1][0]]["recipe"] != name:
            self.batches.append([k])
        else:
            self.batches[-1].append(k)
        self.batch_of[k] = name
        th = cv2.resize(mid[200:420, 250:370], (60, 110))
        self.thumbs[k] = cv2.imencode(".jpg", th)[1].tobytes()
        view = mid.copy()                                  # zoom view: where the color and the fill line were read
        from common.measure import band_rows, PAPER, VIAL_X
        r0, r1 = band_rows(ml)
        cv2.rectangle(view, (VIAL_X[0] - 1, r0 - 1), (VIAL_X[1], r1), (0, 210, 255), 1)      # liquid read here
        cv2.rectangle(view, (PAPER[2] - 1, PAPER[0] - 1), (PAPER[3], PAPER[1]), (255, 255, 255), 1)   # paper
        if fill_line is not None:
            cv2.line(view, (288, fill_line), (337, fill_line), (255, 120, 0), 1)             # fill line
        self.views[k] = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
        self.in_batch += 1
        self.messages.append(f"vial {k + 1} measured ({name})")
        if plan_row is not None:                           # plan: the group ends where the next vial's group differs
            nxt = self.planned(k + 1)
            group_done = nxt is None or nxt["group"] != plan_row["group"]
        else:
            group_done = self.in_batch >= self.batch_size  # no plan (or past its end): every N vials
        if group_done or self.force_break:
            self.end_batch()

    def end_batch(self):
        if not self.batches or self.batches[-1][0] in self.results:
            self.force_break = False
            return
        self.finish_batch()
        idx = self.batches[-1]
        ref = self.vials[idx[-1]]
        for k in idx:
            r = self.results[k]
            if self.ai.ok and not r["is_reference"] and r["verdict"] != "CHECK FILL":   # AI reads the same window
                p, w = self.ai.p_off(self.vials[k]["frame"], self.vials[k]["ml"], ref["frame"], ref["ml"])
                r["ai_p_off"], r["ai_leftover"] = round(p, 3), round(w, 2)
        self.batch_no += 1
        self.in_batch = 0
        self.force_break = False

    def final_status(self, k):
        r = self.results.get(k)
        if r is None:
            return "PENDING"
        if r["is_reference"]:
            return "REFERENCE"
        if k in self.decisions:
            return self.decisions[k]                       # operator override
        if r["verdict"] == "CHECK FILL":
            return "REVIEW"                                # color window above the liquid: a person checks it
        rule_off = r["verdict"] == "OFF-COLOR"
        if "ai_p_off" not in r:
            return "REJECT" if rule_off else "PASS"
        ai_off = r["ai_p_off"] >= 0.5
        if rule_off and ai_off:
            return "REJECT"
        if not rule_off and not ai_off:
            return "PASS"
        return "REVIEW"


class Engine:
    def __init__(self, args):
        self.args = args
        self.ai = AI(args.ai_model)
        self.lock = threading.Lock()
        self.thread = None
        self.running = False
        self.jpeg = self._placeholder("Press Start")
        self.state_text = "stopped"
        self.source_text = ""
        self.last_source = args.video or str(args.camera)      # what the page's Source box shows
        self.raw = None                                         # latest camera frame as it came in (for aligning)
        self.proc = None                                        # latest frame as the inspection sees it
        self.align = None                                       # dict(M=2x3, points=[...], size=[w,h], time=...)
        self.use_align = True                                   # False for organizers' videos (already their camera)
        self.raw_path = ""
        self.snapshots = []                                     # manual measurements (known-mix vials etc.)
        self.prev_refs = []                                     # "prev ..." snapshots: recipe before this run
        # video clips for the demo video, separate from Start/Stop: annotated camera view + status bar, ~12 fps
        self.clip_req = None                                    # "start" / "stop" asked by the page
        self.clip_on, self.clip_name, self.clip_t0, self.last_clip = False, "", 0.0, ""
        self.replay_req = False                                 # "save the last 30 s" asked by the page
        self.CLIP_FPS, self.REPLAY_S = 12, 30
        self.run_video = ""                                     # annotated full-run video (camera runs)
        self.plan, self.plan_recipes = self._load_plan(args.plan)   # run plan: one row per vial (+ mL per tank)
        self.camera_settings, self.session_path, self.session = {}, "", {}
        self.copy_status = ""                                   # auto-copy after Stop (--copy-to / --after-stop)
        self.paused = False
        self._first_status = {}                                 # vial -> what the station said before any click
        self.is_file = False
        self.speed = args.speed
        self.clock_reset = False                                # set when speed changes / resume -> re-time playback
        self.video_t, self.video_len = 0.0, 0.0
        self.ins = OperatorInspector(args.batch_size, self.ai, args.tol)
        self.ins.plan = self.plan                               # show the plan's first vial before Start
        self.out_dir = os.path.abspath(args.out)
        os.makedirs(self.out_dir, exist_ok=True)
        self.log_path = os.path.join(self.out_dir, "operator_log.csv")
        self.align_path = os.path.join(self.out_dir, "camera_align.json")
        if os.path.exists(self.align_path):                     # alignment saved earlier in this output folder
            try:
                a = json.load(open(self.align_path))
                if a.get("on"):
                    a["M"] = np.array(a["M"], dtype=np.float64)
                    self.align = a
            except Exception:
                self.align = None
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow(["time", "event", "batch", "vial", "detail"])

    def log(self, event, batch="", vial="", detail=""):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, batch, vial, detail])

    @staticmethod
    def _software_version():
        import subprocess, platform
        try:
            commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                                    text=True, timeout=5).stdout.strip()
        except Exception:
            commit = ""
        return dict(git_commit=commit, python=platform.python_version(), opencv=cv2.__version__)

    @staticmethod
    def _placeholder(text):
        img = np.full((480, 640, 3), 40, np.uint8)
        cv2.putText(img, text, (200, 240), 0, 0.9, (220, 220, 220), 2, cv2.LINE_AA)
        return cv2.imencode(".jpg", img)[1].tobytes()

    # ---- control
    def start(self, source, batch_size, speed):
        if self.running:
            return "already running"
        self.last_source = str(source)
        self.paused, self.speed = False, speed
        self.ins = OperatorInspector(batch_size, self.ai, self.args.tol)
        self._first_status = {}
        for name, od, lab in self.prev_refs:                     # keep the start-of-run reference across Stop/Start
            self.ins.add_previous_recipe(name, od, lab)
        self.ins.plan = self.plan                                # planned recipe per vial (operator can adjust)
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(source, speed), daemon=True)
        self.thread.start()
        self.log("start", detail=f"source={source} batch_size={batch_size} {self.ai.msg}")
        return "started"

    def stop(self):
        self.running = False
        self.paused = False
        return "stopping"

    def pause(self, on):
        if not self.running:
            return "not running"
        if not self.is_file:
            return "pause works for video files only (a live line keeps moving)"
        self.paused = bool(on)
        if not self.paused:
            self.clock_reset = True
        self.log("paused" if self.paused else "resumed", detail=f"t={self.video_t:.1f}s")
        return "paused" if self.paused else "resumed"

    # ---- camera alignment
    def prep(self, frame):
        """Frame as the inspection sees it: aligned to the organizers' camera if alignment is on, else 640x480."""
        if self.align is not None and self.use_align:
            M = self.align["M"]
            w, h = self.align["size"]
            if frame.shape[1] != w or frame.shape[0] != h:      # camera resolution changed: scale the points
                M = M.copy(); M[:, :2] *= w / frame.shape[1]
            return cv2.warpAffine(frame, M, (640, 480), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return frame if frame.shape[:2] == (480, 640) else cv2.resize(frame, (640, 480))

    def set_align(self, points=None, off=False):
        if off:
            self.align = None
            json.dump({"on": False, "time": time.strftime("%Y-%m-%d %H:%M:%S")}, open(self.align_path, "w"))
            self.log("align_off")
            return "alignment off - using the camera picture as it is"
        if self.raw is None:
            return "no camera picture yet - press Start first"
        h, w = self.raw.shape[:2]
        try:
            M = similarity_from_2pts(np.array(points, float), REF_PTS)
        except Exception as e:
            return f"could not align: {e}"
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        angle = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        self.align = dict(on=True, M=M, points=[list(map(float, p)) for p in points], size=[w, h],
                          scale=round(scale, 3), angle_deg=round(angle, 1), time=time.strftime("%Y-%m-%d %H:%M:%S"))
        json.dump({**self.align, "M": M.tolist()}, open(self.align_path, "w"), indent=1)
        self.log("align_on", detail=f"points={points} scale={scale:.3f} angle={angle:.1f}deg camera={w}x{h}")
        warn = ""
        if not 0.4 <= scale <= 2.5:
            warn = " - WARNING: big zoom difference, move the camera closer/farther so the holder looks about the same size"
        elif abs(angle) > 8:
            warn = " - WARNING: camera is tilted, level it if you can"
        return f"aligned (zoom x{scale:.2f}, turn {angle:.1f} deg){warn}"

    def align_check_jpeg(self):
        """Half the aligned live picture, half the organizers' picture: the holder, vial and paper should line up."""
        if self.raw is None:
            return self._placeholder("No picture yet")
        if not self.use_align:
            return self._placeholder("Alignment is for the camera")
        live = self.prep(self.raw)
        ref = cv2.imread(REF_FRAME)
        if ref is None:
            ref = np.zeros_like(live)
        mix = cv2.addWeighted(live, 0.5, ref, 0.5, 0)
        for (x, y) in REF_PTS.astype(int):
            cv2.circle(mix, (x, y), 5, (0, 0, 255), 1)
        cv2.rectangle(mix, (305, 272), (325, 318), (0, 210, 255), 1)
        cv2.rectangle(mix, (420, 275), (440, 305), (255, 255, 255), 1)
        return cv2.imencode(".jpg", mix, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def take_snapshot(self, label):
        """Measure whatever vial is in the holder right now (vials placed by hand, e.g. known mixes)."""
        if self.proc is None:
            return "no picture yet - press Start first"
        from common.measure import measure, lab_of_T
        img = self.proc.copy()
        ml = auto_ml(img)
        m = measure(img, ml)
        lab = lab_of_T(np.clip(m["T"], 0, 1))
        n = len(self.snapshots) + 1
        stamp = time.strftime("%Y%m%d_%H%M%S")
        snap_dir = os.path.join(self.out_dir, "snapshots")
        os.makedirs(snap_dir, exist_ok=True)
        cv2.imwrite(os.path.join(snap_dir, f"snap{n:02d}_{stamp}.png"), img)
        if self.raw is not None:
            cv2.imwrite(os.path.join(snap_dir, f"snap{n:02d}_{stamp}_raw.png"), self.raw)
        is_ref = (label or "").lower().startswith("ref")          # a "ref ..." snapshot IS the new reference
        ref = None if is_ref else next((x for x in reversed(self.snapshots) if x["label"].lower().startswith("ref")), None)
        dE = None if ref is None else round(float(np.linalg.norm(lab - np.array(ref["lab"]))), 1)
        row = dict(n=n, time=time.strftime("%H:%M:%S"), label=label or f"snapshot {n}", band=f"{ml:.0f} mL band",
                   T=[round(float(x), 4) for x in m["T"]], OD=[round(float(x), 4) for x in m["OD"]],
                   lab=[round(float(x), 2) for x in lab], dE_vs_ref=dE, ref=ref["label"] if ref else "",
                   aligned=self.align is not None, video_t=round(self.video_t, 1))
        self.snapshots.append(row)
        if (label or "").lower().startswith("prev"):            # start of run: recipe that ran before this session
            from demo.live_inspect import srgb_lin
            od = -np.log(np.clip(srgb_lin(m["liquid_rgb"]) / srgb_lin(m["paper_rgb"]), 1e-4, None))
            name = label[4:].strip(" :-") or "previous recipe"
            self.prev_refs = [p for p in self.prev_refs if p[0] != name] + [(name, od, lab)]
            with self.lock:
                self.ins.add_previous_recipe(name, od, lab)
            self.log("previous_recipe", detail=name)
        path = os.path.join(self.out_dir, "snapshots.csv")
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["n", "time", "label", "band", "T_r", "T_g", "T_b", "OD_r", "OD_g", "OD_b", "L", "a", "b",
                            "dE_vs_ref", "ref_label", "aligned", "video_t", "image"])
            w.writerow([n, row["time"], row["label"], row["band"], *row["T"], *row["OD"], *row["lab"], dE, row["ref"],
                        row["aligned"], row["video_t"], f"snapshots/snap{n:02d}_{stamp}.png"])
        self.log("snapshot", detail=f"{row['label']} Lab={row['lab']} dE_vs_ref={dE}")
        return f"saved snapshot {n}: {row['label']}" + (f" (dE {dE} vs '{row['ref']}')" if dE is not None else "")

    @staticmethod
    def _load_plan(path):
        """Run plan CSV, ONE ROW PER VIAL in fill order - exactly what is typed into the line:
             vial, clear_ml, yellow_ml, red_ml, blue_ml   (+ optional: color, code, group, purpose, total_ml)
        Vials in a row with the same mL (or the same 'group') form one group; the group's LAST vial is its clean
        reference. Returns (rows in order, {code: row})."""
        if not path:
            return [], {}
        with open(path, newline="", encoding="utf-8-sig") as f:
            rows = [{k.strip(): (v or "").strip() for k, v in r.items() if k}
                    for r in csv.DictReader(f) if (r.get("vial") or "").strip().isdigit()]
        rows.sort(key=lambda r: int(r["vial"]))
        tanks = [k for k in (rows[0] if rows else {}) if k.endswith("_ml") and k != "total_ml"]
        g, last = 0, None
        for r in rows:
            mix = tuple(r.get(t, "") or "0" for t in tanks)
            if not r.get("group"):                       # no group column: a new group whenever the mix changes
                if mix != last:
                    g += 1
                r["group"] = f"G{g}"
            last = mix
            r["code"] = r.get("code") or r.get("color") or "/".join(mix)
            r["tanks"] = {t[:-3]: r.get(t, "") for t in tanks}
        recipes = {}
        for r in rows:
            recipes.setdefault(r["code"], r)
        return rows, recipes

    def recipe_info(self, code):
        """'Light yellow (clear 50, yellow 20 mL)' for a recipe code from the plan, else just the code."""
        r = self.plan_recipes.get(code)
        if not r:
            return code
        ml = ", ".join(f"{t} {v}" for t, v in r["tanks"].items() if v not in ("", "0"))
        return f"{r.get('color') or code}" + (f" ({ml} mL)" if ml else "")

    def _batch_recipe_name(self, batch):
        """Recipe of a station batch (e.g. 'B1' -> 'Dark blue'): operator entry, else the plan of its first vial."""
        ins = self.ins
        if not batch:
            return ""
        if ins.batch_recipe.get(batch):
            return ins.batch_recipe[batch]
        ks = sorted(k for k, b in ins.batch_of.items() if b == batch)
        p = ins.vials[ks[0]].get("planned") if ks else None
        return (p or {}).get("color", "")

    def plan_shift(self, delta):
        """Operator: the line skipped a planned vial (+1) or filled an extra one (-1) - realign the plan."""
        with self.lock:
            self.ins.plan_offset += int(delta)
        self.log("plan_shift", detail=f"offset={self.ins.plan_offset}")
        return f"plan shifted: next vial is planned vial {len(self.ins.vials) + 1 + self.ins.plan_offset}"

    def _write_session(self, final=False):
        """session_<time>.json: everything needed to understand / repeat this run (written at Start and at Stop)."""
        if not self.session_path:
            return
        ins = self.ins
        s = dict(self.session)
        s.update(updated=time.strftime("%Y-%m-%d %H:%M:%S"), finished=final, state=self.state_text,
                 camera_settings=self.camera_settings,
                 alignment=None if self.align is None or not self.use_align else
                 {k: v for k, v in self.align.items() if k != "M"},
                 batch_size=ins.batch_size, plan_file=self.args.plan or "",
                 plan=[{k: v for k, v in r.items() if k != "tanks"} for r in self.plan], plan_offset=ins.plan_offset,
                 recipes_by_batch={b: dict(code=c, info=self.recipe_info(c)) for b, c in ins.batch_recipe.items()},
                 previous_recipe=[p[0] for p in self.prev_refs], vials=len(ins.vials),
                 pulled=sum(ins.final_status(k) == "REJECT" for k in range(len(ins.vials))),
                 review=sum(ins.final_status(k) == "REVIEW" for k in range(len(ins.vials))),
                 raw_video=os.path.basename(self.raw_path), run_video=os.path.basename(self.run_video),
                 snapshots=len(self.snapshots))
        with open(self.session_path, "w") as f:
            json.dump(s, f, indent=1, default=str)

    def request_clip(self, on):
        if not self.running:
            return "press Start first"
        self.clip_req = "start" if on else "stop"
        return "clip recording starting" if on else "clip saved"

    def request_replay(self):
        if not self.running:
            return "press Start first"
        self.replay_req = True
        return f"saving the last {self.REPLAY_S} s"

    def _banner(self):
        """One line for the clip's status bar: what the station is doing + the latest batch result."""
        ins = self.ins
        done = [idx for idx in ins.batches if idx[0] in ins.results]
        if not done:
            return self.state_text, "", (200, 200, 200)
        idx = done[-1]
        b = ins.vials[idx[0]]["recipe"]
        st = {k: ins.final_status(k) for k in idx}
        pull = [ins.vials[k]["n"] for k in idx if st[k] == "REJECT"]
        rev = [ins.vials[k]["n"] for k in idx if st[k] == "REVIEW"]
        if pull:
            k = next(k for k in idx if st[k] == "REJECT")
            r = ins.results[k]
            why = (f" - leftover from {r['cause']}" if r.get("cause") and not str(r["cause"]).startswith("start")
                   else "")
            return self.state_text, f"Batch {b}: PULL vial {', '.join(map(str, pull))} to scrap{why}", (60, 60, 230)
        if rev:
            return self.state_text, f"Batch {b}: vial {', '.join(map(str, rev))} to operator REVIEW", (0, 170, 240)
        return self.state_text, f"Batch {b}: all vials PASS", (80, 200, 80)

    def _clip_frame(self, view):
        state, result, color = self._banner()
        bar = np.full((64, view.shape[1], 3), 24, np.uint8)
        cv2.putText(bar, state[:70], (10, 24), 0, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
        if result:
            cv2.circle(bar, (18, 47), 7, color, -1)
            cv2.putText(bar, result[:70], (32, 53), 0, 0.6, color, 2, cv2.LINE_AA)
        return np.vstack([view, bar])

    def _clips_dir(self):
        d = os.path.join(self.out_dir, "clips")
        os.makedirs(d, exist_ok=True)
        return d

    def set_speed(self, speed):
        self.speed = max(0.25, min(float(speed), 200.0))
        self.clock_reset = True
        return f"speed {self.speed:g}x"

    def new_batch(self):
        with self.lock:
            if self.ins.in_batch == 0:
                return "no vials in the current batch yet"
            self.ins.end_batch()
        self.log("new_batch_pressed", batch=f"B{self.ins.batch_no - 1}")
        return "batch closed"

    def set_recipe(self, batch, recipe):
        with self.lock:
            self.ins.batch_recipe[batch] = recipe
        self.log("recipe_entered", batch, "", recipe)
        return "ok"

    def set_sorter(self, vial, lane):
        with self.lock:
            self.ins.sorter[int(vial) - 1] = lane
        self.log("sorter_lane", self.ins.batch_of.get(int(vial) - 1, ""), vial, lane)
        return "ok"

    def decide(self, batch=None, vial=None, action=None):
        with self.lock:
            if vial is not None:
                k = int(vial) - 1
                before = self.ins.final_status(k) if k not in self.ins.decisions else None
                self.ins.decisions[k] = action
                self.log("vial_decision", self.ins.batch_of.get(k, ""), vial, action)
                self._save_example(k, action, before)
            elif batch:
                self.ins.decisions[batch] = action
                self.log("batch_decision", batch, "", action)
        return "ok"

    def _save_example(self, k, action, station_status):
        """Flywheel: keep the operator's Pass / Reject as a labeled example (vial + reference frame). Used only by
        flywheel/retrain.py between shifts - nothing is trained while the line runs. Never stops the station."""
        if self.args.no_examples:
            return
        try:
            ins = self.ins
            idx = next((b for b in ins.batches if k in b), None)
            if idx is None or k == idx[-1] or k not in ins.results:
                return
            if station_status is None:                     # operator changed an earlier click: keep what the
                station_status = self._first_status.get(k, "")   # station itself said, not the earlier click
            self._first_status.setdefault(k, station_status)
            run_id = os.path.basename(self.session_path)[8:-5] if self.session_path else time.strftime("%Y%m%d_%H%M%S")
            batch = ins.batch_of.get(k, "")
            path = save_example(self.out_dir, run_id, k + 1, idx[-1] + 1, batch, action, station_status,
                                ins.vials[k], ins.vials[idx[-1]], ins.results[k], self._batch_recipe_name(batch),
                                time.strftime("%Y-%m-%d %H:%M:%S"))
            if path:
                self.log("example_saved", batch, k + 1, f"{action} ({station_status})")
        except Exception as e:                             # saving an example must never break the station
            self.log("example_error", "", k + 1, f"{type(e).__name__}: {e}")

    # ---- loop
    def _run(self, source, speed):
        source = str(source).strip()
        hardware_camera = is_hardware_camera(source)
        live_url = is_network_camera(source)
        is_file = not hardware_camera and not live_url           # network camera streams count as live
        self.is_file = is_file
        # alignment is for OUR camera: live, or our own raw recordings of it - never the organizers' videos
        self.use_align = (not is_file) or os.path.basename(str(source)).startswith("raw_camera_")
        if is_file and self.use_align and self.align is None:     # replaying a recording: use the alignment saved
            side = os.path.join(os.path.dirname(os.path.abspath(str(source))), "camera_align.json")   # next to it
            try:
                a = json.load(open(side))
                if a.get("on"):
                    a["M"] = np.array(a["M"], dtype=np.float64)
                    self.align = a
                    self.log("align_loaded", detail=side)
            except Exception:
                pass
        try:
            if hardware_camera:
                cap, camera_index, settings = open_linewatch_camera(source)
                self.camera_settings = settings
                self.source_text = f"camera {source} (/dev/video{camera_index})"
                self.log("camera_ready", detail=f"source={source} resolved=/dev/video{camera_index} settings={settings}")
            else:
                cap = cv2.VideoCapture(source)
                self.source_text = "network camera" if live_url else os.path.basename(source)
        except RuntimeError as exc:
            self.source_text = f"camera {source}" if hardware_camera else source
            self.state_text = str(exc)
            self.jpeg = self._placeholder("No camera signal")
            self.running = False
            self.log("camera_error", detail=str(exc))
            return
        if not cap.isOpened():
            self.state_text = f"could not open {self.source_text}"
            self.jpeg = self._placeholder("No video")
            self.running = False
            cap.release()
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fps = fps if 1 <= fps <= 120 else 30.0
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
        self.video_len = n_frames / fps if is_file and n_frames > 0 else 0.0
        win = int(round(fps))
        import collections
        buf = collections.deque(maxlen=int(3 * fps))
        scores = collections.deque(maxlen=win)
        start, peak, last_leave, fidx = None, 0.0, -10 ** 9, 0
        on, off_min, min_fill_s = 20.0, 8.0, 5.0         # same as demo/live_inspect.py
        det = LevelFillDetector(fps) if self.args.detector == "level" else None
        t_base, f_base, last_jpeg = time.time(), 0, 0.0
        stamp0 = time.strftime('%Y%m%d_%H%M%S')
        self.session_path = os.path.join(self.out_dir, f"session_{stamp0}.json")
        self.session = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), source=str(source), source_text=self.source_text,
                            fps=fps, live=not is_file, ai=self.ai.msg, detector=self.args.detector, tol=self.args.tol,
                            software=self._software_version())
        self._write_session()
        writer = None                                     # raw recording (camera runs): every frame as it came in
        import collections as _c
        clip_writer, replay = None, _c.deque(maxlen=self.CLIP_FPS * self.REPLAY_S)   # replay = jpeg bytes
        if not is_file and not self.args.no_record:
            stamp = time.strftime('%Y%m%d_%H%M%S')
            self.raw_path = os.path.join(self.out_dir, f"raw_camera_{stamp}.mp4")
            self.run_video = os.path.join(self.out_dir, f"run_annotated_{stamp}.mp4")   # Start -> Stop, with status bar
        else:
            self.raw_path, self.run_video = "", ""
        run_writer = None
        self.state_text = "waiting for a vial"

        def vial_done(s0, leave_f, line):
            with self.lock:
                closed_before = len(self.ins.results)
                self.ins.add_vial(steady_frames(buf, leave_f, fps), s0, leave_f, line)
                self.ins.vials[-1]["video_s"] = round(leave_f / fps, 1) if is_file else None
                now_w = time.time()                       # wall clock: match vials to CSI logs / videos
                self.ins.vials[-1]["leave_time"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_w))
                self.ins.vials[-1]["start_time"] = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(now_w - max(0, leave_f - s0) / fps))
                if len(self.ins.results) > closed_before:
                    b = f"B{self.ins.batch_no - 1}"
                    self.log("batch_done", b, "", self._batch_summary(b))

        while self.running:
            if self.paused:                               # hold the picture and the detector exactly where they are
                self.state_text = f"PAUSED at {fidx / fps:.1f} s"
                time.sleep(0.05)
                continue
            if self.clock_reset:                          # resumed or speed changed: re-time playback from here
                t_base, f_base, self.clock_reset = time.time(), fidx, False
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if self.raw_path:
                if writer is None:
                    writer = cv2.VideoWriter(self.raw_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (frame.shape[1], frame.shape[0]))
                    self.log("raw_recording", detail=self.raw_path)
                writer.write(frame)
            self.raw = frame
            frame = self.prep(frame)                      # aligned to the organizers' camera if alignment is on
            self.proc = frame
            buf.append((fidx, frame))
            nv = len(self.ins.vials)
            if det:                                       # default: change from the empty vial (finds clear fills too)
                ev = det.update(frame, fidx)
                if ev and (ev[2] - ev[1]) / fps >= min_fill_s:
                    vial_done(ev[1], ev[2], ev[3])
                self.state_text = (f"FILLING vial {nv + 1}  (batch B{self.ins.batch_no})" if det.state == "filling" else
                                   f"vial {nv + 1} filled - {level_label(det.level)}" if det.state == "full"
                                   else "waiting for a vial")
            else:                                         # --detector color: the older darker/colorful-bottom score
                s = liquid_score(frame) if holder_present(frame) else 0.0
                scores.append((fidx, s))
                if len(scores) == win:
                    f0, s0 = scores[0]
                    if start is None:
                        if f0 - last_leave > 4 * fps and s0 > on and all(v > on * 0.5 for _, v in scores):
                            start, peak = f0, s0
                            self.state_text = f"FILLING vial {nv + 1}  (batch B{self.ins.batch_no})"
                    else:
                        peak = max(peak, s0)
                        leave = max(off_min, 0.35 * peak)
                        if s0 < leave and all(v < leave for _, v in scores):
                            if (f0 - start) / fps >= min_fill_s:
                                vial_done(start, f0, None)
                                last_leave = f0
                            start, peak = None, 0.0
                            self.state_text = "waiting for a vial"
            self.video_t = fidx / fps
            now = time.time()
            if now - last_jpeg > 0.08:                    # ~12 frames/s to the page
                view = frame.copy()
                cv2.rectangle(view, (305, 272), (325, 318), (0, 210, 255), 1)
                cv2.putText(view, f"t={fidx / fps:6.1f}s", (8, 20), 0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                if det and det.state in ("filling", "full") and det.level is not None:   # fill line (volume)
                    cv2.line(view, (288, det.level), (337, det.level), (255, 120, 0), 2)
                self.jpeg = cv2.imencode(".jpg", view, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes()
                last_jpeg = now
                cf = self._clip_frame(view)                   # demo-video clips: camera view + status bar
                if self.run_video:                            # whole run, automatically, for the video edit
                    if run_writer is None:
                        run_writer = cv2.VideoWriter(self.run_video, cv2.VideoWriter_fourcc(*"mp4v"), self.CLIP_FPS,
                                                     (cf.shape[1], cf.shape[0]))
                        self.log("run_video", detail=self.run_video)
                    run_writer.write(cf)
                replay.append(cv2.imencode(".jpg", cf, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes())
                if self.clip_req == "start" and clip_writer is None:
                    self.clip_name = f"clip_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    clip_writer = cv2.VideoWriter(os.path.join(self._clips_dir(), self.clip_name),
                                                  cv2.VideoWriter_fourcc(*"mp4v"), self.CLIP_FPS,
                                                  (cf.shape[1], cf.shape[0]))
                    self.clip_on, self.clip_t0, self.clip_req = True, time.time(), None
                    self.log("clip_start", detail=self.clip_name)
                if clip_writer is not None:
                    clip_writer.write(cf)
                if self.clip_req == "stop" and clip_writer is not None:
                    clip_writer.release(); clip_writer = None
                    self.clip_on, self.clip_req, self.last_clip = False, None, self.clip_name
                    self.log("clip_saved", detail=self.clip_name)
                if self.replay_req:
                    name = f"last{self.REPLAY_S}s_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                    w = cv2.VideoWriter(os.path.join(self._clips_dir(), name), cv2.VideoWriter_fourcc(*"mp4v"),
                                        self.CLIP_FPS, (cf.shape[1], cf.shape[0]))
                    for j in list(replay):
                        w.write(cv2.imdecode(np.frombuffer(j, np.uint8), cv2.IMREAD_COLOR))
                    w.release()
                    self.replay_req, self.last_clip = False, name
                    self.log("replay_saved", detail=f"{name} ({len(replay) / self.CLIP_FPS:.0f} s)")
            fidx += 1
            if is_file:                                   # play recordings at speed x real time
                target = t_base + (fidx - f_base) / (fps * max(self.speed, 0.1))
                if target > time.time():
                    time.sleep(target - time.time())
        if det and det.state in ("filling", "full") and (fidx - 1 - det.start) / fps >= min_fill_s:
            vial_done(det.start, fidx - 1, final_level(det))
        elif not det and start is not None and (fidx - 1 - start) / fps >= min_fill_s:
            vial_done(start, fidx - 1, None)
        with self.lock:
            if self.ins.in_batch:
                self.ins.end_batch()
                b = f"B{self.ins.batch_no - 1}"
                self.log("batch_done", b, "", self._batch_summary(b))
        if writer is not None:
            writer.release()
        if run_writer is not None:
            run_writer.release()
        if clip_writer is not None:                       # run ended while a clip was recording: keep it
            clip_writer.release()
            self.last_clip = self.clip_name
            self.log("clip_saved", detail=self.clip_name)
        self.clip_on, self.clip_req, self.replay_req = False, None, False
        cap.release()
        self.state_text = "finished" if self.running else "stopped"
        self.running = False
        self.paused = False
        self.log("stop", detail=f"{len(self.ins.vials)} vials")
        self._write_results()
        self.session["stopped"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._write_session(final=True)
        if self.args.copy_to or self.args.after_stop:
            threading.Thread(target=self._copy_out, daemon=True).start()

    def _copy_out(self):
        """After Stop: copy everything in the output folder to --copy-to (USB stick / synced folder), then run
        --after-stop (e.g. an upload command). Only new or changed files are copied."""
        import shutil, subprocess
        try:
            if self.args.copy_to:
                dest = os.path.join(os.path.abspath(self.args.copy_to), os.path.basename(self.out_dir))
                self.copy_status = f"copying to {dest} ..."
                n = 0
                for root, _, files in os.walk(self.out_dir):
                    for f in files:
                        src = os.path.join(root, f)
                        dst = os.path.join(dest, os.path.relpath(src, self.out_dir))
                        if (not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src)
                                or os.path.getmtime(dst) < os.path.getmtime(src)):
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(src, dst)
                            n += 1
                import glob
                for pattern in self.args.copy_extra or []:              # e.g. the line program's own log files
                    for src in glob.glob(os.path.expanduser(pattern)):
                        if os.path.isfile(src):
                            dst = os.path.join(dest, "extra", os.path.basename(src))
                            os.makedirs(os.path.dirname(dst), exist_ok=True)
                            shutil.copy2(src, dst)
                            n += 1
                self.copy_status = f"copied {n} new/changed file(s) to {dest}"
                self.log("copied", detail=self.copy_status)
            if self.args.after_stop:
                self.copy_status += ("; " if self.copy_status else "") + "running upload command ..."
                r = subprocess.run(self.args.after_stop, shell=True, capture_output=True, text=True, timeout=3600)
                ok = "upload done" if r.returncode == 0 else f"upload FAILED (code {r.returncode}): {r.stderr.strip()[-200:]}"
                self.copy_status = self.copy_status.replace("running upload command ...", ok)
                self.log("after_stop", detail=f"code={r.returncode} {r.stderr.strip()[-300:]}")
            if self.args.copy_to:                         # refresh the log copy so it includes the lines above
                shutil.copy2(self.log_path, os.path.join(dest, os.path.basename(self.log_path)))
        except Exception as e:                            # never lose the run because a copy failed
            self.copy_status = f"COPY FAILED: {e} - the files are still in {self.out_dir}"
            self.log("copy_error", detail=str(e))

    def _batch_summary(self, b):
        ks = [k for k, n in self.ins.batch_of.items() if n == b]
        st = [self.ins.final_status(k) for k in ks]
        return " ".join(f"v{k + 1}:{s}" for k, s in zip(ks, st))

    def _write_results(self):
        rows = self.snapshot()["vials"]
        if rows:
            with open(os.path.join(self.out_dir, "operator_results.csv"), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
            self._write_run_log(rows)

    def _write_run_log(self, rows):
        """run_log_<time>.csv in the same layout as the team's run-log sheet, filled in automatically.
        Only 'Tinted by eye' and 'Notes' are left for a person."""
        label = {"REJECT": "PULL", "PASS": "PASS", "REVIEW": "REVIEW", "REFERENCE": "REFERENCE"}
        stamp = os.path.basename(self.session_path)[8:-5] if self.session_path else time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.out_dir, f"run_log_{stamp}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Vial #", "Group", "Recipe", "Clear mL", "Yellow mL", "Red mL", "Blue mL", "Total mL", "Purpose",
                        "Time filled", "Tinted by eye? (Y/N)", "Station said", "Notes", "Planned vial", "Batch (station)",
                        "dE vs reference", "AI % off", "Cause", "Leftover %", "Fill", "Operator decision"])
            for r in rows:
                k = int(r["vial"]) - 1
                p = self.ins.vials[k]["planned"] if k < len(self.ins.vials) else None
                t = (p or {}).get("tanks", {})
                said = label.get(r["status"], r["status"])
                if r.get("cause") and r["status"] == "REJECT":
                    said += f" (leftover from {r['cause']}" + (f" - {r['cause_recipe']})" if r.get("cause_recipe") else ")")
                w.writerow([r["vial"], (p or {}).get("group", ""), (p or {}).get("color", "") or r.get("recipe", ""),
                            t.get("clear", ""), t.get("yellow", ""), t.get("red", ""), t.get("blue", ""),
                            (p or {}).get("total_ml", ""), (p or {}).get("purpose", ""),
                            (r.get("start_time") or r.get("time") or "")[-8:], "", said, "",
                            r.get("planned_vial", ""), r["batch"], r.get("dE", ""),
                            "" if r.get("ai_p_off") is None else round(100 * r["ai_p_off"]),
                            r.get("cause") or "", "" if r.get("leftover_pct") is None else r["leftover_pct"],
                            r.get("fill", ""), r.get("operator", "")])

    # ---- state for the page
    def snapshot(self):
        with self.lock:
            ins = self.ins
            vials = []
            for k, v in enumerate(ins.vials):
                r = ins.results.get(k, {})
                vials.append(dict(
                    vial=v["n"], batch=v["recipe"], time=v["t"], video_s=v.get("video_s"),
                    fill=(level_label(v["fill_line"]) if v.get("fill_line") is not None
                          else ("60-80 mL" if v["ml"] >= 60 else "30-40 mL")),
                    fill_pct=level_pct(v.get("fill_line")), measured_as=("60-80 mL" if v["ml"] >= 60 else "30-40 mL"),
                    ref_vial=(ins.vials[next(b for b in ins.batches if k in b)[-1]]["n"] if k in ins.results else None),
                    status=ins.final_status(k), rule=r.get("verdict", "PENDING"), dE=r.get("dE_ref"),
                    ai_p_off=r.get("ai_p_off"), cause=r.get("cause"), cause_sure=r.get("cause_sure"),
                    low_fill=r.get("low_fill"), ref_low_fill=r.get("ref_low_fill"),
                    cause_2nd=r.get("cause_2nd"),
                    leftover_pct=None if r.get("leftover") is None else round(100 * r["leftover"]),
                    operator=ins.decisions.get(k, ""), sorter=ins.sorter.get(k, ""),
                    recipe=(ins.batch_recipe.get(v["recipe"]) or (v["planned"] or {}).get("code", "")),
                    recipe_info=(self.recipe_info(ins.batch_recipe[v["recipe"]]) if ins.batch_recipe.get(v["recipe"])
                                 else self.recipe_info((v["planned"] or {}).get("code", ""))),
                    planned_vial=(v["planned"] or {}).get("vial", ""),
                    start_time=v.get("start_time", ""), leave_time=v.get("leave_time", ""),
                    cause_recipe=self._batch_recipe_name(r.get("cause") or "")))
            batches = []
            for idx in ins.batches:
                b = ins.vials[idx[0]]["recipe"]
                st = [ins.final_status(k) for k in idx]
                done = idx[0] in ins.results
                codes = list(dict.fromkeys((ins.vials[k]["planned"] or {}).get("code", "") for k in idx))
                bcode = ins.batch_recipe.get(b) or " + ".join(c for c in codes if c)
                batches.append(dict(batch=b, recipe=bcode,
                                    recipe_info=(self.recipe_info(bcode) if len(codes) == 1 or ins.batch_recipe.get(b)
                                                 else bcode),
                                    vials=[ins.vials[k]["n"] for k in idx], done=done,
                                    n_reject=st.count("REJECT"), n_review=st.count("REVIEW"),
                                    decision=ins.decisions.get(b, "")))
            judged = [v for v in vials if v["status"] in ("PASS", "REJECT", "REVIEW")]
            tally = dict(
                ours_pull=sum(v["status"] == "REJECT" or v["operator"] == "REJECT" for v in judged),
                ours_review=sum(v["status"] == "REVIEW" for v in judged),
                sorter_logged=sum(1 for v in vials if v["sorter"]),
                sorter_scrap=sum(v["sorter"] == "scrap" for v in vials))
            return dict(tally=tally, state=self.state_text, running=self.running, source=self.source_text, ai=self.ai.msg,
                        last_source=self.last_source, camera_source=str(self.args.camera),
                        paused=self.paused, is_file=self.is_file, speed=self.speed,
                        video_t=round(self.video_t, 1), video_len=round(self.video_len, 1),
                        detector=self.args.detector,
                        align=None if self.align is None else {**{k: v for k, v in self.align.items() if k != "M"},
                                                               "applied": self.use_align},
                        raw_path=os.path.basename(self.raw_path) if self.raw_path else "",
                        clip_on=self.clip_on, clip_name=self.clip_name, copy_status=self.copy_status,
                        copy_to=self.args.copy_to or "",
                        run_video=os.path.basename(self.run_video) if self.run_video and self.running else "",
                        clip_s=round(time.time() - self.clip_t0) if self.clip_on else 0, last_clip=self.last_clip,
                        snapshots=self.snapshots[-12:],
                        ai_on=self.ai.ok, batch_size=ins.batch_size, current_batch=f"B{ins.batch_no}",
                        current_recipe=ins.batch_recipe.get(f"B{ins.batch_no}", ""),
                        plan_recipes=[dict(code=c, info=self.recipe_info(c)) for c in self.plan_recipes],
                        next_planned=(lambda r: None if r is None else dict(vial=r["vial"], info=self.recipe_info(r["code"]),
                                      purpose=r.get("purpose", "")))(ins.planned(len(ins.vials))),
                        plan_len=len(self.plan), plan_offset=ins.plan_offset,
                        plan_file=os.path.basename(self.args.plan) if self.args.plan else "",
                        in_batch=ins.in_batch, vials=vials, batches=batches, log=os.path.basename(self.log_path),
                        messages=list(ins.messages))


# ----------------------------------------------------------------------------------------------- web server
def make_handler(engine):
    page = open(os.path.join(HERE, "operator.html"), encoding="utf-8").read()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            b = body if isinstance(body, bytes) else body.encode()
            self.send_response(code); self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store"); self.send_header("Content-Length", str(len(b)))
            self.end_headers(); self.wfile.write(b)

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/?"):
                return self._send(200, page, "text/html; charset=utf-8")
            if self.path.startswith("/state"):
                return self._send(200, json.dumps(engine.snapshot()))
            if self.path.startswith("/frame.jpg"):
                return self._send(200, engine.jpeg, "image/jpeg")
            if self.path.startswith("/raw.jpg"):            # camera picture as it comes in (for aligning)
                if engine.raw is None:
                    return self._send(200, engine._placeholder("Press Start first"), "image/jpeg")
                return self._send(200, cv2.imencode(".jpg", engine.raw, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes(),
                                  "image/jpeg")
            if self.path.startswith("/ref.jpg"):            # organizers' picture with the 2 target corners marked
                ref = cv2.imread(REF_FRAME)
                for (x, y), t in zip(REF_PTS.astype(int), ("1", "2")):
                    cv2.circle(ref, (x, y), 7, (0, 0, 255), 2)
                    cv2.putText(ref, t, (x - 4, y + 24), 0, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                return self._send(200, cv2.imencode(".jpg", ref)[1].tobytes(), "image/jpeg")
            if self.path.startswith("/aligncheck.jpg"):
                return self._send(200, engine.align_check_jpeg(), "image/jpeg")
            if self.path.startswith("/view/"):              # full camera frame of one vial, for the zoom view
                k = int(self.path.split("/")[2].split(".")[0]) - 1
                return self._send(200, engine.ins.views.get(k, b""), "image/jpeg")
            if self.path.startswith("/thumb/"):
                k = int(self.path.split("/")[2].split(".")[0]) - 1
                return self._send(200, engine.ins.thumbs.get(k, b""), "image/jpeg")
            if self.path.startswith("/stream"):
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        j = engine.jpeg
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(j)).encode() + b"\r\n\r\n" + j + b"\r\n")
                        time.sleep(0.08)
                except (BrokenPipeError, ConnectionResetError):
                    return
            if self.path.startswith("/log.csv"):
                return self._send(200, open(engine.log_path, "rb").read(), "text/csv")
            self._send(404, "{}")

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/start":
                src = d.get("source") or engine.args.video or str(engine.args.camera)
                msg = engine.start(src, int(d.get("batch_size") or engine.args.batch_size), float(d.get("speed") or 1))
            elif self.path == "/stop":
                msg = engine.stop()
            elif self.path == "/pause":
                msg = engine.pause(d.get("on", True))
            elif self.path == "/align":
                msg = engine.set_align(d.get("points"), bool(d.get("off")))
            elif self.path == "/snapshot":
                msg = engine.take_snapshot(str(d.get("label", "")).strip()[:60])
            elif self.path == "/planshift":
                msg = engine.plan_shift(d.get("delta", 0))
            elif self.path == "/clip":
                msg = engine.request_clip(bool(d.get("on")))
            elif self.path == "/replay":
                msg = engine.request_replay()
            elif self.path == "/speed":
                msg = engine.set_speed(d.get("speed") or 1)
            elif self.path == "/newbatch":
                msg = engine.new_batch()
            elif self.path == "/recipe":
                msg = engine.set_recipe(d.get("batch"), d.get("recipe", ""))
            elif self.path == "/sorter":
                msg = engine.set_sorter(d.get("vial"), d.get("lane", ""))
            elif self.path == "/decide":
                msg = engine.decide(d.get("batch"), d.get("vial"), d.get("action"))
            else:
                return self._send(404, "{}")
            self._send(200, json.dumps({"msg": msg}))
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0", help="camera number or stable /dev/v4l/by-id path")
    ap.add_argument("--video", help="default video file for the page (e.g. vial_fill_video_5.mp4)")
    ap.add_argument("--ai-model", help="pair_model.pt from lane6_classifier/pair_model (optional)")
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--speed", type=float, default=8, help="playback speed for video files")
    ap.add_argument("--tol", type=float, default=7.0)
    ap.add_argument("--out", default=os.path.join(ROOT, "..", "operator_output"))
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--autostart", action="store_true", help="start right away (video if given, else camera)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--no-examples", action="store_true",
                    help="don't save Pass / Reject clicks as training examples (flywheel; saved by default)")
    ap.add_argument("--no-record", action="store_true",
                    help="don't save the camera videos (raw + annotated full-run video are saved by default)")
    ap.add_argument("--copy-to", help="after every Stop, copy the output folder here (e.g. a USB stick or synced folder)")
    ap.add_argument("--after-stop", help="shell command to run after every Stop (e.g. an rclone upload)")
    ap.add_argument("--plan", help="run plan CSV: batch,recipe,name,<tank>_ml... - tags each batch with its recipe")
    ap.add_argument("--copy-extra", action="append",
                    help="file or pattern to copy to <copy-to>/extra after every Stop (e.g. the line program's log)")
    ap.add_argument("--detector", choices=["level", "color"], default="level",
                    help="level (default): change from the empty vial, finds clear/pale fills and reads the fill line; "
                         "color: the older darker/colorful-bottom score")
    a = ap.parse_args()
    eng = Engine(a)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(eng))
    url = f"http://localhost:{a.port}"
    print(f"Operator station running: {url}   ({eng.ai.msg})   log -> {eng.log_path}\nPress Ctrl+C here to quit.", flush=True)
    if a.autostart:
        eng.start(a.video or str(a.camera), a.batch_size, a.speed)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        eng.stop()


if __name__ == "__main__":
    main()
