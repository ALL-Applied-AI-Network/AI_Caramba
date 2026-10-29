"""Melting Rainbow - live vial inspection (the demo for the final video).

Watches the vial_fill camera (a webcam / USB camera, or a recorded video), finds each fill as it happens,
measures the liquid color the same way as our validation (common/measure.py), and at the end of every batch
says for each vial: GOOD or OFF-COLOR, which earlier recipe most likely caused it, and the estimated leftover in mL.
It writes an annotated video, a results CSV and a log.

  Recorded video (e.g. organizers' video 5) with the recipe plan:
    python demo/live_inspect.py --video <dataset>/vial_fill/videos/vial_fill_video_5.mp4 ^
        --plan <dataset>/vial_fill/labels/video5_vial_recipes.csv --out demo_out --speed 8
  Live camera (camera 0) with a plan you type up for the run (CSV with columns recipe_id,total_ml; one row per vial,
  in the order they will be filled):
    python demo/live_inspect.py --camera 0 --plan my_run.csv --out demo_out --show
  Live camera with NO recipe list - every 3 vials is one batch (added Oct 26):
    python demo/live_inspect.py --camera 0 --batch-size 3 --out demo_out --show
    Batches are named B1, B2, ...; the cause reads "B3" = leftover from batch 3. Fill level is found from the picture
    (see auto_ml) and the leftover is reported in % of the vial instead of mL.

How each piece works and where its numbers come from:
  vial present ... the black vial holder is in place: two patches of it (rows 220-245, cols 258-280 and 342-362)
                   are dark (< 100). With no holder those patches see white paper (~165). Checked on video 5:
                   holder 39-57, no holder 161-166. Stops gaps between batches from counting as fills.
  fill found ..... (default, Oct 26) lane2_color/detect_fills_level.py: the vial column compared with the same vial
                   while empty - any liquid, even clear, changes it. Also reads the fill line (volume).
                   Found every vial in videos 1-5 (19/3/6/6/30); the old color score missed 3 pale/clear fills in
                   video 3. --detector color = the older lane2_color/detect_fills.py signal (darker / more colorful
                   vial bottom), run online with a 1 s confirmation window.
  measured ....... common/measure.py (liquid in front of the paper / paper), median of 3 frames picked from the
                   last ~2 s before the vial leaves where the picture is steadiest (vial not moving).
  decision ....... lane6_classifier/approach_a.py: compare to the batch's LAST vial (organizers' clean-reference
                   convention), off-color if dE >= 7 (cutoff from videos 1-4 noise). So results for a batch come
                   out when the batch ends (the plan says when the recipe changes), like a batch release check.
  cause .......... earlier recipe whose reference color best explains the shift (same as eval_video5.py), with a
                   confidence: "unsure" (shown as R9?) when the runner-up explains it almost as well
                   (ratio >= 0.6, set on the twin - lane6_classifier/approach_a.py CAUSE_SURE_RATIO).
  leftover mL .... w_hat x vial volume (rough: about +/- 13 points of w on the twin).
  fill level ..... without a plan (auto_ml): liquid in the top of the measuring band (rows 272-285) means a 60-80 mL
                   fill -> upper band; empty glass there means 30-40 mL -> lower band. Empty glass in front of the paper
                   reads 84-97% of the paper, liquid 0-65% (video 5's 30 vials; it only reads fill level, never the
                   good / off decision), so the line is at 75%.
Nothing is trained or tuned on video 5 here.
"""
import argparse, collections, csv, os, sys, time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.measure import measure, lab_of_T, PAPER, VIAL_X, band_rows
from lane2_color.detect_fills import liquid_score
from lane2_color.detect_fills_level import LevelFillDetector, level_label, level_pct, final_level
from lane6_classifier.approach_a import TOL, CAUSE_SURE_RATIO, cause_with_confidence

GREEN, RED, GRAY, WHITE, YELLOW = (60, 200, 60), (40, 40, 230), (150, 150, 150), (255, 255, 255), (0, 210, 255)


def holder_present(img):
    return img[220:245, 258:280].mean() < 100 and img[220:245, 342:362].mean() < 100


def auto_ml(img):
    """No plan: 70 if the top of the upper measuring band is liquid, else 30 (picks the band in common/measure.py)."""
    p = PAPER
    paper = np.median(img[p[0]:p[1], p[2]:p[3]].reshape(-1, 3), axis=0)
    top = np.median(img[272:286, VIAL_X[0]:VIAL_X[1]].reshape(-1, 3), axis=0) / np.maximum(paper, 1)
    return 70.0 if top.min() < 0.75 else 30.0


def srgb_lin(v):
    v = np.asarray(v, float) / 255
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def load_plan(path):
    if not path:
        return None
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [(int(float(r["recipe_id"])), float(r["total_ml"])) for r in rows]


class Inspector:
    def __init__(self, plan, tol=TOL, fps=30.0, batch_size=None):
        self.plan, self.tol, self.fps, self.batch_size = plan, tol, fps, batch_size
        self.vials = []            # measured vials, in order
        self.batches = []          # list of lists of vial indices
        self.refs = {}             # recipe_id -> reference vial (last vial of its batch)
        self.order = []            # recipe order as batches finish
        self.results = {}          # vial index -> result dict
        self.messages = collections.deque(maxlen=4)

    def recipe_of(self, k):
        if self.plan and k < len(self.plan):
            return self.plan[k]
        return (None, 70.0)

    def add_vial(self, frames, start_f, leave_f, fill_line=None):
        k = len(self.vials)
        rid, ml = self.recipe_of(k)
        if self.batch_size:                              # no recipe list: batch number instead, fill level from picture
            rid = f"B{k // self.batch_size + 1}"
            ml = float(np.median([auto_ml(img) for img in frames]))
        Ts, L, P = [], [], []
        for img in frames:
            m = measure(img, ml)
            Ts.append(m["T"]); L.append(m["liquid_rgb"]); P.append(m["paper_rgb"])
        T, Lm, Pm = np.median(Ts, 0), np.median(L, 0), np.median(P, 0)
        v = dict(n=k + 1, recipe=rid, ml=ml, start_frame=start_f, leave_frame=leave_f, T=T,
                 OD=-np.log(np.clip(srgb_lin(Lm) / srgb_lin(Pm), 1e-4, None)), lab=lab_of_T(np.clip(T, 0, 1)),
                 fill_line=fill_line)
        self.vials.append(v)
        new_batch = (not self.batches or self.vials[self.batches[-1][0]]["recipe"] != rid
                     or (self.plan is None and not self.batch_size and self._jump(v)))
        if new_batch:
            if self.batches and self.plan is None:
                self.finish_batch()          # no plan: the previous batch ends here
            self.batches.append([k])
        else:
            self.batches[-1].append(k)
        self.messages.append(f"vial {k + 1} measured" + (f" ({self._name(rid)}, {ml:.0f} mL)" if rid is not None else ""))
        if self.batch_size:
            if (k + 1) % self.batch_size == 0:     # every Nth vial ends a batch -> release it
                self.finish_batch()
        elif self.plan is not None:
            nxt = self.recipe_of(k + 1)[0] if k + 1 < len(self.plan) else "end"
            if nxt != rid:                  # plan says this was the batch's last vial -> release the batch
                self.finish_batch()

    @staticmethod
    def color_unreadable(v):
        """Fill line below the window where the color is read -> that window is empty glass, the color means nothing."""
        return v.get("fill_line") is not None and v["fill_line"] > band_rows(v["ml"])[1]

    def _jump(self, v):                    # no plan: new batch when the color jumps a lot
        last = self.vials[-2] if len(self.vials) > 1 else None
        return last is not None and np.linalg.norm(v["lab"] - last["lab"]) > 25

    def add_previous_recipe(self, name, OD, lab):
        """Start of a run: a clean vial of the recipe that ran BEFORE this session (measured by hand) becomes the
        earliest 'earlier batch', so the first batch can be checked for carryover and named like any other."""
        key = f"{name} (prev)"
        self.refs[key] = dict(OD=np.asarray(OD, float), lab=np.asarray(lab, float))
        if key in self.order:
            self.order.remove(key)
        self.order.insert(0, key)                         # oldest: batches measured in this run stay more recent
        self.messages.append(f"previous recipe registered: {name}")

    def finish_batch(self):
        if not self.batches or self.batches[-1][0] in self.results:
            return
        idx = self.batches[-1]
        rid = self.vials[idx[0]]["recipe"]
        key = rid if rid is not None else f"batch{len(self.order) + 1}"   # recipe id, or "B3" with --batch-size
        ref = self.vials[idx[-1]]
        prev_keys = list(self.order)
        self.refs[key] = ref; self.order.append(key)
        n_off = 0
        for k in idx:
            v = self.vials[k]
            dE = float(np.linalg.norm(v["lab"] - ref["lab"]))
            low = self.color_unreadable(v) or self.color_unreadable(ref)
            res = dict(is_reference=k == idx[-1], dE_ref=round(dE, 1),
                       verdict="REFERENCE" if k == idx[-1] else ("CHECK FILL" if low else
                                                                 "OFF-COLOR" if dE >= self.tol else "GOOD"),
                       leftover=None, residual_ml=None, cause=None,
                       low_fill=self.color_unreadable(v), ref_low_fill=self.color_unreadable(ref))
            if prev_keys and not res["is_reference"] and not low:
                d = v["OD"] - ref["OD"]
                u = self.refs[prev_keys[-1]]["OD"] - ref["OD"]
                w = float(np.clip(d @ u / (u @ u), 0, 1)) if u @ u > 1e-9 else 0.0
                res["leftover"] = round(w, 2)
                res["residual_ml"] = None if self.batch_size else round(w * v["ml"], 1)   # volume unknown w/o plan
                if res["verdict"] == "OFF-COLOR":
                    best, ratio, second, _ = cause_with_confidence(v["OD"], ref["OD"],
                                                                    {c: self.refs[c]["OD"] for c in prev_keys})
                    name = self._name
                    res["cause"] = name(best)
                    res["cause_ratio"] = round(ratio, 2)
                    res["cause_sure"] = ratio < CAUSE_SURE_RATIO
                    res["cause_2nd"] = name(second) if second is not None else None
            elif res["verdict"] == "OFF-COLOR":
                res["cause"] = "start of run (no earlier batch to compare)"
            n_off += res["verdict"] == "OFF-COLOR"
            self.results[k] = res
        self.messages.append(f"batch {self._name(key)} released: {len(idx)} vials, {n_off} off-color")

    @staticmethod
    def _name(c):
        return f"R{c}" if isinstance(c, (int, np.integer)) else str(c)

    def rows(self):
        out = []
        for k, v in enumerate(self.vials):
            r = self.results.get(k, {})
            out.append(dict(vial=v["n"], recipe=v["recipe"], total_ml=v["ml"], start_frame=v["start_frame"],
                            leave_frame=v["leave_frame"], verdict=r.get("verdict", "PENDING"),
                            dE_ref=r.get("dE_ref"), cause=r.get("cause"),
                            cause_confidence=("sure" if r.get("cause_sure") else "unsure") if r.get("cause_ratio") is not None else None,
                            cause_ratio=r.get("cause_ratio"), cause_2nd=r.get("cause_2nd"), leftover_est=r.get("leftover"),
                            residual_ml_est=r.get("residual_ml"),
                            fill_line_row=v.get("fill_line"), fill_level_pct=level_pct(v.get("fill_line")),
                            T_r=round(float(v["T"][0]), 3), T_g=round(float(v["T"][1]), 3), T_b=round(float(v["T"][2]), 3)))
        return out


def draw(frame, ins, state, fidx, fps):
    h, w = frame.shape[:2]
    pw = 360
    canvas = np.full((h, w + pw, 3), 28, np.uint8)
    canvas[:, :w] = frame
    cv2.rectangle(canvas, (0, 0), (w, 48), (0, 0, 0), -1)
    cv2.putText(canvas, f"Melting Rainbow - vial inspection   t = {fidx / fps:6.1f} s", (8, 19), 0, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(canvas, state, (8, 40), 0, 0.5, YELLOW, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (305, 272), (325, 318), YELLOW, 1)          # where the color is read
    x0 = w + 10
    cols = [0, 34, 82, 180, 228, 290]                                   # vial, recipe, result, dE, cause, mL
    for c, t in zip(cols, ["vial", "recipe" if not ins.batch_size else "batch", "result", "dE", "cause",
                           "left mL" if not ins.batch_size else "left %"]):
        cv2.putText(canvas, t, (x0 + 14 + c, 20), 0, 0.42, WHITE, 1, cv2.LINE_AA)
    for i, r in enumerate(ins.rows()[-16:]):
        y = 44 + i * 22
        c = {"GOOD": GREEN, "OFF-COLOR": RED, "REFERENCE": GRAY}.get(r["verdict"], YELLOW)
        cv2.circle(canvas, (x0 + 4, y - 4), 6, c, -1)
        off = r["verdict"] == "OFF-COLOR"
        cause = (r["cause"] or "") if off else ""
        if off and r.get("cause_confidence") == "unsure":
            cause += "?"                                                # two recipes explain it about equally
        vals = [str(r["vial"]), Inspector._name(r["recipe"]) if r["recipe"] is not None else "?",
                {"REFERENCE": "reference"}.get(r["verdict"], r["verdict"]),
                "" if r["dE_ref"] is None or r["verdict"] == "REFERENCE" else f"{r['dE_ref']:.1f}",
                cause if len(cause) <= 6 else "?",
                (f"{r['residual_ml_est']:.0f}" if r["residual_ml_est"] is not None
                 else f"{100 * r['leftover_est']:.0f}%" if r["leftover_est"] is not None else "") if off else ""]
        for cx, t in zip(cols, vals):
            cv2.putText(canvas, t, (x0 + 14 + cx, y), 0, 0.42, c if cx == 82 else WHITE, 1, cv2.LINE_AA)
    for i, m in enumerate(list(ins.messages)[::-1]):
        cv2.putText(canvas, m, (x0, h - 12 - 18 * i), 0, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
    return canvas


def steady_frames(buf, leave_f, fps):
    """From the buffered frames before the vial left, pick 3 (about 0.2 s apart) where the vial area changes least."""
    cand = [(f, img) for f, img in buf if leave_f - 2.0 * fps <= f <= leave_f - 0.3 * fps]
    if len(cand) < 3:
        cand = list(buf)[-3:]
        return [img for _, img in cand]
    best, best_s = None, 1e9
    step = max(1, int(0.2 * fps))
    for i in range(0, len(cand) - 2 * step):
        trio = [cand[i][1], cand[i + step][1], cand[i + 2 * step][1]]
        roi = [t[250:330, 295:335].astype(np.int16) for t in trio]
        s = np.abs(roi[0] - roi[1]).mean() + np.abs(roi[1] - roi[2]).mean()
        if s < best_s:
            best, best_s = trio, s
    return best


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--video"); src.add_argument("--camera", type=int)
    ap.add_argument("--plan", help="CSV with recipe_id,total_ml per vial in run order (organizers' recipe CSV works)")
    ap.add_argument("--batch-size", type=int, help="no recipe list: every N vials is one batch (its last vial = reference)")
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--speed", type=int, default=1, help="write every Nth frame to the output video (8 = 8x faster)")
    ap.add_argument("--show", action="store_true", help="show a live window (press q to stop)")
    ap.add_argument("--no-video", action="store_true", help="skip writing the annotated video")
    ap.add_argument("--detector", choices=["level", "color"], default="level",
                    help="level (default, Oct 26): change from the empty vial - also finds clear / pale fills. "
                         "color: the older darker/colorful-bottom score")
    a = ap.parse_args()
    if not a.plan and not a.batch_size:
        sys.exit("give --plan (recipe list) or --batch-size N. Guessing batches from color alone was tested on video 5 "
                 "and was unreliable")

    os.makedirs(a.out, exist_ok=True)
    cap = cv2.VideoCapture(a.video if a.video else a.camera)
    if not cap.isOpened():
        sys.exit("could not open the video / camera")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps < 1 or fps > 120:
        fps = 30.0
    plan = None if a.batch_size else load_plan(a.plan)
    ins = Inspector(plan, a.tol, fps, a.batch_size)
    log = open(os.path.join(a.out, "inspection_log.txt"), "w", buffering=1)
    say = lambda m: (print(m, flush=True), log.write(m + "\n"))
    say(f"started {time.ctime()}  source={a.video or f'camera {a.camera}'}  "
        + (f"batch size={a.batch_size} (no recipe list)" if a.batch_size else f"plan={a.plan}") + f"  cutoff dE={a.tol}")

    win = int(round(fps))                                  # 1 s confirmation window (detect_fills.py)
    on, off_min = 20.0, 8.0
    min_fill_s = 5.0      # detect_fills.py uses 8 s; light recipes (R14) only score for the last ~7 s, and the
                          # holder check now stops between-batch gaps from looking like fills
    det = LevelFillDetector(fps) if a.detector == "level" else None
    last_frame = None
    buf = collections.deque(maxlen=int(3 * fps))           # last 3 s of frames
    scores = collections.deque(maxlen=win)                 # (frame, score) waiting for confirmation
    start, peak, last_leave = None, 0.0, -10 ** 9
    writer, fidx, t0 = None, 0, time.time()
    state = "waiting for a vial"

    def vial_done(s0, leave_f, line):
        ins.add_vial(steady_frames(buf, leave_f, fps), s0, leave_f, line)
        v = ins.vials[-1]
        say(f"vial {v['n']}: {Inspector._name(v['recipe'])} frames {s0}-{leave_f}  T={np.round(v['T'], 3)}"
            + (f"  {level_label(line)}" if det else ""))

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        last_frame = frame
        frame = cv2.resize(frame, (640, 480)) if frame.shape[:2] != (480, 640) else frame
        buf.append((fidx, frame))
        nv = len(ins.vials)
        filling_text = f"FILLING vial {nv + 1}" + (f"  (batch B{nv // a.batch_size + 1})" if a.batch_size
                                                   else f"  (R{ins.recipe_of(nv)[0]})" if plan else "")
        if det:                                            # fill level: change from the empty vial
            ev = det.update(frame, fidx)
            if ev and (ev[2] - ev[1]) / fps >= min_fill_s:
                vial_done(ev[1], ev[2], ev[3])
            state = (filling_text if det.state == "filling" else
                     f"vial {nv + 1} filled ({level_label(det.level)}) - waiting for it to leave" if det.state == "full"
                     else "waiting for a vial")
        else:                                              # older color score
            s = liquid_score(frame) if holder_present(frame) else 0.0
            scores.append((fidx, s))
            if len(scores) == win:
                f0, s0 = scores[0]
                if start is None:
                    if f0 - last_leave > 4 * fps and s0 > on and all(v > on * 0.5 for _, v in scores):
                        start, peak = f0, s0
                        state = filling_text
                else:
                    peak = max(peak, s0)
                    leave = max(off_min, 0.35 * peak)
                    if s0 < leave and all(v < leave for _, v in scores):
                        if (f0 - start) / fps >= min_fill_s:
                            vial_done(start, f0, None)
                            last_leave = f0
                        start, peak, state = None, 0.0, "waiting for a vial"
        if not a.no_video or a.show:
            canvas = draw(frame, ins, state, fidx, fps)
            if not a.no_video and fidx % a.speed == 0:
                if writer is None:
                    writer = cv2.VideoWriter(os.path.join(a.out, "inspection.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (canvas.shape[1], canvas.shape[0]))
                writer.write(canvas)
            if a.show:
                cv2.imshow("Melting Rainbow", canvas)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        fidx += 1
    if det and det.state in ("filling", "full") and (fidx - 1 - det.start) / fps >= min_fill_s:   # video ended full
        vial_done(det.start, fidx - 1, final_level(det))
    elif not det and start is not None and (fidx - 1 - start) / fps >= min_fill_s:
        vial_done(start, fidx - 1, None)
    ins.finish_batch()
    if writer is not None:
        for _ in range(int(fps * 3)):                            # hold the final table for 3 s
            writer.write(draw(cv2.resize(last_frame, (640, 480)), ins, "run finished", fidx, fps))
        writer.release()
    rows = ins.rows()
    with open(os.path.join(a.out, "inspection_results.csv"), "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["vial"]); wcsv.writeheader(); wcsv.writerows(rows)
    judged = [r for r in rows if r["verdict"] in ("GOOD", "OFF-COLOR")]
    say(f"finished {time.ctime()}  ({time.time() - t0:.0f} s)  vials {len(rows)}  judged {len(judged)}  "
        f"off-color {sum(r['verdict'] == 'OFF-COLOR' for r in judged)}")
    for r in rows:
        say(f"  vial {r['vial']:>2} {Inspector._name(r['recipe'])}: {r['verdict']:<9} dE {r['dE_ref']}  cause {r['cause']}"
            + (f" ({r['cause_confidence']}, ratio {r['cause_ratio']}, 2nd {r['cause_2nd']})" if r.get("cause_confidence") else "")
            + (f"  leftover {r['residual_ml_est']} mL" if not a.batch_size
               else f"  leftover {r['leftover_est']}" + (f" ({r['total_ml']:.0f}-mL band)" if r['total_ml'] else "")))
    if plan and len(rows) != len(plan):
        say(f"WARNING: found {len(rows)} fills but the plan lists {len(plan)} vials - check the recipe assignment")
    log.close()
    if a.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
