# Melting Rainbow — runbook for the CSI computer

How to set up and run our off-color / carryover check (the **operator station**) on a CSI computer, from a fresh copy of this repo.
Typed commands are in `grey boxes`. Everything else is explanation.

| | |
|---|---|
| What it does | Watches the vial-fill camera, finds each fill, compares every vial with its batch's clean vial, shows **PASS / PULL / REVIEW** on a web page, saves everything |
| Needs | Python 3.10+, internet once (for `git clone` + `pip`), a camera that sees the vial holder with the white paper behind it |
| Time to set up | ~10 min (most of it is `pip install`) |
| Where results go | `results/operator_output/` inside the repo folder |

---

## 1. Get the code (once)

Open a terminal (**Windows:** Start → type `cmd` → Command Prompt. **Linux:** Terminal).

`git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-cruz-rios-s-team-d24d4d83ca.git melting-rainbow-repo`

`cd melting-rainbow-repo`

No git on the computer? On the GitHub page: green **Code** button → **Download ZIP** → unzip → open a terminal in that folder.

## 2. Install Python packages (once)

Check Python: `python --version` (Linux: `python3 --version`). Needs 3.10 or newer.

**Windows**

`python -m venv venv`

`venv\Scripts\activate`

`pip install opencv-python numpy`

`pip install torch`

**Linux**

`python3 -m venv venv`

`source venv/bin/activate`

`pip install opencv-python numpy`

`pip install torch --index-url https://download.pytorch.org/whl/cpu`

- The last line (torch) is only for the AI second opinion (~200 MB on Windows). If it fails or takes too long, **skip it** — the station still works on the rule alone and the page says "AI off: … No module named 'torch'" (tested).
- Linux: if the `--index-url` line fails, `pip install torch` also works but downloads much more (GPU version, ~2 GB).
- If `pip install opencv-python` fails on a Linux machine without a screen, use `pip install opencv-python-headless`.
- Next time you open a terminal, only re-run the `activate` line (from the repo folder).

## 3. Start the station

From the repo folder, with the venv active:

**Windows**

`python melting-rainbow\operator_ui\operator_app.py --camera 0 --batch-size 3 --ai-model melting-rainbow\operator_ui\pair_model.pt --out results\operator_output`

**Linux**

On the configured CSI computer, use the launcher. It selects the stable Vial Fill camera, applies the tested 640×480 MJPG / automatic-exposure / locked-white-balance setup, and starts capture in this UI:

`./start_capture.sh`

Direct equivalent:

`python3 melting-rainbow/operator_ui/operator_app.py --camera /dev/v4l/by-id/usb-4K_USB_CAMERA_HD_USB_CAMERA_01.00.00-video-index0 --batch-size 3 --ai-model melting-rainbow/operator_ui/pair_model.pt --out results/operator_output --autostart`

- The terminal prints `Operator station running: http://localhost:8765`. **Leave this terminal open** — closing it stops the station.
- The browser opens the page. If not, open a browser and go to `localhost:8765`.
- Loading takes 10–20 s. The pill at the top says **AI on** (or AI off without torch — fine).
- Windows shortcut instead of typing: double-click `windows\run_operator_station.bat` (camera 0) or `windows\run_operator_station_usbcam.bat` (camera 1). If Windows blocks the .bat (Smart App Control), use the typed command above.

## 4. Pick the camera

Top bar: **Source = Camera**, and the box next to it is the camera number or stable Linux device path.

| Box | Usually |
|---|---|
| `0` | the first camera on the computer (a laptop's own camera, or the only USB camera) |
| `1` | a second camera (e.g. a USB webcam on a laptop) |
| `2` | a third camera |
| `/dev/v4l/by-id/...` | a stable Linux camera path that survives camera renumbering (used by `start_capture.sh`) |
| `rtsp://...` | a network camera — paste its stream address (ask CSI staff) |

`start_capture.sh` starts the live feed automatically. After a manual Stop, press **Start** to begin again. Under the picture it must say `recording raw video → raw_camera_….mp4` — that file is the data.
Wrong picture → **Stop**, change the box, **Start**. Black picture / "could not open" → close other apps using the camera (Camera, Teams, Zoom), try the next number.

## 5. Line the camera up (Align)

Our measuring spots are fixed places in the organizers' fill-camera picture.

- **Using CSI's own fill camera (same view as the organizers' videos)?** Probably no alignment needed. Do the check in step 6; align only if the yellow box is off.
- **Using another camera (laptop / webcam)?** Point it at the **black vial holder, the vial, and the white paper behind it**, from about the same angle as the organizers' camera, 30–50 cm away, and tape it down. Then:
  1. Press **Align camera**.
  2. On the left picture (your camera) click the **bottom-left corner** of the holder's black front face, then the **bottom-right corner** (the right picture marks them as 1 and 2).
  3. **Save alignment** and look at the overlay: holder, vial and paper should line up. WARNING → move/level the camera, **New picture**, click again.
  4. Won't line up after two tries? Skip it — the raw video lets us align and re-run afterwards.
- The alignment is saved (`results/operator_output/camera_align.json`) and reused on the next start. It is only used for the live camera and our own `raw_camera_*.mp4` files — never for the organizers' videos.

## 6. Check the first vial

Fill one vial. When it appears in the list on the right, click its picture to zoom in. **The yellow box must sit on the liquid in front of the white paper.** If it sits on glass, metal or the black holder, redo step 5.

## 7. Run batches

**Default — the station is never told the recipe:** `./start_capture.sh`

- Every **3 vials is one batch** (the "Vials / batch" box); the batch's **last vial is its clean reference**. That is all the station knows — not what the recipe is or what color it should be.
- Leave the recipe dropdown empty.
- A batch was shorter than 3 → press **New batch now** right after its last vial.
- Keep **your own record** of what was filled (recipe and mL per vial), e.g. [`runs/2026-10-30_plan.csv`](runs/2026-10-30_plan.csv) (30 vials, 10 recipes, 10 batches of 3, ~28 min). It is used only **afterwards**, to score the run — it is not given to the station.

**Optional — with a run plan** (only where giving the station the recipe list is allowed): `./start_capture.sh --plan <plan.csv>`, one row per vial in fill order (`vial, clear_ml, yellow_ml, red_ml, blue_ml, …, group`). The page then shows "Next on the line: vial N"; vials with the same `group` are one batch, so batches can be any size; **▶ / ◀** fix a skipped / extra vial.

**During the run:**

- When a group finishes, each vial shows **PASS → good**, **PULL → scrap** (with likely cause and leftover %), or **REVIEW** (look, then click Pass / Reject). Then **Release batch** or **Hold batch**.
- If the line's Teledyne color check shows a pass/fail, click under each vial what it said (**good / scrap**) — that is the baseline. If there is no Teledyne decision, skip those buttons.
- On paper, per vial: number, recipe, time, tinted by eye (Y/N), anything odd.
- Don't change settings mid-run.

## 8. Known-mix vials (optional)

**Easiest: have the line fill them** (vials 28–29 of [`runs/2026-10-30_plan.csv`](runs/2026-10-30_plan.csv)): light yellow with exactly 10 % and 20 % dark blue, right after three light yellows (25–27) so the lines hold the same base color. Afterwards they are scored against vial 27 (a clean light yellow) — nothing to do by hand during the run. **By hand instead:**

For checking the "% leftover" estimate against the truth. Fill to ≥ 60 mL, stir well, place in the holder, wait 5 s, type a label, press **Measure vial in holder**:

1. `prev clean R12` (a clean R12 vial)
2. `ref clean R14` (a clean R14 vial — label must start with **ref**)
3. `R14 65 + R12 5`, `R14 60 + R12 10`, `R14 50 + R12 20` (write the exact mL actually used)

Saved to `results/operator_output/snapshots.csv` + `snapshots/`.

## 8b. Recording for the video (clips + screen recording)

**Automatic, every camera run (Start → Stop):** two videos in `results/operator_output/`:

| File | What | Size (30-min run) |
|---|---|---|
| `raw_camera_<time>.mp4` | every camera frame, no drawings — the data, can be re-run later | ~200 MB |
| `run_annotated_<time>.mp4` | the camera view + measuring box + fill line + status bar with each batch result, ~12 frames/s — for editing the video | ~110 MB |

Sizes measured on the organizers' 640×480 video (raw ≈ 6.7 MB/min, annotated ≈ 3.7 MB/min). A 30-vial / 10-recipe run took **28 min** in the organizers' video 5 (≈ 56 s per vial). Have ~1 GB free.


**Clips from the station (works on any computer, nothing to install)** — buttons under the camera picture, separate from Start / Stop:

| Button | What it saves |
|---|---|
| **● Record clip** → **■ Stop clip** | a video of the camera view with the yellow box, fill line, status and the latest batch result (e.g. "Batch B2: PULL vial 4, 5 to scrap – leftover from B1") |
| **Save last 30 s** | the 30 seconds that just happened — press it right after a PULL appears, no need to have been recording |

Saved to `results/operator_output/clips/` (`clip_<time>.mp4`, `last30s_<time>.mp4`, about 12 frames/s). The run keeps going; a clip still recording when you press Stop is saved.

**Screen recording of the page — built in, saved with everything else:**

1. Leave **"Record screen with Start"** ticked (top bar, on by default).
2. `start_capture.sh` starts the camera right away — use that to check the picture, then press **Stop**.
3. When the line is ready, press **Start**. The browser asks what to share: pick **this tab** (Chrome) or the **browser window / entire screen** (Firefox / Ubuntu dialog) → **Share**.
4. Check the line under the picture: **"● screen recording → screen_….webm · N MB saved"** — N must keep going up.
5. **Stop** first finishes the screen recording ("screen recording saved: …"), then stops the run, then copies everything to the USB stick (step 8c).

- Saved as `results/operator_output/screen_<time>.webm`, written to disk every 2 seconds — if the browser or computer crashes, everything up to then is kept. About 5 MB per minute (≤ 340 MB for 30 min).
- **● Record screen** (next to the clip buttons) starts / stops it by hand at any time.
- Said "not started" or shows a red **SCREEN RECORDING PROBLEM**? The run itself is not affected — use the Ubuntu recorder below as a backup (and keep the phone filming).
- Backup, Ubuntu's own recorder: **Print Screen** → video camera icon → **Record**; red button top right stops it; saved to `~/Videos/Screencasts/` (copy it to the stick by hand). Windows: **Win + Alt + R**.

## 8c. Copy the results to a USB stick automatically

1. **Plug a USB stick into the CSI computer before starting.**
2. Start as usual: `./start_capture.sh`. The terminal prints
   `Results will be copied to the USB stick after every Stop: /media/<user>/<STICK>/melting_rainbow`
   and the page shows "copies to USB after Stop".
3. After **every Stop**, the station copies `results/operator_output` (videos, CSVs, clips, snapshots — only new or changed files) to `<STICK>/melting_rainbow/operator_output/`. The page shows "copied N file(s) …".
4. Wait for that message before unplugging the stick (a 30-min run is ~300 MB; copying takes under a minute on most sticks).

- No stick found → the terminal says so and everything stays in `results/operator_output` (plug the stick in and restart, or copy by hand).
- Choose a different place: `COPY_TO=/path/to/folder ./start_capture.sh`. Turn it off: `COPY_TO=none ./start_capture.sh`.
- Windows: add `--copy-to E:\melting_rainbow` to the start command (E: = the stick).
- If a copy fails, the page says so — nothing is lost, the files are still in `results/operator_output`.

## 8d. What gets recorded (metadata)

| File | Contents |
|---|---|
| `operator_results.csv` | per vial: batch, recipe + mL (from the plan / dropdown), **start and end time (date + clock)**, status, rule ΔE, AI %, cause, leftover %, fill level, operator decision |
| `run_log_<time>.csv` | **our run-log sheet, filled in automatically**: vial, group, recipe, mL, purpose, time, what the station said (+ ΔE, AI %, cause, leftover, fill). Only "Tinted by eye" and "Notes" are left to fill by hand — open it from the USB stick, or paste it into the Drive run-log sheet |
| `operator_log.csv` | every event and click with time (start, recipe changes, batch results, Pass/Reject, Release/Hold, clips, copies) |
| `session_<time>.json` | the whole run: start / stop time, camera settings (resolution, fps, exposure, white balance), alignment, plan, recipes per batch, "prev" recipe, counts, video file names, software version (git commit) |
| `raw_camera_*.mp4`, `run_annotated_*.mp4`, `screen_*.webm`, `clips/`, `snapshots/` | videos and pictures (step 8b) |

## 9. Finish — don't skip

1. Press **Stop**, wait until the page says **stopped** (this closes the raw video file properly). Only then close the terminal.
2. Copy the whole `results/operator_output` folder off the computer (USB stick, or upload to our Drive folder). It contains:
   - `raw_camera_*.mp4` — every camera frame (≈ 9 MB/min at 1280×720)
   - `operator_results.csv` — every vial: status, rule, ΔE, AI %, cause, leftover, fill level
   - `operator_log.csv` — every click and event with time
   - `camera_align.json`, `snapshots.csv`, `snapshots/` (if used)
   - `clips/` — the clips from **Record clip** / **Save last 30 s**
   - `run_annotated_*.mp4` — the whole run with the status bar (for the video)
   - Already on the USB stick if one was plugged in at start (step 8c) — check the page says "copied …".
3. Photo of the paper notes.
4. Don't leave our results on the CSI computer if they ask you not to: delete `results/` after copying (ask staff first).

## If something goes wrong

| Problem | Fix |
|---|---|
| `python` not found | Windows: try `py` instead of `python`. Linux: `python3`. |
| `No module named cv2` / `numpy` | venv not active → run the `activate` line from step 2, or re-run the `pip install` lines |
| Page doesn't open | Browser → `localhost:8765`. Port busy → add `--port 8766` to the start command and open `localhost:8766` |
| "AI off" | torch not installed — fine, the rule still decides. Install torch later if wanted |
| Black picture / wrong camera | Step 4 |
| Vials not being detected | Check step 6 (camera view). The detector needs the vial column below the holder in view and a still camera |
| Station window closed by accident | Start it again (step 3). Alignment is remembered; a new raw video starts |
| Camera can't be set up at all | Film the vial + white paper with a phone on a stand for the whole run (landscape, don't touch it) and keep the paper log — the system can be run on that video afterwards |

## Run on a recorded video (practice / afterwards)

**Windows** `python melting-rainbow\operator_ui\operator_app.py --video "C:\path\to\video.mp4" --autostart --speed 8 --batch-size 3 --ai-model melting-rainbow\operator_ui\pair_model.pt --out results\replay`

**Linux** `python3 melting-rainbow/operator_ui/operator_app.py --video /path/to/video.mp4 --autostart --speed 8 --batch-size 3 --ai-model melting-rainbow/operator_ui/pair_model.pt --out results/replay`

Pause / Resume and the speed box work for videos. A `raw_camera_*.mp4` replays with the alignment saved next to it.

---

**What's in this repo:** `melting-rainbow/` = our station and analysis code (see `melting-rainbow/operator_ui/README.txt` for how each part works and where every number comes from). `windows/` = double-click launchers. `frank_recipe_classifier/` = Frank's EfficientNet recipe-classifier pipeline. `docs/` = results and how it works. `archive/` = early experiments.
