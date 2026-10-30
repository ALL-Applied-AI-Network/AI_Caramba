# Melting Rainbow — runbook for the CSI computer

How to set up and run our off-color / carryover check (the **operator station**) on a CSI computer, from a fresh copy of this repo.
Typed commands are in `grey boxes`. Everything else is explanation.

| | |
|---|---|
| What it does | Watches the vial-fill camera, finds each fill, compares every vial with its batch's clean vial, shows **PASS / PULL / REVIEW** on a web page, saves everything — including every operator Pass / Reject as a training example ([step 10](#10-retrain-between-shifts-the-flywheel)) |
| Needs | Python 3.10+, internet once (for `git clone` + `pip`), a camera that sees the vial holder with the white paper behind it |
| Time to set up | ~10 min (most of it is `pip install`) |
| Where results go | `results/operator_output/` inside the repo folder |

---

## 1. Get the code (once)

Open a terminal (**Windows:** Start → type `cmd` → Command Prompt. **Linux:** Terminal).

`git clone https://github.com/ALL-Applied-AI-Network/msoe-ai-club-rockwell-automation-nvidia-hackath-cruz-rios-s-team-d24d4d83ca.git melting-rainbow-repo`

`cd melting-rainbow-repo`

No git on the computer? On the GitHub page: green **Code** button → **Download ZIP** → unzip → open a terminal in that folder.

**Until the flywheel is merged into `main`:** switch to its branch before starting the station:

`git fetch origin flywheel-safe-retrain`

`git checkout flywheel-safe-retrain`

Check with `git branch` — the line with `*` must say `flywheel-safe-retrain`. (ZIP instead: on GitHub pick the branch in the branch menu first, then **Code → Download ZIP**.) Already have the repo? Run the two lines in it, then `git pull`.

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

**With a run plan (recommended):** a CSV with **one row per vial, in fill order** — the first columns are exactly what is typed into the line:

| vial | clear_ml | yellow_ml | red_ml | blue_ml | total_ml | color | code | group | purpose |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 0 | 0 | 50 | 70 | Dark blue | DB | G1 | start |

Example: [`runs/2026-10-30_plan.csv`](runs/2026-10-30_plan.csv) (30 vials, 10 recipes, ~28 min). Start with

`./start_capture.sh --plan runs/2026-10-30_plan.csv`

- Each vial is tagged with its planned recipe and mL; the page shows **"Next on the line: vial N of 30 — …"**.
- **Groups:** vials with the same `group` are judged together and the group's **last vial is its clean reference** — so groups can be any size (not just 3). Without a `group` column, a new group starts whenever the mL change.
- The line skipped a planned vial → press **▶**; it filled an extra vial → press **◀** (keeps the plan lined up with the real vials).
- Mixed groups (e.g. known mixes judged against one clean vial) need the `group` column.

**Without a plan:** every N vials (the "Vials / batch" box) is one group, the recipe is picked in the dropdown at each change, and **New batch now** ends a group early.

**During the run (both ways):**

- When a group finishes, each vial shows **PASS → good**, **PULL → scrap** (with likely cause and leftover %), or **REVIEW** (look, then click Pass / Reject). Then **Release batch** or **Hold batch**.
- **Every Pass / Reject click is saved as a training label** (`results/operator_output/flywheel/`). Click what you actually see in the zoom view, not what the station said. Settle every REVIEW — those are the most useful examples. Clicked the wrong one? Click the right one; the last click counts. Nothing is trained during the run.
- If the line's Teledyne color check shows a pass/fail, click under each vial what it said (**good / scrap**) — that is the baseline. If there is no Teledyne decision, skip those buttons.
- On paper, per vial: number, recipe, time, tinted by eye (Y/N), anything odd.
- Don't change settings mid-run.

## 8. Known-mix vials (optional)

**Easiest: put them in the plan** (vials 27–30, the last group of [`runs/2026-10-30_plan.csv`](runs/2026-10-30_plan.csv)): the line itself dispenses light yellow with exactly 10 % and 20 % dark blue, then two light yellows — the last one is the group's clean reference. Run a light-yellow group right before them (vials 25–26) so leftover in the lines is the same base color and the 10 % / 20 % stay exact. Nothing to do by hand. **By hand instead:**

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

**Screen recording of the whole page** (for the "operator using the station" shot):

- **CSI computer (Linux / Ubuntu):** press **Print Screen** → in the bar that opens, switch to the **video camera** icon → pick the screen or window → **Record**. Stop with the red button at the top right of the screen. Saved to `~/Videos/Screencasts/`. Older systems: **Ctrl + Alt + Shift + R** starts / stops (may stop on its own after 30 s).
- **Windows:** **Win + Alt + R** starts / stops (Xbox Game Bar, saved to `Videos\Captures`), or Snipping Tool → **Record**.
- Copy the screen recordings off the computer together with `results/operator_output` (step 9).

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
| `raw_camera_*.mp4`, `run_annotated_*.mp4`, `clips/`, `snapshots/` | videos and pictures (step 8b) |
| `flywheel/examples.csv` + `flywheel/images/` | one row per operator Pass / Reject: label, what the station said (status, ΔE, AI %), and the vial + reference frames (PNG, ~0.5 MB per vial). Input for step 10. Turn off with `--no-examples` |

## 9. Finish — don't skip

1. Press **Stop**, wait until the page says **stopped** (this closes the raw video file properly). Only then close the terminal.
2. Copy the whole `results/operator_output` folder off the computer (USB stick, or upload to our Drive folder). It contains:
   - `raw_camera_*.mp4` — every camera frame (≈ 9 MB/min at 1280×720)
   - `operator_results.csv` — every vial: status, rule, ΔE, AI %, cause, leftover, fill level
   - `operator_log.csv` — every click and event with time
   - `camera_align.json`, `snapshots.csv`, `snapshots/` (if used)
   - `clips/` — the clips from **Record clip** / **Save last 30 s**
   - `run_annotated_*.mp4` — the whole run with the status bar (for the video)
   - `flywheel/` — the operator's Pass / Reject clicks as training examples (step 10). **Keep each run's folder separate** (e.g. rename to `operator_output_2026-10-30`) so runs can be told apart later.
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
| `example_error` in `operator_log.csv` | a click couldn't be saved as an example (e.g. disk full). The station keeps running and the click still counts on the page; free space and keep going |
| Don't want clicks saved (e.g. a practice run) | add `--no-examples` to the start command |
| Camera can't be set up at all | Film the vial + white paper with a phone on a stand for the whole run (landscape, don't touch it) and keep the paper log — the system can be run on that video afterwards |

## 10. Retrain between shifts (the flywheel)

The station **never changes its own model while the line runs.** Between shifts, the clicks from step 7 are used to train a new model, and it only replaces the current one if it is **at least as good** on a fixed test set. Details: [`melting-rainbow/flywheel/README.md`](melting-rainbow/flywheel/README.md).

**Where:** any computer — a laptop CPU takes ~5 min per retrain (2,000 twin images + your examples). It does **not** have to be the CSI computer. Needs the venv from step 2 plus:

`pip install pandas`

**What you need** (from the team Drive, `01 Team Handoff`):

| | Where | Unzip to |
|---|---|---|
| Twin dataset | `04_twin_datasets/twin_dataset_v2.zip` | `twin_dataset_v2/` |
| Video 5 test pack | `03_validation_video5/video5_eval.zip` | `video5_eval/` |
| Run results | each run's `results/operator_output` folder (USB stick / Drive) | anywhere |

**1. Dry run — nothing changes** (Linux shown; on Windows use `\` and `python`):

`python3 melting-rainbow/flywheel/retrain.py --twin twin_dataset_v2 --gate-pack video5_eval --examples operator_output_2026-10-30`

- More runs: list them all after `--examples`.
- The terminal prints a table, also saved as `results/flywheel_models/retrain_report_<time>.md`:

| Gate set | Vials | Off-color | Current: misses / false alarms | Candidate: misses / false alarms |
|---|---|---|---|---|
| video5 | 20 | 10 | 0 / 1 | 0 / 1 |

- Last line says **PASSED the gate** (candidate no worse on misses AND false alarms) or **KEPT the current model** (candidate was worse — nothing to do, try again after more runs).

**2. Use the new model** — only after a PASSED dry run, same command plus `--promote`:

`python3 melting-rainbow/flywheel/retrain.py --twin twin_dataset_v2 --gate-pack video5_eval --examples operator_output_2026-10-30 --promote`

- It trains again, re-checks the gate, and only if it passes replaces `melting-rainbow/operator_ui/pair_model.pt`. The old model goes to `results/flywheel_models/history/pair_model_replaced_<time>.pt`.
- The station uses the new model from its **next Start** (restart `./start_capture.sh`). The page pill shows `AI on: pair_model.pt (small)`.
- Write in the run log which report promoted it, and commit the new `pair_model.pt` (79 KB) with the report's numbers in the message so everyone runs the same model.

**3. Undo** (station looks worse on the line): copy the old model back, then restart the station:

`cp results/flywheel_models/history/pair_model_replaced_<time>.pt melting-rainbow/operator_ui/pair_model.pt` (Windows: `copy`)

**Keep the test honest**

- Once there are a few runs, keep one aside as extra test data: `--gate-examples operator_output_<that run>`. It is **never** trained on (the script refuses if the same vial is in both lists), so the gate checks real line data, not just video 5.
- Don't promote on a run with few clicks and a lucky table: 20 test vials catch a model that got clearly worse, not small differences.
- Only the PASS / PULL decision learns from clicks. The leftover % does not (no true leftover is known for real vials).

## Run on a recorded video (practice / afterwards)

**Windows** `python melting-rainbow\operator_ui\operator_app.py --video "C:\path\to\video.mp4" --autostart --speed 8 --batch-size 3 --ai-model melting-rainbow\operator_ui\pair_model.pt --out results\replay`

**Linux** `python3 melting-rainbow/operator_ui/operator_app.py --video /path/to/video.mp4 --autostart --speed 8 --batch-size 3 --ai-model melting-rainbow/operator_ui/pair_model.pt --out results/replay`

Pause / Resume and the speed box work for videos. A `raw_camera_*.mp4` replays with the alignment saved next to it.

---

**What's in this repo:** `melting-rainbow/` = our station and analysis code (see `melting-rainbow/operator_ui/README.txt` for how each part works and where every number comes from). `windows/` = double-click launchers. `frank_recipe_classifier/` = Frank's EfficientNet recipe-classifier pipeline. `docs/` = results and how it works. `archive/` = early experiments.
