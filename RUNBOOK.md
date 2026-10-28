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

`python3 melting-rainbow/operator_ui/operator_app.py --camera 0 --batch-size 3 --ai-model melting-rainbow/operator_ui/pair_model.pt --out results/operator_output`

- The terminal prints `Operator station running: http://localhost:8765`. **Leave this terminal open** — closing it stops the station.
- The browser opens the page. If not, open a browser and go to `localhost:8765`.
- Loading takes 10–20 s. The pill at the top says **AI on** (or AI off without torch — fine).
- Windows shortcut instead of typing: double-click `windows\run_operator_station.bat` (camera 0) or `windows\run_operator_station_usbcam.bat` (camera 1). If Windows blocks the .bat (Smart App Control), use the typed command above.

## 4. Pick the camera

Top bar: **Source = Camera**, and the box next to it is the camera number.

| Box | Usually |
|---|---|
| `0` | the first camera on the computer (a laptop's own camera, or the only USB camera) |
| `1` | a second camera (e.g. a USB webcam on a laptop) |
| `2` | a third camera |
| `rtsp://...` | a network camera — paste its stream address (ask CSI staff) |

Press **Start**. Under the picture it must say `recording raw video → raw_camera_….mp4` — that file is the data.
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

| Batch | Recipe | Clear (T100) / Orange (T200) / Blue (T400) mL | Looks | Why |
|---|---|---|---|---|
| 1 | R12 | 10 / 50 / 10 = 70 | dark orange | start (no earlier batch → cause "unknown") |
| 2 | R14 | 60 / – / 10 = 70 | pale blue | dark → pale: first vial should be tinted (high carryover risk) |
| 3 | R11 | 10 / 10 / 50 = 70 | dark blue | pale → dark: checks we don't false-alarm (low risk) |
| 4 (optional) | R12 | 10 / 50 / 10 = 70 | dark orange | blue → orange: should be caught again |

Backups: R13 (– / 50 / 20), R9 (10 / 35 / 35). Keep fills ≥ 60 mL so the liquid covers the color window.

- Before each batch's first vial: pick the recipe in **Recipe on the line now**.
- Every 3 vials = one batch (the batch's last vial is its clean reference). Different batch size → press **New batch now** at the recipe change.
- When a batch finishes, each vial shows **PASS → good**, **PULL → scrap** (with likely cause and leftover %), or **REVIEW** (look, then click Pass / Reject). Then **Release batch** or **Hold batch**.
- If the line's Teledyne color check shows a pass/fail, click under each vial what it said (**good / scrap**) — that is the baseline. If there is no Teledyne decision, skip those buttons.
- On paper, per vial: number, recipe, time, tinted by eye (Y/N), anything odd.
- Don't change settings mid-run.

## 8. Known-mix vials (optional)

For checking the "% leftover" estimate against the truth. Fill to ≥ 60 mL, stir well, place in the holder, wait 5 s, type a label, press **Measure vial in holder**:

1. `prev clean R12` (a clean R12 vial)
2. `ref clean R14` (a clean R14 vial — label must start with **ref**)
3. `R14 65 + R12 5`, `R14 60 + R12 10`, `R14 50 + R12 20` (write the exact mL actually used)

Saved to `results/operator_output/snapshots.csv` + `snapshots/`.

## 9. Finish — don't skip

1. Press **Stop**, wait until the page says **stopped** (this closes the raw video file properly). Only then close the terminal.
2. Copy the whole `results/operator_output` folder off the computer (USB stick, or upload to our Drive folder). It contains:
   - `raw_camera_*.mp4` — every camera frame (≈ 9 MB/min at 1280×720)
   - `operator_results.csv` — every vial: status, rule, ΔE, AI %, cause, leftover, fill level
   - `operator_log.csv` — every click and event with time
   - `camera_align.json`, `snapshots.csv`, `snapshots/` (if used)
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

**What's in this repo:** `melting-rainbow/` = our station and analysis code (see `melting-rainbow/operator_ui/README.txt` for how each part works and where every number comes from). `windows/` = double-click launchers. `train.py`, `evaluate.py`, `split_dataset.py` = Frank's EfficientNet recipe-classifier pipeline.
