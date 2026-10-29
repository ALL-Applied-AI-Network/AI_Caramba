# Results — every number and where it comes from

All result files referenced here are in [`docs/results/`](results). Commands re-create them from this repo plus the organizers' dataset.

## 1. Validation on real video 5 (the organizers' only labeled video)

**Setup.** 30 vials, filled in batches of 3; each batch's **last vial is its clean reference**, so 20 vials are judged: **10 off-color, 10 good** (eye check, [`melting-rainbow/labels/truth_by_eye.csv`](../melting-rainbow/labels/truth_by_eye.csv)). Video 5 was **never used for training or tuning**; each model version was scored on it once.

![All 30 vials of video 5](images/video5_all_30_vials.png)

| Method | Off-color caught (of 10) | Good wrongly pulled (of 10) | Cause named right | Source |
|---|---|---|---|---|
| Existing color check | ~0 expected | ~0 | — | Organizers' description (not measurable: the line shows no per-vial decision) |
| Rule (vs. batch reference, ΔE ≥ 7) | **10** | 2 | 6 of 10 | [`rule_video5.csv`](results/rule_video5.csv) |
| AI comparison model (difference-only) | **10** | 1 | — | [`pair_model_video5_relative.csv`](results/pair_model_video5_relative.csv) |
| **Operator station (rule + AI)** | **10** PULL | **1** PULL + 2 REVIEW | 6 of 10 | [`operator_station_video5.csv`](results/operator_station_video5.csv) |
| Recipe classifier (EfficientNet, first attempt) | 9 | 6 | — | Frank's Oct 26 run (provisional: re-scoring with matching preprocessing) |

Re-create:
```
python melting-rainbow/lane6_classifier/eval_video5.py <dataset>/csi-vial-fill-dataset melting-rainbow/labels/truth_by_eye.csv
python melting-rainbow/operator_ui/operator_app.py --video <dataset>/.../vial_fill_video_5.mp4 --autostart --speed 200 --ai-model melting-rainbow/operator_ui/pair_model.pt
```

**The misses and false alarms, explained**
- Vial 1 (the one good vial pulled): the organizers' recording starts mid-session, so the station had no earlier batch to compare its first batch with. In a live run the operator now measures one clean vial of the previous recipe at the start ("prev" vial, added Oct 28), so the first batch is checked and its cause named like every other batch (tested on video 5 frames: vials 4–5 → "R5", leftover 78 % / 49 %).
- Vials 2 and 13 (REVIEW): rule and AI disagreed; both truly good. The operator decides.
- Cause wrong for 4 of 10: when several earlier recipes have similar colors, the leftover's source is ambiguous; the page marks these "unsure, maybe B2".

## 2. Digital-twin test (generalization to unseen recipes)

2,000 Isaac Sim renders (every recipe, and every recipe-to-recipe leftover from 5 % to 100 %), camera exposure calibrated to the real camera. **Leave-one-recipe-out**: train on 9 recipes, test on the 10th, repeat for all 10.

| AI model | Misses | False alarms | Source |
|---|---|---|---|
| **Difference-only comparison (used)** | **89 / 1,090 (8 %)** | **48 / 910 (5 %)** | [`pair_model_loro_relative.csv`](results/pair_model_loro_relative.csv) |
| EfficientNet comparison that could see absolute color | 191 / 1,090 (18 %) | 840 / 910 (92 %) | Frank's Oct 26 run (DGX Spark) |

Lesson: a model trained on simulation must **compare**, not memorize simulated colors (twin vs. real colors differ by ΔE 16–52 even after calibration).

## 3. Finding the vials (fill detection) on all organizers' videos

| Video | Vials (organizers' list) | Found |
|---|---|---|
| 1 | 19 | 19 |
| 2 | 3 | 3 |
| 3 | 6 | **6** (the older color-based detector found 3 — it missed pale/clear fills) |
| 4 | 6 | 6 |
| 5 | 30 | 30 (start/end within ~1–5 s of the organizers' labels) |

## 4. Live on the CSI line

| Date | What ran | Camera | Result |
|---|---|---|---|
| Oct 28 | Station on the CSI computer (`start_capture.sh`), recipe changes: dark orange → light blue → dark orange → dark blue (+ a green test). No recipe colors were entered — the station only compares vials with their own batch | Line's own vial-fill USB camera (640×480, white balance locked) | *Files being collected from the CSI computer — results will be added here* |
| Oct 30 | 30 vials / 10 recipes, filmed; known-mix vials to check the leftover % | same | *to be added* |

## 5. Scope and next steps

| Today | Next |
|---|---|
| Each batch is compared with its own last vial (the organizers' convention for a clean reference) | Optional purge-check vial for batches where the whole batch could be tinted |
| First batch of a run uses a "prev" vial measured at the start | Read the previous recipe straight from the line's controller (PLC) |
| Batches come from a fixed size or the operator's "New batch" | Batch and recipe changes from the PLC; diverter acts on PULL automatically |
| Leftover % estimated (±13 points on the twin) | Checked against real known-mix vials on Oct 30 |
| Validated on the organizers' 30-vial labeled video + live CSI runs | Every operator Pass / Reject becomes a labeled real example for retraining (feedback flywheel) |
| Very small fills (liquid below the color window) go to REVIEW with "check the volume" | Fill-line reading in mL once known-volume vials are measured |
| Works with any dye colors — no color list needed | Warn *before* risky changeovers (dark → pale) so the purge can be adjusted |
