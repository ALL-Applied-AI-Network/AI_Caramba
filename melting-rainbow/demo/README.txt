LIVE VIAL INSPECTION DEMO  (demo/live_inspect.py)  - Oct 24
================================================================
What it does
  Watches the vial_fill camera (or a recorded video), finds each fill, reads the liquid color in front of the
  white paper, and when a batch ends shows for every vial: GOOD / OFF-COLOR, the likely cause (which earlier
  recipe left liquid behind) and a rough leftover in mL. It saves:
    inspection.mp4           the video with the results panel drawn next to it
    inspection_results.csv   one row per vial
    inspection_log.txt       what happened, with frame numbers

Run it (Windows, double-click):
  01_code\windows\run_demo_video5.bat   organizers' video 5 -> about 1 minute, writes an 8x-speed video (~3.5 min)
  01_code\windows\run_demo_camera.bat   live USB camera 0, NO recipe list: every 3 vials = one batch (Oct 26)

The plan file = the recipes in the order the vials will be filled, one row per vial:
  recipe_id,total_ml
  6,40
  ...
  (The organizers' video5_vial_recipes.csv works as a plan as it has those two columns.)
  Either a plan OR a fixed batch size is required. We tried guessing batches from color alone on video 5 and it
  was unreliable (wrong batches, 11 flags of which several wrong).

NO RECIPE LIST (added Oct 26):  --batch-size 3   (every 3 vials is one batch; the 3rd vial is the reference)
  - The model never needed the recipe: it compares each vial with its batch's last vial. The recipe was only used
    to know where batches start and the fill volume.
  - Batches are named B1, B2, ...; the cause reads "B8" = leftover from batch 8 ("B8?" = unsure).
  - Fill level is read from the picture: liquid in the top of the measuring band (rows 272-285) = 60-80 mL band,
    empty glass there = 30-40 mL band. Empty glass reads 84-97% of the paper, liquid 0-65% (video 5's 30 vials),
    line at 75%. It only picks which rows to measure, never the good/off decision.
  - Leftover is shown in % of the vial (volume isn't known without the recipe).
  - Checked on video 5 (Oct 26): exactly the same verdicts and causes as with the recipe list
    (0 misses, 2 false alarms, cause 7/10 with B-numbers in place of recipe numbers).

Result on video 5 (checked Oct 24): 30/30 fills found, same as our validation run:
  0 missed off-color vials, 2 false alarms (vials 1 and 2 - first batch, no earlier recipe to compare with),
  cause right for 7 of 10 off-color vials (the validation script got 6/10; vial 16 flipped to the right answer
  because the demo picks the steadiest frames instead of fixed offsets). Borderline: vial 13 dE 7.0 vs cutoff 7.
  Nothing was tuned on video 5.

Where the numbers come from
  vial present .. black holder patches dark (<100); on video 5: holder 39-57, no holder 161-166.
                  Needed because the empty gap between batches (no vial, dark machine) looked like a fill.
  fill found .... lane2_color/detect_fills.py signal and thresholds; minimum fill 5 s (8 s missed the last,
                  very light R14 vial, which only scores for its final ~7 s).
  color read .... common/measure.py (yellow box in the video), median of the 3 steadiest frames in the last 2 s.
  verdict ....... lane6_classifier/approach_a.py: compared with the batch's LAST vial, off-color if dE >= 7
                  (cutoff from videos 1-4 noise).
  cause / mL .... same math as lane6_classifier/eval_video5.py. mL is rough (+/- ~13 points of leftover on the twin).
  cause confidence (added Oct 26) .. "R9?" on screen = unsure: the 2nd-best recipe explains the color shift almost
                  as well (ratio >= 0.6, threshold set on the twin, lane6_classifier/approach_a.py). On video 5 (scored
                  once): sure on 5 vials, 4 right; unsure on 5, 3 right; the right recipe is in the top 2 for 9 of 10.

Real hardware
  The color box assumes the organizers' camera view (640x480, vial at x ~310, paper behind it). With a
  different camera position, run with --show and line the yellow box up on the liquid in front of the paper
  before trusting results.
