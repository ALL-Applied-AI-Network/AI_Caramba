OPERATOR STATION (operator_ui/)  - Oct 26 (evening update: pause, zoom, fill-level detection)
====================================================
What it is
  A browser page for the line operator. It watches the fill camera (or a recording), and after every batch of
  vials (default 3) shows each vial as:
    PASS      rule and AI agree it matches the batch's clean reference vial
    REJECT    both say off-color - with the likely cause ("leftover from B8") and leftover %
    REVIEW    they disagree - the operator looks at the thumbnail and clicks Pass or Reject
    Reference the batch's last vial, assumed clean (what the others are compared to)
  The operator then clicks Release batch / Hold batch. Every result and click is written to
  operator_output/operator_log.csv (also "download log" on the page).

Run it (Windows, double-click in 01_code\windows\):
  run_operator_station.bat   live camera 0 -> page opens at http://localhost:8765, press Start
  run_operator_video5.bat    same page, playing organizers' video 5 at 8x (for practice / the video)
  The AI second opinion turns on when operator_ui\pair_model.pt exists (make it on the Spark:
  python lane6_classifier/pair_model/pair_model.py train --data twin_dataset_v2 --out pair_model.pt,
  then copy it into operator_ui\). Without it - or without torch - the page runs on the rule alone and says so.
  Buttons: Start / Stop, "New batch now" (ends the batch early at a recipe change), Release / Hold per batch,
  Pass / Reject per vial (overrides, marked with a pencil).

Run it on the configured CSI Linux computer:
  ./start_capture.sh
  This launches this operator UI with the stable Vial Fill /dev/v4l/by-id path, the Linewatch-tested 640x480 MJPG
  camera setup, automatic exposure, locked white balance, raw recording, and the included AI model.

Operator in the loop (organizers' guidance, Oct 26)
  Our system can't drive the line's sorter or read its sensors yet. The line's existing Teledyne sorter compares color
  with a static, non-updatable color list and has no link to the recipe, so it sends almost every vial to the "good"
  location. So the page:
    - says PULL -> scrap for vials the operator should take out of the good lane, PASS -> good for the rest;
    - has "Recipe on the line now" - the operator enters the recipe at each change, so batches and causes carry the
      recipe (e.g. "leftover from B8 (R12)");
    - has "Sorter sent it to: good / scrap" under each vial - the operator records what the existing sorter did.
      That is the MEASURED BASELINE for the video/summary; the tally box compares it with ours.
  PLC integration (letting the system drive the diverter) is the after-hackathon step.

Checked on video 5 (Oct 26, rule + Oct 26 small AI model, 60x playback; nothing tuned on video 5):
  all 10 off-color vials -> REJECT (0 missed)
  vial 1 -> REJECT (false alarm, first batch: no earlier recipe to compare with)
  vials 2 and 13 -> REVIEW (rule and AI disagreed; both truly good by our eye check)
  the other 7 good vials -> PASS
  So the AI second opinion turns 2 automatic false alarms (rule alone) into 1, plus 2 for a human to check.

Where things come from
  fill detection, vial-present check, measurement, rule, cause: demo/live_inspect.py (see its README)
  AI: lane6_classifier/pair_model/pair_model.py (vial crop + reference crop -> P(off), off if >= 50%), trained on
      the twin only. Small model: twin leave-one-recipe-out 14% misses / 6% false alarms; video 5 0 misses / 2 FA.
      The AI numbers on this page can differ a little from pair_model_video5.csv because the station picks the
      steadiest frames of each fill instead of a fixed frame.
  Needs: python with opencv-python + numpy (the same as the demo). AI needs torch (+ timm for the EfficientNet model).
  No internet needed; the page is served from this computer only (127.0.0.1).

Added Oct 26 evening
  Pause / Resume   video files only: freezes the picture AND the detector at that frame; Resume continues from the
                   same spot (Stop ends the run; Start begins again from the start). Disabled for a live camera,
                   because the real line keeps moving. Speed can be changed while it runs (type a number, press Enter).
                   A progress bar shows where you are in the video. Pause/resume are written to the operator log.
  Zoom             click any vial picture: the vial and its batch's clean reference vial side by side, zoom 1-12x
                   (scroll, +/-, double-click), drag to move - both pictures move together. Yellow box = where the
                   color is read, white box = the paper it is divided by, blue line = fill line. Arrow keys = other
                   vials, Esc = close. The picture is the exact frame the color was measured on.
  Fill detection   default is now --detector level (lane2_color/detect_fills_level.py): the vial is compared with the
                   same vial while it was empty, so clear and pale liquid count too. Videos 1-5 vial counts:
                   19/19, 3/3, 6/6 (was 3/6), 6/6, 30/30. --detector color = the older color score.
  Fill reading     each vial shows "high fill (line behind the holder)" or "partial fill (line at N% of the visible
                   vial)". Geometry only - not converted to mL yet (needs vials of known volume, e.g. at CSI).
  CHECK FILL       if the fill line is BELOW the window where the color is read, that window is empty glass and the
                   color means nothing -> the vial goes to REVIEW with "fill line is below the color window - check
                   the volume" (the AI is skipped too: it reads the same window). Same if the batch's reference is low.
                   Found in video 3: vials 4-6 are small fills (33-39%) - the color verdicts there were not real.
  Video 5 re-checked once with all of this: unchanged - 10 of 10 off-color PULL, vial 1 PULL (false alarm,
  first batch), 2 REVIEW, 7 PASS. The new detector reads the same colors as the old one there (max difference
  0.013 in transmittance, identical verdicts).

Added Oct 28 (for the CSI session)
  Raw recording    every CAMERA run saves the plain camera video to <output>\raw_camera_<date_time>.mp4 (every frame,
                   unaligned) so the whole session can be re-run later. ~9 MB per minute at 1280x720.
                   --no-record turns it off. Replaying it: drag the raw_camera_*.mp4 onto run_operator_any_video.bat;
                   the station uses the camera_align.json saved in the same folder.
  Align camera     button at the top. Our measuring windows are fixed spots in the organizers' camera picture. With our
                   own webcam: Start the camera, press Align camera, click the BOTTOM-LEFT then BOTTOM-RIGHT corner of
                   the black holder's front face, Save, and check the 50/50 overlay (holder, vial, paper line up).
                   Maps those two corners onto the organizers' (259,258) and (371,270) - measured on videos 1-5, +/-2 px -
                   with shift + scale + rotation. Saved in <output>\camera_align.json and reused next start.
                   Only used for the camera and our own raw_camera_*.mp4 files, never for the organizers' videos.
                   Tested Oct 28 with a fake "other camera" (video 1 scaled 1.35x, turned 3 deg, shifted, 1280x720):
                   clicking in the page gave x0.74 / -3.2 deg (true 0.741 / -3.0); all 8 vials got the same
                   PASS/PULL/REVIEW as the original video; color differences within ~2-4 dE (two good vials read
                   dE ~6 instead of 2-5: an imperfect alignment adds a little noise - still under the 7 cutoff).
                   Limits: same viewing direction only (no perspective fix); the white paper must be behind the vial.
  Snapshot         "Measure vial in holder" with a label: measures whatever vial sits in the holder right now (known
                   mixes placed by hand - the fill detector only sees vials being filled). Label the clean one
                   "ref ..."; the next ones show dE against it. Saved to <output>\snapshots.csv + snapshots\*.png.

Added Oct 28 (evening)
  Record clip / Save last 30 s   buttons under the camera picture, separate from Start/Stop. A clip = the camera view
                   with the measuring box, fill line and a status bar with the latest batch result, ~12 frames/s, saved
                   to <output>/clips/. "Save last 30 s" writes the 30 s that just happened (kept in memory as JPEGs).
  prev vial        a snapshot labeled "prev <name>" registers the recipe that ran before this session, so the first
                   batch is checked for carryover and gets a cause (instead of "start of run").
  Full-run video   every camera run also writes run_annotated_<time>.mp4 (camera view + status bar, ~12 fps) from Start
                   to Stop, next to raw_camera_<time>.mp4. --no-record turns both off.
  Run plan         --plan runs/<date>_plan.csv: one row per vial in fill order (vial, <tank>_ml..., color, code, group).
                   Groups end where the plan's group changes (last vial = reference); "Next on the line" + < > to realign.
  Metadata         per-vial start/end date+time in operator_results.csv; session_<time>.json per run (camera
                   settings, alignment, plan, counts, software version).
  Auto-copy        --copy-to DIR copies the output folder to DIR after every Stop (USB stick / synced folder);
                   --after-stop "CMD" runs an upload command afterwards (e.g. rclone). Shown on the page; never deletes.
