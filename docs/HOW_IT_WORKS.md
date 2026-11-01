# How it works

```
fill camera ──► 1. find the fill ──► 2. measure color ──► 3. compare with the most recent vial of the same recipe ──► 5. operator page
                 (change from the      (liquid ÷ white       rule: ΔE ≥ 7 → off                      PASS / PULL / REVIEW
                  empty vial)           paper behind it)    4. AI second opinion (twin-trained)       + cause + leftover %
```

## 1. Find each fill — `melting-rainbow/lane2_color/detect_fills_level.py`
The station remembers what the vial looks like **while empty**. Any liquid — even clear — changes that picture, starting at the bottom and rising. A fill **ends** when the picture moves the way a vial swap does (swaps move it ≥ 43 units, pours and pauses between two tanks ≤ 20, so the cutoff is 25 — set on videos 1–4). The highest changed row is the **fill line** (a volume reading; the black holder hides lines above ~60 mL).

## 2. Measure the color — `melting-rainbow/common/measure.py`
Only where the liquid is **in front of the white paper**, divided by the paper right next to the vial. That gives transmittance per color channel (T), which cancels exposure and white-balance changes. ΔE (CIE Lab) compares two vials. Lesson learned: our first measuring box half-covered the dark machine behind the paper — *where* you measure mattered more than any model.

## 3. The rule — `melting-rainbow/lane6_classifier/approach_a.py`
- The station is never told the recipe; its only setting is "every 3 vials is one recipe". Each recipe group's **most recent (last) vial is its reference** (the cleanest, since carryover fades vial by vial; the organizers' convention). Every other vial is compared with it: **off-color if ΔE ≥ 7** (cutoff from the noise between identical vials in videos 1–4).
- **Cause:** dyes add up in optical density (Beer–Lambert), so the shift is matched against each earlier recipe group's reference — the best match is the likely source. "Unsure" when the runner-up explains it almost as well (ratio ≥ 0.6, set on the twin).
- **Leftover %:** how far the vial has moved toward the previous recipe (±13 points on the twin; not yet checked on real mixes).
- **No color database:** the station never needs to know what color a recipe should be, so new recipes and dyes need no setup.
- **Start of a run:** the operator measures one clean vial of the previous recipe ("prev" vial); it becomes the earliest group, so the first recipe group is checked for carryover too.

## 4. The AI second opinion — `melting-rainbow/lane6_classifier/pair_model/pair_model.py`
- Input: two crops (the vial + its recipe group's reference), each divided by its paper. A small CNN encodes both; the decision head sees **only the difference** between them, so it cannot memorize simulated colors.
- Trained **only on the digital twin**: 2,000 Isaac Sim renders of every recipe and every recipe-to-recipe leftover (5–100 %), camera exposure calibrated to the real camera (`melting-rainbow/lane4_twin/`). Label: off if ΔE ≥ 7.
- Tested leave-one-recipe-out on the twin, then scored once on real video 5 (see [Results](RESULTS.md)).
- File: `melting-rainbow/operator_ui/pair_model.pt` (79 KB, runs on a laptop CPU).

## 5. The operator station — `melting-rainbow/operator_ui/`
- **PASS → good** when rule and AI agree it matches; **PULL → scrap** when both say off (with cause and leftover); **REVIEW** when they disagree (DeltaSight = rule + AI) — the operator zooms in next to the reference and clicks Pass / Reject.
- Why an operator: our system can't drive the line's diverter yet, and the organizers asked for an operator-in-the-loop interface first. **Every operator click is a labeled real example** — the start of a training flywheel.
- Also: Release / Hold per recipe group (the screen calls a group a "batch"), recipe entry, pause / resume for recordings, a zoom view, **Align camera** (2 clicks map a different camera onto the organizers' camera view), raw camera recording, snapshots for hand-placed vials, a CSV log of everything.

## What we tried first
1. **Color model per recipe** (`archive/early_experiments/`): predicts each recipe's color from its tank amounts — like the existing static list, it needs every recipe's color in advance.
2. **EfficientNet recipe classifier** (`frank_recipe_classifier/`, trained on Rosie, MSOE's supercomputer): 82 % recipe accuracy on twin images, ~25 % on real video — it learned the *simulated* colors. This showed us the AI must **compare** vials rather than recognize absolute colors, which led to the difference-only model above.
3. **EfficientNet comparison that still saw absolute color:** 92 % false alarms on unseen recipes in the twin — confirming the same lesson.

## Next steps (after the hackathon)
- PLC link so the diverter acts on PULL automatically.
- **Changeover risk warning** before a recipe runs: dark → pale changes (e.g. dark orange → light blue) carry the most visible leftover; warn or extend the purge.
- Retrain the AI on operator-confirmed real vials (the flywheel), stored as small per-vial samples instead of full video.
