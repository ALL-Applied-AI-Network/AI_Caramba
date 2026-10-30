# Feedback flywheel — learns from the line, safely

The station gets better the longer it runs on the line, **without ever changing itself while the line runs.**

```
operator clicks Pass / Reject ──► labeled real example ──► retrain between shifts ──► gate ──► new model (or keep old)
        (station, live)              (examples.py)            (retrain.py)          same fixed test set
```

## 1. Collect (automatic, during operation)
Every **Pass** or **Reject** click on a vial in the operator station saves that vial's frame, its batch's reference frame, and one row in `results/operator_output/flywheel/examples.csv`:

| column | meaning |
|---|---|
| `label` | `off` (Reject) or `good` (Pass) — the operator's call |
| `source` | `review` = settled a REVIEW (the hard cases), `override` = overruled the station, `confirm` = agreed with it |
| `station_status`, `rule_dE`, `ai_p_off` | what the station had said |
| `vial_image`, `ref_image` | lossless PNG frames (≈ 0.5 MB per vial) |

If the operator clicks a vial twice, the last click wins. Saving can never stop the station (errors go to `operator_log.csv`). Turn it off with `--no-examples`.

## 2. Retrain + gate (between shifts, any laptop CPU, a few minutes)
```
python melting-rainbow/flywheel/retrain.py --twin twin_dataset_v2 \
    --examples results/operator_output --gate-pack video5_eval
```
- Trains a **candidate** on the 2,000 twin images + the real examples (each counted 10×, `--real-repeat`).
- Scores the **current** model and the candidate on the same fixed gate set: video 5 (+ any real runs given with `--gate-examples`, which are never trained on).
- **Rule:** the candidate is only accepted if it has **no more misses and no more false alarms** than the current model. Otherwise the current model stays.
- Add `--promote` to let a passing candidate replace `operator_ui/pair_model.pt` (the old one is kept in `results/flywheel_models/history/`). Without it, it's a dry run.
- Every run writes `results/flywheel_models/retrain_report_<time>.md` — the before/after table.

## Why not train live?
- **Wrong clicks**: a few mislabeled vials could quietly make the model worse, mid-shift.
- **Rare defects**: most vials are good, so a self-updating model drifts toward "everything is good".
- **Traceability**: on a production line every model change should be a decision with a record, not a side effect.

## Honest limits
- Video 5 is our only labeled real video, so today it is both our headline result and the gate. As real runs build up, they should take over as the gate (`--gate-examples`) so video 5 stays an untouched check.
- 20 judged vials is a small gate: it catches a model that got clearly worse, not small differences.
- The leftover % is not retrained on real vials (no true leftover is known for them); only the off-color decision is.
