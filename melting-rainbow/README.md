# melting-rainbow/ — the code

| Folder | What it does | Used by the station? |
|---|---|---|
| `operator_ui/` | the operator station: `operator_app.py` (engine + web server), `operator.html` (page), `pair_model.pt` (AI), `reference_frame.jpg` (for Align) | **yes** |
| `demo/` | `live_inspect.py`: the inspection logic (batches, rule, cause) + a command-line demo that writes an annotated video | **yes** |
| `common/` | `measure.py`: color of the liquid in front of the white paper ÷ the paper | **yes** |
| `lane2_color/` | `detect_fills_level.py` (finds each fill + fill line), `detect_fills.py` (older color-based detector, `--detector color`) | **yes** |
| `lane6_classifier/` | `approach_a.py` (the rule + cause), `pair_model/` (AI: train / leave-one-recipe-out / video 5), `eval_video5.py` (scores the rule on video 5), `recipe_model/` (scoring kit for the EfficientNet classifier) | **yes** (rule + AI) |
| `lane4_twin/` | Isaac Sim scripts that render the digital-twin training images (launchers in `windows/isaac_twin/`) | training only |
| `labels/` | `truth_by_eye.csv`: which video 5 vials are off-color (eye check) | scoring only |

How each piece works: [docs/HOW_IT_WORKS.md](../docs/HOW_IT_WORKS.md). Detailed notes: `operator_ui/README.txt`, `demo/README.txt`.
Install: `pip install -r requirements.txt` (torch optional).
