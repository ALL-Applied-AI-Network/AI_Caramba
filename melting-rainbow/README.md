# Melting Rainbow — starter code (Dataset 2, Fluid Mixing Station)

Tested on the REAL dataset (video 5) on Oct 16, 2026. Real results are in the team doc
("What we tested"). The mock dataset is still here for quick setup checks.

## Setup (everyone)
```
pip install opencv-python numpy pandas matplotlib usd-core
```
Put the real dataset next to this folder, unchanged:
```
csi-vial-fill-dataset/{README.md, videos.csv, vial_fill/, tank_level/, cap_seal/}
```
Don't commit the dataset to git (add it to .gitignore).

## Run order
| Lane | Command | Output |
|---|---|---|
| 1 Sync | `python lane1_sync/sync.py ../csi-vial-fill-dataset` | `sync_index.csv` + gap/drift report |
| 2 Color | `python lane2_color/color_baseline.py ../csi-vial-fill-dataset --video --box 300 300 28 30 --preview` | `vial_colors.csv`, `crop_preview.jpg` |
| 2b Volume | `python lane2_color/volume_checks.py ../csi-vial-fill-dataset` | `fill_time_bands.csv`, `tank_levels.csv` |
| 2c Color range | `python lane2_color/color_range.py vial_colors.csv` | lightest-to-darkest list, pairs the camera can't separate |
| 3 Carryover | `python lane3_carryover/carryover.py vial_colors.csv --mix od` | H1 vs H2 verdict, `transition_risk.csv` |
| 4 Twin | from the twin root: `python <repo>/lane4_twin/make_variants.py`, then run the lines in `render_commands.txt` with Isaac Sim | one twin file per recipe + renders |
| 5 Eval | `python lane5_eval/eval_loro.py vial_colors.csv --mix od --w 0.54 --truth labels/truth_by_eye.csv` | misses / false alarms / cause accuracy, `loro_results.csv` |

Lanes 3 and 5 need lane 2's `vial_colors.csv`.

Notes from the real run:
- The dataset unzips into `csi-vial-fill-dataset/csi-vial-fill-dataset/` (double folder). Point scripts at the inner one.
- Box `300 300 28 30` sits in the liquid with white paper behind it. Lower boxes see a dark background and every dark recipe reads as black.
- The black vial holder hides the liquid line for most fills (about 45-75 mL), so volume is checked by fill time and tank level instead (volume_checks.py).
- 30 mL fills (recipe 5): the default box sits on the liquid's top edge and reads too light. Use `--box 300 340 28 25` for those.
- `labels/truth_by_eye.csv` = which of the 30 vials look off, checked by eye (10 vials, incl. 2nd vials of R6 and R14).

## Try it without the real data
```
python tools/make_mock_dataset.py mock_data
python lane1_sync/sync.py mock_data
python lane2_color/color_baseline.py mock_data --box 315 330 35 60 --preview   # mock has no video: stills mode
python lane3_carryover/carryover.py vial_colors.csv
python lane5_eval/eval_loro.py vial_colors.csv --w 0.3
python lane4_twin/recolor_liquid.py list tools/mock_station.usda
```
The mock injects carryover = 30% of the previous recipe's mix, so lane 3 should report H1 with w ≈ 0.3.
