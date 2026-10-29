# Archive — early experiments (not used by the final system)

Kept for history and transparency. Nothing in the operator station imports these files.

| Path | What it was |
|---|---|
| `early_experiments/README_starter_code_oct16.md` | the original starter-code plan ("lanes") from Oct 16 |
| `early_experiments/lane1_sync/` | lining up the three cameras' timestamps |
| `early_experiments/lane2_color/` | first color baseline, color-range check, fill-time / tank-level volume checks |
| `early_experiments/lane3_carryover/` + `common/mixing.py` | first carryover model (predict mixed colors from tank amounts) |
| `early_experiments/lane5_eval/` | leave-one-recipe-out test of that first color model |
| `early_experiments/tools/` | mock dataset generator for testing scripts without the real data |

They still run from this folder (same relative layout), e.g. `python archive/early_experiments/lane5_eval/eval_loro.py ...`.
