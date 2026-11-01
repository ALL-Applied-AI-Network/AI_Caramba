# Recipe classifier (EfficientNet-B0)

Our first AI approach: classify which recipe a vial image shows, then flag a vial as off-color when the predicted
recipe differs from the planned one. Trained on Rosie, MSOE's supercomputer.

| File | What it does |
|---|---|
| `split_dataset.py` | splits `recipe_dataset/<recipe>/*.png` into `dataset/train`, `dataset/val`, `dataset/test` |
| `train.py` | fine-tunes timm EfficientNet-B0 (224 px, 10 epochs) |
| `evaluate.py` | accuracy on `dataset/test` |

Run from inside this folder (paths are relative): `cd frank_recipe_classifier && python split_dataset.py && python train.py && python evaluate.py`.

**What it taught us:** ~82 % recipe accuracy on twin images but ~25 % on the real video — the model learned the
digital twin's absolute colors, which differ from the real camera. That is why the final system *compares* each vial
with its batch's clean vial instead (see [How it works](../docs/HOW_IT_WORKS.md#what-we-tried-first)).
Honest-split and video 5 scoring scripts for this model: `melting-rainbow/lane6_classifier/recipe_model/`.
