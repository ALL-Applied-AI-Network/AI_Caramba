"""Lane 5: leave-one-recipe-out evaluation (as the dataset page recommends).

For each held-out recipe:
  1. Estimate tank colors from the OTHER 9 recipes' clean vials (no peeking).
  2. Predict the held-out recipe's expected color from its volumes alone.
  3. Flag a vial if dE(observed, expected) > threshold (threshold learned on the other recipes).
  4. If flagged, claim a cause: "carryover from R<prev>" if blending in the previous
     recipe's color explains the vial better than the clean prediction; else "unexplained".
Ground truth used here: vial 1 of each recipe after the first is a defect (dataset page:
first vial after a recipe change is discolored). CHECK THIS VISUALLY ON THE REAL DATA.

Reports false alarms, missed defects and cause accuracy separately.

Usage: python lane5_eval/eval_loro.py vial_colors.csv [--mix rgb|od] [--w 0.3]
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.mixing import fit_tank_colors, predict, to_space, from_space, dE

ap = argparse.ArgumentParser()
ap.add_argument("colors"); ap.add_argument("--mix", default="rgb", choices=["rgb", "od"])
ap.add_argument("--w", type=float, default=0.3, help="carryover fraction (from lane 3)")
ap.add_argument("--truth", help="CSV with columns vial,is_defect (checked by eye). Default: vial 1 of each recipe after the first")
a = ap.parse_args()
df = pd.read_csv(a.colors)
bgr = lambda r: np.array([r["B"], r["G"], r["R"]], float)
recipes = sorted(df.recipe_id.unique())
first_recipe = recipes[0]
if a.truth:
    t = pd.read_csv(a.truth).set_index("vial")["is_defect"].astype(bool)
    df["is_defect"] = df.vial.map(t).fillna(False)
else:
    df["is_defect"] = (df.pos_in_batch == 1) & (df.recipe_id != first_recipe)

out = []
for held in recipes:
    train = df[df.recipe_id != held]
    T = fit_tank_colors([r for _, r in train[train.pos_in_batch == 3].iterrows()], a.mix)
    # threshold: midpoint between clean (pos 2,3) and defect dE on the TRAIN recipes
    tr_dE = train.apply(lambda r: dE(predict(r, T, a.mix), bgr(r)), axis=1)
    clean_max = tr_dE[~train.is_defect].quantile(0.95)
    defect_min = tr_dE[train.is_defect].quantile(0.05) if train.is_defect.any() else clean_max * 2
    thr = (clean_max + defect_min) / 2
    i = recipes.index(held)
    prev = recipes[i - 1] if i > 0 else None
    for _, r in df[df.recipe_id == held].iterrows():
        exp = predict(r, T, a.mix)
        e = dE(exp, bgr(r))
        flagged = e > thr
        cause = None
        if flagged:
            cause = "unexplained"
            if prev is not None:
                prev_ref = bgr(df[(df.recipe_id == prev) & (df.pos_in_batch == 3)].iloc[0])
                blend = from_space((1 - a.w) * to_space(exp, a.mix) + a.w * to_space(prev_ref, a.mix), a.mix)
                if dE(blend, bgr(r)) < e:
                    cause = f"carryover from R{prev}"
        out.append(dict(vial=r.vial, recipe=held, pos=r.pos_in_batch, dE=round(e, 1),
                        thr=round(thr, 1), flagged=flagged, truth=r.is_defect, cause=cause))

res = pd.DataFrame(out)
res.to_csv("loro_results.csv", index=False)
tp = ((res.flagged) & (res.truth)).sum(); fp = ((res.flagged) & (~res.truth)).sum()
fn = ((~res.flagged) & (res.truth)).sum()
caught = res[(res.flagged) & (res.truth)]
right_cause = (caught.cause == caught.recipe.map(lambda c: f"carryover from R{recipes[recipes.index(c)-1]}")).sum()
print(f"defects: {int(res.truth.sum())} | caught: {tp} | MISSED: {fn} | FALSE ALARMS: {fp} (of {int((~res.truth).sum())} clean vials)")
print(f"cause correct on caught defects: {right_cause}/{tp}")
print("\nper-vial results -> loro_results.csv")
print(res[(res.flagged) | (res.truth)].to_string(index=False))
