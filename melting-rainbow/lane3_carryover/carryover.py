"""Lane 3: explain each first-vial color using the recipe that ran before it.

For every recipe change, models  first_vial = (1 - w) * current_clean + w * residue
and compares two hypotheses for the residue:
  H1: the previous recipe's final MIXTURE color (its clean reference vial)
  H2: the previous recipe's LAST-DISPENSED tank color (estimated from the references)
Fits one shared w per hypothesis (least squares) and reports which explains the
first vials better (lower dE). Also prints a predicted first-vial dE for every
recipe pair -> the "flush recommended" table.

Usage: python lane3_carryover/carryover.py vial_colors.csv [--mix rgb|od]
"""
import argparse, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common.mixing import to_space, from_space, fit_tank_colors, last_tank, dE

ap = argparse.ArgumentParser()
ap.add_argument("colors"); ap.add_argument("--mix", default="rgb", choices=["rgb", "od"])
a = ap.parse_args()
df = pd.read_csv(a.colors)
refs = df[df.pos_in_batch == 3].set_index("recipe_id")
T = fit_tank_colors([r for _, r in refs.iterrows()], a.mix)
bgr = lambda r: np.array([r["B"], r["G"], r["R"]], float)

recipes = sorted(refs.index)
trans = []
for prev, cur in zip(recipes, recipes[1:]):
    first = df[(df.recipe_id == cur) & (df.pos_in_batch == 1)].iloc[0]
    trans.append(dict(prev=prev, cur=cur,
                      first=to_space(bgr(first), a.mix),
                      clean=to_space(bgr(refs.loc[cur]), a.mix),
                      H1=to_space(bgr(refs.loc[prev]), a.mix),
                      H2=T[last_tank(refs.loc[prev].dispense_order)],
                      first_bgr=bgr(first)))

def fit_w(h):
    num = sum(np.dot(t["first"] - t["clean"], t[h] - t["clean"]) for t in trans)
    den = sum(np.dot(t[h] - t["clean"], t[h] - t["clean"]) for t in trans)
    return float(np.clip(num / den, 0, 1))

res = {}
for h in ("H1", "H2"):
    w = fit_w(h)
    errs = [dE(from_space((1 - w) * t["clean"] + w * t[h], a.mix), t["first_bgr"]) for t in trans]
    res[h] = (w, errs)
    print(f"{h}: w = {w:.2f}  mean dE(predicted vs actual first vial) = {np.mean(errs):.2f}")
noco = [dE(from_space(t["clean"], a.mix), t["first_bgr"]) for t in trans]
print(f"no-carryover baseline: mean dE = {np.mean(noco):.2f}")

best = min(res, key=lambda h: np.mean(res[h][1]))
print(f"\nbest hypothesis: {best}\n\nper transition (dE of predicted first vial vs actual):")
for t, e1, e2, e0 in zip(trans, res["H1"][1], res["H2"][1], noco):
    print(f"  R{t['prev']:>2} -> R{t['cur']:>2}:  no-carryover {e0:5.1f} | H1 {e1:5.1f} | H2 {e2:5.1f}")

# Risk table: predicted first-vial dE for ANY recipe pair (before filling)
w = res[best][0]
risk = pd.DataFrame(index=recipes, columns=recipes, dtype=float)
for p in recipes:
    for c in recipes:
        if p == c: continue
        clean = to_space(bgr(refs.loc[c]), a.mix)
        resid = to_space(bgr(refs.loc[p]), a.mix) if best == "H1" else T[last_tank(refs.loc[p].dispense_order)]
        risk.loc[p, c] = dE(from_space((1 - w) * clean + w * resid, a.mix), bgr(refs.loc[c]))
risk.to_csv("transition_risk.csv")
print("\nwrote transition_risk.csv (rows = previous recipe, cols = next recipe, value = predicted first-vial dE)")
