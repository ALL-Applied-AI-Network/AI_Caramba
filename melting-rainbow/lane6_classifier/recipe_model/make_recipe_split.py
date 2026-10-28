"""Make an HONEST train / val / test split of our twin dataset for a recipe classifier (Frank's EfficientNet).

Two problems with splitting twin_dataset_v2/images/good at random (found Oct 25):
  1. Copies leak. 600 of the 1,315 "good" images are 60 augmented copies each of only 10 clean renders, so a random
     split puts copies of the SAME render in train and test. Fix: split by render (labels.csv column raw_image) -
     all copies of one render go to the same split.
  2. Wrong recipe labels. 282 "good" images are >= 50% the PREVIOUS recipe's liquid (they were called "good"
     because in the twin they looked within dE 5 of clean). Fine for "does it look off?", wrong for "which recipe?".
     Fix: keep only vials with leftover < --max-leftover (default 0.10), whatever their good/off label.

Split rule (per recipe): the clean render's copies always go to TRAIN (there is only one clean render per recipe).
The other renders are shuffled (fixed --seed) and dealt out by render: test first, then val, then the rest to train,
so every recipe with >= 3 renders gets at least one test and one val render.

Run:  python make_recipe_split.py --data twin_dataset_v2 --out recipe_split
Out:  recipe_split/{train,val,test}/recipe_<id>/*.jpg   (same folder layout torchvision ImageFolder expects)
      recipe_split/split_manifest.csv                  (which image / render went where, with its leftover)
"""
import argparse, collections, csv, os, random, shutil

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="the unzipped twin_dataset_v2 folder (has labels.csv and images/)")
ap.add_argument("--out", default="recipe_split")
ap.add_argument("--max-leftover", type=float, default=0.10)
ap.add_argument("--test-frac", type=float, default=0.2)
ap.add_argument("--val-frac", type=float, default=0.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

rows = list(csv.DictReader(open(os.path.join(a.data, "labels.csv"), newline="")))
keep = [r for r in rows if float(r["carryover"]) < a.max_leftover]
renders = collections.defaultdict(lambda: collections.defaultdict(list))   # recipe -> raw_image -> rows
for r in keep:
    renders[int(r["recipe_id"])][r["raw_image"]].append(r)

rng = random.Random(a.seed)
plan = []
for rid in sorted(renders):
    clean = [k for k, v in renders[rid].items() if v[0]["kind"] == "clean"]
    other = sorted(k for k in renders[rid] if k not in clean)
    rng.shuffle(other)
    n_test = max(1, round(a.test_frac * len(other))) if len(other) >= 2 else 0
    n_val = max(1, round(a.val_frac * len(other))) if len(other) - n_test >= 2 else 0
    for i, k in enumerate(other):
        split = "test" if i < n_test else "val" if i < n_test + n_val else "train"
        plan += [(split, rid, k, r) for r in renders[rid][k]]
    plan += [("train", rid, k, r) for k in clean for r in renders[rid][k]]

if os.path.exists(a.out):
    shutil.rmtree(a.out)
with open(os.path.join(os.makedirs(a.out) or a.out, "split_manifest.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["split", "recipe_id", "raw_image", "image", "kind", "prev_recipe_id", "carryover"])
    for split, rid, k, r in plan:
        dst = os.path.join(a.out, split, f"recipe_{rid}")
        os.makedirs(dst, exist_ok=True)
        shutil.copy2(os.path.join(a.data, r["image"]), dst)
        w.writerow([split, rid, k, r["image"], r["kind"], r["prev_recipe_id"], r["carryover"]])

print(f"kept {len(keep)} of {len(rows)} images (leftover < {a.max_leftover})")
print(f"{'recipe':>7} | " + " | ".join(f"{s:>16}" for s in ("train img/renders", "val img/renders", "test img/renders")))
for rid in sorted(renders):
    cells = []
    for s in ("train", "val", "test"):
        p = [x for x in plan if x[0] == s and x[1] == rid]
        cells.append(f"{len(p):>9} / {len({x[2] for x in p}):<6}")
    print(f"{'R' + str(rid):>7} | " + " | ".join(cells))
tot = {s: (sum(x[0] == s for x in plan), len({x[2] for x in plan if x[0] == s})) for s in ("train", "val", "test")}
print("total   | " + " | ".join(f"{tot[s][0]:>9} / {tot[s][1]:<6}" for s in ("train", "val", "test")))
print(f"wrote {a.out}/ (train, val, test) and split_manifest.csv")
