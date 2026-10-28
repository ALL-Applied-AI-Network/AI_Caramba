"""Score a RECIPE classifier (Frank's EfficientNet-B0) ONCE on real video 5. Never train or tune on these images.

Uses our validation pack (03_validation_video5/video5_eval.zip -> video5_eval/frames + video5_eval.csv):
30 real frames, one per vial, with the recipe that SHOULD be in the vial, the recipe before the batch, and our
eye-check truth (10 off-color vials: 4, 5, 7, 10, 16, 19, 22, 25, 28, 29).

How a recipe classifier becomes an off-color check (so it can be compared with Approach A):
  flag OFF-COLOR when the predicted recipe != the planned recipe
  cause = the predicted recipe (it "looks like" that recipe); right if it equals the previous batch's recipe
Reported separately, as the MAIC Dataset 2 page asks: misses, false alarms, cause - plus plain recipe accuracy
on the 20 clean vials (the twin -> real check).

Run (in the same venv as train.py, from the folder with the weights):
  python score_video5_recipe.py --weights model.pth --pack video5_eval
!! The image preprocessing below must MATCH evaluate.py (resize size, normalization). Defaults: 224x224,
   ImageNet mean/std, timm efficientnet_b0 - the usual setup. Change --size / --arch / --no-normalize if not.
Out: video5_recipe_scores.csv + a printed summary. Save both with the model version and date.
"""
import argparse, csv, os, sys
import torch
from PIL import Image
from torchvision import transforms

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True, help="saved model: a state_dict (.pth) or a whole saved model")
ap.add_argument("--pack", default="video5_eval", help="unzipped video5_eval folder")
ap.add_argument("--arch", default="efficientnet_b0", help="timm model name used in train.py")
ap.add_argument("--size", type=int, default=224)
ap.add_argument("--no-normalize", action="store_true", help="use if evaluate.py does NOT normalize with ImageNet mean/std")
ap.add_argument("--classes", nargs="+",
                default=["recipe_10", "recipe_11", "recipe_12", "recipe_13", "recipe_14",
                         "recipe_5", "recipe_6", "recipe_7", "recipe_8", "recipe_9"],
                help="class order printed by train.py (ImageFolder sorts folder names alphabetically)")
ap.add_argument("--out", default="video5_recipe_scores.csv")
a = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
obj = torch.load(a.weights, map_location=dev, weights_only=False)
if isinstance(obj, torch.nn.Module):
    model = obj
else:
    import timm
    sd = obj.get("state_dict", obj.get("model_state_dict", obj)) if isinstance(obj, dict) else obj
    sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    model = timm.create_model(a.arch, pretrained=False, num_classes=len(a.classes))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        sys.exit(f"weights don't fit {a.arch} with {len(a.classes)} classes: missing {missing[:3]}, unexpected {unexpected[:3]}")
model.eval().to(dev)

tf = [transforms.Resize((a.size, a.size)), transforms.ToTensor()]
if not a.no_normalize:
    tf.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
tf = transforms.Compose(tf)
cls_id = [int(c.split("_")[-1]) for c in a.classes]

rows = list(csv.DictReader(open(os.path.join(a.pack, "video5_eval.csv"), newline="")))
out = []
with torch.no_grad():
    for r in rows:
        x = tf(Image.open(os.path.join(a.pack, r["image"])).convert("RGB")).unsqueeze(0).to(dev)
        p = torch.softmax(model(x), 1)[0].cpu()
        k = int(p.argmax())
        pred, planned = cls_id[k], int(r["recipe_id"])
        prev = int(r["prev_recipe_id"]) if r["prev_recipe_id"] else None
        truth_off = r["truth_off_color_by_eye"] == "1"
        flag = pred != planned
        out.append(dict(vial=int(r["vial"]), planned_recipe=planned, prev_recipe=prev, truth=("off-color" if truth_off else "good"),
                        predicted_recipe=pred, confidence=round(float(p[k]), 3),
                        prob_planned=round(float(p[cls_id.index(planned)]), 3),
                        flag=("OFF-COLOR" if flag else "good"),
                        cause_right=(pred == prev) if (flag and truth_off and prev is not None) else ""))

with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)

good = [o for o in out if o["truth"] == "good"]
off = [o for o in out if o["truth"] == "off-color"]
print(f"{'vial':>4} {'plan':>5} {'pred':>5} {'conf':>5}  truth      flag")
for o in out:
    print(f"{o['vial']:>4} {'R' + str(o['planned_recipe']):>5} {'R' + str(o['predicted_recipe']):>5} {o['confidence']:>5}  "
          f"{o['truth']:<10} {o['flag']}" + (f"  cause {'right' if o['cause_right'] else 'wrong'}" if o["cause_right"] != "" else ""))
caught = [o for o in off if o["flag"] == "OFF-COLOR"]
print(f"\nRecipe accuracy on the {len(good)} clean vials (twin -> real check): "
      f"{sum(o['predicted_recipe'] == o['planned_recipe'] for o in good)}/{len(good)}")
print(f"Misses:       {len(off) - len(caught)}/{len(off)} off-color vials called good")
print(f"False alarms: {sum(o['flag'] == 'OFF-COLOR' for o in good)}/{len(good)} good vials flagged")
print(f"Cause right:  {sum(o['cause_right'] is True for o in caught)}/{len(caught)} caught vials named the previous recipe")
print("Compare: Approach A on the same 30 vials = 0 misses, 2 false alarms, cause 6/10 (live demo 7/10).")
print(f"Saved {a.out}")
