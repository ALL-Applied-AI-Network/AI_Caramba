"""Vial-vs-reference model (the AI part of Approach A): trained on the twin, scored once on real video 5.

Idea: don't ask "which recipe is this?" (absolute twin colors don't match the real camera, dE 16-52, Oct 24).
Ask "does this vial look different from its batch's clean reference vial?" The model sees TWO small crops - the vial
and the reference - both divided by the white paper next to the vial (common/measure.py), so camera exposure and
twin-vs-real brightness mostly cancel. Outputs: P(off-color) and the leftover fraction.

Data (twin_dataset_v2, made Oct 22):
  vial      = any image in labels.csv (clean, or with 5-100% leftover of another recipe)
  reference = a DIFFERENT random augmented copy of the clean render of the same recipe (like a different camera moment)
  label off = visibly different from clean: dE >= 7 between vial and clean reference, measured the corrected way
              (paper-normalized band, Oct 24). 7 = Approach A's cutoff from videos 1-4 noise. (Cruz's call Oct 22:
              "label off-color by visible difference".) Leftover target = the true carryover fraction from labels.csv.
Validation: leave-one-recipe-out on the twin (MAIC Dataset 2 testing), then ONE score on real video 5
(video5_eval pack: each vial vs its batch's last vial; truth = our eye check).

  python pair_model.py loro  --data twin_dataset_v2                      # 10 folds, prints misses / false alarms
  python pair_model.py train --data twin_dataset_v2 --out pair_model.pt  # all recipes
  python pair_model.py video5 --model pair_model.pt --pack video5_eval   # ONCE per model version
Runs on CPU in a few minutes (DGX Spark / any laptop with torch). Needs: torch, opencv-python, numpy, pandas.

--arch effnet (added Oct 26): same two-input idea with a pretrained EfficientNet-B0 (timm) as the shared encoder,
96x96 crops. Needs timm + a GPU to be quick (DGX Spark). Judge it by `loro` on the twin, NOT by video 5; score
video 5 once at the end. Checkpoints now store the arch, so `video5` and the operator UI load either kind.
"""
import argparse, os, sys, random
import cv2, numpy as np
import torch, torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))     # repo layout: melting-rainbow/lane6_classifier/pair_model/
sys.path.insert(0, HERE)                                       # or a flat folder with common/measure.py next to this file
from common.measure import band_rows, VIAL_X, PAPER, measure, lab_of_T

TOL = 7.0
SIZE = 24                       # small CNN input
SIZES = {"small": 24, "effnet": 96}
IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def crop(img, total_ml, size=SIZE):
    """Paper-normalized crop of the liquid band (same rows / columns as common/measure.py), SIZE x SIZE, float 0-1.2."""
    r0, r1 = band_rows(total_ml)
    p = PAPER
    paper = np.median(img[p[0]:p[1], p[2]:p[3]].reshape(-1, 3), axis=0).astype(np.float32)
    c = img[r0:r1, VIAL_X[0] - 4:VIAL_X[1] + 4].astype(np.float32) / np.maximum(paper, 1)
    c = cv2.resize(np.clip(c, 0, 1.2), (size, size), interpolation=cv2.INTER_AREA)
    return c[..., ::-1].transpose(2, 0, 1).copy()            # RGB, channels first


def load_twin(root, size=SIZE):
    import pandas as pd             # only needed for training / scoring, not by the operator station
    lab = pd.read_csv(os.path.join(root, "labels.csv"))
    X, T = [], []
    for r in lab.itertuples():
        img = cv2.imread(os.path.join(root, r.image))
        X.append(crop(img, r.total_ml, size)); T.append(measure(img, r.total_ml)["T"])
    lab["crop_i"] = range(len(lab))
    return lab, np.stack(X), np.stack(T)


def make_pairs(lab, X, T, recipes, rng):
    """One pair per vial image: (vial crop, crop of a different clean copy of the same recipe)."""
    clean = {rid: g.crop_i.tolist() for rid, g in lab[lab.kind == "clean"].groupby("recipe_id")}
    A, B, y, w = [], [], [], []
    for r in lab[lab.recipe_id.isin(recipes)].itertuples():
        choices = [i for i in clean[r.recipe_id] if i != r.crop_i]
        j = rng.choice(choices)
        dE = np.linalg.norm(lab_of_T(np.clip(T[r.crop_i], 0, 1)) - lab_of_T(np.clip(T[j], 0, 1)))
        A.append(X[r.crop_i]); B.append(X[j]); y.append(float(dE >= TOL)); w.append(float(r.carryover))
    return (torch.tensor(np.stack(A)), torch.tensor(np.stack(B)), torch.tensor(y), torch.tensor(w))


class PairNet(nn.Module):
    """Shared small CNN on each crop; head sees [vial, reference, vial - reference] features.
    relative=True: head sees only [vial - reference, |vial - reference|], so it can't learn what each recipe's
    absolute color looks like - only how the vial differs from its own reference (added Oct 26)."""
    def __init__(self, relative=False):
        super().__init__()
        self.relative = relative
        self.enc = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Linear(64 if relative else 96, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        f = torch.cat([fa - fb, (fa - fb).abs()], 1) if self.relative else torch.cat([fa, fb, fa - fb], 1)
        o = self.head(f)
        return o[:, 0], torch.sigmoid(o[:, 1])                # off-color logit, leftover fraction


class PairEffNet(nn.Module):
    """Same idea with a pretrained EfficientNet-B0 encoder (shared by vial and reference)."""
    def __init__(self, pretrained=True, relative=False, freeze=False):
        super().__init__()
        import timm
        self.relative = relative
        self.enc = timm.create_model("efficientnet_b0", pretrained=pretrained, num_classes=0)   # 1280 features
        if freeze:                                    # use ImageNet features as-is, train only the small head
            for q in self.enc.parameters():
                q.requires_grad = False
        self.head = nn.Sequential(nn.Linear((2 if relative else 3) * 1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, a, b):
        a = (a - IMNET_MEAN.to(a.device)) / IMNET_STD.to(a.device)
        b = (b - IMNET_MEAN.to(b.device)) / IMNET_STD.to(b.device)
        fa, fb = self.enc(a), self.enc(b)
        f = torch.cat([fa - fb, (fa - fb).abs()], 1) if self.relative else torch.cat([fa, fb, fa - fb], 1)
        o = self.head(f)
        return o[:, 0], torch.sigmoid(o[:, 1])


def make_net(arch, pretrained=True, relative=False, freeze=False):
    return PairEffNet(pretrained, relative, freeze) if arch == "effnet" else PairNet(relative)


def save_model(net, arch, path):
    torch.save({"arch": arch, "size": SIZES[arch], "relative": net.relative, "state_dict": net.state_dict()}, path)


def load_model(path, device="cpu"):
    """Returns (net, arch, crop size). Reads new checkpoints and the Oct 26 small-CNN state_dict."""
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "arch" in obj:
        net = make_net(obj["arch"], pretrained=False, relative=obj.get("relative", False))
        net.load_state_dict(obj["state_dict"])
        return net.to(device).eval(), obj["arch"], obj["size"]
    net = PairNet(); net.load_state_dict(obj)
    return net.to(device).eval(), "small", SIZE


def fit(data, epochs=25, seed=0, arch="small", pretrained=True, relative=False, freeze=False, color_jitter=0.0):
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A, B, y, w = data[:4]
    # optional 5th item: 1 where the leftover target is known (twin), 0 where it isn't (real operator examples)
    m = data[4] if len(data) > 4 else torch.ones_like(w)
    net = make_net(arch, pretrained, relative, freeze).to(dev)
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.Adam(params, 2e-3) if arch == "small" else torch.optim.AdamW(params, 1e-3 if freeze else 3e-4)
    bs = 64 if arch == "small" else 32
    pos = float(y.mean()); pw = torch.tensor((1 - pos) / max(pos, 1e-3)).to(dev)
    for _ in range(epochs):
        net.train()
        if freeze and arch == "effnet":
            net.enc.eval()                            # frozen backbone: keep its BatchNorm statistics fixed too
        perm = torch.randperm(len(y))
        for k in range(0, len(y), bs):
            i = perm[k:k + bs]
            # augmentation: small per-image brightness change (camera drift between vial and reference)
            ga = 1 + 0.06 * torch.randn(len(i), 1, 1, 1); gb = 1 + 0.06 * torch.randn(len(i), 1, 1, 1)
            # color jitter applied the SAME way to vial and reference: changes the absolute color (like twin vs
            # real camera) but not the difference, so the model has to judge the difference
            gc = (1 + color_jitter * torch.randn(len(i), 3, 1, 1)).clamp(0.3, 3) if color_jitter else 1.0
            logit, wl = net((A[i] * ga * gc).to(dev), (B[i] * gb * gc).to(dev))
            yi, wi, mi = y[i].to(dev), w[i].to(dev), m[i].to(dev)
            mse = ((wl - wi) ** 2 * mi).sum() / mi.sum().clamp(min=1.0)    # = mse_loss when every target is known
            loss = nn.functional.binary_cross_entropy_with_logits(logit, yi, pos_weight=pw) + 2.0 * mse
            opt.zero_grad(); loss.backward(); opt.step()
    return net.eval().cpu()


def predict(net, A, B, bs=128):
    out_l, out_w = [], []
    with torch.no_grad():
        for k in range(0, len(A), bs):
            l, w = net(A[k:k + bs], B[k:k + bs]); out_l.append(l); out_w.append(w)
    return torch.cat(out_l), torch.cat(out_w)


def score(net, data):
    A, B, y, w = data
    logit, wl = predict(net, A, B)
    pred = (logit > 0).float()
    return dict(n=len(y), off=int(y.sum()), misses=int(((y == 1) & (pred == 0)).sum()),
                false_alarms=int(((y == 0) & (pred == 1)).sum()), leftover_mae=float((wl - w).abs().mean()))


def main():
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["loro", "train", "video5"])
    ap.add_argument("--data", help="unzipped twin_dataset_v2 folder")
    ap.add_argument("--model", default="pair_model.pt")
    ap.add_argument("--out", default="pair_model.pt")
    ap.add_argument("--pack", default="video5_eval")
    ap.add_argument("--epochs", type=int, help="default 25 (small) / 8 (effnet)")
    ap.add_argument("--arch", choices=["small", "effnet"], default="small")
    ap.add_argument("--no-pretrained", action="store_true", help="effnet without downloading ImageNet weights")
    ap.add_argument("--relative", action="store_true",
                    help="head sees only vial-minus-reference features (can't memorize recipe colors)")
    ap.add_argument("--freeze", action="store_true", help="effnet: keep ImageNet features fixed, train only the head")
    ap.add_argument("--color-jitter", type=float, default=0.0,
                    help="random per-channel color change applied identically to vial and reference, e.g. 0.3")
    a = ap.parse_args()
    rng = random.Random(0)
    epochs = a.epochs or (25 if a.arch == "small" else 8)
    pre = not a.no_pretrained

    if a.mode in ("loro", "train"):
        lab, X, T = load_twin(a.data, SIZES[a.arch])
        recipes = sorted(lab.recipe_id.unique())
        if a.mode == "loro":
            rows = []
            for hold in recipes:
                net = fit(make_pairs(lab, X, T, [r for r in recipes if r != hold], rng), epochs, arch=a.arch, pretrained=pre,
                          relative=a.relative, freeze=a.freeze, color_jitter=a.color_jitter)
                s = score(net, make_pairs(lab, X, T, [hold], rng)); s["held_out"] = f"R{hold}"; rows.append(s)
                print(f"held out R{hold}: {s['n']} vials ({s['off']} off)  misses {s['misses']}  "
                      f"false alarms {s['false_alarms']}  leftover MAE {s['leftover_mae']:.3f}", flush=True)
            tag = ("" if a.arch == "small" else "_" + a.arch) + ("_rel" if a.relative else "") + \
                  ("_frz" if a.freeze else "") + (f"_cj{a.color_jitter:g}" if a.color_jitter else "")
            out_csv = f"pair_model_loro{tag}.csv"
            d = pd.DataFrame(rows); d.to_csv(out_csv, index=False)
            print(f"TOTAL leave-one-recipe-out: {d.n.sum()} vials, misses {d.misses.sum()}/{d.off.sum()}, "
                  f"false alarms {d.false_alarms.sum()}/{d.n.sum() - d.off.sum()}, leftover MAE {d.leftover_mae.mean():.3f}"
                  f"  -> {out_csv}")
        else:
            net = fit(make_pairs(lab, X, T, recipes, rng), epochs, arch=a.arch, pretrained=pre,
                          relative=a.relative, freeze=a.freeze, color_jitter=a.color_jitter)
            save_model(net, a.arch, a.out); print("saved", a.out, f"({a.arch})")
        return

    # video5: score ONCE. Each vial vs its batch's last vial (the reference), like Approach A.
    net, arch, size = load_model(a.model)
    ev = pd.read_csv(os.path.join(a.pack, "video5_eval.csv"))
    ref_vial = ev.groupby("recipe_id").vial.max().to_dict()
    C = {r.vial: crop(cv2.imread(os.path.join(a.pack, r.image)), r.total_ml, size) for r in ev.itertuples()}
    rows = []
    for r in ev.itertuples():
        rv = ref_vial[r.recipe_id]
        if r.vial == rv:
            continue
        with torch.no_grad():
            logit, wl = net(torch.tensor(C[r.vial])[None], torch.tensor(C[rv])[None])
        p = float(torch.sigmoid(logit)); pred = p >= 0.5
        rows.append(dict(vial=r.vial, recipe=r.recipe_id, reference_vial=rv, p_off=round(p, 3),
                         prediction="off-color" if pred else "good",
                         truth="off-color" if r.truth_off_color_by_eye == 1 else "good",
                         leftover_est=round(float(wl), 2), residual_ml_est=round(float(wl) * r.total_ml, 1)))
    out_csv = f"pair_model_video5{'' if arch == 'small' else '_' + arch}.csv"
    d = pd.DataFrame(rows); d.to_csv(out_csv, index=False)
    print(d.to_string(index=False))
    t, p = d.truth == "off-color", d.prediction == "off-color"
    print(f"\nvideo 5 ({len(d)} judged vials; each batch's last vial is the reference): "
          f"misses {(t & ~p).sum()}/{t.sum()}, false alarms {(~t & p).sum()}/{(~t).sum()}  -> {out_csv}  [{arch}]")
    print("Compare: Approach A = 0 misses, 2 false alarms (vials 1-2).")


if __name__ == "__main__":
    main()
