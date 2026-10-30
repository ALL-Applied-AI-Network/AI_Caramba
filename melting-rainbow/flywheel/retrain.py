"""Gated retraining (the flywheel's second half, Oct 30). Run BETWEEN SHIFTS, never while the line runs.

  1. Train a CANDIDATE comparison model on the digital twin + the operator-confirmed real vials
     (flywheel/examples.py writes those when the operator clicks Pass / Reject in the station).
  2. Score the CURRENT model and the candidate on the same fixed gate set:
       - the video 5 pack (each vial vs its batch's last vial, truth = our eye check), and
       - optionally real examples passed with --gate-examples (kept OUT of training).
  3. The candidate replaces the current model only if it has no more misses AND no more false alarms than the
     current model on that gate set, and only with --promote. The old model is kept in <out-dir>/history/.
     Otherwise the station keeps the current model. Every run writes a report either way.

  python flywheel/retrain.py --twin twin_dataset_v2 --examples results/operator_output \
      --gate-pack video5_eval --current operator_ui/pair_model.pt            # dry run: report only
  ... add --promote to let a candidate that passes the gate replace operator_ui/pair_model.pt

Runs on a laptop CPU in a few minutes (small model). Needs torch, opencv-python, numpy, pandas.
"""
import argparse, json, os, random, shutil, sys, time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "lane6_classifier", "pair_model"))
import pair_model as pm                      # noqa: E402
from flywheel.examples import load_examples  # noqa: E402


def real_pairs(examples, size):
    """Operator examples -> (vial crops, reference crops, label, leftover 0, mask 0: leftover unknown)."""
    A, B, y = [], [], []
    for r in examples:
        a, b = cv2.imread(r["vial_image"]), cv2.imread(r["ref_image"])
        if a is None or b is None:
            print(f"  skipped (image missing): run {r['run_id']} vial {r['vial']}")
            continue
        A.append(pm.crop(a, float(r["vial_ml"]), size)); B.append(pm.crop(b, float(r["ref_ml"]), size))
        y.append(1.0 if r["label"] == "off" else 0.0)
    if not A:
        return None
    n = len(y)
    return (torch.tensor(np.stack(A)), torch.tensor(np.stack(B)), torch.tensor(y),
            torch.zeros(n), torch.zeros(n))


def video5_pairs(pack, size):
    """Video 5 pack -> (vial crops, reference crops, truth). Reference = each batch's last vial; it isn't judged."""
    import pandas as pd
    ev = pd.read_csv(os.path.join(pack, "video5_eval.csv"))
    ref_vial = ev.groupby("recipe_id").vial.max().to_dict()
    img = {r.vial: (cv2.imread(os.path.join(pack, r.image)), r.total_ml) for r in ev.itertuples()}
    A, B, y = [], [], []
    for r in ev.itertuples():
        rv = ref_vial[r.recipe_id]
        if r.vial == rv:
            continue
        A.append(pm.crop(img[r.vial][0], img[r.vial][1], size)); B.append(pm.crop(img[rv][0], img[rv][1], size))
        y.append(float(r.truth_off_color_by_eye))
    return torch.tensor(np.stack(A)), torch.tensor(np.stack(B)), torch.tensor(y)


def gate_score(net, sets):
    """Misses / false alarms per gate set and in total (off-color if P(off) >= 50%, as in the station)."""
    out, tot = {}, dict(n=0, off=0, misses=0, false_alarms=0)
    for name, (A, B, y) in sets.items():
        logit, _ = pm.predict(net, A, B)
        pred = logit > 0
        t = y == 1
        s = dict(n=len(y), off=int(t.sum()), misses=int((t & ~pred).sum()), false_alarms=int((~t & pred).sum()))
        out[name] = s
        for k in tot:
            tot[k] += s[k]
    out["total"] = tot
    return out


def concat(*parts):
    parts = [p for p in parts if p is not None]
    return tuple(torch.cat([p[i] for p in parts]) for i in range(5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--twin", required=True, help="unzipped twin_dataset_v2 folder")
    ap.add_argument("--examples", nargs="*", default=[], help="operator_output folders (or examples.csv) to TRAIN on")
    ap.add_argument("--gate-pack", required=True, help="video5_eval folder (never trained on)")
    ap.add_argument("--gate-examples", nargs="*", default=[], help="real examples used ONLY for the gate")
    ap.add_argument("--current", default=os.path.join(ROOT, "operator_ui", "pair_model.pt"))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(ROOT), "results", "flywheel_models"))
    ap.add_argument("--real-repeat", type=int, default=10,
                    help="each real example counts this many times (a few real vials vs 2,000 twin images)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--promote", action="store_true", help="replace --current if the candidate passes the gate")
    a = ap.parse_args()

    stamp = time.strftime("%Y%m%d_%H%M%S")
    os.makedirs(a.out_dir, exist_ok=True)
    cur, arch, size = pm.load_model(a.current)
    relative = getattr(cur, "relative", False)
    print(f"current model: {a.current} ({arch}, relative={relative})")

    # ---- training data: twin + operator examples (oversampled)
    train_ex = load_examples(a.examples)
    gate_ex = load_examples(a.gate_examples)
    overlap = {(r["run_id"], r["vial"]) for r in train_ex} & {(r["run_id"], r["vial"]) for r in gate_ex}
    if overlap:
        sys.exit(f"the same vials are in --examples and --gate-examples ({len(overlap)}): a model can't be judged "
                 f"on what it trained on")
    lab, X, T = pm.load_twin(a.twin, size)
    twin = pm.make_pairs(lab, X, T, sorted(lab.recipe_id.unique()), random.Random(a.seed))
    twin = twin + (torch.ones(len(twin[2])),)
    real = real_pairs(train_ex, size)
    n_real = 0 if real is None else len(real[2])
    data = concat(twin, *([real] * a.real_repeat if real is not None else []))
    print(f"training: {len(twin[2])} twin pairs + {n_real} real examples x{a.real_repeat} "
          f"({0 if real is None else int(real[2].sum())} off-color)")
    if n_real == 0:
        print("  no real examples yet - the candidate is a plain twin retrain")

    torch.manual_seed(a.seed)
    cand = pm.fit(data, a.epochs, seed=a.seed, arch=arch, relative=relative)
    cand_path = os.path.join(a.out_dir, f"pair_model_candidate_{stamp}.pt")
    pm.save_model(cand, arch, cand_path)

    # ---- gate: same fixed test set for both models
    sets = {"video5": video5_pairs(a.gate_pack, size)}
    g = real_pairs(gate_ex, size)
    if g is not None:
        sets["real_holdout"] = g[:3]
    s_cur, s_cand = gate_score(cur, sets), gate_score(cand, sets)
    c, n = s_cur["total"], s_cand["total"]
    passed = n["misses"] <= c["misses"] and n["false_alarms"] <= c["false_alarms"]

    promoted = False
    if passed and a.promote:
        hist = os.path.join(a.out_dir, "history")
        os.makedirs(hist, exist_ok=True)
        shutil.copy2(a.current, os.path.join(hist, f"pair_model_replaced_{stamp}.pt"))
        shutil.copy2(cand_path, a.current)
        promoted = True

    verdict = ("PROMOTED - the station uses the new model from its next start" if promoted else
               "PASSED the gate - run again with --promote to use it" if passed else
               "KEPT the current model - the candidate was worse on the gate set")
    report = dict(time=stamp, current=os.path.abspath(a.current), candidate=cand_path, arch=arch,
                  twin_pairs=len(twin[2]), real_examples=n_real, real_repeat=a.real_repeat,
                  gate_sets={k: v["n"] for k, v in sets.items()}, current_scores=s_cur, candidate_scores=s_cand,
                  passed=passed, promoted=promoted, verdict=verdict,
                  rule="candidate misses <= current misses AND candidate false alarms <= current false alarms")
    with open(os.path.join(a.out_dir, f"retrain_report_{stamp}.json"), "w") as f:
        json.dump(report, f, indent=2)
    lines = [f"# Retrain {stamp}", "", f"Training: {len(twin[2])} twin pairs + {n_real} real operator examples "
             f"(x{a.real_repeat})", "", "| Gate set | Vials | Off-color | Current: misses / false alarms | "
             "Candidate: misses / false alarms |", "|---|---|---|---|---|"]
    for k in s_cur:
        lines.append(f"| {k} | {s_cur[k]['n']} | {s_cur[k]['off']} | {s_cur[k]['misses']} / {s_cur[k]['false_alarms']}"
                     f" | {s_cand[k]['misses']} / {s_cand[k]['false_alarms']} |")
    lines += ["", f"**{verdict}**", "", f"Rule: {report['rule']}."]
    with open(os.path.join(a.out_dir, f"retrain_report_{stamp}.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
