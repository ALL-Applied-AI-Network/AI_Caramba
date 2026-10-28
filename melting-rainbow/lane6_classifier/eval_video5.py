"""Score approach A once on real video 5 (validation). Needs the corrected measurements first:
   python lane6_classifier/eval_video5.py <dataset_root> <truth_by_eye.csv>
Reference vial of each batch = its last vial (organizers' clean-reference convention); previous recipe's
reference gives the leftover direction. Nothing here is fitted on video 5.
"""
import sys, os, cv2, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.measure import measure, lab_of_T
from lane6_classifier.approach_a import TOL

root, truth_csv = sys.argv[1], sys.argv[2]
rec = pd.read_csv(os.path.join(root, "vial_fill", "labels", "video5_vial_recipes.csv"))
cap = cv2.VideoCapture(os.path.join(root, "vial_fill", "videos", "vial_fill_video_5.mp4"))
lin = lambda v: np.where(v / 255 <= 0.04045, (v / 255) / 12.92, ((v / 255 + 0.055) / 1.055) ** 2.4)
M = {}
for _, r in rec.iterrows():
    Ts, liq, pap = [], [], []
    for off in (21, 15, 9):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(r.end_frame) - off); m = measure(cap.read()[1], r.total_ml)
        Ts.append(m["T"]); liq.append(m["liquid_rgb"]); pap.append(m["paper_rgb"])
    T, L, P = np.median(Ts, 0), np.median(liq, 0), np.median(pap, 0)
    M[int(r.vial)] = dict(T=T, OD=-np.log(np.clip(lin(L) / lin(P), 1e-4, None)), ml=r.total_ml, rid=int(r.recipe_id))
order = list(dict.fromkeys(rec.recipe_id.astype(int)))
ref = {rid: int(rec[rec.recipe_id == rid].vial.max()) for rid in order}
truth = pd.read_csv(truth_csv).set_index("vial")
rows = []
for v, m in M.items():
    rid = m["rid"]; rv = ref[rid]
    dE = float(np.linalg.norm(lab_of_T(np.clip(m["T"], 0, 1)) - lab_of_T(np.clip(M[rv]["T"], 0, 1))))
    pred = "off-color" if dE >= TOL else "good"
    w = cause = ok = None
    if order.index(rid) > 0:
        prev = order[order.index(rid) - 1]
        d = m["OD"] - M[rv]["OD"]; u = M[ref[prev]]["OD"] - M[rv]["OD"]
        w = float(np.clip(d @ u / (u @ u), 0, 1))
        if pred == "off-color":
            best = min(order[:order.index(rid)], key=lambda c: np.linalg.norm(
                d - np.clip(d @ (M[ref[c]]["OD"] - M[rv]["OD"]) / ((M[ref[c]]["OD"] - M[rv]["OD"]) @ (M[ref[c]]["OD"] - M[rv]["OD"])), 0, 1)
                * (M[ref[c]]["OD"] - M[rv]["OD"])))
            cause, ok = f"R{best}", best == prev
    rows.append(dict(vial=v, recipe=rid, is_reference=v == rv, dE_ref=round(dE, 1), prediction=pred,
                     truth_by_eye="off-color" if truth.loc[v, "is_defect"] else "good",
                     leftover_est=None if w is None else round(w, 2),
                     residual_ml_est=None if w is None else round(w * m["ml"], 1), cause=cause, cause_correct=ok))
out = pd.DataFrame(rows); out.to_csv("approach_a_video5.csv", index=False)
j = out[~out.is_reference]
print(out.to_string(index=False))
print(f"\ncutoff dE {TOL} | judged {len(j)} | misses {((j.truth_by_eye=='off-color')&(j.prediction=='good')).sum()}"
      f" | false alarms {((j.truth_by_eye=='good')&(j.prediction=='off-color')).sum()}"
      f" | cause correct {int((out.cause_correct==True).sum())}/{int(out.cause.notna().sum())}")
