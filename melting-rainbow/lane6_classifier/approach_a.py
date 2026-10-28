"""Approach A: decide off-color by COMPARISON, estimate the leftover, convert it to mL.

For a vial of recipe r that ran right after recipe p:
  ref_r  = the batch's own reference vial (the organizers' dataset page uses each recipe's LAST vial as the
           clean reference); ref_p = the previous batch's reference vial. Both come from the line itself, so
           camera / twin color differences cancel out.
  colors are measured with common/measure.py (liquid in front of the white paper / paper) -> OD = -ln(T)
  dE_ref  = perceptual difference between the vial and ref_r              -> off-color if >= TOL (7, from videos 1-4 noise)
  w_hat   = leftover fraction: how far the vial's OD has moved from ref_r toward ref_p
            (projection of OD_vial - OD_r onto OD_p - OD_r, clipped 0..1)   (mixture model H1)
  perp    = distance not explained by that leftover (large -> some other cause)
  residual_ml = w_hat * vial volume
The twin is used to CHECK the leftover estimator (true w is known there) and to set the correction.
"""
import numpy as np

# Off-color cutoff, set from INDEPENDENT real data (videos 1-4, unlabeled; not the validation video):
# consecutive same-looking vials there differ by 1.1-6.5 dE (camera/fill noise), so 7 sits just above it.
TOL = 7.0


def features(od_vial, od_ref, od_prev, lab_vial, lab_ref):
    d = np.asarray(od_vial) - od_ref
    u = np.asarray(od_prev) - od_ref
    uu = float(u @ u)
    w = float(np.clip(d @ u / uu, 0, 1)) if uu > 1e-6 else 0.0
    perp = float(np.linalg.norm(d - w * u))
    de = float(np.linalg.norm(np.asarray(lab_vial) - lab_ref))
    return dict(dE_ref=round(de, 2), w_hat=round(w, 3), perp=round(perp, 3), sep_prev=round(float(np.sqrt(uu)), 3))


def decide(f, tol=TOL):
    return "off-color" if f["dE_ref"] >= tol else "good"


# Cause with a confidence (added Oct 26). For an off-color vial, every candidate earlier recipe c is tried as the
# leftover: unexplained(c) = how much of the color shift is NOT explained by mixing in c's reference color.
# ratio = best / second-best unexplained. A small ratio means one recipe explains the shift much better than any
# other -> "sure"; close to 1 means two recipes explain it about equally -> "unsure".
# CAUSE_SURE_RATIO = 0.6 was set on the TWIN (twin_dataset_v2, 1,035 visibly-off vials, all 9 other recipes as
# candidates, Oct 26): "sure" on 60% of vials with 73% right vs 39% right when "unsure". Not tuned on video 5.
CAUSE_SURE_RATIO = 0.6


def cause_with_confidence(od_vial, od_ref, candidate_refs):
    """candidate_refs: {name: OD of that recipe's reference}. Returns (best, ratio, second, ranked list)."""
    d = np.asarray(od_vial) - od_ref
    r = {}
    for c, od_c in candidate_refs.items():
        u = np.asarray(od_c) - od_ref
        w = np.clip(d @ u / (u @ u), 0, 1) if u @ u > 1e-9 else 0.0
        r[c] = float(np.linalg.norm(d - w * u))
    ranked = sorted(r, key=r.get)
    if len(ranked) == 1:
        return ranked[0], 0.0, None, ranked
    return ranked[0], r[ranked[0]] / max(r[ranked[1]], 1e-9), ranked[1], ranked
