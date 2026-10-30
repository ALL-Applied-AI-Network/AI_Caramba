"""Operator-confirmed vials -> labeled training examples (the feedback flywheel, Oct 30).

Every time the operator clicks Pass or Reject on a vial in the operator station, the station saves:
  - the vial's camera frame and its batch's reference frame (PNG, lossless - color is what we measure)
  - one row in <out>/flywheel/examples.csv with the label and what the station had said

Nothing is trained during operation. These examples are used later, between shifts, by retrain.py, which only
replaces the station's model if the new one is at least as good on a fixed test set (see flywheel/README.md).

Layout:
  <out>/flywheel/examples.csv
  <out>/flywheel/images/<run>_v007.png         vial 7 of that run
  <out>/flywheel/images/<run>_v009_ref.png     its batch reference (saved once per batch)
"""
import csv
import os

import cv2

FIELDS = ["time", "run_id", "vial", "batch", "ref_vial", "label", "source", "station_status", "rule_dE",
          "ai_p_off", "vial_ml", "ref_ml", "recipe", "vial_image", "ref_image"]


def save_example(out_dir, run_id, vial_no, ref_no, batch, action, station_status, vial, ref, result=None,
                 recipe="", when=""):
    """Save one operator decision. action: 'PASS' (label good) or 'REJECT' (label off-color).
    vial / ref: the station's vial dicts (need 'frame' and 'ml'). Returns the examples.csv path, or None if skipped."""
    if action not in ("PASS", "REJECT") or vial.get("frame") is None or ref.get("frame") is None:
        return None
    d = os.path.join(out_dir, "flywheel")
    img_dir = os.path.join(d, "images")
    os.makedirs(img_dir, exist_ok=True)
    v_name = f"{run_id}_v{int(vial_no):03d}.png"
    r_name = f"{run_id}_v{int(ref_no):03d}_ref.png"
    cv2.imwrite(os.path.join(img_dir, v_name), vial["frame"])
    if not os.path.exists(os.path.join(img_dir, r_name)):
        cv2.imwrite(os.path.join(img_dir, r_name), ref["frame"])
    # where the label came from: the operator settled a REVIEW, overruled the station, or confirmed it
    if station_status == "REVIEW":
        source = "review"
    elif (station_status == "REJECT") == (action == "REJECT"):
        source = "confirm"
    else:
        source = "override"
    result = result or {}
    row = dict(time=when, run_id=run_id, vial=int(vial_no), batch=batch, ref_vial=int(ref_no),
               label="off" if action == "REJECT" else "good", source=source, station_status=station_status,
               rule_dE=result.get("dE_ref", ""), ai_p_off=result.get("ai_p_off", ""),
               vial_ml=vial.get("ml", ""), ref_ml=ref.get("ml", ""), recipe=recipe,
               vial_image=f"images/{v_name}", ref_image=f"images/{r_name}")
    path = os.path.join(d, "examples.csv")
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return path


def load_examples(paths):
    """Read one or more flywheel folders (or examples.csv files). If the operator clicked a vial more than once,
    the LAST click wins. Image paths come back absolute."""
    rows = {}
    for p in paths or []:
        csv_path = p if p.endswith(".csv") else os.path.join(p, "examples.csv")
        if not os.path.exists(csv_path) and os.path.isdir(os.path.join(p, "flywheel")):
            csv_path = os.path.join(p, "flywheel", "examples.csv")     # an operator_output folder was given
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"no examples.csv in {p}")
        base = os.path.dirname(os.path.abspath(csv_path))
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                if r.get("label") not in ("good", "off"):
                    continue
                r["vial_image"] = os.path.join(base, r["vial_image"])
                r["ref_image"] = os.path.join(base, r["ref_image"])
                rows[(r["run_id"], r["vial"])] = r
    return list(rows.values())
