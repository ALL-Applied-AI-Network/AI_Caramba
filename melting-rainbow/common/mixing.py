"""Shared color-mixing helpers.

Two mixing assumptions, so the team can test which fits the real line better:
  rgb : mixture color = volume-weighted average of the tank colors in BGR
  od  : Beer-Lambert style, average in optical density (-log(I/255)); closer to how dyes mix
Tank colors are not known (the recipe system only stores tank numbers), so we
ESTIMATE them by least squares from the clean reference vials of known recipes.
"""
import numpy as np, cv2

TANKS = ("t100_ml", "t200_ml", "t400_ml")

def to_space(bgr, mode):
    bgr = np.clip(np.asarray(bgr, float), 1, 255)
    return -np.log(bgr / 255.0) if mode == "od" else bgr

def from_space(v, mode):
    v = np.asarray(v, float)
    return np.clip(255.0 * np.exp(-v), 0, 255) if mode == "od" else np.clip(v, 0, 255)

def fractions(row):
    v = np.array([0 if (row[t] != row[t]) or row[t] == "" else float(row[t]) for t in TANKS])
    return v / v.sum()

def fit_tank_colors(ref_rows, mode):
    """ref_rows: iterable of rows with tank volumes + B,G,R of a clean vial. Returns 3x3 (tank x BGR-space)."""
    F = np.array([fractions(r) for r in ref_rows])
    Y = np.array([to_space([r["B"], r["G"], r["R"]], mode) for r in ref_rows])
    T, *_ = np.linalg.lstsq(F, Y, rcond=None)
    return T

def predict(row, T, mode):
    return from_space(fractions(row) @ T, mode)

def last_tank(order):
    return {"T100": 0, "T200": 1, "T400": 2}[order.split(">")[-1]]

def to_lab(bgr):
    px = np.uint8([[np.clip(bgr, 0, 255)]])
    L, A, B = cv2.cvtColor(px, cv2.COLOR_BGR2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, A - 128, B - 128])

def dE(bgr1, bgr2):
    return float(np.linalg.norm(to_lab(bgr1) - to_lab(bgr2)))
