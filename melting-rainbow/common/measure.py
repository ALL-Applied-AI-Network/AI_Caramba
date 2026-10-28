"""Measure a vial's liquid color the corrected way (Oct 24): ONLY where the liquid is in front of the white
paper, divided by the paper right next to the vial. Used identically for real frames and twin renders.

Why (found Oct 23): the old box (x300 y300 28x30) straddled the bottom edge of the white paper (~row 320 at
the vial), so half of it saw the liquid against the dark machine. Real paper is steady between vials
(+/- ~5%), so dividing by it also removes exposure / white-balance differences.

Returns transmittance T (per channel, liquid / paper; 1 = as bright as the paper) and optical density
OD = -ln(T), which adds up linearly for dyes (Beer-Lambert), so mixtures and leftovers are easy to model.
"""
import numpy as np
import cv2

VIAL_X = (305, 325)            # vial center columns (real and twin FillCam: liquid projects to x ~310)
PAPER = (275, 305, 420, 440)   # rows, cols of white paper beside the vial (same in real and twin)


def band_rows(total_ml):
    """Rows where the liquid is in front of the paper. 60-80 mL: below the black holder, above the paper
    edge. 30-40 mL: the liquid top sits lower, so only the thin strip just above the paper edge."""
    return (272, 305) if total_ml >= 60 else (306, 318)


def measure(img_bgr, total_ml):
    r0, r1 = band_rows(total_ml)
    liq = np.median(img_bgr[r0:r1, VIAL_X[0]:VIAL_X[1]].reshape(-1, 3), axis=0)[::-1].astype(float)
    p = PAPER
    pap = np.median(img_bgr[p[0]:p[1], p[2]:p[3]].reshape(-1, 3), axis=0)[::-1].astype(float)
    T = np.clip(liq / np.maximum(pap, 1.0), 1e-3, 1.2)
    return dict(liquid_rgb=liq, paper_rgb=pap, T=T, OD=-np.log(T))


def lab_of_T(T):
    """CIE Lab of the paper-normalized color (paper = white 255), for perceptual dE."""
    px = np.uint8([[np.clip(np.round(np.asarray(T) * 255), 0, 255)]])
    L, A, B = cv2.cvtColor(px, cv2.COLOR_RGB2LAB)[0, 0].astype(float)
    return np.array([L * 100 / 255, A - 128, B - 128])
