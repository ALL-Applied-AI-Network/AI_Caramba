"""Shared recipe / color / fill-level math for the digital-twin renders (Lane 4).

Every number here has a source:
  RECIPES ........ organizers' vial_fill/labels/video5_vial_recipes.csv (t100/t200/t400 mL, run order)
  BLUE_R7 ........ the twin's own liquid color (/FillStation/_materials/USD_blue_liquid_approximation)
                   treated as recipe 7 strength (50 mL blue of 70 mL), because the twin's animation
                   pours 50 mL blue then 20 mL clear
  ORANGE ......... CALIBRATED Oct 22 on R6 only (orange-only recipe): with the camera at exposure:time
                   1/256 s, the color ladder (36 known liquid colors rendered in the twin) was inverted to find
                   the liquid color whose render matches the real clean R6 vial (56, 24, 7). Held-out check on the
                   8 recipes not used (all but R6, R14): mean dE 7.6 (was 8.5 with the guess), worst R12 20.7.
                   Blue was NOT re-fitted from R14: fitting it made the held-out recipes worse (mean 9.2).
  od mixing ...... dye-style mixing: darkness OD = -ln(color) adds by volume share
                   (fit real video 5 better than RGB averaging: error 6.8 vs 13.8, our Oct 16 run)
  FRAME_FOR_ML ... our reading of the twin's liquid height vs frame (70 mL is the most the animation shows)
Carryover: a vial that keeps a fraction w of the PREVIOUS recipe's mixture (hypothesis H1, best fit Oct 16)
is mixed the same dye way: OD = (1-w)*OD_current + w*OD_previous.
"""
import numpy as np

BLUE_R7 = np.array([0.006, 0.20, 0.30])
ORANGE_GUESS = np.array([0.35, 0.12, 0.02])    # original guess, kept for the record (not used)
OD_BLUE = -np.log(np.clip(BLUE_R7, 1e-3, 1)) / (50 / 70)
OD_ORANGE = np.array([1.5092, 5.4573, 10.637])       # calibrated on R6 (darkness per unit orange share), see above
CAMERA_EV = -8                   # camera exposure:time = 2**-8 = 1/256 s (calibrated on R6 + R14)

# recipe_id: (T100 clear, T200 orange, T400 blue) mL  -- video 5 run order R5 -> R14
RECIPES = {5: (10, 10, 10), 6: (30, 10, 0), 7: (20, 0, 50), 8: (20, 20, 20), 9: (10, 35, 35),
           10: (45, 15, 15), 11: (10, 10, 50), 12: (10, 50, 10), 13: (0, 50, 20), 14: (60, 0, 10)}
RUN_ORDER = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

# fill level -> twin frame. 70 mL uses 678 (after the pour ends at 661, as the teammate's script did).
FRAME_FOR_ML = {30: 217, 40: 290, 60: 619, 70: 678}


def total_ml(rid):
    return sum(RECIPES[rid])


def od(rid):
    c, o, b = RECIPES[rid]
    t = c + o + b
    return (o / t) * OD_ORANGE + (b / t) * OD_BLUE


def color(rid, prev_rid=None, carry=0.0):
    """Liquid RGB (0-1) for recipe rid, optionally with fraction `carry` of prev_rid's mixture left over."""
    d = od(rid)
    if prev_rid is not None and carry > 0:
        d = (1 - carry) * d + carry * od(prev_rid)
    return tuple(float(x) for x in np.round(np.exp(-d), 4))


def frame_for(rid):
    ml = total_ml(rid)
    return FRAME_FOR_ML[min(FRAME_FOR_ML, key=lambda m: abs(m - ml))]
