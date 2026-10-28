"""Fill detection by CHANGE FROM THE EMPTY VIAL (Oct 26) - finds clear and pale fills that the color score misses.

Why: liquid_score() in detect_fills.py looks for a darker / more colorful vial bottom, so very pale or clear liquid
(organizers' video 3: a pale-yellow fill and two small clear fills) never triggers it. Any liquid - even clear - bends
the light through the vial, so the picture of the vial changes compared with the same vial while it was empty.

How:
  - Keep a picture of the vial column while it is empty (BASE, rows 240-405, cols 295-330, blurred).
  - A fill STARTS when the bottom of the vial (rows 360-405) differs from BASE for ~1 s while the top of the column
    is unchanged (liquid comes in at the bottom; a vial swap changes the whole column at once).
  - While filling, the FILL LINE = the highest row where the column still differs from BASE, counting up from the
    bottom. That is the volume reading. The black holder hides the line above ~row 268, so big fills read "high fill";
    smaller ones read as % of the visible vial height. Not converted to mL yet (needs known volumes - CSI session).
  - The vial LEAVES when the column moves a lot (a swap; a pause between two tanks and the next pour is much
    smaller), or when it looks empty again. After the scene is still again for ~1 s the
    new (empty) vial becomes the new BASE.
Thresholds were set by looking at videos 1-4 (unlabeled), not video 5.
"""
import collections
import cv2
import numpy as np

R0, R1, C0, C1 = 240, 405, 295, 330      # vial column below the holder
BOTTOM = 360 - R0                        # rows 360-405 = vial bottom (first place liquid shows)
UPPER = 300 - R0                         # rows 240-300 = upper column (changes only on a swap or a big fill)
HIDDEN_ROW = 268                         # above this the black holder hides the liquid line


def column(img):
    return cv2.GaussianBlur(img[R0 - 3:R1 + 3, C0 - 3:C1 + 3], (5, 5), 0)[3:-3, 3:-3].astype(np.float32)


def row_diff(a, b):
    return np.abs(a - b).mean(axis=(1, 2))           # one number per row


def fill_line(rows_diff, thr=12.0):
    """Highest row (frame coords) of the changed region that touches the bottom, or None."""
    on = np.convolve(rows_diff > thr, np.ones(5) / 5, "same") > 0.6
    if not on[-20:].any():
        return None
    top = len(on) - 1
    gap = 0
    while top > 0 and gap < 8:                      # walk up, allow small gaps (bubbles, reflections)
        top -= 1
        gap = 0 if on[top] else gap + 1
    return R0 + top + gap


class LevelFillDetector:
    START, MOVE, STILL = 12.0, 10.0, 2.5
    LEAVE = 25.0     # a vial swap / carrier move changes the column by 35-72; a pour (even a 2nd tank after a pause) <= ~15

    def __init__(self, fps):
        self.fps = fps
        self.win = max(1, int(round(fps)))            # ~1 s
        self.base = None
        self.state = "settling"                        # settling -> waiting -> filling -> (vial leaves) -> settling
        self.hist = collections.deque(maxlen=self.win)   # recent columns (motion check)
        self.trig = collections.deque(maxlen=self.win)   # recent start tests
        self.start = None
        self.level = None
        self.peak_bottom = 0.0
        self.still_n = 0
        self.last_still = None                         # last frame where the filled vial sat still (for 'gone')
        self.wait_from = None                          # frame the empty vial was first remembered
        self.level_ok = False                          # False if the fill began <2 s after that (e.g. run started
                                                       # mid-fill): the 'empty' picture may already hold liquid
        self.score = 0.0                               # shown on screen: bottom difference from the empty vial

    def update(self, img, fidx):
        """Feed one 640x480 frame. Returns ('fill_end', start_frame, leave_frame, fill_line_row) when a vial leaves."""
        col = column(img)
        motion = row_diff(col, self.hist[0]).mean() if len(self.hist) == self.hist.maxlen else 0.0
        self.hist.append(col)
        moving = motion > self.MOVE
        self.still_n = 0 if motion > self.STILL else self.still_n + 1
        ev = None

        if self.state == "settling":                  # wait until the new empty vial sits still, then remember it
            if self.still_n >= self.win:
                self.base, self.state, self.wait_from = col.copy(), "waiting", fidx
                self.trig.clear()
            self.score = 0.0
            return None

        d = row_diff(col, self.base)
        bottom, upper = float(d[BOTTOM:].mean()), float(d[:UPPER].mean())
        self.score = bottom

        if self.state == "waiting":
            if upper > self.START and moving:          # whole column changed: carrier moved / vial swapped
                self.state, self.still_n = "settling", 0
                return None
            self.trig.append(bottom > self.START and upper < self.START)
            if len(self.trig) == self.trig.maxlen and all(self.trig):
                self.state, self.start, self.peak_bottom = "filling", fidx - self.win + 1, bottom
                self.level = fill_line(d)
                self.level_ok = self.start - self.wait_from >= 2 * self.win
            elif self.still_n >= self.win and bottom < 4:   # follow slow light changes while empty
                self.base = 0.95 * self.base + 0.05 * col
            return None

        # filling (liquid still coming in, the column keeps changing) / full (column still again)
        self.peak_bottom = max(self.peak_bottom, bottom)
        if self.state == "filling" and self.still_n >= self.win:
            self.state = "full"
        if self.still_n >= self.win // 2 and bottom > 0.5 * self.peak_bottom:
            self.last_still = fidx
            lv = fill_line(d)
            if lv is not None:
                self.level = lv
        leaving = self.state == "full" and motion > self.LEAVE    # the full vial is carried away
        gone = bottom < max(6.0, 0.35 * self.peak_bottom) and self.still_n >= self.win // 2   # back to empty
        if leaving or gone:
            leave = fidx - self.win // 2 if leaving else (self.last_still or fidx - self.win)
            ev = ("fill_end", self.start, leave, self.level if self.level_ok else None)
            self.state, self.still_n, self.start, self.level, self.last_still = "settling", 0, None, None, None
        return ev


def final_level(det):
    """Fill line to report when a run ends with a vial still in place."""
    return det.level if det.level_ok else None


def level_pct(row):
    """Fill line as % of the vial's visible height (bottom row 405 = 0 %, holder edge row 268 = 100 %)."""
    if row is None:
        return None
    return int(round(100 * min(1.0, max(0.0, (R1 - row) / (R1 - HIDDEN_ROW)))))


def level_label(row):
    """Plain-English fill reading from the fill line row (geometry only - no volume calibration yet)."""
    if row is None:
        return "fill line not seen"
    if row <= HIDDEN_ROW + 2:
        return "high fill (line behind the holder)"
    return f"partial fill (line at {level_pct(row)}% of the visible vial)"
