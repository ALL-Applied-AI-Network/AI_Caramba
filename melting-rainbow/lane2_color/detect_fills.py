"""Find vial fills in an UNLABELED vial_fill video (videos 1-4), so they can be used without touching video 5.

Signal: the vial bottom region (rows 360-400, cols 295-330). An empty vial there reads bright grey (~85-100,
low color); once liquid reaches the bottom it turns darker and/or colorful (liquid_score > 20). A fill = from the bottom first going dark (start)
until an empty vial is back in place (the full vial has moved on). We measure the full vial 15 frames
before it leaves (same idea as video 5's end_frame - 15).
Checked against video 5's labels (see README) before use.
"""
import cv2, numpy as np

def liquid_score(img):
    """0 for an empty vial bottom (bright, grey); grows when liquid makes it darker or more colorful."""
    b = img[360:400, 295:330].reshape(-1, 3).mean(0)
    return float(max(0.0, 85.0 - b.mean()) + 1.5 * max(0.0, (b.max() - b.min()) - 25.0))

def detect(video_path, step=5, on=20.0, off=8.0, min_fill_s=8.0):
    cap = cv2.VideoCapture(video_path); n = int(cap.get(7)); fps = cap.get(5) or 30.0
    vals, f = [], 0
    while True:                                   # read sequentially (seeking every frame is slow)
        ok = cap.grab()
        if not ok: break
        if f % step == 0:
            ok, img = cap.retrieve()
            vals.append((f, liquid_score(img)))
        f += 1
    fills, start, peak = [], None, 0.0
    win = max(1, int(fps / step))
    for i, (f, b) in enumerate(vals):
        if start is None:
            just_left = fills and (f - fills[-1]["leave_frame"]) < 4 * fps   # same vial wobbling as it leaves
            if not just_left and b > on and all(v > on * 0.5 for _, v in vals[i:i + win]):   # liquid ~1 s
                start, peak = f, b
        else:
            peak = max(peak, b)
            leave = max(off, 0.35 * peak)          # empty vial = score falls well below this fill's level
            if b < leave and all(v < leave for _, v in vals[i:i + win]):
                if (f - start) / fps >= min_fill_s:
                    fills.append(dict(start_frame=start, leave_frame=f, measure_frame=max(start, f - 15),
                                      fill_to_leave_s=round((f - start) / fps, 1)))
                start, peak = None, 0.0
    if start is not None and (vals[-1][0] - start) / fps >= min_fill_s:      # video ended with a full vial
        fills.append(dict(start_frame=start, leave_frame=vals[-1][0], measure_frame=vals[-1][0] - 15,
                          fill_to_leave_s=round((vals[-1][0] - start) / fps, 1), ended_with_video=True))
    return fills, fps
