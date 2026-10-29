"""Lane 1: match every vial_fill still to the nearest tank_level and cap_seal still.

Uses each camera's image_index.csv (filename, timestamp, video_id) from the dataset.
Output: sync_index.csv + a printed gap report.

Usage: python lane1_sync/sync.py <dataset_root> [out_csv]
"""
import sys
import pandas as pd

root = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else "sync_index.csv"

def load(cam):
    df = pd.read_csv(f"{root}/{cam}/image_index.csv")
    df = df[df["video_id"].astype(str) != "unassigned"].copy()   # README: some rows may be 'unassigned'
    df["video_id"] = df["video_id"].astype(int)
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed")
    except (ValueError, TypeError):
        # Fallback: parse from filename {camera}_{YYYYMMDD}_{HHMMSS}_{microseconds}.jpg (README section 5)
        parts = df["filename"].str.extract(r"_(\d{8})_(\d{6})_(\d{1,6})\.")
        df["timestamp"] = pd.to_datetime(parts[0] + parts[1] + parts[2].str.ljust(6, "0"),
                                         format="%Y%m%d%H%M%S%f")
    return df.sort_values("timestamp").rename(columns={"filename": f"{cam}_file"})

fill = load("vial_fill")
merged = fill
for cam in ("tank_level", "cap_seal"):
    other = load(cam).rename(columns={"timestamp": f"{cam}_ts"})
    merged = pd.merge_asof(merged.sort_values("timestamp"), other.sort_values(f"{cam}_ts"),
                           left_on="timestamp", right_on=f"{cam}_ts", by="video_id",
                           direction="nearest")
    merged[f"{cam}_gap_s"] = (merged[f"{cam}_ts"] - merged["timestamp"]).dt.total_seconds()

merged.to_csv(out, index=False)

print(f"wrote {out}: {len(merged)} vial_fill frames")
for cam in ("tank_level", "cap_seal"):
    g = merged.groupby("video_id")[f"{cam}_gap_s"]
    print(f"\n{cam} gap vs vial_fill (s), per video:")
    print(g.agg(["count", "median", lambda s: s.abs().max()]).rename(columns={"<lambda_0>": "max_abs"}).round(3))
    bad = merged[merged[f"{cam}_gap_s"].abs() > 1.0]
    print(f"  frames with |gap| > 1 s (missing/misaligned): {len(bad)}")
# Drift: does the median offset change across a video? Compare first vs last quarter.
for cam in ("tank_level", "cap_seal"):
    for vid, d in merged.groupby("video_id"):
        q = max(len(d) // 4, 1)
        early, late = d[f"{cam}_gap_s"].iloc[:q].median(), d[f"{cam}_gap_s"].iloc[-q:].median()
        if pd.notna(early) and abs(late - early) > 0.25:
            print(f"  possible drift: {cam} video {vid}: offset {early:+.2f}s -> {late:+.2f}s")
