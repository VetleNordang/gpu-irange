"""
Quick sanity-check for YouTube-8M iRangeGraph .bin files.

Checks every size partition that exists under exectable_data/video/ and
exectable_data/audi/. Run from the repo root:

    python3 python/youtube8m/check_yt8m_bins.py
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path("exectable_data")
SIZES = ["1m", "2m", "4m", "8m"]

ok = True

for size in SIZES:
    vid_dir = ROOT / "video" / size
    aud_dir = ROOT / "audi"  / size

    # Skip sizes that were not produced yet
    if not vid_dir.exists() and not aud_dir.exists():
        continue

    print(f"\n--- {size} ---")

    checks = [
        (vid_dir / f"youtube_rgb_{size}.bin",      "vec"),
        (vid_dir / f"youtube_rgb_query.bin",        "vec"),
        (vid_dir / f"youtube_rgb_attr_{size}.bin",  "attr"),
        (aud_dir / f"yt_aud_{size}.bin",            "vec"),
        (aud_dir / f"yt_aud_query.bin",             "vec"),
        (aud_dir / f"yt_aud_attr_{size}.bin",       "attr"),
    ]

    for path, kind in checks:
        if not path.exists():
            print(f"  MISSING   {path.relative_to(ROOT)}")
            ok = False
            continue

        with open(path, "rb") as f:
            raw = f.read()

        if kind == "vec":
            n, d = struct.unpack("ii", raw[:8])
            good = len(raw) == 8 + n * d * 4
            tag  = "OK      " if good else "BAD SIZE"
            print(f"  {tag}  {path.relative_to(ROOT)}   n={n:>9,}  d={d}")
            if not good:
                ok = False

        else:  # attr
            arr       = np.frombuffer(raw, dtype=np.int32)
            sorted_ok = bool(np.all(arr[:-1] <= arr[1:]))
            tag       = "OK      " if sorted_ok else "NOT SORTED"
            print(f"  {tag}  {path.relative_to(ROOT)}   "
                  f"n={len(arr):>9,}  min={arr.min()}  max={arr.max()}")
            if not sorted_ok:
                ok = False

print()
print("=" * 42)
print("Result:", "ALL GOOD" if ok else "PROBLEMS FOUND")
print("=" * 42)
sys.exit(0 if ok else 1)