#!/usr/bin/env python3
"""
Sanity-check all datasets: file sizes, attribute sorting, and duplicate vectors.

Run from repo root:
    python3 python/check_datasets.py
"""

import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path("executable_data")
LOG = Path("logs/check_datasets.log")
LOG.parent.mkdir(parents=True, exist_ok=True)
_log = open(LOG, "w")

def out(msg=""):
    print(msg)
    _log.write(msg + "\n")
    _log.flush()

def load_vec(path):
    with open(path, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
        data = np.frombuffer(f.read(n * d * 4), dtype=np.float32).reshape(n, d)
    return n, d, data

def check_duplicates(data):
    hashes = [hash(row.tobytes()) for row in data]
    unique = len(set(hashes))
    return unique, len(data) - unique

def check_vec(path, check_dupes=True):
    if not path.exists():
        out(f"  MISSING    {path.relative_to(ROOT)}")
        return False
    n, d, data = load_vec(path)
    rel = str(path.relative_to(ROOT))
    if check_dupes:
        unique, dupes = check_duplicates(data)
        dupe_pct = dupes / n * 100
        tag = "OK        " if dupes == 0 else "DUPES     "
        out(f"  {tag} {rel:<55} n={n:>9,}  d={d:>4}  dupes={dupes:>7,} ({dupe_pct:.1f}%)")
    else:
        out(f"  OK        {rel:<55} n={n:>9,}  d={d:>4}")
    return True

def check_attr(path):
    if not path.exists():
        out(f"  MISSING    {path.relative_to(ROOT)}")
        return False
    arr = np.frombuffer(path.read_bytes(), dtype=np.int32)
    sorted_ok = bool(np.all(arr[:-1] <= arr[1:]))
    rel = str(path.relative_to(ROOT))
    tag = "OK        " if sorted_ok else "NOT SORTED"
    out(f"  {tag} {rel:<55} n={len(arr):>9,}  min={arr.min()}  max={arr.max()}")
    return sorted_ok

out("=" * 80)
out("Dataset sanity check")
out("=" * 80)

all_ok = True

# YouTube-8M audio + video
for size in ["1m", "2m", "4m", "8m"]:
    vid_dir = ROOT / "video" / size
    aud_dir = ROOT / "audi" / size
    if not vid_dir.exists() and not aud_dir.exists():
        continue
    out(f"\n--- YouTube {size} ---")
    for path, kind in [
        (vid_dir / f"youtube_rgb_{size}.bin",     "vec"),
        (vid_dir / f"youtube_rgb_query.bin",       "query"),
        (vid_dir / f"youtube_rgb_attr_{size}.bin", "attr"),
        (aud_dir / f"yt_aud_{size}.bin",           "vec"),
        (aud_dir / f"yt_aud_query.bin",            "query"),
        (aud_dir / f"yt_aud_attr_{size}.bin",      "attr"),
    ]:
        if kind == "vec":
            ok = check_vec(path, check_dupes=True)
        elif kind == "query":
            ok = check_vec(path, check_dupes=False)
        else:
            ok = check_attr(path)
        all_ok = all_ok and ok

# GIST1M
for size in ["250k", "500k", "750k", "1000k"]:
    gist_dir = ROOT / "gist1m" / size
    if not gist_dir.exists():
        continue
    out(f"\n--- GIST {size} ---")
    for path, kind in [
        (gist_dir / f"gist_base_{size}.bin",  "vec"),
        (gist_dir / f"gist_query_{size}.bin", "query"),
    ]:
        ok = check_vec(path, check_dupes=(kind == "vec"))
        all_ok = all_ok and ok

out()
out("=" * 80)
out("Result: " + ("ALL GOOD" if all_ok else "PROBLEMS FOUND"))
out("=" * 80)
out(f"Log saved to: {LOG}")
_log.close()
sys.exit(0 if all_ok else 1)
