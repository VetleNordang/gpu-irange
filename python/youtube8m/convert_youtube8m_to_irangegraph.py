#!/usr/bin/env python3
"""
Download YouTube-8M frame-level data and convert to iRangeGraph .bin files.

Downloads TFRecord shards directly from data.yt8m.org, then extracts
per-frame RGB (1024-dim) and audio (128-dim) features, sorts by L2-norm,
and writes iRangeGraph-compatible binary files.

No train/val/test split is enforced:
  - Training shards  → base vectors  (the searchable database)
  - Validation shard → query vectors (guaranteed non-overlapping with train,
                                      YouTube-8M never puts the same video in
                                      more than one split)

Output layout (mirrors the GIST partitioned structure):
  out_root/
    video/
      1m/  youtube_rgb_1m.bin   youtube_rgb_query.bin   youtube_rgb_attr_1m.bin
      2m/  youtube_rgb_2m.bin   youtube_rgb_query.bin   youtube_rgb_attr_2m.bin
      # 4m and 8m are commented out — uncomment + download more shards if needed
    audi/
      1m/  yt_aud_1m.bin        yt_aud_query.bin        yt_aud_attr_1m.bin
      2m/  yt_aud_2m.bin        yt_aud_query.bin        yt_aud_attr_2m.bin

iRangeGraph vector format   : int32 n, int32 d, float32[n * d]
iRangeGraph attribute format: raw int32[n], no header (sorted ascending)

Usage (runs download + conversion end-to-end):
  python3 convert_youtube8m_to_irangegraph.py \
      --raw_dir  exectable_data/yt8m_raw \
      --out_root exectable_data \
      [--mirror  us|eu|asia]   # default: us
"""

import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

RGB_DIM   = 1024
AUDIO_DIM = 128

# YouTube-8M stores features as uint8 encoding float in ~[-2, 2]:
#   encode: round(x * 255/4 + 127.5)   decode: x * 4/255 - 2
_DEQUANT_SCALE  = 4.0 / 255.0
_DEQUANT_OFFSET = -2.0

# Rough number of frames per TFRecord shard (used to estimate how many to fetch).
# ~1 000 videos/shard × ~150 frames/video on average.
_FRAMES_PER_SHARD    = 150_000
_TOTAL_TRAIN_SHARDS  = 3_844
_TOTAL_VAL_SHARDS    = 1_085


# ── Download ──────────────────────────────────────────────────────────────────

def _shard_fraction(needed_frames: int, total_shards: int, safety: float = 1.4) -> int:
    """Return Y for 'shard=1,Y' so we download enough shards for needed_frames."""
    needed_shards = max(1, int(needed_frames / _FRAMES_PER_SHARD * safety) + 1)
    if needed_shards >= total_shards:
        return 1  # download everything
    return max(1, total_shards // needed_shards)


def download_shards(partition: str, target_dir: Path, shard_frac: int, mirror: str) -> None:
    """
    Download YouTube-8M TFRecord shards using the official download.py script.

    Runs:  curl data.yt8m.org/download.py | shard=1,<Y> partition=2/frame/<partition> \
               mirror=<mirror> python3
    Files land in target_dir (script is executed from there).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob("*.tfrecord"))
    if existing:
        print(f"  {len(existing)} .tfrecord file(s) already in {target_dir} — skipping download.")
        return

    env = {
        **os.environ,
        "partition": f"2/frame/{partition}",
        "mirror":    mirror,
        "shard":     f"1,{shard_frac}",
    }
    cmd = "curl -s data.yt8m.org/download.py | python3"
    print(f"  Running: {cmd}  [shard=1,{shard_frac} partition=2/frame/{partition} mirror={mirror}]",
          flush=True)
    result = subprocess.run(cmd, shell=True, env=env, cwd=str(target_dir))
    if result.returncode != 0:
        raise RuntimeError(f"Download failed for partition '{partition}' (exit {result.returncode})")

    downloaded = list(target_dir.glob("*.tfrecord"))
    print(f"  Downloaded {len(downloaded)} shard(s) into {target_dir}", flush=True)
    if not downloaded:
        raise RuntimeError(f"No .tfrecord files found in {target_dir} after download.")


# ── TFRecord parsing ──────────────────────────────────────────────────────────

def _decode_record(raw):
    """Return (rgb, audio) float32 arrays for one SequenceExample."""
    import tensorflow as tf

    _, seqs = tf.io.parse_single_sequence_example(
        raw,
        context_features={"id": tf.io.FixedLenFeature([], tf.string)},
        sequence_features={
            "rgb":   tf.io.FixedLenSequenceFeature([], tf.string),
            "audio": tf.io.FixedLenSequenceFeature([], tf.string),
        },
    )
    rgb_bytes   = seqs["rgb"].numpy()
    audio_bytes = seqs["audio"].numpy()
    T = len(rgb_bytes)
    if T == 0:
        return (np.zeros((0, RGB_DIM), dtype=np.float32),
                np.zeros((0, AUDIO_DIM), dtype=np.float32))

    rgb   = np.empty((T, RGB_DIM),   dtype=np.float32)
    audio = np.empty((T, AUDIO_DIM), dtype=np.float32)
    for i in range(T):
        rgb[i]   = np.frombuffer(rgb_bytes[i],   dtype=np.uint8)
        audio[i] = np.frombuffer(audio_bytes[i], dtype=np.uint8)

    rgb   = rgb   * _DEQUANT_SCALE + _DEQUANT_OFFSET
    audio = audio * _DEQUANT_SCALE + _DEQUANT_OFFSET
    return rgb, audio


def collect_to_memmap(tfrecord_dir: Path, rgb_mm: np.memmap,
                      audio_mm: np.memmap, max_frames: int) -> int:
    """Stream .tfrecord files from tfrecord_dir into memmaps. Returns frame count."""
    import tensorflow as tf

    paths = sorted(tfrecord_dir.glob("*.tfrecord"))
    if not paths:
        raise FileNotFoundError(f"No .tfrecord files in {tfrecord_dir}")

    count = 0
    for path in paths:
        if count >= max_frames:
            break
        print(f"  {path.name}  (frames so far: {count:,})", flush=True)
        for raw in tf.data.TFRecordDataset(str(path)):
            if count >= max_frames:
                break
            rgb_f, audio_f = _decode_record(raw)
            n = len(rgb_f)
            if n == 0:
                continue
            end    = min(count + n, max_frames)
            actual = end - count
            rgb_mm[count:end]   = rgb_f[:actual]
            audio_mm[count:end] = audio_f[:actual]
            count = end

    rgb_mm.flush()
    audio_mm.flush()
    print(f"  → {count:,} frames collected from {tfrecord_dir}", flush=True)
    return count


# ── Attribute computation ─────────────────────────────────────────────────────

def compute_attr(mm: np.memmap, count: int, chunk: int) -> np.ndarray:
    """L2-norm × 1000, quantised to int32. Computed chunk-by-chunk to save RAM."""
    attr = np.empty(count, dtype=np.float32)
    for s in range(0, count, chunk):
        e = min(s + chunk, count)
        attr[s:e] = np.linalg.norm(mm[s:e].astype(np.float32), axis=1)
    return np.round(attr * 1000.0).astype(np.int32)


# ── Binary writers ────────────────────────────────────────────────────────────

def write_vec_bin(path: Path, arr: np.ndarray) -> None:
    """iRangeGraph vector format: int32 n, int32 d, float32[n × d]."""
    arr = np.asarray(arr, dtype=np.float32)
    n, d = arr.shape
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("ii", n, d))
        arr.tofile(f)


def write_attr_bin(path: Path, arr: np.ndarray) -> None:
    """Raw int32 attribute array — no header."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(arr, dtype=np.int32).tofile(str(path))


def write_sorted_partition(mm: np.memmap, order: np.ndarray,
                           n: int, out_path: Path, chunk: int) -> None:
    """Write n vectors from mm reindexed by order[:n], in chunks to avoid RAM spikes."""
    dim = mm.shape[1]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("ii", n, dim))
        for s in range(0, n, chunk):
            e   = min(s + chunk, n)
            idx = order[s:e]
            # Fetch in roughly sequential order to reduce random disk seeks,
            # then un-permute back to the desired sorted output order.
            seq  = np.argsort(idx)
            data = mm[idx[seq]].astype(np.float32)
            data = data[np.argsort(seq)]
            f.write(data.tobytes())
    print(f"    wrote {n:,} × {dim}-dim  →  {out_path.name}", flush=True)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download YouTube-8M and convert to iRangeGraph .bin files"
    )
    p.add_argument(
        "--raw_dir", default="exectable_data/yt8m_raw",
        help="Where to store downloaded TFRecord shards (default: exectable_data/yt8m_raw)",
    )
    p.add_argument(
        "--out_root", default="exectable_data",
        help="Output root for .bin files (default: exectable_data)",
    )
    p.add_argument(
        "--mirror", default="us", choices=["us", "eu", "asia"],
        help="Download mirror (default: us)",
    )
    p.add_argument(
        "--sizes", default="1000000,2000000",
        # Uncomment 4000000 and 8000000 below (and increase disk + download shards)
        # if you need larger partitions — each requires ~16 GB and ~32 GB of disk.
        # default="1000000,2000000,4000000,8000000",
        help="Comma-separated partition sizes (default: 1m and 2m)",
    )
    p.add_argument(
        "--num_queries", type=int, default=1000,
        help="Number of query vectors (default: 1000)",
    )
    p.add_argument(
        "--chunk_size", type=int, default=500_000,
        help="Row-chunk size for processing (default: 500 000)",
    )
    p.add_argument(
        "--keep_raw", action="store_true",
        help="Keep raw TFRecord files after conversion (default: delete them)",
    )
    return p.parse_args()


def size_label(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}m"
    if n % 1_000 == 0:
        return f"{n // 1_000}k"
    return str(n)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    sizes   = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]
    max_n   = max(sizes)
    raw_dir = Path(args.raw_dir)
    out_root = Path(args.out_root)

    train_dir = raw_dir / "train"
    val_dir   = raw_dir / "validate"
    tmp_dir   = raw_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Download ──────────────────────────────────────────────────────
    print("\n=== Step 1: Downloading training shards (base vectors) ===")
    # Auto-calculate how many shards we need for max_n frames.
    train_frac = _shard_fraction(max_n, _TOTAL_TRAIN_SHARDS)
    print(f"  Need ~{max_n:,} frames  →  shard=1,{train_frac} "
          f"(≈{_TOTAL_TRAIN_SHARDS // train_frac} shards × {_FRAMES_PER_SHARD:,} frames each)")
    download_shards("train", train_dir, train_frac, args.mirror)

    print("\n=== Step 1b: Downloading validation shard (query vectors) ===")
    # 1–2 validation shards are more than enough for 1 000 queries.
    # YouTube-8M guarantees no video appears in both train and validate,
    # so there is zero overlap with the base vectors.
    val_frac = _shard_fraction(args.num_queries * 3, _TOTAL_VAL_SHARDS)
    download_shards("validate", val_dir, val_frac, args.mirror)

    # ── Step 2: Collect base frames into memmap ───────────────────────────────
    print(f"\n=== Step 2: Collecting up to {max_n:,} base frames ===")
    rgb_tmp   = tmp_dir / "rgb_base.dat"
    audio_tmp = tmp_dir / "audio_base.dat"

    rgb_mm   = np.memmap(rgb_tmp,   dtype=np.float32, mode="w+", shape=(max_n, RGB_DIM))
    audio_mm = np.memmap(audio_tmp, dtype=np.float32, mode="w+", shape=(max_n, AUDIO_DIM))

    base_n = collect_to_memmap(train_dir, rgb_mm, audio_mm, max_n)
    if base_n < min(sizes):
        print(f"ERROR: only {base_n:,} frames collected — need at least {min(sizes):,}.\n"
              "  Try increasing the download by raising the shard fraction "
              "(lower the second number in shard=1,Y).", file=sys.stderr)
        sys.exit(1)

    # ── Step 3: Collect query frames ──────────────────────────────────────────
    print(f"\n=== Step 3: Collecting {args.num_queries} query frames ===")
    rgb_q_tmp   = tmp_dir / "rgb_query.dat"
    audio_q_tmp = tmp_dir / "audio_query.dat"

    rgb_q_mm   = np.memmap(rgb_q_tmp,   dtype=np.float32, mode="w+",
                           shape=(args.num_queries, RGB_DIM))
    audio_q_mm = np.memmap(audio_q_tmp, dtype=np.float32, mode="w+",
                           shape=(args.num_queries, AUDIO_DIM))

    q_n = collect_to_memmap(val_dir, rgb_q_mm, audio_q_mm, args.num_queries)
    if q_n < args.num_queries:
        print(f"  WARNING: only {q_n} query frames available — using all {q_n}.", flush=True)

    # ── Step 4: Attributes and sort order ─────────────────────────────────────
    print("\n=== Step 4: Computing L2-norm attributes and sort order ===")
    rgb_attr   = compute_attr(rgb_mm,   base_n, args.chunk_size)
    audio_attr = compute_attr(audio_mm, base_n, args.chunk_size)

    rgb_order   = np.argsort(rgb_attr[:base_n],   kind="stable")
    audio_order = np.argsort(audio_attr[:base_n], kind="stable")

    print(f"  RGB   attr: min={rgb_attr.min()},  max={rgb_attr.max()}")
    print(f"  Audio attr: min={audio_attr.min()}, max={audio_attr.max()}")

    # Load queries into RAM — small: 1000 × 1024 × 4 ≈ 4 MB
    rgb_queries   = np.array(rgb_q_mm[:q_n],   dtype=np.float32)
    audio_queries = np.array(audio_q_mm[:q_n], dtype=np.float32)

    # ── Step 5: Write partitions ──────────────────────────────────────────────
    print("\n=== Step 5: Writing partitions ===")
    for size in sizes:
        if size > base_n:
            print(f"\nSKIP {size_label(size)}: only {base_n:,} frames collected "
                  f"(need {size:,}). Download more shards.")
            continue

        label = size_label(size)
        print(f"\n--- {label}  ({size:,} vectors) ---")

        # Video — RGB, 1024-dim
        vid_dir = out_root / "video" / label
        write_sorted_partition(rgb_mm, rgb_order, size,
                               vid_dir / f"youtube_rgb_{label}.bin", args.chunk_size)
        write_vec_bin(vid_dir / f"youtube_rgb_query.bin", rgb_queries)
        write_attr_bin(vid_dir / f"youtube_rgb_attr_{label}.bin",
                       rgb_attr[rgb_order[:size]])

        # Audio — 128-dim
        aud_dir = out_root / "audi" / label
        write_sorted_partition(audio_mm, audio_order, size,
                               aud_dir / f"yt_aud_{label}.bin", args.chunk_size)
        write_vec_bin(aud_dir / f"yt_aud_query.bin", audio_queries)
        write_attr_bin(aud_dir / f"yt_aud_attr_{label}.bin",
                       audio_attr[audio_order[:size]])

        print(f"  {label} done.", flush=True)

    # ── Step 6: Cleanup ───────────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)
    if not args.keep_raw:
        print(f"\nRemoving raw TFRecord files in {raw_dir} ...")
        shutil.rmtree(raw_dir, ignore_errors=True)
        print("Removed.")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n=== Done ===")
    print("Files written:")
    for size in sizes:
        if size > base_n:
            continue
        label = size_label(size)
        for mod, name in [("video", f"youtube_rgb_{label}.bin"),
                          ("audi",  f"yt_aud_{label}.bin")]:
            p = out_root / mod / label / name
            if p.exists():
                mb = p.stat().st_size / 1e6
                print(f"  {p.relative_to(out_root)}  ({mb:.0f} MB)")

    print("\nVector .bin   : int32 n, int32 d, then n×d float32")
    print("Attribute .bin: raw int32[n], no header, sorted ascending")
    print("\nBuild an index with:")
    print("  ./tests/buildindex --data_path exectable_data/video/1m/youtube_rgb_1m.bin \\")
    print("      --index_file exectable_data/video/1m/youtube_rgb_1m.index \\")
    print("      --M 32 --ef_construction 500 --threads <N>")


if __name__ == "__main__":
    main()
