import argparse
import os
import struct
from pathlib import Path

import numpy as np


DEFAULT_BASE_FVECS = "/workspaces/irange/exectable_data/gist1m/gist/gist_base.fvecs"
DEFAULT_QUERY_FVECS = "/workspaces/irange/exectable_data/gist1m/gist/gist_query.fvecs"
DEFAULT_OUT_ROOT = "/workspaces/irange/exectable_data/gist1m"
DEFAULT_SIZES = [250000, 500000, 750000, 1000000]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GIST .fvecs to iRangeGraph-compatible .bin files"
    )
    parser.add_argument("--base_fvecs", default=DEFAULT_BASE_FVECS, help="Path to gist_base.fvecs")
    parser.add_argument("--query_fvecs", default=DEFAULT_QUERY_FVECS, help="Path to gist_query.fvecs")
    parser.add_argument("--out_root", default=DEFAULT_OUT_ROOT, help="Output root (e.g., exectable_data/gist1m)")
    parser.add_argument(
        "--sizes",
        default=",".join(str(x) for x in DEFAULT_SIZES),
        help="Comma-separated dataset sizes, e.g. 250000,500000,750000,1000000",
    )
    return parser.parse_args()


def read_fvecs(path: str) -> np.ndarray:
    """
    Read .fvecs into a float32 array of shape (n, d).

    fvecs row layout:
      int32 d, then d float32 values
    """
    raw = np.fromfile(path, dtype=np.int32)
    if raw.size == 0:
        raise ValueError(f"Empty file: {path}")

    d = int(raw[0])
    if d <= 0:
        raise ValueError(f"Invalid dimension {d} in {path}")

    row_size = d + 1
    if raw.size % row_size != 0:
        raise ValueError(
            f"Corrupt fvecs file: int32 count {raw.size} is not divisible by row size {row_size}"
        )

    rows = raw.reshape(-1, row_size)
    dims = rows[:, 0]
    if not np.all(dims == d):
        raise ValueError(f"Inconsistent dimensions in {path}")

    vecs = rows[:, 1:].view(np.float32)
    return np.ascontiguousarray(vecs)


def write_vec_bin(path: Path, arr: np.ndarray) -> None:
    """
    iRangeGraph vector format:
      int32 n
      int32 d
      float32[n, d]
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array")

    path.parent.mkdir(parents=True, exist_ok=True)
    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", int(n)))
        f.write(struct.pack("i", int(d)))
        arr.tofile(f)


def write_attr_bin(path: Path, arr: np.ndarray) -> None:
    """
    Raw int32 attributes, no header.
    """
    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        arr.tofile(f)


def build_attributes(vectors: np.ndarray) -> np.ndarray:
    """
    Deterministic attribute from vector L2 norm.
    """
    norms = np.linalg.norm(vectors, axis=1)
    return np.round(norms * 1000.0).astype(np.int32)


def size_to_label(size: int) -> str:
    if size % 1000 != 0:
        return str(size)
    return f"{size // 1000}k"


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    sizes = [int(x.strip()) for x in args.sizes.split(",") if x.strip()]

    print("Reading base vectors...")
    base = read_fvecs(args.base_fvecs)
    print(f"Base shape: {base.shape}")

    print("Reading query vectors...")
    query = read_fvecs(args.query_fvecs)
    print(f"Query shape: {query.shape}")

    if base.shape[1] != query.shape[1]:
        raise ValueError(
            f"Dimension mismatch: base d={base.shape[1]}, query d={query.shape[1]}"
        )

    # Attribute for data vectors (used to sort base vectors for single-attribute index)
    base_attr = build_attributes(base)

    # Attribute for query vectors (not required by single-attribute search,
    # but useful for custom query-side analysis or multi-attribute experiments)
    query_attr = build_attributes(query)

    # Sort data by attribute ascending (required by single-attribute iRangeGraph build/search)
    order = np.argsort(base_attr, kind="stable")
    base_sorted = np.ascontiguousarray(base[order])
    base_attr_sorted = np.ascontiguousarray(base_attr[order])

    print("Data sorted by attribute ascending.")

    for size in sizes:
        if size <= 0:
            raise ValueError(f"Invalid size: {size}")
        if size > base_sorted.shape[0]:
            raise ValueError(
                f"Requested size {size} exceeds available base vectors {base_sorted.shape[0]}"
            )

        label = size_to_label(size)
        out_dir = out_root / label

        base_subset = base_sorted[:size]
        attr_subset = base_attr_sorted[:size]

        base_bin_path = out_dir / f"gist_base_{label}.bin"
        query_bin_path = out_dir / f"gist_query_{label}.bin"
        data_attr_path = out_dir / f"gist_attr_{label}.bin"
        query_attr_path = out_dir / f"gist_query_attr_{label}.bin"

        print(f"\n=== Creating dataset: {label} ({size} vectors) ===")
        write_vec_bin(base_bin_path, base_subset)
        write_vec_bin(query_bin_path, query)
        write_attr_bin(data_attr_path, attr_subset)
        write_attr_bin(query_attr_path, query_attr)

        sorted_ok = bool(np.all(attr_subset[:-1] <= attr_subset[1:])) if size > 1 else True
        print(f"Base bin        : {base_bin_path}")
        print(f"Query bin       : {query_bin_path}")
        print(f"Data attr bin   : {data_attr_path}")
        print(f"Query attr bin  : {query_attr_path}")
        print(f"Sorted attr asc : {sorted_ok}")

    print("\n=== Conversion complete ===")
    print("Generated files are iRangeGraph-compatible:")
    print("- vectors: int32 n, int32 d, then n*d float32")
    print("- attributes: raw int32, one per vector")


if __name__ == "__main__":
    main()