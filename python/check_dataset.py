"""
Validates the format and integrity of iRangeGraph dataset files.

Checks performed:
  - .bin header (n, d) matches actual file size
  - query vectors exist in the base file (exact float match)
  - attribute file has exactly n int32 values
  - attribute values are sorted ascending (1..n or 0..n-1)
  - base vectors are unique (no duplicate rows)

Usage:
  python check_dataset.py --base <base.bin> --query <query.bin> --attr <attr.bin>
"""

import argparse
import struct
import numpy as np
import os
import sys


def read_bin(path: str) -> np.ndarray:
    """Read an iRangeGraph float .bin file. Returns (n, d) float32 array."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    expected = 8 + n * d * 4
    if size != expected:
        raise ValueError(
            f"{path}: file size {size} != expected {expected} (n={n}, d={d})"
        )
    data = np.fromfile(path, dtype=np.float32, offset=8).reshape(n, d)
    return data


def read_attr(path: str) -> np.ndarray:
    """Read an iRangeGraph int32 attribute file. Returns 1-D int32 array."""
    size = os.path.getsize(path)
    n = size // 4
    return np.fromfile(path, dtype=np.int32)


def check_file_size(path: str, n: int, d: int) -> bool:
    expected = 8 + n * d * 4
    actual = os.path.getsize(path)
    ok = actual == expected
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] File size: {actual} bytes (expected {expected})")
    return ok


def check_attr_count(attr: np.ndarray, n: int, label: str) -> bool:
    ok = len(attr) == n
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Attribute count: {len(attr)} (expected {n})")
    if not ok:
        print(f"         Note: if first value looks like n ({attr[0]}), "
              "the attr file may have a stray header int.")
    return ok


def check_attr_sorted(attr: np.ndarray) -> bool:
    ok = bool(np.all(attr[:-1] <= attr[1:]))
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] Attributes sorted ascending")
    if not ok:
        bad = int(np.argmax(attr[:-1] > attr[1:]))
        print(f"         First violation at index {bad}: "
              f"{attr[bad]} > {attr[bad+1]}")
    return ok


def check_attr_range(attr: np.ndarray, n: int) -> bool:
    mn, mx = int(attr.min()), int(attr.max())
    # Accept 0-based (0..n-1) or 1-based (1..n)
    ok = (mn == 0 and mx == n - 1) or (mn == 1 and mx == n)
    status = "OK" if ok else "WARN"
    base = "0-based" if mn == 0 else ("1-based" if mn == 1 else "unknown")
    print(f"  [{status}] Attribute range: [{mn}, {mx}], n={n}, "
          f"indexing={base}")
    if not ok:
        print(f"         Values don't span [0, {n-1}] — attributes may be category IDs rather than ranks.")
    return True


def check_unique_vectors(base: np.ndarray, sample: int = 200_000) -> bool:
    """Check uniqueness on a sample to avoid O(n^2) cost on 1M vectors."""
    if len(base) <= sample:
        sub = base
        label = f"all {len(base)}"
    else:
        idx = np.random.choice(len(base), sample, replace=False)
        sub = base[idx]
        label = f"random sample {sample}/{len(base)}"

    # Use a view trick: treat each row as a single bytes object via np unique
    # on the raw buffer
    _, counts = np.unique(sub, axis=0, return_counts=True)
    n_dup = int((counts > 1).sum())
    ok = n_dup == 0
    status = "OK" if ok else "WARN"
    print(f"  [{status}] Unique base vectors ({label}): "
          f"{n_dup} duplicate rows found")
    return True


def check_queries_in_base(base: np.ndarray, query: np.ndarray,
                           max_check: int = 100) -> bool:
    """Verify query vectors appear in the base set (exact float match)."""
    n_check = min(max_check, len(query))
    base_set = set(map(tuple, base))
    found = sum(1 for q in query[:n_check] if tuple(q) in base_set)
    ok = found == n_check
    status = "OK" if ok else "WARN"
    print(f"  [{status}] Query vectors found in base "
          f"({found}/{n_check} checked)")
    if not ok:
        print(f"         {n_check - found} query vectors not in base (normal for held-out query sets).")
    return True


def run_checks(base_path: str, query_path: str, attr_path: str,
               skip_unique: bool = False) -> None:
    print(f"\n=== Checking dataset ===")
    print(f"  base:  {base_path}")
    print(f"  query: {query_path}")
    print(f"  attr:  {attr_path}\n")

    all_ok = True

    # --- base file ---
    print("[Base file]")
    try:
        base = read_bin(base_path)
        n, d = base.shape
        print(f"  [OK] Header: n={n}, d={d}")
        check_file_size(base_path, n, d)
        if skip_unique:
            print(f"  [SKIP] Unique vector check skipped (--skip-unique)")
        else:
            all_ok &= check_unique_vectors(base)
    except Exception as e:
        print(f"  [FAIL] Could not read base: {e}")
        all_ok = False
        base = None

    # --- query file ---
    print("\n[Query file]")
    try:
        query = read_bin(query_path)
        nq, dq = query.shape
        print(f"  [OK] Header: n={nq}, d={dq}")
        check_file_size(query_path, nq, dq)
        if base is not None:
            if dq != d:
                print(f"  [FAIL] Query dimension {dq} != base dimension {d}")
                all_ok = False
            else:
                all_ok &= check_queries_in_base(base, query)
    except Exception as e:
        print(f"  [FAIL] Could not read query: {e}")
        all_ok = False
        query = None

    # --- attribute file ---
    print("\n[Attribute file]")
    try:
        attr_raw = read_attr(attr_path)
        print(f"  [INFO] Raw attr length: {len(attr_raw)}, "
              f"first 5 values: {attr_raw[:5].tolist()}")

        # Handle stray header int (first value == n)
        if base is not None and len(attr_raw) == n + 1 and int(attr_raw[0]) == n:
            print("  [WARN] Attr file has a stray leading int (== n). "
                  "Stripping it for remaining checks.")
            attr = attr_raw[1:]
        else:
            attr = attr_raw

        if base is not None:
            all_ok &= check_attr_count(attr, n, "attr")
            all_ok &= check_attr_sorted(attr)
            all_ok &= check_attr_range(attr, n)
        else:
            print("  [SKIP] Base file failed; skipping attr cross-checks.")
    except Exception as e:
        print(f"  [FAIL] Could not read attr: {e}")
        all_ok = False

    print(f"\n{'=== ALL CHECKS PASSED ===' if all_ok else '=== SOME CHECKS FAILED ==='}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate iRangeGraph dataset files")
    parser.add_argument("--base", required=True, help="Path to base .bin file")
    parser.add_argument("--query", required=True, help="Path to query .bin file")
    parser.add_argument("--attr", required=True, help="Path to attribute .bin file")
    parser.add_argument("--skip-unique", action="store_true",
                        help="Skip the slow duplicate vector check")
    args = parser.parse_args()

    run_checks(args.base, args.query, args.attr, skip_unique=args.skip_unique)
