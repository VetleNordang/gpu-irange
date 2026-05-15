"""
Validates all .bin files in executable_data/ and checks consistency within each dataset folder.

Binary formats (from utils.h):
  vectors/queries : [int32 n][int32 d][n*d float32]
  attr            : [n int32] raw, no header, pre-sorted ascending
  query_ranges    : [query_nb * 2 int32] pairs of (ql, qr), no header
  groundtruth     : [query_nb * K int32] neighbour ids, no header
  pq_codes        : [int32 n][int32 d][int32 M][int32 nbits][int32 code_size][data...]
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path

QUERY_K = 10
OK = "\033[92m OK\033[0m"
WARN = "\033[93m WARN\033[0m"
ERR = "\033[91m ERR\033[0m"

def tag(ok, msg=""):
    return (OK if ok else ERR) + (f" — {msg}" if msg else "")


def read_vector_file(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n, d = struct.unpack("ii", f.read(8))
    expected = 8 + n * d * 4
    ok = size == expected
    return n, d, ok, size, expected


def read_attr_file(path, expected_n):
    size = os.path.getsize(path)
    n = size // 4
    match = n == expected_n
    remainder = size % 4
    # spot-check: first few values should be sorted
    with open(path, "rb") as f:
        vals = np.frombuffer(f.read(min(40, size)), dtype=np.int32)
    sorted_ok = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
    return n, match, remainder == 0, sorted_ok


def read_query_range_file(path, expected_query_nb):
    size = os.path.getsize(path)
    n_pairs = size // 8
    match = n_pairs == expected_query_nb
    with open(path, "rb") as f:
        raw = np.frombuffer(f.read(min(80, size)), dtype=np.int32)
    pairs = raw.reshape(-1, 2)
    valid_ranges = all(pairs[i, 0] <= pairs[i, 1] for i in range(len(pairs)))
    return n_pairs, match, valid_ranges


def read_groundtruth_file(path, expected_query_nb):
    size = os.path.getsize(path)
    total = size // 4
    k = total // expected_query_nb if expected_query_nb > 0 else 0
    match = (total == expected_query_nb * k) and k > 0
    return total, k, match


def read_pq_codes_file(path):
    size = os.path.getsize(path)
    if size < 20:
        return None, None, None, None, None, False
    with open(path, "rb") as f:
        n, d, M, nbits, code_size = struct.unpack("iiiii", f.read(20))
    expected = 20 + n * code_size
    ok = size == expected
    return n, d, M, nbits, code_size, ok


def validate_dataset(folder: Path):
    print(f"\n{'='*60}")
    print(f"  {folder.relative_to(folder.parents[2])}")
    print(f"{'='*60}")

    issues = 0

    # --- find main vector file ---
    vec_files = [f for f in folder.glob("*.bin")
                 if "attr" not in f.name and "query" not in f.name
                 and "pq" not in f.stem.split("/")[-1]
                 and f.parent == folder]
    query_files = [f for f in folder.glob("*.bin") if "query" in f.name and "attr" not in f.name]
    attr_files = [f for f in folder.glob("*.bin") if "attr" in f.name and "query" not in f.name]
    query_attr_files = [f for f in folder.glob("*.bin") if "attr" in f.name and "query" in f.name]

    data_n, data_d, query_n = 0, 0, 0

    for vf in sorted(vec_files):
        n, d, size_ok, actual, expected = read_vector_file(vf)
        data_n, data_d = n, d
        status = tag(size_ok, f"size mismatch: got {actual}, expected {expected}" if not size_ok else "")
        print(f"  [vectors]  {vf.name}: n={n:,}  d={d}{status}")
        if not size_ok:
            issues += 1

    for qf in sorted(query_files):
        n, d, size_ok, actual, expected = read_vector_file(qf)
        query_n = n
        dim_ok = d == data_d
        status = tag(size_ok and dim_ok,
                     f"dim mismatch: {d} vs data {data_d}" if not dim_ok else
                     f"size mismatch" if not size_ok else "")
        print(f"  [query]    {qf.name}: n={n:,}  d={d}{status}")
        if not (size_ok and dim_ok):
            issues += 1

    for af in sorted(attr_files):
        if data_n == 0:
            print(f"  [attr]     {af.name}: skipped (no vector file found first)")
            continue
        n, match, aligned, sorted_ok = read_attr_file(af, data_n)
        ok = match and aligned
        parts = []
        if not match:
            parts.append(f"count {n:,} != data_n {data_n:,}")
        if not aligned:
            parts.append("file size not multiple of 4")
        if not sorted_ok:
            parts.append("first values not sorted!")
        status = tag(ok and sorted_ok, ", ".join(parts) if parts else "")
        print(f"  [attr]     {af.name}: n={n:,}  sorted={sorted_ok}{status}")
        if not (ok and sorted_ok):
            issues += 1

    for qaf in sorted(query_attr_files):
        if query_n == 0:
            print(f"  [qattr]    {qaf.name}: skipped (no query file found first)")
            continue
        n, match, aligned, _ = read_attr_file(qaf, query_n)
        ok = match and aligned
        status = tag(ok, f"count {n:,} != query_n {query_n:,}" if not match else "")
        print(f"  [qattr]    {qaf.name}: n={n:,}{status}")
        if not ok:
            issues += 1

    # --- index file ---
    for idx in sorted(folder.glob("*.index")):
        size = os.path.getsize(idx)
        print(f"  [index]    {idx.name}: {size/1e6:.1f} MB")

    # --- query ranges ---
    qr_dir = folder / "query_ranges"
    if qr_dir.exists() and query_n > 0:
        suffixes_found = sorted(qr_dir.glob("*.bin"))
        bad = 0
        for qrf in suffixes_found:
            n_pairs, match, valid = read_query_range_file(qrf, query_n)
            if not match or not valid:
                print(f"  [qrange]   {qrf.name}: {ERR} n_pairs={n_pairs}, match={match}, valid_ranges={valid}")
                bad += 1
        if bad == 0:
            print(f"  [qrange]   {len(suffixes_found)} files{tag(True)}")
        issues += bad

    # --- groundtruth ---
    gt_dir = folder / "groundtruth"
    if gt_dir.exists() and query_n > 0:
        gt_files = sorted(gt_dir.glob("*.bin"))
        bad = 0
        for gtf in gt_files:
            total, k, ok = read_groundtruth_file(gtf, query_n)
            if not ok:
                print(f"  [gt]       {gtf.name}: {ERR} total={total}, k={k}")
                bad += 1
        if bad == 0:
            print(f"  [gt]       {len(gt_files)} files (K={k if gt_files else '?'}){tag(True)}")
        issues += bad

    # --- pq codes ---
    pq_dir = folder / "pq"
    if pq_dir.exists():
        for pqf in sorted(pq_dir.glob("*.bin")):
            n, d, M, nbits, code_size, ok = read_pq_codes_file(pqf)
            if n is None:
                print(f"  [pq]       {pqf.name}: too small to parse{ERR}")
                issues += 1
            else:
                match_n = n == data_n if data_n > 0 else True
                status = tag(ok and match_n,
                             f"n={n:,} != data_n={data_n:,}" if not match_n else
                             "size mismatch" if not ok else "")
                print(f"  [pq]       {pqf.name}: n={n:,} M={M} nbits={nbits} code_size={code_size}{status}")
                if not (ok and match_n):
                    issues += 1

    if issues == 0:
        print(f"  Result: all checks passed")
    else:
        print(f"  Result: {issues} issue(s) found")
    return issues


def main():
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parents[1] / "executable_data"
    if not base.exists():
        print(f"Directory not found: {base}")
        sys.exit(1)

    print(f"Scanning: {base}")
    total_issues = 0

    # Only validate size-level dirs (e.g. gist1m/1000k), skip groundtruth/query_ranges/pq subdirs
    skip = {"groundtruth", "query_ranges", "pq"}
    for size_dir in sorted(base.rglob("*")):
        if not size_dir.is_dir():
            continue
        if any(p.name in skip for p in size_dir.parents):
            continue
        if size_dir.name in skip:
            continue
        bin_files = list(size_dir.glob("*.bin")) + list(size_dir.glob("*.index"))
        if bin_files and size_dir.parent != base:
            total_issues += validate_dataset(size_dir)

    print(f"\n{'='*60}")
    if total_issues == 0:
        print(f"  All datasets{OK}")
    else:
        print(f"  Total issues: {total_issues}{ERR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
