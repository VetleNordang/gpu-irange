import os
import struct
import numpy as np

BASE_PATH  = "/workspaces/irange/executable_data/sift1b/bigann_base.bvecs"
QUERY_PATH = "/workspaces/irange/executable_data/sift1b/bigann_query.bvecs"
OUT_ROOT   = "/workspaces/irange/executable_data/sift1b"

SIZES      = [10_000_000, 30_000_000, 50_000_000, 100_000_000]
SIZE_NAMES = ["10m", "30m", "50m", "100m"]
CHUNK      = 1_000_000
SUBDIRS    = ["groundtruth", "pq", "query_ranges",
              "results/analysis", "results/cpu_parallel",
              "results/gpu_normal", "results/gpu_pq", "results/gpu_root"]


def read_bvecs_dim(path):
    with open(path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    return dim


def read_exact(f, n_bytes):
    buf = bytearray()
    while len(buf) < n_bytes:
        chunk = f.read(n_bytes - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


def stream_bvecs(path, max_n, dim):
    bytes_per_vec = 4 + dim
    with open(path, "rb") as f:
        remaining = max_n
        while remaining > 0:
            n = min(CHUNK, remaining)
            buf = read_exact(f, n * bytes_per_vec)
            if not buf:
                break
            got = len(buf) // bytes_per_vec
            arr = np.frombuffer(buf[:got * bytes_per_vec], dtype=np.uint8).reshape(got, bytes_per_vec)
            vecs = arr[:, 4:].astype(np.float32)
            yield vecs
            remaining -= got


def write_size(out_dir, name, n, dim, vec_path_tmp, query_vecs):
    os.makedirs(out_dir, exist_ok=True)
    for s in SUBDIRS:
        os.makedirs(os.path.join(out_dir, s), exist_ok=True)

    vec_out  = os.path.join(out_dir, f"{name}.bin")
    attr_out = os.path.join(out_dir, f"{name}_attr.bin")
    q_out    = os.path.join(out_dir, f"{name}_query.bin")

    # open full mmap, slice to n — avoids relying on silent size truncation
    total = os.path.getsize(vec_path_tmp) // (dim * 4)
    data = np.memmap(vec_path_tmp, dtype=np.float32, mode="r", shape=(total, dim))
    data = data[:n]
    with open(vec_out, "wb") as f:
        f.write(struct.pack("ii", n, dim))
        # write in chunks to avoid huge peak RAM
        for start in range(0, n, CHUNK):
            f.write(data[start:start + CHUNK].tobytes())
    print(f"  {vec_out}  ({os.path.getsize(vec_out) // 1024 // 1024} MB)")

    attrs = np.arange(n, dtype=np.int32)
    with open(attr_out, "wb") as f:
        f.write(attrs.tobytes())
    print(f"  {attr_out}  ({os.path.getsize(attr_out) // 1024 // 1024} MB)")

    nq, qdim = query_vecs.shape
    with open(q_out, "wb") as f:
        f.write(struct.pack("ii", nq, qdim))
        f.write(query_vecs.tobytes())
    print(f"  {q_out}  ({os.path.getsize(q_out) // 1024} KB)")


def read_query(path):
    dim = read_bvecs_dim(path)
    bytes_per_vec = 4 + dim
    size = os.path.getsize(path)
    nq = size // bytes_per_vec
    with open(path, "rb") as f:
        buf = f.read()
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(nq, bytes_per_vec)
    return arr[:, 4:].astype(np.float32)


if __name__ == "__main__":
    max_n = max(SIZES)
    dim   = read_bvecs_dim(BASE_PATH)
    print(f"dim={dim}, reading up to {max_n:,} vectors in chunks of {CHUNK:,}")

    # stream base vectors into a flat temp mmap
    tmp_path = os.path.join(OUT_ROOT, "_tmp_base_100m.bin")
    tmp = np.memmap(tmp_path, dtype=np.float32, mode="w+", shape=(max_n, dim))

    written = 0
    for chunk_vecs in stream_bvecs(BASE_PATH, max_n, dim):
        n = len(chunk_vecs)
        tmp[written:written + n] = chunk_vecs
        written += n
        print(f"  read {written:,} / {max_n:,}", end="\r")
    tmp.flush()
    del tmp  # release write handle before re-opening read-only in write_size
    print(f"\nDone reading: {written:,} vectors")

    print("\nReading query vectors ...")
    query_vecs = read_query(QUERY_PATH)
    print(f"  {len(query_vecs)} query vectors, dim={query_vecs.shape[1]}")

    for size, size_name in zip(SIZES, SIZE_NAMES):
        n = min(size, written)
        name = f"sift_{size_name}"
        out_dir = os.path.join(OUT_ROOT, size_name)
        print(f"\n=== {size_name} ({n:,} vectors) ===")
        write_size(out_dir, name, n, dim, tmp_path, query_vecs)

    print(f"\nCleaning up temp file {tmp_path} ...")
    os.remove(tmp_path)
    print("All done.")
