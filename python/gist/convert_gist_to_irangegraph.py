import os
import struct
import numpy as np


BASE_FVECS = "/workspaces/irange/exectable_data/gist1m/gist/gist_base.fvecs"
QUERY_FVECS = "/workspaces/irange/exectable_data/gist1m/gist/gist_query.fvecs"

OUT_DIR = "/workspaces/irange/exectable_data/gist1m/"
BASE_BIN = os.path.join(OUT_DIR, "gist_base.bin")
QUERY_BIN = os.path.join(OUT_DIR, "gist_query.bin")
ATTR_BIN = os.path.join(OUT_DIR, "gist_attr.bin")


def read_fvecs(path: str) -> np.ndarray:
    """
    Read .fvecs file into shape (n, d), dtype float32.
    Each vector is stored as:
      int32 dimension
      float32[d] components
    """
    with open(path, "rb") as f:
        data = np.fromfile(f, dtype=np.int32)

    if data.size == 0:
        raise ValueError(f"Empty file: {path}")

    d = data[0]
    print(f"Detected dimension {d} in {path}")
    if d <= 0:
        raise ValueError(f"Invalid dimension {d} in {path}")

    # reinterpret same bytes as float32 for vector values
    data = np.fromfile(path, dtype=np.float32)
    dim_as_int = np.fromfile(path, dtype=np.int32, count=1)[0]

    if dim_as_int != d:
        raise ValueError(f"Dimension mismatch in {path}")

    row_size = d + 1
    if data.size % row_size != 0:
        raise ValueError(
            f"File size not divisible by row size in {path}: "
            f"{data.size} floats, row_size={row_size}"
        )

    data = data.reshape(-1, row_size)
    dims = data[:, 0].view(np.int32)

    if not np.all(dims == d):
        raise ValueError(f"Not all vectors in {path} have dimension {d}")

    vecs = data[:, 1:].astype(np.float32, copy=False)
    return vecs


def write_vec_bin(path: str, arr: np.ndarray) -> None:
    """
    iRangeGraph vector format:
      int32 n
      int32 d
      float32[n, d]
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array")

    n, d = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("i", n))
        f.write(struct.pack("i", d))
        arr.tofile(f)


def write_attr_bin(path: str, arr: np.ndarray) -> None:
    """
    Raw int32 attributes, no header.
    """
    arr = np.asarray(arr, dtype=np.int32)
    if arr.ndim != 1:
        raise ValueError("Expected 1D array")
    with open(path, "wb") as f:
        arr.tofile(f)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Reading base vectors...")
    base = read_fvecs(BASE_FVECS)
    print("Base shape:", base.shape)

    print("Reading query vectors...")
    query = read_fvecs(QUERY_FVECS)
    print("Query shape:", query.shape)

    # Create deterministic attribute from vector norm.
    # Multiply to spread values, then cast to int32.
    norms = np.linalg.norm(base, axis=1)
    attr = np.round(norms * 1000).astype(np.int32)

    # Sort base vectors by attribute ascending
    order = np.argsort(attr, kind="stable")
    base_sorted = base[order]
    attr_sorted = attr[order]

    # Define dataset sizes to create
    sizes = [750000, 500000, 250000]
    
    for size in sizes:
        print(f"\n=== Creating {size} vector dataset ===")
        
        # Take first N vectors from sorted base
        base_subset = base_sorted[:size]
        attr_subset = attr_sorted[:size]
        
        # Define output paths
        size_label = f"{size // 1000}k"
        base_bin_path = os.path.join(OUT_DIR, f"gist_base_{size_label}.bin")
        query_bin_path = os.path.join(OUT_DIR, f"gist_query_{size_label}.bin")
        attr_bin_path = os.path.join(OUT_DIR, f"gist_attr_{size_label}.bin")
        
        print(f"Writing base bin ({size} vectors)...")
        write_vec_bin(base_bin_path, base_subset)
        
        print(f"Writing query bin...")
        write_vec_bin(query_bin_path, query)
        
        print(f"Writing attribute bin...")
        write_attr_bin(attr_bin_path, attr_subset)
        
        print(f"Base bin : {base_bin_path}")
        print(f"Query bin: {query_bin_path}")
        print(f"Attr bin : {attr_bin_path}")
        print(f"Sorted attr ascending: {np.all(attr_subset[:-1] <= attr_subset[1:])}")
    
    print("\n=== All datasets created successfully ===")


if __name__ == "__main__":
    main()