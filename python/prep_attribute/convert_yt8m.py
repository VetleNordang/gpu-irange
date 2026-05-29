import os
import glob
import struct
import random
import sys
import tempfile
import numpy as np
import multiprocessing as mp

TF_DIR   = "/workspaces/irange/executable_data/yt_tf"
AUDI_DIR = "/workspaces/irange/executable_data/audi"
VIDE_DIR = "/workspaces/irange/executable_data/video"
SEED     = 42
SIZES    = [1_000_000, 2_000_000, 4_000_000, 6_134_598]
SIZE_NAMES = ["1m", "2m", "4m", "6m"]
NUM_WORKERS = min(8, mp.cpu_count())

MODALITY = "audio"
if "--modality" in sys.argv:
    MODALITY = sys.argv[sys.argv.index("--modality") + 1]

DIM = 128 if MODALITY == "audio" else 1024
EMB_KEY = "mean_audio" if MODALITY == "audio" else "mean_rgb"

print(f"Modality: {MODALITY}  Dim: {DIM}  Workers: {NUM_WORKERS}")


def read_shard_chunk(args):
    import tensorflow as tf
    shard_paths, worker_id, tmp_dir, emb_key, dim = args

    tmp_vec = os.path.join(tmp_dir, f"worker_{worker_id}_vec.bin")
    tmp_ids = os.path.join(tmp_dir, f"worker_{worker_id}_ids.txt")

    count = 0
    with open(tmp_vec, "wb") as fv, open(tmp_ids, "w") as fi:
        for si, path in enumerate(shard_paths):
            try:
                for raw in tf.data.TFRecordDataset(path):
                    ex = tf.train.Example()
                    ex.ParseFromString(raw.numpy())
                    vid_id = ex.features.feature["id"].bytes_list.value[0].decode()
                    vec = ex.features.feature[emb_key].float_list.value[:]
                    if len(vec) != dim:
                        continue
                    fv.write(np.array(vec, dtype=np.float32).tobytes())
                    fi.write(vid_id + "\n")
                    count += 1
            except Exception as e:
                print(f"  Worker {worker_id}: error reading {path}: {e}")

            if (si + 1) % 200 == 0:
                print(f"  Worker {worker_id}: {si+1}/{len(shard_paths)} shards, {count:,} records")

    print(f"  Worker {worker_id}: done — {count:,} records")
    return tmp_vec, tmp_ids, count


def write_dataset(embeddings, out_dir, name_prefix, size_name, dim):
    os.makedirs(out_dir, exist_ok=True)
    for subdir in ["groundtruth", "pq", "query_ranges",
                   "results/analysis", "results/cpu_parallel",
                   "results/gpu_normal", "results/gpu_pq", "results/gpu_root"]:
        os.makedirs(os.path.join(out_dir, subdir), exist_ok=True)

    N = len(embeddings)
    norms  = np.linalg.norm(embeddings, axis=1)
    order  = np.argsort(norms)
    attrs  = np.empty(N, dtype=np.int32)
    attrs[order] = np.arange(N, dtype=np.int32)

    sort_idx   = np.argsort(attrs)
    embeddings = embeddings[sort_idx]
    attrs      = attrs[sort_idx]

    vec_path = os.path.join(out_dir, f"{name_prefix}_{size_name}.bin")
    with open(vec_path, "wb") as f:
        f.write(struct.pack("ii", N, dim))
        f.write(embeddings.tobytes())
    print(f"  Wrote {vec_path}  ({os.path.getsize(vec_path)//1024//1024} MB)")

    attr_path = os.path.join(out_dir, f"{name_prefix}_attr_{size_name}.bin")
    with open(attr_path, "wb") as f:
        f.write(attrs.tobytes())
    print(f"  Wrote {attr_path}  ({os.path.getsize(attr_path)//1024//1024} MB)")

    rng2  = np.random.default_rng(SEED)
    q_idx = rng2.choice(N, size=1000, replace=False)
    q_path = os.path.join(out_dir, f"{name_prefix}_query.bin")
    with open(q_path, "wb") as f:
        f.write(struct.pack("ii", 1000, dim))
        f.write(embeddings[q_idx].tobytes())
    print(f"  Wrote {q_path}  ({os.path.getsize(q_path)//1024} KB)")


if __name__ == "__main__":
    all_files = (
        sorted(glob.glob(os.path.join(TF_DIR, "train*.tfrecord"))) +
        sorted(glob.glob(os.path.join(TF_DIR, "validate*.tfrecord"))) +
        sorted(glob.glob(os.path.join(TF_DIR, "test*.tfrecord")))
    )
    rng = random.Random(SEED)
    rng.shuffle(all_files)
    print(f"Total shards: {len(all_files)}")

    chunks = [all_files[i::NUM_WORKERS] for i in range(NUM_WORKERS)]

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"\nReading shards with {NUM_WORKERS} workers ...")
        args = [(chunks[i], i, tmp_dir, EMB_KEY, DIM) for i in range(NUM_WORKERS)]

        with mp.Pool(NUM_WORKERS) as pool:
            results = pool.map(read_shard_chunk, args)

        total = sum(r[2] for r in results)
        print(f"\nMerging {total:,} records (dedup by video id) ...")

        # Pre-allocate output array; fill row by row to avoid Python list overhead
        all_embeddings = np.empty((total, DIM), dtype=np.float32)
        seen_ids = set()
        write_pos = 0

        for tmp_vec, tmp_ids, count in results:
            if count == 0:
                continue
            with open(tmp_ids) as fi:
                ids = fi.read().splitlines()
            vecs = np.fromfile(tmp_vec, dtype=np.float32).reshape(count, DIM)

            for i, vid_id in enumerate(ids):
                if vid_id in seen_ids:
                    continue
                seen_ids.add(vid_id)
                all_embeddings[write_pos] = vecs[i]
                write_pos += 1

            del vecs

        all_embeddings = all_embeddings[:write_pos]
        print(f"Total unique embeddings: {write_pos:,}")

    if MODALITY == "audio":
        name_prefix, out_root = "yt_aud", AUDI_DIR
    else:
        name_prefix, out_root = "youtube_rgb", VIDE_DIR

    for size, size_name in zip(SIZES, SIZE_NAMES):
        n = min(size, len(all_embeddings))
        print(f"\n=== {size_name} ({n:,} embeddings) ===")
        write_dataset(all_embeddings[:n].copy(), os.path.join(out_root, size_name), name_prefix, size_name, DIM)

    print("\nAll done.")
