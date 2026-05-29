import os
import glob
import tensorflow as tf

TF_DIR = "/workspaces/irange/executable_data/yt_tf"

partitions = {
    "train":    sorted(glob.glob(os.path.join(TF_DIR, "train*.tfrecord"))),
    "validate": sorted(glob.glob(os.path.join(TF_DIR, "validate*.tfrecord"))),
    "test":     sorted(glob.glob(os.path.join(TF_DIR, "test*.tfrecord"))),
}

total = 0
for partition, files in partitions.items():
    count = 0
    for path in files:
        for _ in tf.data.TFRecordDataset(path):
            count += 1
    print(f"{partition}: {count:,} embeddings across {len(files)} shards")
    total += count

print(f"\nTotal: {total:,} embeddings")
