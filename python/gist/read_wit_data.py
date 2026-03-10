import tensorflow_datasets as tfds

ds = tfds.load(
    "wit",
    split="train",
    shuffle_files=False,
    data_dir="/workspaces/irange/tfds_data",
)

print(ds)

for ex in tfds.as_numpy(ds.take(1)):
    print(ex.keys())
    print(ex["image_url"])
    print(ex["page_title"])
    print(ex["original_height"], ex["original_width"])