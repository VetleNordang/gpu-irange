import tensorflow as tf

path = "/workspaces/irange/executable_data/yt_tf/train0000.tfrecord"

for raw in tf.data.TFRecordDataset(path).take(1):
    ex = tf.train.Example()
    ex.ParseFromString(raw.numpy())

    vid_id = ex.features.feature["id"].bytes_list.value[0]
    rgb    = list(ex.features.feature["mean_rgb"].float_list.value)
    audio  = list(ex.features.feature["mean_audio"].float_list.value)

    print(f"id:         {vid_id}")
    print(f"mean_rgb:   dim={len(rgb)},   first5={rgb[:5]}")
    print(f"mean_audio: dim={len(audio)}, first5={audio[:5]}")
