import h5py
import numpy as np
import tensorflow as tf

IMG_KEY = "img"
MSK_KEY = "mask"

def _load_pair(idx, img_dir, msk_dir):
    idx = int(idx)

    with h5py.File(f"{img_dir}/image_{idx}.h5", "r") as f:
        img = f[IMG_KEY][:].astype(np.float32)

    with h5py.File(f"{msk_dir}/mask_{idx}.h5", "r") as f:
        msk = f[MSK_KEY][:].astype(np.float32)

    if msk.ndim == 2:
        msk = msk[..., None]

    mn, mx = img.min(), img.max()
    img = (img - mn) / (mx - mn) if mx > mn else np.zeros_like(img)

    msk = (msk > 0.5).astype(np.float32)
    return img, msk

def make_dataset(ids, img_dir, msk_dir, batch_size=16, shuffle=False):
    def _tf_loader(i):
        img, msk = tf.numpy_function(
            lambda x: _load_pair(x, img_dir, msk_dir),
            [i],
            [tf.float32, tf.float32]
        )
        img.set_shape([128,128,14])
        msk.set_shape([128,128,1])
        return img, msk

    ds = tf.data.Dataset.from_tensor_slices(ids)
    if shuffle:
        ds = ds.shuffle(1000, seed=42)
    ds = ds.map(_tf_loader, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds
