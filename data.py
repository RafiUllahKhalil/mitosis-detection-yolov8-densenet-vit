"""
data.py — Dataset construction and the tf.data input pipeline.

End-to-end training (no feature caching), so augmentation (Table 11) is applied
live on the training stream each epoch. Validation/test are clean (no aug).
"""

import os
import hashlib
import numpy as np
import pandas as pd
import tensorflow as tf

from config import CFG

AUTOTUNE = tf.data.AUTOTUNE


def _collect(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for f in filenames:
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                files.append((os.path.join(dirpath, f), os.path.basename(dirpath)))
    return files


def build_dataframe(root):
    files = _collect(root)
    if not files:
        raise FileNotFoundError(f"No images found under {root}")
    df = pd.DataFrame({"image_path": [p for p, _ in files],
                       "label": [l for _, l in files]})
    df["label_encoded"] = df["label"].apply(
        lambda x: 1 if x == CFG.POS_CLASS_NAME else 0
    )
    return df.sample(frac=1, random_state=CFG.SEED).reset_index(drop=True)


def get_dataframes():
    train_df = build_dataframe(CFG.TRAIN_PATH)
    valid_df = build_dataframe(CFG.VALID_PATH)
    test_df = build_dataframe(CFG.TEST_PATH)
    return train_df, valid_df, test_df


def _decode(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [CFG.HEIGHT, CFG.WIDTH],
                            method=tf.image.ResizeMethod.LANCZOS3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.ensure_shape(image, [CFG.HEIGHT, CFG.WIDTH, CFG.CHANNELS])
    label = tf.cast(label, tf.float32)
    return image, label


# ---- Table 11 augmentation, implemented with tf ops ----
def _augment(image, label):
    # Horizontal / vertical flip (p=0.5 each)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Brightness / contrast +/-15%
    image = tf.image.random_brightness(image, max_delta=0.15)
    image = tf.image.random_contrast(image, lower=0.85, upper=1.15)
    # Rotation +/-20deg and zoom 0.9-1.1x via a random layer stack
    image = _rotate_zoom_translate(image)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _rotate_zoom_translate(image):
    """Combined rotation (+/-20deg) + zoom (0.9-1.1x) + translation (+/-10%),
    implemented as a single affine transform via a pure TF op. This replaced
    an earlier version built from tf.keras.layers.Random{Rotation,Zoom,
    Translation}: those are Keras 3 layers, which convert their input using
    the model's active backend (JAX) -- but inside tf.data.Dataset.map(),
    `image` is a symbolic graph placeholder, not real data yet, so that
    conversion failed. A tf.py_function workaround fixed the crash but forced
    this step to run eagerly per-image (a real throughput cost). Using a raw
    TF op instead avoids the Keras/backend layer entirely, so it stays a
    normal graph op: correct under any Keras backend, and fully parallelizable
    across AUTOTUNE workers like the rest of the augmentation pipeline.
    """
    height = tf.cast(CFG.HEIGHT, tf.float32)
    width = tf.cast(CFG.WIDTH, tf.float32)
    cx, cy = width / 2.0, height / 2.0

    angle = tf.random.uniform([], -20.0, 20.0) * (np.pi / 180.0)  # +/-20 deg
    zoom = tf.random.uniform([], 0.9, 1.1)                         # 0.9-1.1x
    tx = tf.random.uniform([], -0.1, 0.1) * width                  # +/-10%
    ty = tf.random.uniform([], -0.1, 0.1) * height                 # +/-10%

    # ImageProjectiveTransformV3 samples the OUTPUT at the mapped INPUT
    # coordinate, so this is the inverse (output -> input) transform for a
    # rotate + scale + translate about the image center.
    inv_zoom = 1.0 / zoom
    cos_t = tf.cos(angle) * inv_zoom
    sin_t = tf.sin(angle) * inv_zoom
    a0, a1 = cos_t, sin_t
    b0, b1 = -sin_t, cos_t
    a2 = cx - a0 * cx - a1 * cy - tx
    b2 = cy - b0 * cx - b1 * cy - ty
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])

    image = tf.expand_dims(image, 0)
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[1:3],
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )
    image = tf.squeeze(image, 0)
    image = tf.ensure_shape(image, [CFG.HEIGHT, CFG.WIDTH, CFG.CHANNELS])
    return image


def _cache_path_for(df):
    """Build a stable, unique local-disk cache path for this dataframe's
    exact set of images. Caching lives on Colab's local disk (fast, but
    ephemeral -- it's rebuilt automatically each new runtime), not on Drive,
    since Drive is the slow part we're avoiding.
    """
    key = f"{len(df)}|{df.image_path.iloc[0]}|{df.image_path.iloc[-1]}"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    os.makedirs(CFG.CACHE_DIR, exist_ok=True)
    return os.path.join(CFG.CACHE_DIR, f"cache_{h}")


def make_dataset(df, batch_size=None, training=False):
    batch_size = batch_size or CFG.BATCH_SIZE
    paths = df.image_path.values
    labels = df.label_encoded.values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)

    if getattr(CFG, "CACHE_DATASETS", False):
        # Cache the decoded+resized (pre-augmentation) tensors to local disk.
        # The first epoch pays the full Drive-read + decode + resize cost as
        # before; every epoch after that reads from local disk instead,
        # which is typically far faster than Drive I/O. Augmentation still
        # runs fresh every epoch since it's mapped AFTER the cache.
        ds = ds.cache(_cache_path_for(df))

    if training:
        ds = ds.shuffle(1000, seed=CFG.SEED)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    return ds.batch(batch_size).prefetch(AUTOTUNE)
