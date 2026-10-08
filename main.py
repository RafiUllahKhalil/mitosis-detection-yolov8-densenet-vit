"""
main.py — End-to-end pipeline for the Mitosis-DenseViT hybrid.

Run in Colab:
    %cd /content/drive/MyDrive/Split_Folder/mitosis_denseit   # where these files live
    !python main.py

Or from a notebook cell:
    import main
    main.run()
"""

import os
import random
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from config import CFG
from data import get_dataframes, make_dataset
from model import build_densevit
from train import train
from evaluate import plot_training_curves, evaluate


def _seed():
    random.seed(CFG.SEED)
    np.random.seed(CFG.SEED)
    tf.random.set_seed(CFG.TF_SEED)


def run():
    _seed()
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)

    # --- data ---
    print("Building dataframes...")
    train_df, valid_df, test_df = get_dataframes()
    print(f"train={len(train_df)}  valid={len(valid_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts().to_dict())

    train_ds = make_dataset(train_df, training=True)
    val_ds = make_dataset(valid_df, training=False)
    test_ds = make_dataset(test_df, training=False)

    # --- class weights (handles imbalance) ---
    y = train_df.label_encoded.values
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y)
    class_weight = {0: float(cw[0]), 1: float(cw[1])}
    print("class_weight:", class_weight)

    # --- model ---
    print("Building DenseNet121 -> ViT-B/16 hybrid...")
    model = build_densevit(load_pretrained_vit=True)
    model.summary()

    # --- train (two-phase) ---
    history = train(model, train_ds, val_ds, class_weight=class_weight)

    # --- evaluate ---
    plot_training_curves(history)
    metrics = evaluate(model, test_ds, test_df, threshold=0.5)
    print("Final test metrics:", {k: round(v, 4) for k, v in metrics.items()})
    return model, history, metrics


if __name__ == "__main__":
    run()
