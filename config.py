"""
config.py — Central configuration for the Mitosis-DenseViT hybrid.

All values trace directly to the paper:
  * Table 7  — architecture (sizes, dropout rates)
  * Table 10 — training configuration
  * Table 11 — data augmentation
"""

from types import SimpleNamespace

CFG = SimpleNamespace(
    # ---- data ----
    TRAIN_PATH="/content/drive/MyDrive/Split_Folder/train",
    VALID_PATH="/content/drive/MyDrive/Split_Folder/val",
    TEST_PATH="/content/drive/MyDrive/Split_Folder/test",
    # folder-name -> label mapping (must match your subfolders exactly)
    POS_CLASS_NAME="Positive_Mitosis",   # -> 1
    NEG_CLASS_NAME="Negative_Mitosis",   # -> 0

    # ---- image ----
    HEIGHT=224, WIDTH=224, CHANNELS=3,
    IMAGE_SIZE=(224, 224, 3),

    # ---- model (Table 7) ----
    HIDDEN_DIM=768,          # token dimension
    NUM_TOKENS=196,          # 14x14 spatial tokens (+1 CLS -> 197)
    NUM_TRANSFORMER_BLOCKS=12,
    NUM_HEADS=12,
    MLP_DROPOUT_1=0.4,       # after Dense(512)
    MLP_DROPOUT_2=0.3,       # after Dense(128)
    ENCODER_DROPOUT=0.1,     # inside transformer blocks
    VIT_PRETRAINED="google/vit-base-patch16-224",

    # ---- training (Table 10) ----
    OPTIMIZER="adamw",
    LR=1e-5,
    WEIGHT_DECAY=1e-4,
    BATCH_SIZE=16,
    EPOCHS=50,
    LABEL_SMOOTHING=0.05,
    EARLYSTOP_PATIENCE=10,
    RLROP_FACTOR=0.5,
    RLROP_PATIENCE=4,
    # DenseNet frozen for the first N epochs (Table 10: 5-10), then fine-tune
    FROZEN_WARMUP_EPOCHS=8,
    # during fine-tune, how many trailing transformer blocks to unfreeze
    UNFREEZE_LAST_VIT_BLOCKS=4,
    # unfreeze the last DenseNet dense-block ("conv5") during fine-tune
    UNFREEZE_DENSENET_FROM="conv5",

    # ---- misc ----
    SEED=42,
    TF_SEED=768,
    OUTPUT_DIR="/content/drive/MyDrive/Split_Folder/denseit_outputs",
    CLASS_NAMES=("Negative_Mitosis", "Positive_Mitosis"),

    # ---- performance ----
    # Cache decoded+resized images to local Colab disk after the first epoch,
    # instead of re-reading + re-decoding from the Drive mount every epoch.
    # Off by default -- turn on once you've sanity-checked your dataset size
    # against Colab's local disk (decoded float32 224x224x3 images are
    # ~590KB each, e.g. 10k images ~= 6GB). Cache is ephemeral: it's rebuilt
    # automatically each new runtime, so there's no cleanup needed.
    CACHE_DATASETS=False,
    CACHE_DIR="/content/tf_cache",
)
