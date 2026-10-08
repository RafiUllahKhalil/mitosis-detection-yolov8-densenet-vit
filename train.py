"""
train.py — Two-phase training schedule (Table 10).

Phase 1 (warmup): DenseNet frozen, train projection + transformer + head.
Phase 2 (fine-tune): unfreeze last DenseNet block + last K transformer blocks,
                     continue at the same low LR.

Optimizer AdamW (lr 1e-5, wd 1e-4), BCE with label smoothing 0.05,
EarlyStopping(patience=10) + ReduceLROnPlateau.
"""

import os
import keras

from config import CFG
from model import build_densevit, set_backbone_frozen, unfreeze_for_finetune


def _optimizer():
    return keras.optimizers.AdamW(
        learning_rate=CFG.LR, weight_decay=CFG.WEIGHT_DECAY
    )


def _compile(model):
    model.compile(
        optimizer=_optimizer(),
        loss=keras.losses.BinaryCrossentropy(
            label_smoothing=CFG.LABEL_SMOOTHING
        ),
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc"),
        ],
    )


def _phase_ckpt_path(tag):
    return os.path.join(CFG.OUTPUT_DIR, f"denseit_{tag}.keras")


def _maybe_resume(model, tag):
    """Best-effort resume: if a checkpoint from an earlier, interrupted run of
    this phase already exists, load it before fit() starts.

    This is NOT a perfect resume -- it restores model + optimizer weights
    from the last checkpoint that improved val_auc, but EarlyStopping's and
    ReduceLROnPlateau's patience counters live in memory and reset here, and
    epoch numbering restarts from 0 (so the stitched history will show a few
    repeated epochs after a resume). What it does guarantee: a Colab
    disconnect costs you at most the epochs since the last improvement, not
    the entire phase.
    """
    ckpt = _phase_ckpt_path(tag)
    if os.path.exists(ckpt):
        print(f"[train] Found existing checkpoint for phase '{tag}' -- "
              f"resuming from it instead of starting fresh: {ckpt}")
        model.load_weights(ckpt)
        return True
    return False


def _callbacks(tag):
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    ckpt = os.path.join(CFG.OUTPUT_DIR, f"denseit_{tag}.keras")
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc", mode="max",
            patience=CFG.EARLYSTOP_PATIENCE, restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=CFG.RLROP_FACTOR,
            patience=CFG.RLROP_PATIENCE, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            ckpt, monitor="val_auc", mode="max", save_best_only=True,
        ),
    ]


def train(model, train_ds, val_ds, class_weight=None):
    """Run the two-phase schedule and return the combined History-like dict."""

    # -------- Phase 1: frozen backbone warmup --------
    print("\n" + "=" * 60)
    print(f"PHASE 1 — frozen DenseNet warmup "
          f"({CFG.FROZEN_WARMUP_EPOCHS} epochs)")
    print("=" * 60)
    set_backbone_frozen(model, frozen=True)
    _compile(model)
    _maybe_resume(model, "phase1")
    hist1 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=CFG.FROZEN_WARMUP_EPOCHS,
        class_weight=class_weight,
        callbacks=_callbacks("phase1"),
        verbose=1,
    )

    # -------- Phase 2: fine-tune --------
    print("\n" + "=" * 60)
    print(f"PHASE 2 — fine-tune (unfreeze last DenseNet block + "
          f"last {CFG.UNFREEZE_LAST_VIT_BLOCKS} ViT blocks)")
    print("=" * 60)
    unfreeze_for_finetune(model)
    _compile(model)  # recompile so trainable changes take effect
    _maybe_resume(model, "finetune")
    remaining = max(1, CFG.EPOCHS - CFG.FROZEN_WARMUP_EPOCHS)
    hist2 = model.fit(
        train_ds, validation_data=val_ds,
        epochs=remaining,
        class_weight=class_weight,
        callbacks=_callbacks("finetune"),
        verbose=1,
    )

    # stitch histories
    history = {}
    for k in hist1.history:
        history[k] = hist1.history[k] + hist2.history.get(k, [])
    final_path = os.path.join(CFG.OUTPUT_DIR, "denseit_final.keras")
    model.save(final_path)
    print(f"\nSaved final model -> {final_path}")
    return history
