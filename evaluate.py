"""
evaluate.py — Test-set evaluation and plots.

Single-sigmoid output, so predictions use a probability threshold (default 0.5),
NOT argmax. THRESHOLD can be lowered to trade precision for recall.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve, auc as sk_auc,
)

from config import CFG


def plot_training_curves(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6))
    a1.plot(history["loss"], marker="o", label="train")
    a1.plot(history["val_loss"], marker="o", label="val")
    a1.set_title("Loss"); a1.set_xlabel("epoch"); a1.legend(); a1.grid(alpha=.3)
    a2.plot(history["accuracy"], marker="o", label="train")
    a2.plot(history["val_accuracy"], marker="o", label="val")
    a2.set_title("Accuracy"); a2.set_xlabel("epoch"); a2.legend(); a2.grid(alpha=.3)
    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "training_curves.png"), dpi=150,
                bbox_inches="tight")
    plt.show()


def evaluate(model, test_ds, test_df, threshold=0.5):
    probs = model.predict(test_ds, verbose=1).ravel()
    preds = (probs > threshold).astype(int)
    y_true = test_df.label_encoded.values.astype(int)

    # confusion matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt="g", cmap="Greens",
                xticklabels=CFG.CLASS_NAMES, yticklabels=CFG.CLASS_NAMES,
                linewidths=1, linecolor="black")
    plt.title("Confusion Matrix"); plt.xlabel("Predicted"); plt.ylabel("True")
    plt.yticks(rotation=0)
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "confusion_matrix.png"), dpi=150,
                bbox_inches="tight")
    plt.show()

    print(classification_report(y_true, preds, target_names=CFG.CLASS_NAMES))

    # scalar metrics
    acc = accuracy_score(y_true, preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average="weighted")
    mcc = matthews_corrcoef(y_true, preds)
    print("=" * 45)
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {p:.4f}")
    print(f"recall   : {r:.4f}")
    print(f"f1       : {f1:.4f}")
    print(f"mcc      : {mcc:.4f}")
    print("=" * 45)

    # ROC
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"DenseViT (AUC={sk_auc(fpr, tpr):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.grid(alpha=.3)
    plt.savefig(os.path.join(CFG.OUTPUT_DIR, "roc_curve.png"), dpi=150,
                bbox_inches="tight")
    plt.show()

    return dict(accuracy=acc, precision=p, recall=r, f1=f1, mcc=mcc,
                auc=sk_auc(fpr, tpr))
