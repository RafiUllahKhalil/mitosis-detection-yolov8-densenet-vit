"""
model.py — DenseNet121 -> ViT-B/16 hybrid (Table 7: "Mitosis-DenseViT").

Flow (Table 7):
    Input 224x224x3
      -> DenseNet121 (no top)            -> 7x7x1024
      -> 1x1 Conv projection to 768      -> 7x7x768
      -> Upsample x2                     -> 14x14x768   (to reach 196 tokens)
      -> Reshape / tokenize              -> 196x768
      -> prepend learnable CLS token     -> 197x768
      -> add positional embedding        -> 197x768
      -> 12 x Transformer encoder block  -> 197x768   (MSA + GeLU MLP, dropout 0.1)
      -> LayerNorm
      -> CLS token extraction            -> 768
      -> Dense 512 (GeLU) -> Dropout 0.4
      -> Dense 128 (GeLU) -> Dropout 0.3
      -> Dense 64  (GeLU)
      -> Dense 1   (Sigmoid)

Note on the 7x7 -> 14x14 step: a 1x1 conv cannot change spatial size, so an
explicit upsample produces the 14x14 (=196 tokens) that Table 7's next row shows.
"""

import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model

from config import CFG


# ----------------------------------------------------------------------
# Transformer encoder block (standard ViT block: pre-norm, MSA, MLP)
# ----------------------------------------------------------------------
def transformer_block(x, name, hidden=768, heads=12, mlp_ratio=4, dropout=0.1):
    # --- Multi-head self-attention ---
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln1")(x)
    y = layers.MultiHeadAttention(
        num_heads=heads, key_dim=hidden // heads, dropout=dropout,
        name=f"{name}_msa"
    )(y, y)
    x = layers.Add(name=f"{name}_add1")([x, y])

    # --- MLP (GeLU) ---
    y = layers.LayerNormalization(epsilon=1e-6, name=f"{name}_ln2")(x)
    y = layers.Dense(hidden * mlp_ratio, activation="gelu", name=f"{name}_mlp1")(y)
    y = layers.Dropout(dropout, name=f"{name}_drop")(y)
    y = layers.Dense(hidden, name=f"{name}_mlp2")(y)
    x = layers.Add(name=f"{name}_add2")([x, y])
    return x


# ----------------------------------------------------------------------
# CLS token + positional embedding as a small custom layer
# ----------------------------------------------------------------------
class ClsAndPosEmbed(layers.Layer):
    def __init__(self, num_tokens, hidden, **kw):
        super().__init__(**kw)
        self.num_tokens = num_tokens
        self.hidden = hidden

    def build(self, input_shape):
        self.cls = self.add_weight(
            name="cls_token", shape=(1, 1, self.hidden),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )
        self.pos = self.add_weight(
            name="pos_embed", shape=(1, self.num_tokens + 1, self.hidden),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True,
        )

    def call(self, x):
        b = keras.ops.shape(x)[0]
        cls = keras.ops.broadcast_to(self.cls, [b, 1, self.hidden])
        x = keras.ops.concatenate([cls, x], axis=1)   # (b, 197, 768)
        return x + self.pos

    def get_config(self):
        c = super().get_config()
        c.update(num_tokens=self.num_tokens, hidden=self.hidden)
        return c


def build_densevit(load_pretrained_vit=True):
    H = CFG.HIDDEN_DIM

    inp = layers.Input(shape=CFG.IMAGE_SIZE, name="input_image")

    # --- DenseNet121 backbone (no top) ---
    # Build as its OWN model (no input_tensor) so it stays a single nested layer
    # named "densenet121" that freeze/unfreeze helpers can target. Passing
    # input_tensor= would inline all DenseNet layers into the top model.
    backbone = keras.applications.DenseNet121(
        include_top=False, weights="imagenet",
        input_shape=CFG.IMAGE_SIZE, name="densenet121",
    )
    feat = backbone(inp)                          # (7, 7, 1024)

    # --- 1x1 conv projection to 768, then upsample to 14x14 ---
    x = layers.Conv2D(H, 1, padding="same", name="proj_1x1")(feat)   # 7x7x768
    x = layers.UpSampling2D(size=2, interpolation="bilinear",
                            name="upsample_2x")(x)                    # 14x14x768

    # --- tokenize: 14x14x768 -> 196x768 ---
    x = layers.Reshape((CFG.NUM_TOKENS, H), name="tokenize")(x)

    # --- CLS token + positional embedding -> 197x768 ---
    x = ClsAndPosEmbed(CFG.NUM_TOKENS, H, name="cls_pos")(x)

    # --- 12 transformer encoder blocks ---
    for i in range(CFG.NUM_TRANSFORMER_BLOCKS):
        x = transformer_block(
            x, name=f"encoder_{i}", hidden=H,
            heads=CFG.NUM_HEADS, dropout=CFG.ENCODER_DROPOUT,
        )

    # --- final norm + CLS extraction ---
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)
    cls_out = layers.Lambda(lambda t: t[:, 0], name="cls_extract")(x)   # (b, 768)

    # --- MLP classification head (Table 7) ---
    h = layers.Dense(512, activation="gelu", name="head_dense1")(cls_out)
    h = layers.Dropout(CFG.MLP_DROPOUT_1, name="head_drop1")(h)
    h = layers.Dense(128, activation="gelu", name="head_dense2")(h)
    h = layers.Dropout(CFG.MLP_DROPOUT_2, name="head_drop2")(h)
    h = layers.Dense(64, activation="gelu", name="head_dense3")(h)
    out = layers.Dense(1, activation="sigmoid", name="output")(h)

    model = Model(inp, out, name="Mitosis_DenseViT")

    if load_pretrained_vit:
        _load_vit_encoder_weights(model)

    return model


def _load_vit_encoder_weights(model):
    """Copy pretrained ViT-B/16 transformer-encoder weights into our blocks.

    Only the encoder blocks (attention + MLP) transfer; patch/position embeddings
    do not, because our tokens come from DenseNet features, not image patches.
    Best-effort: if transformers/torch is unavailable or shapes differ, we skip
    and keep random init (the model still trains, just without the warm start).
    """
    try:
        from transformers import TFViTModel
        vit = TFViTModel.from_pretrained(CFG.VIT_PRETRAINED, use_safetensors=False)
    except Exception as e:  # noqa: BLE001
        print(f"[model] Skipping pretrained ViT load ({type(e).__name__}: {e}). "
              f"Using random-init transformer blocks.")
        return

    # NOTE: mapping HF ViT layer weights to Keras MultiHeadAttention requires
    # reshaping q/k/v kernels. This is done defensively; on any mismatch we skip.
    try:
        hf_layers = vit.vit.encoder.layer
        copied = 0
        for i in range(min(len(hf_layers), CFG.NUM_TRANSFORMER_BLOCKS)):
            # MLP weights map cleanly (Dense <-> Dense)
            hf = hf_layers[i]
            mlp1 = model.get_layer(f"encoder_{i}_mlp1")
            mlp2 = model.get_layer(f"encoder_{i}_mlp2")
            # TF Keras Dense layers store weights as .kernel (in, out) — no
            # transpose needed. (.weight + .T is the PyTorch nn.Linear pattern;
            # using it here against a TFViTModel silently failed every time,
            # via the except below, leaving these blocks at random init.)
            mlp1.set_weights([hf.intermediate.dense.kernel.numpy(),
                              hf.intermediate.dense.bias.numpy()])
            mlp2.set_weights([hf.output.dense.kernel.numpy(),
                              hf.output.dense.bias.numpy()])
            copied += 1
        print(f"[model] Loaded pretrained MLP weights for {copied} encoder blocks. "
              f"(Attention kept random-init for shape safety.)")
    except Exception as e:  # noqa: BLE001
        print(f"[model] Partial ViT load failed ({type(e).__name__}: {e}). "
              f"Continuing with random-init blocks.")


# ----------------------------------------------------------------------
# Freeze / unfreeze helpers for the two-phase schedule (Table 10)
# ----------------------------------------------------------------------
def set_backbone_frozen(model, frozen=True):
    """Freeze/unfreeze the entire DenseNet121 backbone."""
    bb = model.get_layer("densenet121")
    bb.trainable = not frozen


def unfreeze_for_finetune(model):
    """Fine-tune phase: unfreeze last DenseNet block + last K transformer blocks."""
    bb = model.get_layer("densenet121")
    bb.trainable = True
    # freeze all DenseNet layers before the final dense block
    started = False
    for lyr in bb.layers:
        if lyr.name.startswith(CFG.UNFREEZE_DENSENET_FROM):
            started = True
        lyr.trainable = started

    # keep only the last K transformer blocks trainable
    last_k = set(range(CFG.NUM_TRANSFORMER_BLOCKS - CFG.UNFREEZE_LAST_VIT_BLOCKS,
                       CFG.NUM_TRANSFORMER_BLOCKS))
    for i in range(CFG.NUM_TRANSFORMER_BLOCKS):
        trainable = i in last_k
        for suffix in ["ln1", "msa", "add1", "ln2", "mlp1", "drop", "mlp2", "add2"]:
            try:
                model.get_layer(f"encoder_{i}_{suffix}").trainable = trainable
            except ValueError:
                pass
