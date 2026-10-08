# Sequential YOLO + Mitosis-DenseViT Pipeline

A two-stage deep learning pipeline for mitosis detection and classification in breast histopathology images:

1. **Detection** ([`detection/`](detection)) — a YOLOv8-based detector (YOLO-det) that localizes candidate mitotic figures.
2. **Classification** ([`classification/`](classification)) — **Mitosis-DenseViT**, a hybrid DenseNet121 + ViT-B/16 classifier that re-examines each candidate patch from stage 1 and classifies it as mitotic or non-mitotic.

This code accompanies the manuscript *"Sequential Deep Learning Framework Using YOLO, DenseNet, and ViT for Mitosis Detection and Classification,"* currently under review. Citation details will be added here once the paper is published.

## Pipeline overview

```
Histopathology image
      |
      v
[ detection/  ]  YOLOv8 (YOLO-det)
      |  candidate mitotic-figure patches
      v
[classification/]  Mitosis-DenseViT (DenseNet121 + ViT-B/16)
      |
      v
Mitotic / Non-mitotic
```

See each subfolder's own README for stage-specific setup and usage:
- [`detection/README.md`](detection/README.md)
- Classification architecture and usage are documented below.

## Classification stage architecture (Mitosis-DenseViT)

```
Input 224x224x3
  -> DenseNet121 (no top)            -> 7x7x1024
  -> 1x1 Conv projection to 768      -> 7x7x768
  -> Upsample x2                     -> 14x14x768
  -> Reshape / tokenize              -> 196x768
  -> prepend learnable CLS token     -> 197x768
  -> add positional embedding        -> 197x768
  -> 12 x Transformer encoder block  -> 197x768
  -> LayerNorm -> CLS token extraction -> 768
  -> Dense 512 (GeLU) -> Dropout 0.4
  -> Dense 128 (GeLU) -> Dropout 0.3
  -> Dense 64  (GeLU)
  -> Dense 1   (Sigmoid)
```

DenseNet121 learns local morphological/textural features; the transformer encoder (initialized where possible from pretrained ViT-B/16 weights) learns long-range contextual relationships between patches. Training uses a two-phase schedule: a frozen-backbone warmup, then fine-tuning of the last DenseNet block and the last few transformer blocks.

## Repository structure

| Path | Purpose |
|---|---|
| `detection/Custom_YOLOV_8.ipynb` | YOLOv8 training notebook for the detection stage |
| `detection/README.md` | Detection-stage setup and dataset access notes |
| `classification/config.py` | Central configuration (paths, architecture, training hyperparameters) |
| `classification/data.py` | Dataset construction and `tf.data` input pipeline, including augmentation |
| `classification/model.py` | Model definition (`build_densevit`) and freeze/unfreeze helpers |
| `classification/train.py` | Two-phase training loop |
| `classification/evaluate.py` | Test-set evaluation, confusion matrix, ROC curve |
| `classification/main.py` | End-to-end entry point |
| `classification/Mitosis_Dense_ViT_.ipynb` | Notebook version of the classification pipeline (e.g. for Colab) |

## Installation

```bash
pip install -r requirements.txt
```

Developed and tested with Python 3.10+, TensorFlow/Keras 3, Ultralytics YOLOv8, and the Hugging Face `transformers` library for pretrained ViT weight initialization.

## Usage — classification stage

1. Organize your data as:
   ```
   <root>/train/Positive_Mitosis/*.png
   <root>/train/Negative_Mitosis/*.png
   <root>/val/...
   <root>/test/...
   ```
2. Update the paths and settings in `classification/config.py` to match your environment (the defaults reflect the authors' own Colab/Google Drive setup and will need to be changed).
3. Run:
   ```bash
   cd classification
   python main.py
   ```

## Usage — detection stage

See [`detection/README.md`](detection/README.md) for setup, including how to supply your own Roboflow API key via an environment variable (never hardcode a key in the notebook).

## Data availability

- **Public benchmark datasets** used to train and evaluate this model are freely available from their original sources: [MITOS-12](http://ludo17.free.fr/mitos_2012/dataset.html), [MITOS-14](https://mitos-atypia-14.grand-challenge.org/), and [TUPAC-16](http://tupac.tue-image.nl/).
- **Custom pathologist-verified annotations** generated during this study (candidate-patch labels reviewed by an expert pathologist) are not included in this repository, as their release was not covered under the terms of the original pathologist collaboration.

## License

This code is released under the MIT License — see [LICENSE](LICENSE).

## Contact

Questions about this code can be directed to the corresponding author, Muhammad Sajjad (muhammad.sajjad@icp.edu.pk).
