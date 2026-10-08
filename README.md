# Sequential YOLO + Mitosis-DenseViT Pipeline

A two-stage deep learning pipeline for mitosis detection and classification in breast histopathology images:

1. **Detection** (`Custom_YOLOV_8.ipynb`): a YOLOv8-based detector (YOLO-det) that localizes candidate mitotic figures.
2. **Classification** (`Mitosis_Dense_ViT_.ipynb` and the `.py` files): **Mitosis-DenseViT**, a hybrid DenseNet121 + ViT-B/16 classifier that re-examines each candidate patch from stage 1 and labels it mitotic or non-mitotic.

This code accompanies the manuscript *"Sequential Deep Learning Framework Using YOLO, DenseNet, and ViT for Mitosis Detection and Classification,"* currently under review. Citation details will be added here once the paper is published.

## Pipeline overview

```
Histopathology image
      |
      v
Custom_YOLOV_8.ipynb        YOLOv8 (YOLO-det)
      |  candidate mitotic-figure patches
      v
Mitosis-DenseViT            DenseNet121 + ViT-B/16
      |
      v
Mitotic / Non-mitotic
```

## Repository contents

| File | Stage | Purpose |
|---|---|---|
| `Custom_YOLOV_8.ipynb` | Detection | YOLOv8 training notebook |
| `config.py` | Classification | Paths, architecture and training hyperparameters |
| `data.py` | Classification | Dataset construction and `tf.data` pipeline, including augmentation |
| `model.py` | Classification | Model definition (`build_densevit`) and freeze/unfreeze helpers |
| `train.py` | Classification | Two-phase training loop |
| `evaluate.py` | Classification | Test-set evaluation, confusion matrix, ROC curve |
| `main.py` | Classification | End-to-end entry point |
| `Mitosis_Dense_ViT_.ipynb` | Classification | Notebook version of the classification pipeline (e.g. for Colab) |
| `requirements.txt` | Both | Python dependencies |

## Installation

```bash
pip install -r requirements.txt
```

Developed with Python 3.10+, TensorFlow/Keras 3, Ultralytics YOLOv8, and the Hugging Face `transformers` library (used for pretrained ViT weight initialization).

## Detection stage (YOLOv8)

The training data for this stage was annotated and exported through Roboflow. The notebook downloads it with an API key supplied as an environment variable. **Never commit a real key to this repository.**

```bash
export ROBOFLOW_API_KEY="your_own_roboflow_api_key"
```

Replace `YOUR_DATASET_ID` in the notebook's download cell with your own Roboflow dataset/project ID, then run the cells in order: GPU check, install `ultralytics`, download the dataset, train.

## Classification stage (Mitosis-DenseViT)

Architecture:

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

DenseNet121 learns local morphological and textural features; the transformer encoder (initialized where possible from pretrained ViT-B/16 weights) learns long-range context between patches. Training uses a two-phase schedule: a frozen-backbone warmup, then fine-tuning of the last DenseNet block and the last few transformer blocks.

Usage:

1. Organize your data as:
   ```
   <root>/train/Positive_Mitosis/*.png
   <root>/train/Negative_Mitosis/*.png
   <root>/val/...
   <root>/test/...
   ```
2. Update the paths and settings in `config.py` for your environment. The defaults reflect the authors' own Colab/Google Drive setup and must be changed.
3. Run:
   ```bash
   python main.py
   ```

## Data availability

- **Public benchmark datasets** used to train and evaluate this pipeline are freely available from their original sources: [MITOS-12](http://ludo17.free.fr/mitos_2012/dataset.html), [MITOS-14](https://mitos-atypia-14.grand-challenge.org/), and [TUPAC-16](http://tupac.tue-image.nl/).
- **Custom pathologist-verified annotations** generated during this study are not included in this repository, because their release was not covered under the terms of the original pathologist collaboration.

## License

Released under the MIT License. See [LICENSE](LICENSE).

## Contact

Questions about this code can be directed to the corresponding author, Muhammad Sajjad (muhammad.sajjad@icp.edu.pk).
