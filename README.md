# Detection stage — YOLOv8 (YOLO-det)

This is the first stage of the pipeline: a YOLOv8-based detector that localizes candidate mitotic figures in breast histopathology images. Candidate patches produced here are passed to the [classification stage](../classification) for cell-level mitotic/non-mitotic classification.

## Setup

```bash
pip install ultralytics roboflow
```

## Dataset access

The training data for this stage was annotated and exported via Roboflow. The notebook downloads it using an API key passed as an environment variable — **never commit a real key to this repository**.

Set your own key before running the notebook:

```bash
export ROBOFLOW_API_KEY="your_own_roboflow_api_key"
```

Then replace `YOUR_DATASET_ID` in the notebook's download cell with your own Roboflow dataset/project ID. The original annotated dataset used in the paper is not publicly hosted; researchers wishing to reproduce this stage on the same data should contact the corresponding author (see the main [README](../README.md)).

## Usage

Open `Custom_YOLOV_8.ipynb` in Jupyter or Google Colab and run the cells in order:
1. GPU check
2. Install `ultralytics`
3. Download/prepare the dataset (after setting your API key as above)
4. Train the YOLOv8 model

Training configuration (epochs, image size, model variant) is set directly in the training cell — adjust as needed for your own dataset size and hardware.
