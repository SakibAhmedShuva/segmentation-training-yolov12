# Segmentation Training with YOLOv12

This repository provides a setup for training YOLOv12 segmentation models using the `ultralytics` library. It is based on the workflow demonstrated in the included Jupyter notebook (`damage-29-yolov12-augmented-01.ipynb`), generalized for broader use cases.

## Features

* Utilizes the YOLOv12 architecture for instance segmentation
* Leverages the `ultralytics` Python package for training and utilities
* Supports multi-GPU training
* Configurable training parameters (epochs, batch size, patience, etc.)

## Prerequisites

* Python 3.8+ (Notebook used 3.11)
* NVIDIA GPU(s) with CUDA support
* `pip` (Python package installer)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SakibAhmedShuva/segmentation-training-yolov12.git
   cd segmentation-training-yolov12
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *This will install `ultralytics` and its dependencies, including compatible versions of `torch` and `torchvision`.*

4. **Prepare your Dataset:**
   * Organize your custom dataset according to the YOLO format expected by `ultralytics`. See the [Ultralytics Datasets Guide](https://docs.ultralytics.com/datasets/) for details.
   * Create a `data.yaml` file defining the dataset paths (train, val), number of classes (`nc`), and class names (`names`).

   **Example `data.yaml` structure:**
   ```yaml
   train: path/to/your/dataset/images/train
   val: path/to/your/dataset/images/val
   # test: [optional] path/to/your/dataset/images/test

   # Classes
   nc: N  # number of classes
   names: ['class1', 'class2', ..., 'classN']  # List of class names
   ```
   *Ensure the corresponding label files exist in `path/to/your/dataset/labels/train`, `path/to/your/dataset/labels/val`, etc.*

5. **Prepare Model Configuration:**
   * This project uses the `yolo12x-seg.yaml` configuration as a base. Ensure this file (or your desired YOLOv12 variant configuration) is available. You can typically let `ultralytics` handle downloading standard configurations.

## Training

Execute the training process using the `yolo` command-line interface. Adjust parameters as needed for your specific hardware and dataset.

The command used in the source notebook (adapt `data` path and other parameters as needed):

```bash
yolo task=segment mode=train \
    model=yolo12x-seg.yaml \
    data=path/to/your/data.yaml \
    epochs=1500 \
    device=0,1 \
    batch=16 \
    workers=16 \
    seed=101 \
    patience=300 \
    # project=runs/segment  # Optional: specify project directory
    # name=train_experiment_1  # Optional: specify experiment name
    # resume=True  # Optional: resume from the last checkpoint
```

### Training Parameters Explained

| Parameter | Description |
|-----------|-------------|
| `task=segment` | Specifies the segmentation task |
| `mode=train` | Specifies training mode |
| `model=yolo12x-seg.yaml` | Path to model configuration file or standard model name |
| `data` | Path to your data.yaml file |
| `epochs` | Total number of training epochs |
| `device` | GPU devices (e.g., 0 for single GPU, 0,1 for two GPUs, cpu for CPU) |
| `batch` | Batch size per device (effective batch size = batch × number_of_devices) |
| `workers` | Number of worker threads for data loading |
| `seed` | Random seed for reproducibility |
| `patience` | Epochs to wait for improvement before early stopping |
| `resume=True` | Add to resume training from the latest checkpoint |

## Results

Training results, including weights (`best.pt`, `last.pt`), logs, and validation metrics, will be saved in a directory structure, typically under `runs/segment/train/` (unless customized with `project` and `name` arguments).

## Included Notebook

The Jupyter notebook `damage-29-yolov12-augmented-01.ipynb` contains the original experimental workflow this documentation is based on. It includes specific data download and environment setup steps relevant to that particular run.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Sakib Ahmed Shuva - [GitHub Profile](https://github.com/SakibAhmedShuva)
