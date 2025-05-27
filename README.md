# YOLOv12 Segmentation Training

This repository contains a Jupyter notebook (`yolo12x-seg.ipynb`) that demonstrates how to train YOLOv12 segmentation models using the `ultralytics` library.

## Overview

The notebook provides a complete workflow for training instance segmentation models with YOLOv12, including:

* Dataset preparation and configuration
* Model setup using YOLOv12 architecture
* Training configuration and execution
* Evaluation of model performance
* Export and inference examples

## Prerequisites

* Python 3.8+ (recommended 3.11)
* NVIDIA GPU(s) with CUDA support
* Jupyter Notebook environment
* `pip` (Python package installer)

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/username/yolov12-segmentation-training.git
   cd yolov12-segmentation-training
   ```

2. **Set up your environment:**
   ```bash
   # Create and activate virtual environment (recommended)
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   
   # Install required packages
   pip install ultralytics jupyter matplotlib
   ```

3. **Launch the notebook:**
   ```bash
   jupyter notebook yolo12x-seg.ipynb
   ```

4. **Follow the steps in the notebook** to prepare your dataset, configure training parameters, and run the training process.

## Key Training Parameters

The notebook uses the following key parameters for training (which you can modify according to your needs):

```python
# Example training configuration
!yolo task=segment mode=train \
    model=yolo12x-seg.yaml \
    data=path/to/your/data.yaml \
    epochs=1500 \
    device=0,1 \
    batch=16 \
    workers=16 \
    seed=101 \
    patience=300
```

| Parameter | Description |
|-----------|-------------|
| `model` | YOLOv12 segmentation model configuration |
| `data` | Path to your dataset YAML file |
| `epochs` | Number of training epochs |
| `device` | GPU device indices to use |
| `batch` | Batch size per device |
| `patience` | Early stopping patience |

## Data Format

The notebook expects data in the standard YOLO format:

```
dataset/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
└── data.yaml
```

Where `data.yaml` follows this structure:
```yaml
train: path/to/your/dataset/images/train
val: path/to/your/dataset/images/val
nc: N  # number of classes
names: ['class1', 'class2', ..., 'classN']
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please open an issue or submit a pull request.
