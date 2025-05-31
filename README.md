# Fall & Slip Detection Using ConvLSTM

This repository provides code and configuration files for training and evaluating a ConvLSTM-based fall-and-slip detection system using low-resolution thermal videos. The system distinguishes between high-risk falls and low-risk slips and can be extended to reconstruct smartphone sensor data (e.g., IMU, audio) and generate 3D spatial models for augmented-reality (AR) applications.

---

## Table of Contents

1. [Repository Structure](#repository-structure)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Configuration Files](#configuration-files)  
   - [Single-Environment Config (`config16-6.yaml`)](#single-environment-config)  
   - [Multi-Environment Config (`config2.yaml`)](#multi-environment-config)  
5. [Usage](#usage)  
   - [Training & Validation](#training--validation)  
6. [Config Field Descriptions](#config-field-descriptions)  
7. [GitHub Repository Link and Versioning](#github-repository-link-and-versioning)  
8. [Contact](#contact)

---

## Repository Structure

```
.
├── config/                     # Single-environment configuration
│   ├── config16-6.yaml         
├── config_multi/               # Multi-environment configuration
│   └── configZZ.yaml           
│
├── datasets/                    # Dataset files (from Hugging Face)
├── train_val_process.ipynb  # Notebook to run training & validation
├── model/                   # Model definitions (e.g., ConvLSTM, multi-branch)
├── dataset                # Custom Dataset classes (e.g., Dataset_multi2_2layer)
├── utils                   # Utility scripts (e.g., data loaders, loss functions)
│
├── README.md                    # This README file
└── requirements.txt             # Python dependencies
```

---

## Prerequisites

- **Python 3.10+**  
- **PyTorch 1.10+** (with CUDA support if using GPU)  
- **CUDA Toolkit** (compatible with your GPU and PyTorch version)  
- **Additional Python Libraries** (listed in `requirements.txt`):  
- **GPU (recommended)** for faster training. CPU-only mode is supported but will be significantly slower.

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/your-username/fall-slip-detection.git
   cd fall-slip-detection
   ```

2. **Create a virtual environment (optional but recommended)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Download or prepare your dataset**  
   - Place your thermal videos and annotation CSV files under `datasets/fall_dataset/` following the structure shown above.  
   - Update the paths in `config/config16-6.yaml` or `config/config2.yaml` to point to your local dataset directories.

---

## Configuration Files

This repository uses YAML-based configuration files to specify all parameters for training, validation, and evaluation. Two example configs are provided:

### Single-Environment Config (`config/config16-6.yaml`)

```yaml
base:
  filename: "/home/shisei/work/lstm/config/config16-6.yaml"
  experiment_name: "ConvLSTM_fall_slip"
  run_name: "YOLO+ConvL+Angle"
  output_csv_path: "/home/shisei/work/lstm/outputs/csvs/"
  output_img_path: "/home/shisei/work/lstm/outputs/images/"

train_set:
  annotation_csv_path: "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
  train_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/train/*"
  val_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/val/*"
  event_annotation_csv_path: "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
  event_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/val/*"
  classes:
    0: "usual"
    1: "falling"
    2: "fall"

parameters:
  batch_size: 16
  window_size: 32
  class_num: 3
  num_layers: 1
  hidden_dim: 100
  resnet_output_dim: 16
  lr: 0.0005
  image_f: true
  motion_f: false
  class_weight: true
  transform: false
  epochs: 5
  preprocess_f: false

features:
  - "cls_prob_sitting"
  - "cls_prob_lying"
  - "cls_prob_standing"
  - "angle_change_2"
  - "point_world_z"
```

- **Base Section**  
  - `filename`: Absolute path to this config file.  
  - `experiment_name`: Unique name for this experiment.  
  - `run_name`: Descriptive run identifier (e.g., which model variants).  
  - `output_csv_path`: Directory where CSV result files will be saved.  
  - `output_img_path`: Directory where output images/plots will be saved.

- **train_set Section**  
  - `annotation_csv_path`: Path to CSV containing ground-truth labels (start/end times, fall/slip labels).  
  - `train_video_dir` / `val_video_dir`: Glob pattern pointing to training/validation video files.  
  - `event_annotation_csv_path`: CSV file for event-level annotations (fall/slip).  
  - `event_video_dir`: Glob pattern for event video files (usually same as `val_video_dir`).  
  - `classes`: Dictionary mapping integer labels to class names (0 = usual, 1 = falling, 2 = fall).

- **parameters Section**  
  - `batch_size`: Number of video clips per training batch.  
  - `window_size`: Number of consecutive frames (time steps) fed to the ConvLSTM.  
  - `class_num`: Number of output classes (3 in this case: usual, falling, fall).  
  - `num_layers`: Number of ConvLSTM layers.  
  - `hidden_dim`: Hidden dimension size for the ConvLSTM.  
  - `resnet_output_dim`: Dimension of features extracted by the ResNet backbone.  
  - `lr`: Learning rate for the optimizer.  
  - `image_f`: Whether to use image-based features (true/false).  
  - `motion_f`: Whether to use motion-based features (true/false).  
  - `class_weight`: Whether to apply class weighting in the loss function.  
  - `transform`: Whether to apply data augmentation/transforms during training.  
  - `epochs`: Total number of training epochs.  
  - `preprocess_f`: Whether to run a preprocessing step before training (e.g., cropping, normalization).

- **features Section**  
  A list of feature names used as inputs to the model (e.g., sitting/lying probabilities from YOLO, angle change, world‐coordinate Z position).

---

### Multi-Environment Config (`config/config2.yaml`)

```yaml
base:
  filename: "/home/shisei/work/lstm/config/config2.yaml"
  experiment_name: "multi_dataset_label_fix0217"
  run_name: "LabelFix0217: 8Datasets + class3_RAW_2branch_2layer"
  output_csv_path: "/home/shisei/work/lstm/outputs/csvs/"
  output_img_path: "/home/shisei/work/lstm/outputs/images/"

run_description: "LabelFix0217: 8Datasets + class3_RAW_2branch_2layer"

train_set:
  annotation_csv_path: "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
  train_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/train/*"
  val_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/val/*"
  event_annotation_csv_path: "/mnt/d/mydatasets/fall_dataset/annotation_csv/fall_0713_annotation.csv"
  event_video_dir: "/mnt/d/mydatasets/fall_dataset/fall_0713_split/val/*"
  resnet_fix_params: 0
  icce_val: false
  classes:
    0: "usual"
    1: "emergency"
    2: "caution"
    3: "unknown"

sub_tasks:
  # Example for enabling specific sub-tasks (uncomment as needed)
  # "Gravity": ["Gravity_y_norm"]
  # "Accelerometer": ["Accelerometer_xyz_std_4s"]
  # "Orientation": ["Orientation_pitch", "Orientation_qx"]
  # "Gyroscope": ["Gyroscope_xyz_norm"]
  # "under_falling": ["under_falling"]

label_col: "video_label"
loss_func: "LossFunction_2Layer"     # Options: hierarchical, FocalLoss, etc.
dataset_style: "Dataset_multi2_2layer"

use_data:
  - "corner1_high1_1126"
  - "corner1_low1_1126"
  - "corner2_high_1126"
  - "corner2_low_1126"
  - "corner3_high_1203"
  - "corner3_low_1203"
  - "corner4_high_1203"
  - "corner4_low1_1203"

cut_standing: True
model: "ConvLSTM_2branch_2layer"

criterion:
  main_task_weight: 1
  sub_task_weight: 1

parameters:
  batch_size: 16
  window_size: 32
  class_num: 3
  num_layers: 1
  hidden_dim: 100
  resnet_output_dim: 16
  lr: 0.0005
  class_weight: true
  epochs: 10
  preprocess_f: false
  augmentation: true
  brighten: false

features:
  - "cls_prob_sitting"
  - "cls_prob_lying"
  - "cls_prob_standing"
  # - "abs_angle_change"

val_video_list:
  corner1_high1_1126: [1, 7, 12, 17, 25]
  corner1_low1_1126:  [1, 7, 12, 17, 22]
  corner2_high_1126:  [1, 7, 12, 17, 22]
  corner2_low_1126:   [1, 7, 12, 17, 21]
  corner3_high_1203:  [1, 7, 12, 17, 23]
  corner3_low_1203:   [1, 7, 12, 17, 23]
  corner4_high_1203:  [1, 7, 12, 17, 21]
  corner4_low1_1203:  [1, 7, 12, 17, 22]
```

- **Base, train_set, parameters, and features** are similar to `config16-6.yaml`, but with additional fields:  
  - `run_description`: Short description of this multi-environment experiment.  
  - `train_set.resnet_fix_params`: Flag for freezing ResNet backbone layers (0 = no freeze).  
  - `train_set.icce_val`: Boolean toggle for ICCE validation mode.  
  - `sub_tasks`: Dictionary specifying auxiliary tasks and their associated feature lists (commented out by default).  
  - `label_col`: Column name in the annotation CSV that contains the overall video label.  
  - `loss_func`: Choice of loss function (e.g., `"LossFunction_2Layer"`).  
  - `dataset_style`: Custom Dataset class to use (e.g., `"Dataset_multi2_2layer"`).  
  - `use_data`: List of dataset identifiers (folder names) to include in training/validation.  
  - `cut_standing`: Boolean flag to enable preprocessing that removes standing frames.  
  - `model`: Name of the model architecture class to instantiate (e.g., `"ConvLSTM_2branch_2layer"`).  
  - `criterion.main_task_weight` / `criterion.sub_task_weight`: Weights for hierarchical or multi-task loss functions.  
  - `val_video_list`: Dictionary mapping each dataset identifier to a list of validation video indices.

---

## Usage

### Training & Validation

All training and validation routines are wrapped in the `train_val_process.ipynb` notebook. To start training with a specific configuration, follow these steps:

1. **Activate your virtual environment** (if not already active):  
   ```bash
   source venv/bin/activate
   ```

2. **Launch Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```
   Then open `src/train_val_process.ipynb` in your browser.

3. **In the first code cell**, import the processing function and call it with your desired config file. For example:  
   ```python
   from train_val_process import train_val_process

   # For single-environment experiment:
   train_val_process("config/config16-6.yaml")

   # For multi-environment experiment:
   train_val_process("config/config2.yaml")
   ```

4. **Monitor outputs**  
   - Training/validation metrics (loss, accuracy) will be printed in the notebook.  
   - Result CSV files (e.g., per-epoch evaluation metrics) are saved in the `outputs/csvs/` directory.  
   - Output images (e.g., confusion matrices, ROC curves) are saved in the `outputs/images/` directory.

---

## Config Field Descriptions

Below is a brief summary of the most important configuration fields. Refer to each config file for the complete list of fields.

- **`base.filename`**: Absolute path to this config YAML file.  
- **`base.experiment_name`**: Unique identifier for this experimental run.  
- **`base.run_name`**: Descriptive string indicating model variants or dataset conditions.  
- **`base.output_csv_path`** / **`base.output_img_path`**: Directories for saving CSV results and images.

- **`train_set.annotation_csv_path`**: Path to the annotation CSV for training.  
- **`train_set.train_video_dir`** / **`val_video_dir`**: Glob pattern pointing to training and validation video folders (e.g., `*/train/*`).  
- **`train_set.event_annotation_csv_path`** / **`event_video_dir`**: CSV and folder for event‐level (fall/slip) annotations.  
- **`train_set.classes`**: Dictionary mapping integer labels to class names (e.g., `0: "usual"`, `1: "falling"`, `2: "fall"`).

- **`parameters.batch_size`**: Number of clips in each minibatch.  
- **`parameters.window_size`**: Number of consecutive frames (temporal window length).  
- **`parameters.class_num`**: Total number of output classes.  
- **`parameters.num_layers`**: Number of ConvLSTM layers.  
- **`parameters.hidden_dim`**: Hidden dimension size for ConvLSTM.  
- **`parameters.resnet_output_dim`**: Output dimensionality of ResNet feature extractor.  
- **`parameters.lr`**: Learning rate.  
- **`parameters.image_f`**, **`parameters.motion_f`**: Boolean toggles for using image or motion features.  
- **`parameters.class_weight`**: Boolean toggle for applying class weights in loss function.  
- **`parameters.transform`**: Boolean toggle for data augmentations.  
- **`parameters.epochs`**: Total number of training epochs.  

- **`features`**: List of feature names to include (e.g., YOLO classification probabilities, angle change, world‐coordinate Z).

- **(Multi-Environment Only)**  
  - **`run_description`**: Short description for multi-environment experiments.  
  - **`train_set.resnet_fix_params`**: If set to non-zero, freeze ResNet backbone layers.  
  - **`train_set.icce_val`**: Toggle for ICCE validation mode.  
  - **`sub_tasks`**: Dictionary of auxiliary tasks + feature sets (comment/uncomment as needed).  
  - **`label_col`**: Column in annotation CSV for overall video label.  
  - **`loss_func`**: Name of the loss function class (e.g., `"LossFunction_2Layer"`).  
  - **`dataset_style`**: Name of the Dataset class to use (e.g., `"Dataset_multi2_2layer"`).  
  - **`use_data`**: List of dataset folder identifiers to include.  
  - **`cut_standing`**: Boolean to enable preprocessing step that removes standing frames.  
  - **`model`**: Name of the model architecture class to instantiate (e.g., `"ConvLSTM_2branch_2layer"`).  
  - **`criterion.main_task_weight`** / **`criterion.sub_task_weight`**: Loss‐weighting for hierarchical or multi-task training.  
  - **`val_video_list`**: Dictionary mapping each dataset identifier to a list of validation video indices.

---

## Datasets
  ```
  https://huggingface.co/datasets/Nshisei/multimodal_multiangle_fall_detection_dataset
  ```

## Contact

For questions or clarifications, please contact:

- **Shisei Nakamura**  
- Email: nakamura.shisei25@gmail.com  
- Affiliation: Graduate School of Information Science and Technology, Osaka University

---

Thank you for using this repository. We hope it accelerates your research on fall and slip detection using thermal imaging and related multi-modal approaches.
